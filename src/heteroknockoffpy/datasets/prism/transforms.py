#
#//  transforms.py
#//  heteroknockoffpy
#//

"""Conversions between the PRISM data's on-disk shapes and heteroknockoffpy calls.

- :func:`parameters_to_dict` -- a bundle's long-format parameter rows as
  ``{group: {name: value}}``, pipeline bookkeeping stripped.
- :func:`heteroknockoffpy_kwargs` -- a trial's ``importances`` parameter
  group reshaped into keyword arguments for the matching
  ``heteroknockoffpy.importance.*`` function.
- :func:`importance_function` -- that function, looked up by
  ``importance_method``.
- :func:`relevant_features` -- ground-truth relevant feature indices,
  parsed from a response's generating expression (``fn_str``).
- :func:`split_importance_array` / :func:`combine_importance_arrays` --
  convert between heteroknockoffpy's flat ``(2*p,)`` importance array and
  the stored ``importance``/``importance_k`` pair.
- :func:`importances_to_dict` / :func:`selections_to_dict` -- one bundle's
  ``importances``/``selections`` frame as plain dicts.

**A trial can hold more than one** ``importance_method``. ``torch_grip2``
and ``torch_prism`` are two scores read off the *same* trained model, so
they're written together into one bundle (a third to a half of
``continuous``/``count``/``categorical`` trials). Always group by, or
iterate over, ``importance_method``; never take the first row's method as
*the* method. Likewise, ``parameters["importances"]["method"]`` names the
shared model fit (``"torch_prism_grip2"``), not either importance method --
don't dispatch on it.
"""

import re
from collections.abc import Callable
from typing import Any, TypedDict

import numpy as np
import polars as pl

_PARAMETER_VALUE_COLUMNS = (
    "par_val_int64",
    "par_val_float64",
    "par_val_string",
    "par_val_boolean",
    "par_val_list_float64",
    "par_val_list_int64",
    "par_val_list_string",
    "par_val_list_list_float64",
    "par_val_list_list_int64",
)

# par_group values holding pipeline bookkeeping (config hashes, trial
# indices, file pointers) rather than research-meaningful parameters.
IDENTIFIER_PARAM_GROUPS = ( "_meta", "_prereq" )

# importance_method value -> heteroknockoffpy.importance function name.
IMPORTANCE_FUNCTIONS: dict[str, str] = {
    "LASSO": "lassoImportances",
    "ranger_gini": "rangerGiniImportances",
    "torch_grip2": "grip2Importances",
    "torch_prism": "torchPrismImportances",
    "xgb_prism": "xgbPrismImportances",
    "xgb_score": "xgbImportances",
}


class ImportanceEntry( TypedDict ):
    importance: dict[int, float]
    importance_k: dict[int, float]
#/class ImportanceEntry


class SelectionEntry( TypedDict ):
    fdr_target: float
    selected: dict[int, bool]
#/class SelectionEntry


def _parameter_value( row: dict[str, Any] ) -> Any:
    for column in _PARAMETER_VALUE_COLUMNS:
        value = row.get( column )
        if value is not None:
            return value
        #
    #
    return None
#/def _parameter_value


def parameters_to_dict( df: pl.DataFrame ) -> dict[str, dict[str, Any]]:
    """One bundle's ``parameters`` rows as ``{par_group: {par_name: value}}``.

    Each value is pulled out of whichever ``par_val_*`` column holds it.
    ``_meta``/``_prereq`` rows (pipeline identifiers, not research
    parameters) are dropped. A ``(group, name)`` pair appearing more than
    once collapses to a list in ``par_idx`` order.
    """
    df = df.filter( ~pl.col( "par_group" ).is_in( IDENTIFIER_PARAM_GROUPS ) ).sort( "par_idx" )

    result: dict[str, dict[str, Any]] = {}
    for ( group, name ), sub in df.group_by( ( "par_group", "par_name" ), maintain_order = True ):
        values = [ _parameter_value( row ) for row in sub.iter_rows( named = True ) ]
        result.setdefault( group, {} )[name] = values[0] if len( values ) == 1 else values
    #
    return result
#/def parameters_to_dict


def importances_to_dict( df: pl.DataFrame ) -> dict[str, ImportanceEntry]:
    """One bundle's ``importances`` frame as ``{importance_method: {importance, importance_k}}``."""
    result: dict[str, ImportanceEntry] = {}
    for ( method, ), sub in df.group_by( "importance_method", maintain_order = True ):
        result[method] = {
            "importance": dict( zip( sub["feature_idx"], sub["importance"] ) ),
            "importance_k": dict( zip( sub["feature_idx"], sub["importance_k"] ) ),
        }
    #
    return result
#/def importances_to_dict


def selections_to_dict( df: pl.DataFrame ) -> dict[str, SelectionEntry]:
    """One bundle's ``selections`` frame as ``{importance_method: {fdr_target, selected}}``."""
    result: dict[str, SelectionEntry] = {}
    for ( method, ), sub in df.group_by( "importance_method", maintain_order = True ):
        result[method] = {
            "fdr_target": sub["fdr_target"][0],
            "selected": dict( zip( sub["feature_idx"], sub["selected"] ) ),
        }
    #
    return result
#/def selections_to_dict


_COL_REF = re.compile( r"col\((\d+)\)" )


def relevant_features( fn_str: str ) -> list[int]:
    """Ground-truth relevant feature indices in a response's generating expression.

    ``fn_str`` (the ``y`` group's ``fn_str`` parameter) is an expression over
    ``col(i)``; every column it references is truly relevant, every other
    column is null. E.g.
    ``"(0.99*col(2)*col(7)) + (1.00*col(13)*catmap(col(23), ...))"`` -> ``[2, 7, 13, 23]``.
    """
    return sorted( { int( i ) for i in _COL_REF.findall( fn_str ) } )
#/def relevant_features


def split_importance_array( arr: np.ndarray ) -> tuple[np.ndarray, np.ndarray]:
    """Split heteroknockoffpy's flat ``(2*p,)`` importance array into ``(importance, importance_k)``.

    The first ``p`` entries score ``X``'s columns, the last ``p`` their
    knockoff copies. Inverse of :func:`combine_importance_arrays`.
    """
    p = len( arr ) // 2
    return arr[:p], arr[p:]
#/def split_importance_array


def combine_importance_arrays( importance: np.ndarray, importance_k: np.ndarray ) -> np.ndarray:
    """``(importance, importance_k)`` -> heteroknockoffpy's flat ``(2*p,)`` array.

    The shape every ``heteroknockoffpy.importance.*`` function returns and
    :func:`heteroknockoffpy.importance.wFromImportances` expects. Inverse
    of :func:`split_importance_array`.
    """
    return np.concatenate( [ np.asarray( importance ), np.asarray( importance_k ) ] )
#/def combine_importance_arrays


def importance_function( importance_method: str ) -> Callable[..., np.ndarray]:
    """The ``heteroknockoffpy.importance`` function that produced ``importance_method``.

    Imported lazily, so loading data never pulls in torch/xgboost/R.
    """
    try:
        name = IMPORTANCE_FUNCTIONS[importance_method]
    except KeyError:
        raise ValueError(
            f"unknown importance_method {importance_method!r}; known: {sorted( IMPORTANCE_FUNCTIONS )}"
        ) from None
    #
    from heteroknockoffpy import importance

    return getattr( importance, name )
#/def importance_function


# xgboost model-constructor hyperparameters, which xgbPrismImportances /
# xgbImportances want nested under `model_kwargs` (passed top-level they're
# forwarded to XGBRegressor.fit() instead and raise TypeError there).
_XGB_MODEL_KWARGS = frozenset( {
    "max_depth", "learning_rate", "min_child_weight", "subsample",
    "colsample_bytree", "reg_alpha", "reg_lambda", "gamma", "n_estimators",
} )

# grip2Importances / torchPrismImportances take `layers: Sequence[int]`,
# built from these three stored values, instead of the values themselves.
_LAYERS_METHODS = frozenset( { "torch_grip2", "torch_prism" } )
_LAYERS_KEYS = ( "first_layer_ratio", "layer_count", "layer_ratio" )

# Logged alongside the hyperparameters but never real keyword arguments.
_NON_KWARG_KEYS = frozenset( { "method", "standardize_y", "local_grad_method" } )


def heteroknockoffpy_kwargs( importance_method: str, raw_params: dict[str, Any], p: int ) -> dict[str, Any]:
    """A trial's ``importances`` parameter group as keyword arguments for
    :func:`importance_function` ``(importance_method)``.

    - Bookkeeping keys (``method``, ``standardize_y``, ``local_grad_method``)
      are dropped.
    - ``torch_grip2``/``torch_prism``: ``first_layer_ratio``/``layer_count``/
      ``layer_ratio`` become ``layers``, built exactly as the generating
      pipeline did: ``[max(1, round(p * first_layer_ratio))]``, then
      ``layer_count - 1`` more entries each ``max(1, round(previous * layer_ratio))``.
    - ``xgb_prism``/``xgb_score``: xgboost constructor hyperparameters are
      nested under ``model_kwargs``.
    - ``LASSO``/``ranger_gini``: forwarded as-is.

    :param p: the ``X`` bundle's feature count (``X.width``); only used to
        build ``layers``.
    """
    kwargs = { k: v for k, v in raw_params.items() if k not in _NON_KWARG_KEYS }

    if importance_method in _LAYERS_METHODS:
        first_layer_ratio, layer_count, layer_ratio = ( kwargs.pop( k ) for k in _LAYERS_KEYS )
        layers = [ max( 1, round( p * first_layer_ratio ) ) ]
        for _ in range( layer_count - 1 ):
            layers.append( max( 1, round( layers[-1] * layer_ratio ) ) )
        #
        kwargs["layers"] = layers
    #
    elif importance_method in ( "xgb_prism", "xgb_score" ):
        kwargs["model_kwargs"] = { k: kwargs.pop( k ) for k in list( kwargs ) if k in _XGB_MODEL_KWARGS }
    #
    return kwargs
#/def heteroknockoffpy_kwargs
#
