#
#//  xgbScip.py
#//  heteroknockoffpy
#//
"""
    Sequential Conditional Independence Procedure (SCIP) knockoff generation
    using sequential xgboost models -- a drop-in alternative backend to
    rbridge's ranger-based implementation (scripts/scip.knockoffs.R).

    Generates knockoffs sequentially: each column's knockoff is conditioned on
    all original variables plus all previously generated knockoffs.

    For categorical columns: fit an xgboost.XGBClassifier, then sample
    knockoff categories via row-wise weighted draws from predict_proba
    (utilities.choices_from_weights).
    For numeric columns: fit an xgboost.XGBRegressor, generate a knockoff via
    conditional residual perturbation (normal draw or permutation).

    xgboost sklearn-API constructor kwargs (forwarded from every public
    function's **kwargs to XGBRegressor/XGBClassifier):
      https://xgboost.readthedocs.io/en/latest/python/python_api.html
    Native categorical-feature support (enable_categorical/tree_method):
      https://xgboost.readthedocs.io/en/latest/tutorials/categorical.html

    The most relevant kwargs (forwarded as `model_kwargs`/plain kwargs to the
    XGBRegressor/XGBClassifier constructor, mirroring xgbImportances.py's
    convention):
      - max_depth (int): Maximum tree depth per boosting round. Deeper trees
        fit more complex interactions but overfit faster.
      - learning_rate (float, xgb's "eta"): Step-size shrinkage applied to
        each boosting round's leaf weights.
      - min_child_weight (float): Minimum sum of instance Hessian weight
        needed in a child to keep splitting; larger values make the trees
        more conservative.
      - subsample (float): Fraction of training rows subsampled per boosting round.
      - colsample_bytree (float): Fraction of columns subsampled when
        constructing each tree.
      - reg_alpha (float): L1 regularization on leaf weights.
      - reg_lambda (float): L2 regularization on leaf weights.
      - gamma (float): Minimum loss reduction required to make a further
        split (larger = more conservative).
      - n_estimators (int): Number of boosting rounds.
    All of the above are omitted from the constructor call when not passed in
    kwargs, so xgboost's own defaults apply.
"""

from .utilities import DataFrameLike, _resolve_df, choices_from_weights

import numpy as np
import polars as pl
import pandas as pd
import xgboost

from typing import Literal


def _make_regressor(
    rng: np.random.Generator | None,
    **model_kwargs,
    ) -> xgboost.XGBRegressor:
    """
        :param model_kwargs: Forwarded to xgboost.XGBRegressor -- most
            relevantly max_depth, learning_rate, min_child_weight, subsample,
            colsample_bytree, reg_alpha, reg_lambda, gamma, n_estimators. See
            the module docstring for what each controls, and
            https://xgboost.readthedocs.io/en/latest/python/python_api.html
            for the full parameter reference.
    """
    if rng is not None:
        model_kwargs = dict( model_kwargs )
        model_kwargs.setdefault( 'random_state', rng )
    #
    return xgboost.XGBRegressor(
        objective = 'reg:squarederror',
        tree_method = 'hist',
        enable_categorical = True,
        **model_kwargs,
    )
#/def _make_regressor


def _make_classifier(
    rng: np.random.Generator | None,
    **model_kwargs,
    ) -> xgboost.XGBClassifier:
    """
        :param model_kwargs: Forwarded to xgboost.XGBClassifier -- see
            `_make_regressor` and the module docstring for the relevant kwargs.
    """
    if rng is not None:
        model_kwargs = dict( model_kwargs )
        model_kwargs.setdefault( 'random_state', rng )
    #
    return xgboost.XGBClassifier(
        tree_method = 'hist',
        enable_categorical = True,
        **model_kwargs,
    )
#/def _make_classifier


def _fit_probability_forest(
    scip_pd: pd.DataFrame,
    col: str,
    categories: list[ str ],
    rng: np.random.Generator | None,
    model_kwargs: dict,
    ) -> np.ndarray:
    """
        Fit an XGBClassifier predicting factor column `col` from all other
        columns (which may already contain knockoff columns). `categories` is
        the sorted, fixed set of category labels; classes not present in the
        fitted data still receive a zero-probability column so the returned
        matrix always has shape (n, len(categories)) -- mirrors ranger's
        fixed-levels probability matrix in scip.knockoffs.R.

        Uses tree_method='hist' + enable_categorical=True so the other
        (already-`category`-dtype) explanatory columns are handled by
        xgboost's native categorical splits rather than manual OHE:
        https://xgboost.readthedocs.io/en/latest/tutorials/categorical.html

        Returns: n x k probability matrix, columns ordered by `categories`.
    """
    X_expl: pd.DataFrame = scip_pd.drop( columns = [ col ] )
    y_codes: np.ndarray = pd.Categorical( scip_pd[ col ], categories = categories ).codes.astype( np.int64 )

    model = _make_classifier( rng, **model_kwargs )
    model.fit( X_expl, y_codes )

    proba: np.ndarray = model.predict_proba( X_expl )
    # xgboost only emits columns for classes seen during fit; re-expand to the
    # full fixed category set (matching ranger's fixed-levels probability matrix).
    full_proba: np.ndarray = np.zeros( ( proba.shape[0], len( categories ) ) )
    for i, cls in enumerate( model.classes_ ):
        full_proba[ :, int( cls ) ] = proba[ :, i ]
    #

    # Row-normalize: predict_proba rows already sum to ~1 over the classes
    # xgboost saw during fit, but not exactly in floating point, and
    # choices_from_weights (np.random.Generator.choice) requires an exact sum
    # of 1 -- mirrors the row-normalization guard in .scip.make_choices (R).
    row_sums: np.ndarray = full_proba.sum( axis = 1, keepdims = True )
    row_sums[ row_sums == 0 ] = 1.0
    full_proba = full_proba / row_sums

    return full_proba
#/def _fit_probability_forest


def _fit_regression_forest(
    scip_pd: pd.DataFrame,
    col: str,
    rng: np.random.Generator | None,
    model_kwargs: dict,
    ) -> np.ndarray:
    """
        Fit an XGBRegressor predicting numeric column `col` from all other
        columns (which may already contain knockoff columns).

        Returns: numeric vector of length n (conditional expectations).
    """
    X_expl: pd.DataFrame = scip_pd.drop( columns = [ col ] )
    y: np.ndarray = scip_pd[ col ].to_numpy().astype( np.float64 )

    model = _make_regressor( rng, **model_kwargs )
    model.fit( X_expl, y )

    return model.predict( X_expl )
#/def _fit_regression_forest


def get_forest_conditional_expectations(
    X: DataFrameLike,
    verbose: int = 0,
    verbose_prefix: str = '',
    rng: np.random.Generator | None = None,
    **kwargs,
    ) -> pl.DataFrame:
    """
        Uses sequential (one-per-column, non-chained) xgboost regressors to
        get conditional expectations for each numeric column of X, each fit
        on all *other original* columns. Mirrors
        `rbridge.get_forest_conditional_expectations`, but with
        xgboost.XGBRegressor in place of ranger::ranger.

        :param kwargs: Forwarded to xgboost.XGBRegressor -- see the module
            docstring for the relevant kwargs (max_depth, learning_rate,
            min_child_weight, subsample, colsample_bytree, reg_alpha,
            reg_lambda, gamma, n_estimators), and
            https://xgboost.readthedocs.io/en/latest/python/python_api.html
            for the full reference.
        :returns: DataFrame with non-categorical columns of X replaced by
            their conditional expectations
    """
    X = _resolve_df( X )
    X_pd: pd.DataFrame = X.to_pandas()

    numeric_columns: tuple[ str,... ] = tuple(
        col for col, dtype in X.schema.items() if dtype != pl.Categorical
    )

    if verbose > 0:
        print( verbose_prefix + 'Fitting xgboost conditional expectations' )
    #

    conditional_expectations_dict: dict[ str, np.ndarray ] = {
        col: _fit_regression_forest( X_pd, col, rng, kwargs )
        for col in numeric_columns
    }

    return pl.DataFrame(
        conditional_expectations_dict,
        schema = {
            col: dtype for col, dtype in X.schema.items() if dtype != pl.Categorical
        },
    )
#/def get_forest_conditional_expectations


def get_knockoffs_with_Xk_numeric(
    X: DataFrameLike,
    Xk_numeric: np.ndarray,
    rng: np.random.Generator,
    verbose: int = 0,
    verbose_prefix: str = '',
    **kwargs,
    ) -> pl.DataFrame:
    """
        Categorical-only SCIP: numeric knockoffs are pre-provided; only
        categorical knockoffs are generated here, sequentially, via xgboost.
        Mirrors `rbridge.get_knockoffs_with_Xk_numeric`.

        :param kwargs: Forwarded to xgboost.XGBClassifier -- see the module
            docstring for the relevant kwargs (max_depth, learning_rate,
            min_child_weight, subsample, colsample_bytree, reg_alpha,
            reg_lambda, gamma, n_estimators).
    """
    X = _resolve_df( X )

    numeric_columns: tuple[ str,... ] = tuple(
        col for col, dtype in X.schema.items() if dtype != pl.Categorical
    )
    assert len( numeric_columns ) == Xk_numeric.shape[1]

    scip_pd: pd.DataFrame = X.to_pandas()
    for j, col in enumerate( numeric_columns ):
        scip_pd[ col + '~' ] = Xk_numeric[ :, j ]
    #

    for col, dtype in X.schema.items():
        if dtype != pl.Categorical:
            continue
        #
        ko: str = col + '~'
        categories: list[ str ] = sorted( X[ col ].cat.get_categories().to_list() )

        if verbose > 0:
            print( verbose_prefix + 'xgb SCIP categorical column: {}'.format( col ) )
        #

        probs: np.ndarray = _fit_probability_forest( scip_pd, col, categories, rng, kwargs )
        indices: np.ndarray = choices_from_weights( probs, rng = rng )
        scip_pd[ ko ] = pd.Categorical.from_codes( indices, categories = categories )
    #/for col, dtype in X.schema.items()

    Xk_pd: pd.DataFrame = scip_pd[ [ col + '~' for col in X.columns ] ]
    Xk_pd.columns = list( X.columns )

    return pl.from_pandas( Xk_pd ).cast(
        { col: dtype for col, dtype in X.schema.items() }
    )
#/def get_knockoffs_with_Xk_numeric


def get_knockoffs_SCIP(
    X: DataFrameLike,
    rng: np.random.Generator,
    residuals_method: Literal['normal','permute',] = 'normal',
    verbose: int = 0,
    verbose_prefix: str = '',
    **kwargs,
    ) -> pl.DataFrame:
    """
        Full sequential SCIP knockoff generation (numeric + categorical),
        using sequential xgboost models as a drop-in alternative to
        `rbridge.get_knockoffs_SCIP`.

        Each column's knockoff conditions on all original columns plus all
        previously generated knockoffs (the scip frame grows column-by-column,
        left to right, matching scripts/scip.knockoffs.R's `scip.knockoffs`).

        :param residuals_method: "normal" (default) -- knockoff residual is
            drawn from N(0, sd(residuals, ddof=1)) via `rng.normal` -- or
            "permute" -- knockoff residual is `rng.permutation(residuals)`.
        :param kwargs: Forwarded to xgboost.XGBRegressor (numeric columns) /
            xgboost.XGBClassifier (categorical columns) -- see the module
            docstring for the relevant kwargs (max_depth, learning_rate,
            min_child_weight, subsample, colsample_bytree, reg_alpha,
            reg_lambda, gamma, n_estimators), and
            https://xgboost.readthedocs.io/en/latest/python/python_api.html
            for the full reference.
    """
    X = _resolve_df( X )

    if residuals_method not in ( 'normal', 'permute' ):
        raise ValueError( "Unrecognized residuals_method={}".format( residuals_method ) )
    #

    scip_pd: pd.DataFrame = X.to_pandas()

    for col, dtype in X.schema.items():
        ko: str = col + '~'

        if verbose > 0:
            print( verbose_prefix + 'xgb SCIP column: {}'.format( col ) )
        #

        if dtype == pl.Categorical:
            categories: list[ str ] = sorted( X[ col ].cat.get_categories().to_list() )
            probs: np.ndarray = _fit_probability_forest( scip_pd, col, categories, rng, kwargs )
            indices: np.ndarray = choices_from_weights( probs, rng = rng )
            scip_pd[ ko ] = pd.Categorical.from_codes( indices, categories = categories )
        #
        else:
            cond_exp: np.ndarray = _fit_regression_forest( scip_pd, col, rng, kwargs )
            residuals: np.ndarray = X[ col ].to_numpy().astype( np.float64 ) - cond_exp

            if residuals_method == 'normal':
                scip_pd[ ko ] = cond_exp + rng.normal( 0, residuals.std( ddof = 1 ), size = len( residuals ) )
            #
            else:
                scip_pd[ ko ] = cond_exp + rng.permutation( residuals )
            #/if residuals_method == 'normal'/else
        #/if dtype == pl.Categorical/else
    #/for col, dtype in X.schema.items()

    Xk_pd: pd.DataFrame = scip_pd[ [ col + '~' for col in X.columns ] ]
    Xk_pd.columns = list( X.columns )

    return pl.from_pandas( Xk_pd ).cast(
        { col: dtype for col, dtype in X.schema.items() }
    )
#/def get_knockoffs_SCIP
