from . import utilities
from .utilities import OutcomeDescriptor, DataFrameLike, SeriesOrDataFrameLike, _resolve_df, _resolve_y

import numpy as np
import polars as pl
import torch

from typing import Callable, Iterable, Literal, Self, Sequence

def _localGrad_forNumeric_t(
    j: int,
    X_t: torch.Tensor,
    model: 'object',
    bandwidth: float,
    inv_cov_t: torch.Tensor | None = None,
    drop_first_y: bool = True,
    ) -> torch.Tensor:
    X_minus = X_t.clone()
    X_minus[:, j] -= bandwidth
    X_plus = X_t.clone()
    X_plus[:, j] += bandwidth
    local_grad = ( model.predict_t( X_plus ) - model.predict_t( X_minus ) ) / ( 2.0 * bandwidth )
    if inv_cov_t is None:
        return local_grad.reshape( -1 )
    #
    if drop_first_y:
        local_grad = local_grad[:, 1:] - local_grad[:, 0:1]
    return torch.einsum( 'nk,kl,nl->n', local_grad, inv_cov_t, local_grad )
#/def _localGrad_forNumeric_t


def _localGrad_forCategories_t(
    j: list[ int ],
    X_t: torch.Tensor,
    model: 'object',
    drop_first: bool,
    inv_cov_t: torch.Tensor | None = None,
    drop_first_y: bool = True,
    ohe_vals: dict[ int, tuple[ float, float ] ] | None = None,
    ) -> torch.Tensor:

    local_grad: torch.Tensor

    if ohe_vals is not None:
        # Normalized plug-in: (f(active_h) - f(reference)) / spacing_h.
        # Reference is always "all columns in j at norm0"; drop_first is irrelevant here
        # because we never rely on it to define the reference — we compute it directly.
        ref_X = X_t.clone()
        for k in j:
            ref_X[:, k] = ohe_vals[ k ][ 0 ]
        ref_pred = model.predict_t( ref_X )                    # (n, output_dim)

        scaled_diffs: list[ torch.Tensor ] = []
        for h in range( len( j ) ):
            act_X = X_t.clone()
            for k in j:
                act_X[:, k] = ohe_vals[ k ][ 0 ]
            act_X[:, j[ h ] ] = ohe_vals[ j[ h ] ][ 1 ]
            spacing_h = ohe_vals[ j[ h ] ][ 1 ] - ohe_vals[ j[ h ] ][ 0 ]
            scaled_diffs.append( ( model.predict_t( act_X ) - ref_pred ) / spacing_h )
        #

        if len( scaled_diffs ) == 1:
            local_grad = scaled_diffs[ 0 ]
        else:
            # Pick the category with the largest absolute derivative, preserving sign.
            stack = torch.stack( scaled_diffs, dim=2 )         # (n, output_dim, len(j))
            idx   = stack.abs().argmax( dim=2, keepdim=True )
            local_grad = stack.gather( dim=2, index=idx ).squeeze( 2 )
        #
    else:
        # Legacy unnormalized plug-in: evaluate at 0/1, return amax - amin across all states.
        preds: list[ torch.Tensor ] = []
        for h in range( len( j ) ):
            _X = X_t.clone()
            _X[:, j ] = 0.0
            _X[:, j[ h ] ] = 1.0
            preds.append( model.predict_t( _X ) )
        #
        if drop_first:
            _X = X_t.clone()
            _X[:, j ] = 0.0
            preds.append( model.predict_t( _X ) )
        #
        y_out = torch.stack( preds, dim=2 )                    # (n, output_dim, n_cats)
        local_grad = y_out.amax( dim=2 ) - y_out.amin( dim=2 )
    #

    if inv_cov_t is None:
        return local_grad.reshape( -1 )
    #
    if drop_first_y:
        local_grad = local_grad[:, 1:] - local_grad[:, 0:1]
    return torch.einsum( 'nk,kl,nl->n', local_grad, inv_cov_t, local_grad )
#/def _localGrad_forCategories_t


def _prismImportances_t(
    model: 'object',
    X_all_t: torch.Tensor,
    oheDict: dict,
    local_grad_method: str,
    bandwidth: float | None,
    exponent: float,
    drop_first: bool = True,
    inv_cov_t: torch.Tensor | None = None,
    cat_ohe_vals: dict[ int, tuple[ float, float ] ] | None = None,
    ) -> torch.Tensor:
    """
    Tensor-native PRISM importance computation. Returns shape (p_out,) tensor.
    model must have predict_t and auto_diff_t methods.
    """
    n = X_all_t.shape[0]
    p_out = len( oheDict )

    if local_grad_method == 'auto_diff':
        auto_diff_full_t: torch.Tensor = model.auto_diff_t( X_all_t )  # (n, p_ohe)
    elif local_grad_method == 'bandwidth':
        if bandwidth is None:
            bandwidth = float( n ** -0.2 )
    else:
        raise ValueError( "Unrecognized local_grad_method='{}'".format( local_grad_method ) )
    #

    localGrad_t = torch.zeros( n, p_out, device=X_all_t.device )
    for j_out, col in enumerate( oheDict ):
        col_idx = oheDict[ col ]
        if isinstance( col_idx, int ):
            # numeric
            if local_grad_method == 'auto_diff':
                localGrad_t[:, j_out] = auto_diff_full_t[:, col_idx].reshape( -1 )
            else:
                localGrad_t[:, j_out] = _localGrad_forNumeric_t(
                    j = col_idx,
                    X_t = X_all_t,
                    model = model,
                    bandwidth = bandwidth,
                    inv_cov_t = inv_cov_t,
                )
        else:
            # categorical input variable — always plug-in regardless of local_grad_method
            localGrad_t[:, j_out] = _localGrad_forCategories_t(
                j = list( col_idx ),
                X_t = X_all_t,
                model = model,
                drop_first = drop_first,
                inv_cov_t = inv_cov_t,
                ohe_vals = cat_ohe_vals,
            )
        #
    #

    return ( torch.abs( localGrad_t ) ** exponent ).mean( dim=0 )
#/def _prismImportances_t


def _prismImportances_categorical_t(
    model: 'object',
    X_all_t: torch.Tensor,
    oheDict: dict,
    inv_cov_t: torch.Tensor,
    exponent: float,
    drop_first: bool = True,
    cat_ohe_vals: dict[ int, tuple[ float, float ] ] | None = None,
    ) -> torch.Tensor:
    """
    PRISM-G importance for a categorical outcome via full Jacobian + Mahalanobis distance.

    For each (sample, OHE input column): computes the Mahalanobis distance of the
    logit-contrast Jacobian, where contrasts are taken relative to the first category.
    Categorical input features use the plug-in estimate (_localGrad_forCategories_t).

    model must expose jacobian_t(X_t) -> (n, k, p_ohe).
    """
    n    = X_all_t.shape[0]
    p_out = len( oheDict )

    # Full per-sample Jacobian: (n, k, p_ohe)
    jac_t = model.jacobian_t( X_all_t )
    # Contrasts relative to first class: (n, k-1, p_ohe)
    jac_contrasts = jac_t[:, 1:, :] - jac_t[:, 0:1, :]
    # Mahalanobis per (sample, OHE column): (n, p_ohe)
    # mahal[n, p] = jac_contrasts[n, :, p]^T @ inv_cov @ jac_contrasts[n, :, p]
    mahal_t = torch.einsum( 'nmp, ml, nlp -> np', jac_contrasts, inv_cov_t, jac_contrasts )

    localGrad_t = torch.zeros( n, p_out, device=X_all_t.device )
    for j_out, col in enumerate( oheDict ):
        col_idx = oheDict[ col ]
        if isinstance( col_idx, int ):
            localGrad_t[:, j_out] = mahal_t[:, col_idx]
        else:
            localGrad_t[:, j_out] = _localGrad_forCategories_t(
                j        = list( col_idx ),
                X_t      = X_all_t,
                model    = model,
                drop_first = drop_first,
                inv_cov_t  = inv_cov_t,
                ohe_vals   = cat_ohe_vals,
            )
        #
    #

    return ( localGrad_t ** exponent ).mean( dim=0 )
#/def _prismImportances_categorical_t


def _prism_setup(
    X: DataFrameLike,
    Xk: DataFrameLike,
    y: SeriesOrDataFrameLike,
    layers: Sequence[int],
    outcome_type: Literal['continuous','count','categorical'] | None,
    drop_first: bool,
    ) -> tuple:
    """
    Shared setup for prismWImportances and prismGImportances.

    Returns (X_all_np, y_np, groups, oheDict, loss_func, output_dimension, outcomeDescriptor).
    """
    import torch.nn as nn

    X = _resolve_df(X)
    Xk = _resolve_df(Xk)
    y = _resolve_y(y)

    outcomeDescriptor: OutcomeDescriptor = OutcomeDescriptor.infer(
        y = y,
        outcome_type = outcome_type,
    )
    if outcomeDescriptor.outcome_dimension != 'single':
        raise TypeError("Joint outcomes unavailable")
    #

    assert all( X.schema[col] == Xk.schema[col] for col in X.columns )

    X_all: pl.DataFrame = pl.concat(
        (
            X,
            Xk.rename({ col: col + '~' for col in Xk.columns }),
        ),
        how = 'horizontal',
    )

    X_all_np: np.ndarray = utilities.get_ohe_np( X = X_all, drop_first = drop_first )
    oheDict: dict[ str, int | tuple[ int,... ] ] = utilities.get_oheDict( X = X_all, drop_first = drop_first )

    groups: list[ list[int] ] = [
        [oheDict[col]] if isinstance( oheDict[col], int ) else list( oheDict[col] )
        for col in oheDict
    ]

    y_np: np.ndarray
    if isinstance( y, pl.Series | pl.DataFrame ):
        y_np = y.to_numpy()
    else:
        y_np = np.asarray(y)
    #

    loss_func: nn.Module
    output_dimension: int

    if outcomeDescriptor.outcome_type == 'continuous':
        loss_func = nn.MSELoss()
        output_dimension = 1
        y_np = y_np.reshape(-1)
    #
    elif outcomeDescriptor.outcome_type == 'count':
        loss_func = nn.PoissonNLLLoss( log_input = True )
        output_dimension = 1
        y_np = y_np.reshape(-1).astype( np.int64 )
    #
    elif outcomeDescriptor.outcome_type == 'categorical':
        _y_series: pl.Series = y if isinstance( y, pl.Series ) else y.to_series()
        loss_func = nn.CrossEntropyLoss()
        output_dimension = len( _y_series.cat.get_categories() )
        y_np = _y_series.to_physical().to_numpy().astype( np.int64 )
    #
    else:
        raise ValueError(
            "Unrecognized outcomeDescriptor.outcome_type='{}'".format(
                outcomeDescriptor.outcome_type,
            )
        )
    #

    return X_all_np, y_np, groups, oheDict, loss_func, output_dimension, outcomeDescriptor
#/def _prism_setup


_DEFAULT_LAMBDA_PATH: np.ndarray = np.logspace( 1, -2, 50 )


def prismWImportances(
    X: DataFrameLike,
    Xk: DataFrameLike,
    y: SeriesOrDataFrameLike,
    layers: Sequence[ int ],
    outcome_type: Literal['continuous','count','categorical',] | None = None,
    lambda_path: Sequence[ float ] | None = None,
    a_path: Iterable[ float ] | None = None,
    batch_size: int | None = None,
    epochs: int = 500,
    model_type: str = 'mlp',
    learning_rate: float = 0.01,
    drop_first: bool = True,
    dense_activation: str = 'relu',
    verbose: int = 0,
    ) -> np.ndarray:
    """
    PRISM-W importances: average of group-norm snapshots over a lambda regularization path.

    Trains a single MLP on [X, Xk] → y with an adaptive proximal penalty on the input layer.
    At the end of each lambda stage the group norms ||w[:, group_j]||_F are recorded;
    the final importances are the mean over all snapshots.

    :param lambda_path: Sequence of lambda values. Defaults to logspace(1,-2,50).
    :param a_path: Per-stage input-layer penalty values. If None, uses lambda_path values.
    :param epochs: Total training epochs, distributed as evenly as possible across lambda stages.
    :returns: Array of shape (2*p,) — first p entries for X, last p for Xk.
    """
    from . import torchImportances

    if lambda_path is None:
        lambda_path = _DEFAULT_LAMBDA_PATH
    #

    X_all_np, y_np, groups, _, loss_func, output_dimension, _ = _prism_setup(
        X = X, Xk = Xk, y = y,
        layers = layers,
        outcome_type = outcome_type,
        drop_first = drop_first,
    )

    _mu = X_all_np.mean( axis=0 )
    _sd = np.maximum( X_all_np.std( axis=0 ), 1e-8 )
    X_all_np = ( X_all_np - _mu ) / _sd

    predictionModel: torchImportances.PRISMPredictionModel = torchImportances.PRISMPredictionModel(
        input_size = X_all_np.shape[1],
        layers = list( layers ),
        dense_activation = dense_activation,
        loss_func = loss_func,
        output_dimension = output_dimension,
        learning_rate = learning_rate,
        epochs = epochs,
        model_type = model_type,
        verbose = verbose,
    )

    snapshots: list[ np.ndarray ] = predictionModel.fit(
        X = X_all_np,
        y = y_np,
        groups = groups,
        lambda_path = lambda_path,
        a_path = a_path,
        batch_size = batch_size,
    )

    return np.mean( snapshots, axis = 0 )
#/def prismWImportances


def prismGImportances(
    X: DataFrameLike,
    Xk: DataFrameLike,
    y: SeriesOrDataFrameLike,
    layers: Sequence[ int ],
    outcome_type: Literal['continuous','count','categorical',] | None = None,
    local_grad_method: Literal['auto_diff','bandwidth'] = 'auto_diff',
    lambda_path: Sequence[ float ] | None = None,
    a_path: Iterable[ float ] | None = None,
    batch_size: int | None = None,
    epochs: int = 500,
    bandwidth: float | None = None,
    exponent: float = 1.0,
    model_type: str = 'mlp',
    learning_rate: float = 0.01,
    drop_first: bool = True,
    dense_activation: str = 'relu',
    verbose: int = 0,
    ) -> np.ndarray:
    """
    PRISM-G importances: average of PRISM local-gradient snapshots over a lambda path.

    Same training procedure as prismWImportances; at the end of each lambda stage the
    PRISM importances (auto_diff or bandwidth) of the current model are recorded.
    Delegates snapshot computation to _prismImportances_t.

    :param local_grad_method: 'auto_diff' (exact) or 'bandwidth' (finite difference).
    :param lambda_path: Sequence of lambda values. Defaults to logspace(1,-2,50).
    :param a_path: Per-stage input-layer penalty values. If None, uses lambda_path values.
    :param epochs: Total training epochs, distributed as evenly as possible across lambda stages.
    :param bandwidth: Bandwidth for finite-difference approximation (auto-set if None).
    :param exponent: Power applied to each local gradient value before averaging.
    :returns: Array of shape (2*p,).
    """
    from . import torchImportances

    X_all_np, y_np, groups, oheDict, loss_func, output_dimension, outcomeDescriptor = _prism_setup(
        X = X, Xk = Xk, y = y,
        layers = layers,
        outcome_type = outcome_type,
        drop_first = drop_first,
    )

    _mu = X_all_np.mean( axis=0 )
    _sd = np.maximum( X_all_np.std( axis=0 ), 1e-8 )
    X_all_np = ( X_all_np - _mu ) / _sd

    cat_ohe_vals: dict[ int, tuple[ float, float ] ] = {}
    for _col_idx in oheDict.values():
        if not isinstance( _col_idx, int ):
            for k in _col_idx:
                cat_ohe_vals[ k ] = (
                    float( ( 0.0 - _mu[ k ] ) / _sd[ k ] ),
                    float( ( 1.0 - _mu[ k ] ) / _sd[ k ] ),
                )
    #

    predictionModel: torchImportances.PRISMPredictionModel = torchImportances.PRISMPredictionModel(
        input_size = X_all_np.shape[1],
        layers = list( layers ),
        dense_activation = dense_activation,
        loss_func = loss_func,
        output_dimension = output_dimension,
        learning_rate = learning_rate,
        epochs = epochs,
        model_type = model_type,
        verbose = verbose,
    )

    if outcomeDescriptor.outcome_type == 'categorical':
        def snapshot_fn( model: torchImportances.PRISMPredictionModel, X_t: torch.Tensor ) -> np.ndarray:
            with torch.no_grad():
                _logits = model.predict_t( X_t )
            logit_contrasts = _logits[:, 1:] - _logits[:, 0:1]
            _cov = torch.cov( logit_contrasts.T )
            if _cov.ndim == 0:
                inv_cov_t = torch.tensor( [[ 1.0 / _cov.item() ]], device=X_t.device, dtype=torch.float32 )
            else:
                inv_cov_t = torch.linalg.inv( _cov )
            #
            if local_grad_method == 'auto_diff':
                return _prismImportances_categorical_t(
                    model = model,
                    X_all_t = X_t,
                    oheDict = oheDict,
                    inv_cov_t = inv_cov_t,
                    exponent = exponent,
                    drop_first = drop_first,
                    cat_ohe_vals = cat_ohe_vals,
                ).cpu().numpy()
            else:
                return _prismImportances_t(
                    model = model,
                    X_all_t = X_t,
                    oheDict = oheDict,
                    local_grad_method = 'bandwidth',
                    bandwidth = bandwidth,
                    exponent = exponent,
                    drop_first = drop_first,
                    inv_cov_t = inv_cov_t,
                    cat_ohe_vals = cat_ohe_vals,
                ).cpu().numpy()
            #
        #/def snapshot_fn
    else:
        def snapshot_fn( model: torchImportances.PRISMPredictionModel, X_t: torch.Tensor ) -> np.ndarray:
            return _prismImportances_t(
                model = model,
                X_all_t = X_t,
                oheDict = oheDict,
                local_grad_method = local_grad_method,
                bandwidth = bandwidth,
                exponent = exponent,
                drop_first = drop_first,
                cat_ohe_vals = cat_ohe_vals,
            ).cpu().numpy()
        #/def snapshot_fn
    #

    snapshots: list[ np.ndarray ] = predictionModel.fit(
        X = X_all_np,
        y = y_np,
        groups = groups,
        lambda_path = lambda_path,
        a_path = a_path,
        batch_size = batch_size,
        snapshot_fn = snapshot_fn,
    )

    return np.mean( snapshots, axis = 0 )
#/def prismGImportances


def prismGWImportances(
    X: DataFrameLike,
    Xk: DataFrameLike,
    y: SeriesOrDataFrameLike,
    layers: Sequence[ int ],
    outcome_type: Literal['continuous','count','categorical',] | None = None,
    local_grad_method: Literal['auto_diff','bandwidth'] = 'auto_diff',
    lambda_path: Sequence[ float ] | None = None,
    a_path: Iterable[ float ] | None = None,
    batch_size: int | None = None,
    epochs: int = 500,
    bandwidth: float | None = None,
    exponent: float = 1.0,
    model_type: str = 'mlp',
    learning_rate: float = 0.01,
    drop_first: bool = True,
    dense_activation: str = 'relu',
    verbose: int = 0,
    ) -> tuple[ np.ndarray, np.ndarray ]:
    """
    PRISM-G and PRISM-W importances from a single training pass.

    Identical hyperparameters and model to prismGImportances / prismWImportances.
    At each lambda stage the snapshot_fn records PRISM-W group norms as a side
    effect while returning PRISM-G local-gradient importances as the primary snapshot.

    :returns: (g_importances, w_importances) both of shape (2*p,).
    """
    from . import torchImportances

    if lambda_path is None:
        lambda_path = _DEFAULT_LAMBDA_PATH
    #

    X_all_np, y_np, groups, oheDict, loss_func, output_dimension, outcomeDescriptor = _prism_setup(
        X = X, Xk = Xk, y = y,
        layers = layers,
        outcome_type = outcome_type,
        drop_first = drop_first,
    )

    _mu = X_all_np.mean( axis=0 )
    _sd = np.maximum( X_all_np.std( axis=0 ), 1e-8 )
    X_all_np = ( X_all_np - _mu ) / _sd

    cat_ohe_vals: dict[ int, tuple[ float, float ] ] = {}
    for _col_idx in oheDict.values():
        if not isinstance( _col_idx, int ):
            for k in _col_idx:
                cat_ohe_vals[ k ] = (
                    float( ( 0.0 - _mu[ k ] ) / _sd[ k ] ),
                    float( ( 1.0 - _mu[ k ] ) / _sd[ k ] ),
                )
    #

    predictionModel: torchImportances.PRISMPredictionModel = torchImportances.PRISMPredictionModel(
        input_size = X_all_np.shape[1],
        layers = list( layers ),
        dense_activation = dense_activation,
        loss_func = loss_func,
        output_dimension = output_dimension,
        learning_rate = learning_rate,
        epochs = epochs,
        model_type = model_type,
        verbose = verbose,
    )

    w_snapshots: list[ np.ndarray ] = []

    if outcomeDescriptor.outcome_type == 'categorical':
        def snapshot_fn( model: torchImportances.PRISMPredictionModel, X_t: torch.Tensor ) -> np.ndarray:
            w_snapshots.append( model.get_group_importances( groups ) )
            with torch.no_grad():
                _logits = model.predict_t( X_t )
            logit_contrasts = _logits[:, 1:] - _logits[:, 0:1]
            _cov = torch.cov( logit_contrasts.T )
            if _cov.ndim == 0:
                inv_cov_t = torch.tensor( [[ 1.0 / _cov.item() ]], device=X_t.device, dtype=torch.float32 )
            else:
                inv_cov_t = torch.linalg.inv( _cov )
            #
            if local_grad_method == 'auto_diff':
                return _prismImportances_categorical_t(
                    model = model,
                    X_all_t = X_t,
                    oheDict = oheDict,
                    inv_cov_t = inv_cov_t,
                    exponent = exponent,
                    drop_first = drop_first,
                    cat_ohe_vals = cat_ohe_vals,
                ).cpu().numpy()
            else:
                return _prismImportances_t(
                    model = model,
                    X_all_t = X_t,
                    oheDict = oheDict,
                    local_grad_method = 'bandwidth',
                    bandwidth = bandwidth,
                    exponent = exponent,
                    drop_first = drop_first,
                    inv_cov_t = inv_cov_t,
                    cat_ohe_vals = cat_ohe_vals,
                ).cpu().numpy()
            #
        #/def snapshot_fn
    else:
        def snapshot_fn( model: torchImportances.PRISMPredictionModel, X_t: torch.Tensor ) -> np.ndarray:
            w_snapshots.append( model.get_group_importances( groups ) )
            return _prismImportances_t(
                model = model,
                X_all_t = X_t,
                oheDict = oheDict,
                local_grad_method = local_grad_method,
                bandwidth = bandwidth,
                exponent = exponent,
                drop_first = drop_first,
                cat_ohe_vals = cat_ohe_vals,
            ).cpu().numpy()
        #/def snapshot_fn
    #

    g_snapshots: list[ np.ndarray ] = predictionModel.fit(
        X = X_all_np,
        y = y_np,
        groups = groups,
        lambda_path = lambda_path,
        a_path = a_path,
        batch_size = batch_size,
        snapshot_fn = snapshot_fn,
    )

    return np.mean( g_snapshots, axis=0 ), np.mean( w_snapshots, axis=0 )
#/def prismGWImportances


def _get_localGrad_ohe_matrix_t(
    model:             'object',
    X_all_t:           torch.Tensor,
    x_oheDict:         dict,
    local_grad_method: str,
    bandwidth:         float | None,
    ) -> torch.Tensor:
    """
    Per-sample local gradient matrix for X-only features (not Xk).

    For numeric variables: bandwidth finite-diff or auto_diff gradient (one column each).
    For categorical variables: model-prediction contrast vs. reference category (drop_first=True
    convention, so c-1 columns per variable; category 0 is the reference).

    x_oheDict: oheDict filtered to X columns only (keys without '~').
    Returns tensor of shape (n, p_ohe_x) where p_ohe_x = p_numeric + sum(c_k - 1).
    """
    n = X_all_t.shape[0]
    p_ohe = sum( 1 if isinstance( v, int ) else len( v ) for v in x_oheDict.values() )

    if local_grad_method == 'auto_diff':
        auto_diff_full_t: torch.Tensor = model.auto_diff_t( X_all_t )  # (n, p_all_ohe)
    elif local_grad_method == 'bandwidth':
        if bandwidth is None:
            bandwidth = float( n ** -0.2 )
    else:
        raise ValueError( "Unrecognized local_grad_method='{}'".format( local_grad_method ) )

    grad_t = torch.zeros( n, p_ohe, device=X_all_t.device )
    out_col = 0

    for col, col_idx in x_oheDict.items():
        if isinstance( col_idx, int ):
            # numeric — one gradient column
            if local_grad_method == 'auto_diff':
                grad_t[ :, out_col ] = auto_diff_full_t[ :, col_idx ]
            else:
                X_plus  = X_all_t.clone(); X_plus[  :, col_idx ] += bandwidth
                X_minus = X_all_t.clone(); X_minus[ :, col_idx ] -= bandwidth
                grad_t[ :, out_col ] = (
                    ( model.predict_t( X_plus ) - model.predict_t( X_minus ) ) / ( 2.0 * bandwidth )
                ).reshape( -1 )
            out_col += 1
        else:
            # categorical — c-1 contrast columns (category 0 = reference, dropped)
            cat_indices = list( col_idx )  # OHE col indices for categories 1..c-1
            # reference: set all OHE bits for this variable to 0 (implicit category 0)
            X_ref = X_all_t.clone()
            X_ref[ :, cat_indices ] = 0.0
            pred_ref = model.predict_t( X_ref ).reshape( n )  # (n,)
            for ohe_col in cat_indices:
                X_j = X_all_t.clone()
                X_j[ :, cat_indices ] = 0.0
                X_j[ :, ohe_col      ] = 1.0
                pred_j = model.predict_t( X_j ).reshape( n )
                grad_t[ :, out_col ] = pred_j - pred_ref
                out_col += 1
        #
    #

    return grad_t
#/def _get_localGrad_ohe_matrix_t


def prismGLocalGradients(
    X:                 DataFrameLike,
    Xk:                DataFrameLike,
    y:                 SeriesOrDataFrameLike,
    layers:            'Sequence[int]',
    outcome_type:      'Literal["continuous","count","categorical"] | None' = None,
    local_grad_method: 'Literal["auto_diff","bandwidth"]' = 'bandwidth',
    lambda_path:       'Sequence[float] | None' = None,
    a_path:            'Iterable[float] | None' = None,
    batch_size:        int | None = None,
    epochs:            int = 500,
    bandwidth:         float | None = None,
    model_type:        str = 'mlp',
    learning_rate:     float = 0.01,
    drop_first:        bool = True,
    dense_activation:  str = 'relu',
    verbose:           int = 0,
    ) -> np.ndarray:
    """
    Train a PRISM-G network on (X, Xk, y) and return the per-sample local gradient
    matrix for X only.

    Returns array of shape (n, p_ohe_x) where
      p_ohe_x = p_numeric + sum(c_k - 1 for each categorical variable in X).
    Numeric columns: bandwidth or auto_diff gradient.
    Categorical columns (c-1 per variable): model-prediction contrast vs. category 0
      (drop_first=True convention — category 0 is the implicit reference).
    """
    from . import torchImportances

    X_all_np, y_np, groups, oheDict, loss_func, output_dimension, outcomeDescriptor = _prism_setup(
        X            = X,
        Xk           = Xk,
        y            = y,
        layers       = layers,
        outcome_type = outcome_type,
        drop_first   = drop_first,
    )

    predictionModel: torchImportances.PRISMPredictionModel = torchImportances.PRISMPredictionModel(
        input_size       = X_all_np.shape[1],
        layers           = list( layers ),
        dense_activation = dense_activation,
        loss_func        = loss_func,
        output_dimension = output_dimension,
        learning_rate    = learning_rate,
        epochs           = epochs,
        model_type       = model_type,
        verbose          = verbose,
    )

    predictionModel.fit(
        X           = X_all_np,
        y           = y_np,
        groups      = groups,
        lambda_path = lambda_path,
        a_path      = a_path,
        batch_size  = batch_size,
    )

    # oheDict covers X_all = concat(X, Xk); filter to X columns only (no '~' suffix)
    x_oheDict = { col: idx for col, idx in oheDict.items() if not col.endswith( '~' ) }

    X_all_t = torch.tensor( X_all_np, dtype=torch.float32 ).to( predictionModel.device )

    grad_t = _get_localGrad_ohe_matrix_t(
        model             = predictionModel,
        X_all_t           = X_all_t,
        x_oheDict         = x_oheDict,
        local_grad_method = local_grad_method,
        bandwidth         = bandwidth,
    )

    return grad_t.cpu().numpy()
#/def prismGLocalGradients


def rangerGiniImportances(
    X: DataFrameLike,
    Xk: DataFrameLike,
    y: SeriesOrDataFrameLike,
    outcome_type: Literal['continuous','count','categorical',] | None = None,
    verbose: int = 0,
    **kwargs,
    ) -> np.ndarray:
    from . import rbridge
    return rbridge.rangerGiniImportances(
        X = X,
        Xk = Xk,
        y = y,
        outcome_type = outcome_type,
        verbose = verbose,
        **kwargs,
    )
#/def rangerGiniImportances


def rangerPrismImportances(
    X: DataFrameLike,
    Xk: DataFrameLike,
    y: SeriesOrDataFrameLike,
    outcome_type: Literal['continuous','count','categorical',] | None = None,
    verbose: int = 0,
    **kwargs,
    ) -> np.ndarray:
    from . import rbridge
    return rbridge.rangerPrismImportances(
        X = X,
        Xk = Xk,
        y = y,
        outcome_type = outcome_type,
        verbose = verbose,
        **kwargs,
    )
#/def rangerPrismImportances

def _collapse_cat_importance(
    coef: np.ndarray,
    indices: tuple[int, ...],
    col_name: str,
) -> float:
    idxlist = list(indices)
    if not idxlist:
        raise ValueError(
            f"oheDict[{col_name!r}] is empty — categorical column has only 1 unique value "
            "in the OHE design matrix. Ensure X and Xk have at least 2 distinct values per categorical column."
        )
    return max(np.max(coef[idxlist]), 0) - min(np.min(coef[idxlist]), 0)


def lassoImportances(
    X: DataFrameLike,
    Xk: DataFrameLike,
    y: SeriesOrDataFrameLike,
    outcome_type: Literal['continuous','count','categorical',] | None = None,
    fit_intercept: bool = True,
    exponent: float = 1.0,
    verbose: int = 0,
    **kwargs,
    ) -> np.ndarray:
    X = _resolve_df(X)
    Xk = _resolve_df(Xk)
    y = _resolve_y(y)

    # Resolve outcome type and dimension
    outcomeDescriptor: OutcomeDescriptor = OutcomeDescriptor.infer(
        y = y,
        outcome_type = outcome_type,
    )
    
    if outcomeDescriptor.outcome_dimension != 'single':
        raise TypeError("Joint outcomes unavailable")
    #
    else:
        if outcomeDescriptor.outcome_type != 'categorical':
            # Numeric — normalise to (n,) for width-1, (n, k) for joint
            _y_np = y.to_numpy().reshape( X.shape[0], -1 )
            y = _y_np[:, 0] if _y_np.shape[1] == 1 else _y_np
        #
        # categorical: y stays as polars Series/DataFrame; converted in the elif branch below
    #/switch outcomeDescriptor.outcome_dimension
    
    # Grab Parameters
    n_splits: int = kwargs.get( 'n_splits', 5 )
    max_iter: int
    if outcomeDescriptor.outcome_type == 'count':
        max_iter = kwargs.get( 'max_iter', 200 )
    #
    else:
        max_iter = kwargs.get( 'max_iter', 4000 )
    #
    
    # One hot encode the X, Xk data
    X_all_df: pl.DataFrame = pl.concat(
        (
            X,
            Xk.rename(
                { col: col + '~' for col in Xk.columns }
            ),
        ),
        how = 'horizontal',
    )

    oheDict: dict[ str, int | tuple[ int,...] ] = utilities.get_oheDict(
        X_all_df,
        drop_first = True,
    )

    X_ohe: np.ndarray = utilities.get_ohe_np( X = X_all_df, drop_first = True )
    if not np.isfinite( X_ohe ).all():
        raise ValueError(
            "OHE design matrix contains NaN or Inf — check knockoff generation for numerical instability."
        )
    _zero_var_cols = np.where( X_ohe.var( axis=0 ) == 0 )[0]
    if _zero_var_cols.size:
        raise ValueError(
            f"OHE design matrix has {_zero_var_cols.size} zero-variance column(s) at indices "
            f"{_zero_var_cols.tolist()} — check X/Xk for degenerate features."
        )
    #

    if outcomeDescriptor.outcome_type == 'continuous':
        from sklearn.linear_model import LassoCV
        lassoModel: LassoCV = LassoCV(
            max_iter = max_iter,
            fit_intercept = fit_intercept,
            cv = n_splits,
        )

        if verbose > 0:
            print("Fitting LassoCV:")
            print("  max_iter={}".format(max_iter))
            print("  n_splits={}".format(n_splits))
        #

        lassoModel.fit(
            X = X_ohe,
            y = y,
        )

        # Grab coefficients, and get importances
        lasso_coefficients: np.ndarray = lassoModel.coef_
        p: int
        if len( lasso_coefficients.shape ) == 1:
            p = len( lasso_coefficients )
            lasso_coefficients = lasso_coefficients.reshape( (1,p) )
        #
        elif len( lasso_coefficients.shape ) == 2:
            p = lasso_coefficients.shape[1]
        #
        else:
            raise ValueError("Unexpected lassoModel.coef_.shape={}".format(
                lasso_coefficients.shape
            ))
        #/switch len( lasso_coefficients.shape )

        lasso_importances: np.ndarray
        if lasso_coefficients.shape[0] > 1:
            raise Exception("Bad lasso_coefficients.shape={}".format(lasso_coefficients.shape))
        #
        
        lasso_coefficients = lasso_coefficients.reshape( (p,) )
    #
    elif outcomeDescriptor.outcome_type == 'count':
        from .poissonLasso import PoissonLassoCV
        alphas: np.ndarray = kwargs.get(
            'alphas',
            np.logspace( -4, 2, 10 )
        )
        
        
        glmModel: PoissonLassoCV = PoissonLassoCV(
            fit_intercept = fit_intercept,
            alphas = alphas,
            n_splits = n_splits,
            max_iter = max_iter,
        )

        if verbose > 0:
            print("Fitting PoissonLassoCV")
            print("  max_iter={}".format(max_iter))
            print("  n_splits={}".format(n_splits))
            print("  alphas={}".format(alphas))
        #

        glmModel.fit(
            X = X_ohe,
            y = y,
        )

        lasso_coefficients: np.ndarray = glmModel.coef_
    #
    elif outcomeDescriptor.outcome_type == 'categorical':
        from sklearn.linear_model import LogisticRegressionCV

        logisticModel: LogisticRegressionCV = LogisticRegressionCV(
            #penalty = 'elasticnet',
            l1_ratios = ( 1, ),
            solver = 'saga',
            fit_intercept = fit_intercept,
            max_iter = max_iter,
            cv = n_splits,
            use_legacy_attributes = False,
        )

        if verbose > 0:
            print("Fitting LogisticRegressionCV (L1)")
            print("  max_iter={}".format(max_iter))
            print("  n_splits={}".format(n_splits))
        #

        logisticModel.fit(
            X = X_ohe,
            y = y.to_numpy().ravel(),
        )

        # coef_ shape: (n_classes, n_features) or (1, n_features) for binary
        
        lasso_coefficients: np.ndarray = logisticModel.coef_
        if len( lasso_coefficients.shape ) == 1:
            ...
        #
        elif lasso_coefficients.shape[0] == 1:
            lasso_coefficients = lasso_coefficients.reshape(-1)
        #
        else:
            # Multi-class (k >= 3): Mahalanobis distance on contrasted coefficients.
            # coef_ shape: (k, p) → contrast against first class → (k-1, p)
            coef_contrast: np.ndarray = lasso_coefficients[1:, :] - lasso_coefficients[0:1, :]
            if coef_contrast.shape[0] == 1:
                # Degenerate case: k=2 but returned as (2, p); just take abs of single contrast
                lasso_coefficients = np.abs( coef_contrast[0, :] )
            else:
                # Covariance from predicted log probabilities (one-hot convention: subtract first column)
                _log_proba: np.ndarray = logisticModel.predict_log_proba(
                    X_ohe
                )  # (n, k)
                _log_proba_contrast: np.ndarray = _log_proba[:, 1:] - _log_proba[:, 0:1]  # (n, k-1)
                _cov: np.ndarray = np.cov( _log_proba_contrast, rowvar=False )  # (k-1, k-1)
                inv_cov: np.ndarray = np.linalg.inv( _cov )
                # Mahalanobis distance for each feature: sqrt( v^T inv_cov v ) over (k-1,) vectors
                lasso_coefficients = np.sqrt(
                    np.einsum( 'kp,kl,lp->p', coef_contrast, inv_cov, coef_contrast )
                )
        #/switch lasso_coefficients.shape
    #
    else:
        raise ValueError(
            "Unrecognized outcomeDescriptor.outcome_type='{}'".format(
                outcomeDescriptor.outcome_type,
            )
        )
    #
    
    # Collapse the ohe categories
    importances = np.fromiter(
        (
            np.abs( lasso_coefficients[ oheDict[col] ] )
            if isinstance( oheDict[col], int )
            else _collapse_cat_importance( lasso_coefficients, oheDict[col], col )
            for col in X_all_df.columns
        ),
        dtype = float,
    )**exponent

    return importances
#/def lassoImportances

def ridgeImportances(
    X: DataFrameLike,
    Xk: DataFrameLike,
    y: SeriesOrDataFrameLike,
    outcome_type: Literal['continuous','count','categorical',] | None = None,
    fit_intercept: bool = True,
    exponent: float = 1.0,
    verbose: int = 0,
    **kwargs,
    ) -> np.ndarray:
    """
    Ridge (L2-penalised) analogue of lassoImportances.

    - continuous: sklearn RidgeCV
    - count:      sklearn PoissonRegressor cross-validated via GridSearchCV
                  (neg_mean_poisson_deviance scoring)
    - categorical: sklearn LogisticRegressionCV with penalty='l2', solver='lbfgs'

    All other logic (OHE, oheDict collapsing, multi-class Mahalanobis, exponent)
    is identical to lassoImportances.
    """
    X = _resolve_df(X)
    Xk = _resolve_df(Xk)
    y = _resolve_y(y)

    outcomeDescriptor: OutcomeDescriptor = OutcomeDescriptor.infer(
        y = y,
        outcome_type = outcome_type,
    )

    if outcomeDescriptor.outcome_dimension != 'single':
        raise TypeError("Joint outcomes unavailable")

    if outcomeDescriptor.outcome_type != 'categorical':
        _y_np = y.to_numpy().reshape( X.shape[0], -1 )
        y = _y_np[:, 0] if _y_np.shape[1] == 1 else _y_np

    n_splits: int = kwargs.get( 'n_splits', 5 )
    max_iter: int = kwargs.get( 'max_iter', 4000 )

    X_all_df: pl.DataFrame = pl.concat(
        (
            X,
            Xk.rename( { col: col + '~' for col in Xk.columns } ),
        ),
        how = 'horizontal',
    )

    oheDict: dict[ str, int | tuple[ int,...] ] = utilities.get_oheDict(
        X_all_df,
        drop_first = True,
    )

    X_ohe: np.ndarray = utilities.get_ohe_np( X = X_all_df, drop_first = True )
    if not np.isfinite( X_ohe ).all():
        raise ValueError(
            "OHE design matrix contains NaN or Inf — check knockoff generation for numerical instability."
        )
    _zero_var_cols = np.where( X_ohe.var( axis=0 ) == 0 )[0]
    if _zero_var_cols.size:
        raise ValueError(
            f"OHE design matrix has {_zero_var_cols.size} zero-variance column(s) at indices "
            f"{_zero_var_cols.tolist()} — check X/Xk for degenerate features."
        )
    #

    ridge_coefficients: np.ndarray

    if outcomeDescriptor.outcome_type == 'continuous':
        from sklearn.linear_model import RidgeCV
        alphas: np.ndarray = kwargs.get(
            'alphas',
            np.logspace( -4, 4, 13 ),
        )

        if verbose > 0:
            print("Fitting RidgeCV:")
            print("  n_splits={}".format( n_splits ))
            print("  alphas={}".format( alphas ))

        ridgeModel: RidgeCV = RidgeCV(
            alphas = alphas,
            fit_intercept = fit_intercept,
            cv = n_splits,
        )
        ridgeModel.fit( X = X_ohe, y = y )
        ridge_coefficients = ridgeModel.coef_.reshape( -1 )

    elif outcomeDescriptor.outcome_type == 'count':
        from sklearn.linear_model import PoissonRegressor
        from sklearn.model_selection import GridSearchCV
        from sklearn.preprocessing import StandardScaler
        alphas = kwargs.get( 'alphas', np.logspace( -4, 2, 10 ) )

        if verbose > 0:
            print("Fitting PoissonRegressor (L2) via GridSearchCV:")
            print("  max_iter={}".format( max_iter ))
            print("  n_splits={}".format( n_splits ))
            print("  alphas={}".format( alphas ))

        # Standardize before Poisson GLM: the log link causes weight divergence
        # during L-BFGS when numeric and binary OHE columns are on different scales.
        _scaler = StandardScaler()
        X_ohe_scaled = _scaler.fit_transform( X_ohe )

        poissonModel = GridSearchCV(
            PoissonRegressor(
                fit_intercept = fit_intercept,
                max_iter = max_iter,
            ),
            param_grid = { 'alpha': alphas },
            cv = n_splits,
            scoring = 'neg_mean_poisson_deviance',
        )
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings( 'ignore', category = RuntimeWarning )
            poissonModel.fit( X = X_ohe_scaled, y = y )
        #

        ridge_coefficients = poissonModel.best_estimator_.coef_
        if not np.isfinite( ridge_coefficients ).all():
            raise ValueError(
                "PoissonRegressor produced non-finite coefficients — "
                "check data or increase regularization."
            )
        #

    elif outcomeDescriptor.outcome_type == 'categorical':
        from sklearn.linear_model import LogisticRegressionCV

        if verbose > 0:
            print("Fitting LogisticRegressionCV (L2):")
            print("  max_iter={}".format( max_iter ))
            print("  n_splits={}".format( n_splits ))

        logisticModel: LogisticRegressionCV = LogisticRegressionCV(
            penalty = 'l2',
            solver = 'lbfgs',
            fit_intercept = fit_intercept,
            max_iter = max_iter,
            cv = n_splits,
            use_legacy_attributes = False,
        )
        logisticModel.fit(
            X = X_ohe,
            y = y.to_numpy().ravel(),
        )

        ridge_coefficients = logisticModel.coef_
        if len( ridge_coefficients.shape ) == 1:
            pass
        elif ridge_coefficients.shape[0] == 1:
            ridge_coefficients = ridge_coefficients.reshape( -1 )
        else:
            # Multi-class (k >= 3): Mahalanobis distance on contrasted coefficients
            coef_contrast: np.ndarray = ridge_coefficients[1:, :] - ridge_coefficients[0:1, :]
            if coef_contrast.shape[0] == 1:
                ridge_coefficients = np.abs( coef_contrast[0, :] )
            else:
                # Covariance from predicted log probabilities (one-hot convention: subtract first column)
                _log_proba: np.ndarray = logisticModel.predict_log_proba( X_ohe )  # (n, k)
                _log_proba_contrast: np.ndarray = _log_proba[:, 1:] - _log_proba[:, 0:1]  # (n, k-1)
                _cov: np.ndarray = np.cov( _log_proba_contrast, rowvar=False )  # (k-1, k-1)
                inv_cov: np.ndarray = np.linalg.inv( _cov )
                ridge_coefficients = np.sqrt(
                    np.einsum( 'kp,kl,lp->p', coef_contrast, inv_cov, coef_contrast )
                )
        #/switch ridge_coefficients.shape

    else:
        raise ValueError(
            "Unrecognized outcomeDescriptor.outcome_type='{}'".format(
                outcomeDescriptor.outcome_type,
            )
        )

    importances = np.fromiter(
        (
            np.abs( ridge_coefficients[ oheDict[col] ] )
            if isinstance( oheDict[col], int )
            else _collapse_cat_importance( ridge_coefficients, oheDict[col], col )
            for col in X_all_df.columns
        ),
        dtype = float,
    )**exponent

    return importances
#/def ridgeImportances

def elasticImportances(
    X: DataFrameLike,
    Xk: DataFrameLike,
    y: SeriesOrDataFrameLike,
    outcome_type: Literal['continuous','count','categorical',] | None = None,
    l1_ratio: float = 0.5,
    fit_intercept: bool = True,
    exponent: float = 1.0,
    verbose: int = 0,
    **kwargs,
    ) -> np.ndarray:
    """
    Elastic-net importance measures.

    Delegates to ``lassoImportances`` when ``l1_ratio=1`` and to
    ``ridgeImportances`` when ``l1_ratio=0``; otherwise fits elastic-net models:

    - continuous: sklearn ElasticNetCV
    - count:      PoissonLassoCV with L1_wt=l1_ratio (statsmodels elastic-net GLM)
    - categorical: sklearn LogisticRegressionCV with penalty='elasticnet',
                   l1_ratios=[l1_ratio], solver='saga'
    """
    X = _resolve_df(X)
    Xk = _resolve_df(Xk)
    y = _resolve_y(y)

    if l1_ratio == 1.0:
        return lassoImportances(
            X = X, Xk = Xk, y = y,
            outcome_type = outcome_type,
            fit_intercept = fit_intercept,
            exponent = exponent,
            verbose = verbose,
            **kwargs,
        )
    if l1_ratio == 0.0:
        return ridgeImportances(
            X = X, Xk = Xk, y = y,
            outcome_type = outcome_type,
            fit_intercept = fit_intercept,
            exponent = exponent,
            verbose = verbose,
            **kwargs,
        )

    outcomeDescriptor: OutcomeDescriptor = OutcomeDescriptor.infer(
        y = y,
        outcome_type = outcome_type,
    )

    if outcomeDescriptor.outcome_dimension != 'single':
        raise TypeError("Joint outcomes unavailable")

    if outcomeDescriptor.outcome_type != 'categorical':
        _y_np = y.to_numpy().reshape( X.shape[0], -1 )
        y = _y_np[:, 0] if _y_np.shape[1] == 1 else _y_np

    n_splits: int = kwargs.get( 'n_splits', 5 )
    max_iter: int
    if outcomeDescriptor.outcome_type == 'count':
        max_iter = kwargs.get( 'max_iter', 200 )
    else:
        max_iter = kwargs.get( 'max_iter', 4000 )

    X_all_df: pl.DataFrame = pl.concat(
        (
            X,
            Xk.rename( { col: col + '~' for col in Xk.columns } ),
        ),
        how = 'horizontal',
    )

    oheDict: dict[ str, int | tuple[ int,...] ] = utilities.get_oheDict(
        X_all_df,
        drop_first = True,
    )

    X_ohe: np.ndarray = utilities.get_ohe_np( X = X_all_df, drop_first = True )
    if not np.isfinite( X_ohe ).all():
        raise ValueError(
            "OHE design matrix contains NaN or Inf — check knockoff generation for numerical instability."
        )
    _zero_var_cols = np.where( X_ohe.var( axis=0 ) == 0 )[0]
    if _zero_var_cols.size:
        raise ValueError(
            f"OHE design matrix has {_zero_var_cols.size} zero-variance column(s) at indices "
            f"{_zero_var_cols.tolist()} — check X/Xk for degenerate features."
        )
    #

    elastic_coefficients: np.ndarray

    if outcomeDescriptor.outcome_type == 'continuous':
        from sklearn.linear_model import ElasticNetCV

        if verbose > 0:
            print("Fitting ElasticNetCV:")
            print("  l1_ratio={}".format( l1_ratio ))
            print("  max_iter={}".format( max_iter ))
            print("  n_splits={}".format( n_splits ))

        elasticModel: ElasticNetCV = ElasticNetCV(
            l1_ratio = l1_ratio,
            max_iter = max_iter,
            fit_intercept = fit_intercept,
            cv = n_splits,
        )
        elasticModel.fit( X = X_ohe, y = y )
        elastic_coefficients = elasticModel.coef_.reshape( -1 )

    elif outcomeDescriptor.outcome_type == 'count':
        from .poissonLasso import PoissonLassoCV
        alphas: np.ndarray = kwargs.get( 'alphas', np.logspace( -4, 2, 10 ) )

        if verbose > 0:
            print("Fitting PoissonLassoCV (elastic-net):")
            print("  l1_ratio={}".format( l1_ratio ))
            print("  max_iter={}".format( max_iter ))
            print("  n_splits={}".format( n_splits ))
            print("  alphas={}".format( alphas ))

        glmModel: PoissonLassoCV = PoissonLassoCV(
            fit_intercept = fit_intercept,
            alphas = alphas,
            n_splits = n_splits,
            max_iter = max_iter,
            L1_wt = l1_ratio,
        )
        glmModel.fit( X = X_ohe, y = y )
        elastic_coefficients = glmModel.coef_

    elif outcomeDescriptor.outcome_type == 'categorical':
        from sklearn.linear_model import LogisticRegressionCV

        if verbose > 0:
            print("Fitting LogisticRegressionCV (elastic-net):")
            print("  l1_ratio={}".format( l1_ratio ))
            print("  max_iter={}".format( max_iter ))
            print("  n_splits={}".format( n_splits ))

        logisticModel: LogisticRegressionCV = LogisticRegressionCV(
            penalty = 'elasticnet',
            l1_ratios = ( l1_ratio, ),
            solver = 'saga',
            fit_intercept = fit_intercept,
            max_iter = max_iter,
            cv = n_splits,
            use_legacy_attributes = False,
        )
        logisticModel.fit(
            X = X_ohe,
            y = y.to_numpy().ravel(),
        )

        elastic_coefficients = logisticModel.coef_
        if len( elastic_coefficients.shape ) == 1:
            pass
        elif elastic_coefficients.shape[0] == 1:
            elastic_coefficients = elastic_coefficients.reshape( -1 )
        else:
            coef_contrast: np.ndarray = elastic_coefficients[1:, :] - elastic_coefficients[0:1, :]
            if coef_contrast.shape[0] == 1:
                elastic_coefficients = np.abs( coef_contrast[0, :] )
            else:
                # Covariance from predicted log probabilities (one-hot convention: subtract first column)
                _log_proba: np.ndarray = logisticModel.predict_log_proba( X_ohe )  # (n, k)
                _log_proba_contrast: np.ndarray = _log_proba[:, 1:] - _log_proba[:, 0:1]  # (n, k-1)
                _cov: np.ndarray = np.cov( _log_proba_contrast, rowvar=False )  # (k-1, k-1)
                inv_cov: np.ndarray = np.linalg.inv( _cov )
                elastic_coefficients = np.sqrt(
                    np.einsum( 'kp,kl,lp->p', coef_contrast, inv_cov, coef_contrast )
                )
        #/switch elastic_coefficients.shape

    else:
        raise ValueError(
            "Unrecognized outcomeDescriptor.outcome_type='{}'".format(
                outcomeDescriptor.outcome_type,
            )
        )

    importances = np.fromiter(
        (
            np.abs( elastic_coefficients[ oheDict[col] ] )
            if isinstance( oheDict[col], int )
            else _collapse_cat_importance( elastic_coefficients, oheDict[col], col )
            for col in X_all_df.columns
        ),
        dtype = float,
    )**exponent

    return importances
#/def elasticImportances

# -- W stats for every importance measure

def wFromImportances(
    importances: np.ndarray,
    W_method: Literal['difference','signed_max'] = 'difference',
    verbose: int = 0
    ) -> np.ndarray:
    """
        :param np.ndarray importances: Importance measures, likely from ``importancesFromModel()``
        :param Literal['difference','signed_max'] W_method: How to calculate W statistics from importance measures, given the two most common methods.
        :param int verbose: How much to print out, for mostly for debugging.
        :returns: W statistics, half the length of `importances`, the same length as the original number of variables
        :rtype: np.ndarray
        
        Converts arbitrary importances to W statistics for the knockoff procedure.
    """
    p: int = len( importances ) // 2
    W_out: np.ndarray
    if W_method == 'difference':
        W_out = importances[ : p ] - importances[ p: ]
    #
    elif W_method == 'signed_max':
        W_out = np.zeros( shape = ( p,) )
        for j in range(p):
            if importances[ j ] > importances[ j+p ]:
                W_out[ j ] = importances[ j ]
            #
            elif importances[ j ] < importances[ j+p ]:
                W_out[ j ] = importances[ j+p ]
            #/switch importances[ j ] - importances[ j+p ]
        #/for j in range(p)
    else:
        raise ValueError("Unrecognized W_method={}".format(W_method))
    #
    return W_out
#/def wFromImportances

def selection_threshold(
    W: np.ndarray,
    fdr: float,
    offset: float = 1.0,
    ) -> float:
    """
        From knockpy.knockoff_stats.data_dependent_threshold
        
        :param offset: Adjustment. From `knockpy`:
            If offset = 0, control the modified FDR.
            If offset = 1 (default), controls the FDR exactly.
        https://github.com/amspector100/knockpy/blob/master/knockpy/knockoff_stats.py
    """
    # sort by abs values
    absW = np.abs(W)
    inds = np.argsort(-absW, kind="stable")
    negatives = np.cumsum(W[inds] <= 0)
    positives = np.cumsum(W[inds] > 0)
    positives[positives == 0] = 1  # Don't divide by 0
    # calc hat fdrs
    hat_fdrs = (negatives + offset) / positives
    # Minimum threshold such that hat_fdr <= nominal level
    if np.any(hat_fdrs <= fdr):
        T = absW[inds[np.where(hat_fdrs <= fdr)[0].max()]]
        if T == 0:
            T = np.min(W[W > 0])
    else:
        T = np.inf
    return T
#/def selection_threshold
