from .. import utilities
from ..utilities import OutcomeDescriptor, DataFrameLike, SeriesOrDataFrameLike, _resolve_df, _resolve_y

import numpy as np
import polars as pl
import torch

from typing import Iterable, Literal, Sequence

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


def _ridge_inv_cov_t( cov: torch.Tensor ) -> torch.Tensor:
    """Inverse of a Mahalanobis covariance, ridge-regularized relative to its own
    scale so a momentarily-degenerate (near-)constant logit-contrast doesn't produce
    an exactly-singular matrix or blow up the inverse. Trace-relative rather than a
    fixed absolute epsilon since `cov` is built from raw model logits (unlike e.g.
    the standardized-input eps=1e-8 floors elsewhere in this module).
    """
    if cov.ndim == 0:
        eps = max( 1e-6 * abs( cov.item() ), 1e-12 )
        return torch.tensor( [[ 1.0 / ( cov.item() + eps ) ]], device=cov.device, dtype=cov.dtype )
    #
    k = cov.shape[0]
    eps = max( ( 1e-6 * torch.trace( cov ) / k ).item(), 1e-12 )
    return torch.linalg.inv( cov + eps * torch.eye( k, device=cov.device, dtype=cov.dtype ) )
#/def _ridge_inv_cov_t


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
    bandwidth_exponent: float = 0.2,
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
            bandwidth = float( n ** -bandwidth_exponent )
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

    # -- OHE-encode X and Xk independently (not a single concatenated frame), so
    #    X_all_np/groups/oheDict lay out as [X's own columns, Xk's own columns] --
    #    the contract assumed by wFromImportances, calculatorOps.py's torch_prism_gw
    #    row-building, and _PRISMNetworkPairwise/_PRISMNetworkAdditive's
    #    `p = input_size // 2` split. Encoding a single concatenated frame instead
    #    groups columns by numeric-vs-categorical across X and Xk jointly (per
    #    get_ohe_df/get_oheDict's canonical "non-categorical first, then categorical"
    #    order), which only coincides with an X/Xk split when a dataset has no
    #    categorical columns.
    categorical_columns: tuple[ str, ... ] = tuple(
        col for col, dtype in X.schema.items() if dtype == pl.Categorical
    )
    categories_override: dict[ str, list[ str ] ] | None = None
    if categorical_columns:
        # Union of categories present in either X or Xk, so both sides encode to
        # the same dummy-column count even if a category is missing from one side's
        # realized sample.
        categories_override = {
            col: sorted(
                pl.concat( ( X[ col ], Xk[ col ] ), how = 'vertical' )
                .cast( pl.Utf8 ).unique().drop_nulls().to_list()
            )
            for col in categorical_columns
        }
    #

    X_np: np.ndarray = utilities.get_ohe_np(
        X = X, drop_first = drop_first, categories_override = categories_override,
    )
    Xk_np: np.ndarray = utilities.get_ohe_np(
        X = Xk, drop_first = drop_first, categories_override = categories_override,
    )
    X_all_np: np.ndarray = np.concatenate( ( X_np, Xk_np ), axis = 1 )

    X_oheDict: dict[ str, int | tuple[ int,... ] ] = utilities.get_oheDict(
        X = X, drop_first = drop_first, categories_override = categories_override,
    )
    p_ohe_x: int = X_np.shape[1]
    oheDict: dict[ str, int | tuple[ int,... ] ] = dict( X_oheDict )
    for col, idx in X_oheDict.items():
        oheDict[ col + '~' ] = (
            idx + p_ohe_x if isinstance( idx, int )
            else tuple( i + p_ohe_x for i in idx )
        )
    #

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
    model_type: str = 'pairwise',
    n_warmup: int = 0,
    vertical_prefit: bool = False,
    prefit_noise_std: float = 0.01,
    reset_optimizer: bool = True,
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

    :param model_type: see torchImportances.PRISMPredictionModel docstring for the full
        list ('mlp', 'pairwise', 'additive').
    :param lambda_path: Sequence of lambda values. Defaults to logspace(1,-2,50).
    :param a_path: Per-stage input-layer penalty values. If None, uses lambda_path values.
    :param epochs: Total training epochs, distributed as evenly as possible across lambda stages.
    :returns: Array of shape (2*p,) — first p entries for X, last p for Xk.
    """
    from . import torchImportances

    if lambda_path is None:
        lambda_path = _DEFAULT_LAMBDA_PATH
    #

    X_all_np, y_np, groups, oheDict, loss_func, output_dimension, _ = _prism_setup(
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
        n_warmup = n_warmup,
        vertical_prefit = vertical_prefit,
        prefit_noise_std = prefit_noise_std,
        reset_optimizer = reset_optimizer,
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


def prismWImportancesPerOHE(
    X: DataFrameLike,
    Xk: DataFrameLike,
    y: SeriesOrDataFrameLike,
    layers: Sequence[ int ],
    outcome_type: Literal['continuous','count','categorical',] | None = None,
    lambda_path: Sequence[ float ] | None = None,
    a_path: Iterable[ float ] | None = None,
    batch_size: int | None = None,
    epochs: int = 500,
    model_type: Literal['mlp','pairwise',] = 'mlp',
    n_warmup: int = 0,
    vertical_prefit: bool = False,
    prefit_noise_std: float = 0.01,
    reset_optimizer: bool = True,
    learning_rate: float = 0.01,
    drop_first: bool = True,
    dense_activation: str = 'relu',
    verbose: int = 0,
    ) -> np.ndarray:
    """
    PRISM-W importances, but every OHE dummy column is treated as its own independent
    variable instead of being grouped back to one score per original variable.

    Identical training procedure to prismWImportances, except `groups` is built as one
    singleton per OHE column (a categorical variable's dummy columns are NOT bundled
    into a shared group), so group_regularization/get_group_importances regularize and
    report each dummy column independently. Collinearity between dummies of the same
    variable is accepted; the group-lasso-style column penalty already regularizes it.

    Only 'mlp' and 'pairwise' are supported: 'additive' does not extend naturally to
    a per-dummy treatment.

    :param model_type: 'mlp' or 'pairwise' only.
    :param lambda_path: Sequence of lambda values. Defaults to logspace(1,-2,50).
    :param a_path: Per-stage input-layer penalty values. If None, uses lambda_path values.
    :param epochs: Total training epochs, distributed as evenly as possible across lambda stages.
    :returns: Array of shape (2*p_ohe,) — first p_ohe entries for X's OHE-expanded columns,
        last p_ohe for Xk's. p_ohe is the total OHE-expanded width per side (numeric columns
        contribute 1 entry each, a K-category column contributes K-1 entries under
        drop_first=True), NOT the original variable count.
    """
    from . import torchImportances

    if model_type not in ( 'mlp', 'pairwise' ):
        raise ValueError(
            "prismWImportancesPerOHE only supports model_type in ('mlp','pairwise'); "
            "got {!r}".format( model_type )
        )
    #

    if lambda_path is None:
        lambda_path = _DEFAULT_LAMBDA_PATH
    #

    X_all_np, y_np, _grouped_groups, oheDict, loss_func, output_dimension, _ = _prism_setup(
        X = X, Xk = Xk, y = y,
        layers = layers,
        outcome_type = outcome_type,
        drop_first = drop_first,
    )

    _mu = X_all_np.mean( axis=0 )
    _sd = np.maximum( X_all_np.std( axis=0 ), 1e-8 )
    X_all_np = ( X_all_np - _mu ) / _sd

    # Flatten oheDict: one singleton group per OHE column, instead of grouping a
    # categorical variable's dummy columns together.
    groups: list[ list[int] ] = []
    for col in oheDict:
        col_idx = oheDict[ col ]
        if isinstance( col_idx, int ):
            groups.append( [ col_idx ] )
        else:
            groups.extend( [ idx ] for idx in col_idx )
        #
    #/for col in oheDict

    predictionModel: torchImportances.PRISMPredictionModel = torchImportances.PRISMPredictionModel(
        input_size = X_all_np.shape[1],
        layers = list( layers ),
        dense_activation = dense_activation,
        loss_func = loss_func,
        output_dimension = output_dimension,
        learning_rate = learning_rate,
        epochs = epochs,
        model_type = model_type,
        n_warmup = n_warmup,
        vertical_prefit = vertical_prefit,
        prefit_noise_std = prefit_noise_std,
        reset_optimizer = reset_optimizer,
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
#/def prismWImportancesPerOHE


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
    model_type: str = 'pairwise',
    n_warmup: int = 0,
    vertical_prefit: bool = False,
    prefit_noise_std: float = 0.01,
    reset_optimizer: bool = True,
    learning_rate: float = 0.01,
    drop_first: bool = True,
    dense_activation: str = 'relu',
    verbose: int = 0,
    bandwidth_exponent: float = 0.2,
    ) -> np.ndarray:
    """
    PRISM-G importances: average of PRISM local-gradient snapshots over a lambda path.

    Same training procedure as prismWImportances; at the end of each lambda stage the
    PRISM importances (auto_diff or bandwidth) of the current model are recorded.
    Delegates snapshot computation to _prismImportances_t.

    :param model_type: see torchImportances.PRISMPredictionModel docstring for the full
        list ('mlp', 'pairwise', 'additive').
    :param local_grad_method: 'auto_diff' (exact) or 'bandwidth' (finite difference).
    :param lambda_path: Sequence of lambda values. Defaults to logspace(1,-2,50).
    :param a_path: Per-stage input-layer penalty values. If None, uses lambda_path values.
    :param epochs: Total training epochs, distributed as evenly as possible across lambda stages.
    :param bandwidth: Bandwidth for finite-difference approximation (auto-set if None).
    :param exponent: Power applied to each local gradient value before averaging.
    :param bandwidth_exponent: Exponent used for the auto-set bandwidth (n ** -bandwidth_exponent)
        when bandwidth is None. Ignored if bandwidth is given explicitly.
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
        n_warmup = n_warmup,
        vertical_prefit = vertical_prefit,
        prefit_noise_std = prefit_noise_std,
        reset_optimizer = reset_optimizer,
        verbose = verbose,
    )

    if outcomeDescriptor.outcome_type == 'categorical':
        def snapshot_fn( model: torchImportances.PRISMPredictionModel, X_t: torch.Tensor ) -> np.ndarray:
            with torch.no_grad():
                _logits = model.predict_t( X_t )
            logit_contrasts = _logits[:, 1:] - _logits[:, 0:1]
            _cov = torch.cov( logit_contrasts.T )
            inv_cov_t = _ridge_inv_cov_t( _cov )
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
                    bandwidth_exponent = bandwidth_exponent,
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
                bandwidth_exponent = bandwidth_exponent,
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
    model_type: str = 'pairwise',
    n_warmup: int = 0,
    vertical_prefit: bool = False,
    prefit_noise_std: float = 0.01,
    reset_optimizer: bool = True,
    learning_rate: float = 0.01,
    drop_first: bool = True,
    dense_activation: str = 'relu',
    verbose: int = 0,
    bandwidth_exponent: float = 0.2,
    ) -> tuple[ np.ndarray, np.ndarray ]:
    """
    PRISM-G and PRISM-W importances from a single training pass.

    Identical hyperparameters and model to prismGImportances / prismWImportances.
    At each lambda stage the snapshot_fn records PRISM-W group norms as a side
    effect while returning PRISM-G local-gradient importances as the primary snapshot.

    :param model_type: see torchImportances.PRISMPredictionModel docstring for the full
        list ('mlp', 'pairwise', 'additive').
    :param bandwidth_exponent: Exponent used for the auto-set bandwidth (n ** -bandwidth_exponent)
        when bandwidth is None. Ignored if bandwidth is given explicitly.
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
        n_warmup = n_warmup,
        vertical_prefit = vertical_prefit,
        prefit_noise_std = prefit_noise_std,
        reset_optimizer = reset_optimizer,
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
            inv_cov_t = _ridge_inv_cov_t( _cov )
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
                    bandwidth_exponent = bandwidth_exponent,
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
                bandwidth_exponent = bandwidth_exponent,
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
    layers:            Sequence[int],
    outcome_type:      Literal["continuous","count","categorical"] | None = None,
    local_grad_method: Literal["auto_diff","bandwidth"] = 'bandwidth',
    lambda_path:       Sequence[float] | None = None,
    a_path:            Sequence[float] | None = None,
    batch_size:        int | None = None,
    epochs:            int = 500,
    bandwidth:         float | None = None,
    model_type:        str = 'pairwise',
    n_warmup:          int = 0,
    vertical_prefit:   bool = False,
    prefit_noise_std:  float = 0.01,
    reset_optimizer:   bool = True,
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
        n_warmup         = n_warmup,
        vertical_prefit  = vertical_prefit,
        prefit_noise_std = prefit_noise_std,
        reset_optimizer  = reset_optimizer,
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
