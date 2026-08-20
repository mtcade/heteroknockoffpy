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
    ) -> torch.Tensor:
    """
    Tensor-native PRISM importance computation. Returns shape (p_out,) tensor.
    model must have predict_t and auto_diff_t methods.

    `bandwidth` is used exactly as given -- there is no sample-size- or
    column-scale-derived auto-bandwidth. `X_all_t` is already standardized to
    unit variance by the callers below, so bandwidth=1.0 (the proposal's
    central difference at +/-1) is the natural default; re-deriving a
    bandwidth from `n` on top of that would double-scale it.
    """
    n = X_all_t.shape[0]
    p_out = len( oheDict )

    if local_grad_method == 'auto_diff':
        auto_diff_full_t: torch.Tensor = model.auto_diff_t( X_all_t )  # (n, p_ohe)
    elif local_grad_method == 'bandwidth':
        if bandwidth is None:
            raise ValueError(
                "local_grad_method='bandwidth' requires an explicit bandwidth "
                "(no auto-scaling from n is applied)."
            )
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


_DEFAULT_N_BLOCKS:    int   = 30
_LAMBDA_LOGUNIF_LOW:  float = 1e-3
_LAMBDA_LOGUNIF_HIGH: float = 1e-1
_A_UNIF_LOW:          float = 0.3
_A_UNIF_HIGH:         float = 1.0


def _resolve_lambda_a_path(
    lambda_path: Sequence[ float ] | None,
    a_path: Sequence[ float ] | None,
    rng: np.random.Generator | None,
    n_blocks: int | None = None,
    calibrate: bool = False,
    lambda_min: float | None = None,
    lambda_max: float | None = None,
    a_min: float | None = None,
    a_max: float | None = None,
    ) -> tuple[ list[float] | None, list[float] | None ]:
    """
    Resolve the (lambda_path, a_path) BSS schedule per the PRISM proposal's Monte
    Carlo scheme: lambda_b ~ LogUniform(lambda_min, lambda_max), a_b ~
    Uniform(a_min, a_max), drawn INDEPENDENTLY per block -- not a_b = lambda_b
    (the old mirroring default let a > 1 slip through whenever lambda_path
    ranged above 1).

    calibrate=True defers path resolution entirely to
    PRISMPredictionModel.fit() (GRIP2 Eq. 5's gradient-ratio calibration needs
    a post-warmup model + real data, neither of which exist yet at this call
    site) -- this function only validates in that case and returns
    (None, None); lambda_path/a_path/lambda_min/lambda_max/a_min/a_max must
    all be omitted so calibration has sole authority over the (lambda, a)
    support. n_blocks is the one calibrate-compatible parameter (it controls
    how many blocks calibration itself samples/averages over) but is always
    incompatible with an explicit lambda_path/a_path, calibrate or not --
    n_blocks only means something when a path is being auto-generated.

    A `def` default can't itself express "sample fresh values each call" -- so
    lambda_path=None / a_path=None are handled dynamically here rather than as a
    fixed array: omitting either triggers a new random draw from `rng`
    (np.random.default_rng() if rng is None) at call time. Callers who want a
    reproducible, inspectable path should draw one themselves (mirroring the
    LogUniform/Uniform formulas above) and pass it in explicitly rather than
    relying on this default. a_path=None always draws independently at the
    resolved lambda_path's length, even when lambda_path was supplied explicitly.
    """
    if calibrate:
        for _name, _val in (
            ( 'lambda_path', lambda_path ), ( 'a_path', a_path ),
            ( 'lambda_min', lambda_min ), ( 'lambda_max', lambda_max ),
            ( 'a_min', a_min ), ( 'a_max', a_max ),
        ):
            if _val is not None:
                raise ValueError(
                    "_resolve_lambda_a_path: calibrate=True determines {0} itself "
                    "(via GRIP2 Eq. 5's gradient-ratio calibration); got an "
                    "explicit {0}={1!r}. Pass calibrate=False to use your own "
                    "value, or drop it to let calibration choose it.".format( _name, _val )
                )
            #
        #/for _name, _val
    #/if calibrate

    if n_blocks is not None and lambda_path is not None:
        raise ValueError(
            "_resolve_lambda_a_path: n_blocks only applies when lambda_path is "
            "auto-generated (lambda_path=None); got n_blocks={} with an explicit "
            "lambda_path of length {}. Drop n_blocks, or drop lambda_path to let "
            "it control the auto-generated path length.".format( n_blocks, len( lambda_path ) )
        )
    #
    if n_blocks is not None and a_path is not None:
        raise ValueError(
            "_resolve_lambda_a_path: n_blocks only applies when a_path is "
            "auto-generated (a_path=None); got n_blocks={} with an explicit "
            "a_path of length {}.".format( n_blocks, len( a_path ) )
        )
    #

    if calibrate:
        return None, None
    #

    _rng = rng if rng is not None else np.random.default_rng()
    _n_blocks = n_blocks if n_blocks is not None else _DEFAULT_N_BLOCKS

    if lambda_path is None:
        _lam_lo = lambda_min if lambda_min is not None else _LAMBDA_LOGUNIF_LOW
        _lam_hi = lambda_max if lambda_max is not None else _LAMBDA_LOGUNIF_HIGH
        log_low, log_high = np.log10( _lam_lo ), np.log10( _lam_hi )
        _lp = list( 10.0 ** _rng.uniform( log_low, log_high, size=_n_blocks ) )
    else:
        _lp = list( lambda_path )
    #

    if a_path is None:
        _a_lo = a_min if a_min is not None else _A_UNIF_LOW
        _a_hi = a_max if a_max is not None else _A_UNIF_HIGH
        _ap = list( _rng.uniform( _a_lo, _a_hi, size=len( _lp ) ) )
    else:
        _ap = list( a_path )
    #

    return _lp, _ap
#/def _resolve_lambda_a_path


def prismWImportances(
    X: DataFrameLike,
    Xk: DataFrameLike,
    y: SeriesOrDataFrameLike,
    layers: Sequence[ int ],
    outcome_type: Literal['continuous','count','categorical',] | None = None,
    lambda_path: Sequence[ float ] | None = None,
    a_path: Iterable[ float ] | None = None,
    n_blocks: int | None = None,
    calibrate: bool = False,
    lambda_min: float | None = None,
    lambda_max: float | None = None,
    a_min: float | None = None,
    a_max: float | None = None,
    calibrate_rmin: float = 0.01,
    calibrate_rmax: float = 1.0,
    batch_size: int | None = None,
    epochs: int = 500,
    total_steps: int | None = None,
    model_type: str = 'mlp',
    n_warmup: int = 5000,
    vertical_prefit: bool = False,
    prefit_noise_std: float = 0.01,
    reset_optimizer: bool = True,
    learning_rate: float = 0.01,
    drop_first: bool = True,
    dense_activation: str = 'relu',
    verbose: int = 0,
    weight: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
    categorical_collapse_method: Literal['l2_norm','range'] = 'l2_norm',
    ) -> np.ndarray:
    """
    PRISM-W importances: average of group-norm snapshots over a lambda regularization path.

    Trains a single MLP on [X, Xk] → y with an adaptive proximal penalty on the input layer.
    At the end of each lambda stage the group norms are recorded; the final importances are
    the mean over all snapshots.

    :param model_type: see torchImportances.PRISMPredictionModel docstring for the full
        list ('mlp', 'pairwise', 'additive').
    :param categorical_collapse_method: How a categorical (multi-column OHE) group's
        per-category column norms collapse into one importance value; ignored for
        numeric (singleton) groups, which always use their own column norm.
        - 'l2_norm' (default): ||w[:, group_j]||_F, the Frobenius norm of the whole
          group's weight block -- equivalently the L2 norm of the group's per-column
          norm vector. Aggregates signal across every category, so it's stronger
          when a categorical variable's true effect is spread across several/all of
          its categories, but gets diluted when the effect is concentrated in one
          category.
        - 'range': a zero-anchored max-min spread over the group's per-column norms
          (`max(max_j, 0) - min(min_j, 0)`) -- isolates the single most extreme
          category's column norm instead of aggregating. Stronger when only one
          category actually deviates; discards other categories' real signal when
          the effect is spread across several.
    :param lambda_path: Sequence of lambda values. If None (default), a fresh path of
        `n_blocks` values is drawn from LogUniform(1e-3, 1e-1) via `rng` at call time --
        see `_resolve_lambda_a_path`. Pass an explicit sequence for a reproducible,
        inspectable path instead of relying on this dynamic default.
    :param a_path: Per-stage input-layer penalty values. If None (default), drawn
        independently from Uniform(0.3, 1) via `rng`, at the same length as the
        resolved `lambda_path` (NOT mirrored from lambda_path's values).
    :param n_blocks: Number of BSS blocks/stages to draw when `lambda_path` is None
        (including under `calibrate=True`, where it also controls how many blocks
        the calibration pilot pass averages its gradient ratio over). Incompatible
        with an explicit `lambda_path`/`a_path` -- raises `ValueError` if both are given.
    :param calibrate: If True, runs GRIP2 Eq. 5's gradient-ratio calibration (on the
        post-warmup model) to derive `lambda_path` instead of requiring it; `a_path` is
        still drawn from `Uniform(a_min, a_max)` and the calibration ratio is averaged
        over it. Mutually exclusive with `lambda_path`, `a_path`, `lambda_min`,
        `lambda_max`, `a_min`, `a_max` -- raises `ValueError` if any of those are given
        alongside `calibrate=True`, since calibration determines all of them itself.
    :param lambda_min: Lower bound for `lambda_path`'s `LogUniform` draw when
        `lambda_path` is None and `calibrate=False`. Defaults to this module's
        `_LAMBDA_LOGUNIF_LOW` (1e-3) when omitted. Per GRIP2, `lambda`'s range has no
        universal fixed bound (unlike `a`'s fixed upper bound of 1) -- both endpoints
        are always context-dependent, whether set manually here or derived by `calibrate`.
    :param lambda_max: Upper bound for that same draw; defaults to `_LAMBDA_LOGUNIF_HIGH`
        (1e-1) when omitted.
    :param a_min: Lower bound for `a_path`'s `Uniform` draw when `a_path` is None.
        Defaults to this module's `_A_UNIF_LOW` (0.3) when omitted. GRIP2's own
        recommended default is 0.1; this codebase's historical default is 0.3.
    :param a_max: Upper bound for that same draw; defaults to `_A_UNIF_HIGH` (1.0) when
        omitted -- GRIP2 always fixes `a`'s upper bound at the literal constant 1, so
        this is exposed for override/symmetry with `a_min` rather than because the
        paper ever varies it.
    :param calibrate_rmin: Target lower bound for the gradient-ratio
        `||grad_W R|| / ||grad_W L_pred||` that `calibrate=True` solves for (GRIP2
        Eq. 5's `r_min`). Only consulted when `calibrate=True`.
    :param calibrate_rmax: Target upper bound for that same ratio (`r_max`). Paper
        recommends 0.20 for exact/well-conditioned knockoffs, 1.0 (this function's
        default) for approximate/ill-conditioned ones. Only consulted when `calibrate=True`.
        Ignored if `lambda_path` is given explicitly.
    :param epochs: Full-batch-equivalent training passes. Converted internally to a raw
        step budget (`epochs * ceil(n / effective_batch_size)`), which is what's actually
        distributed -- as evenly as possible in raw-step units, not whole epochs -- across
        lambda stages, so a block can end mid-epoch instead of being quantized to whole
        passes over the data. Ignored if `total_steps` is given.
    :param total_steps: Overrides the `epochs`-derived step budget with an exact step
        count, independent of `n`/`batch_size` (mirrors HuggingFace Trainer's `max_steps`
        overriding `num_train_epochs`). None (default) falls back to the `epochs`-derived
        budget above. Either way, changing `n_blocks`/`lambda_path` length only changes how
        finely this fixed total budget is sliced across BSS blocks, never the total itself.
    :param rng: Seeds both the default lambda/a-path draw above and torch's global RNG
        (via `PRISMPredictionModel`) for full run-to-run reproducibility. Unseeded if omitted.
    :returns: Array of shape (2*p,) — first p entries for X, last p for Xk.
    """
    from . import torchImportances

    lambda_path, a_path = _resolve_lambda_a_path(
        lambda_path, a_path, rng, n_blocks,
        calibrate = calibrate, lambda_min = lambda_min, lambda_max = lambda_max,
        a_min = a_min, a_max = a_max,
    )
    _resolved_n_blocks = n_blocks if n_blocks is not None else _DEFAULT_N_BLOCKS
    _resolved_a_min    = a_min if a_min is not None else _A_UNIF_LOW
    _resolved_a_max    = a_max if a_max is not None else _A_UNIF_HIGH

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
        rng = rng,
    )

    snapshots: list[ np.ndarray ] = predictionModel.fit(
        X = X_all_np,
        y = y_np,
        groups = groups,
        lambda_path = lambda_path,
        a_path = a_path,
        calibrate = calibrate,
        n_blocks = _resolved_n_blocks,
        a_min = _resolved_a_min,
        a_max = _resolved_a_max,
        calibrate_rmin = calibrate_rmin,
        calibrate_rmax = calibrate_rmax,
        batch_size = batch_size,
        total_steps = total_steps,
        weight = weight,
        categorical_collapse_method = categorical_collapse_method,
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
    n_blocks: int | None = None,
    calibrate: bool = False,
    lambda_min: float | None = None,
    lambda_max: float | None = None,
    a_min: float | None = None,
    a_max: float | None = None,
    calibrate_rmin: float = 0.01,
    calibrate_rmax: float = 1.0,
    batch_size: int | None = None,
    epochs: int = 500,
    total_steps: int | None = None,
    model_type: Literal['mlp','pairwise',] = 'mlp',
    n_warmup: int = 5000,
    vertical_prefit: bool = False,
    prefit_noise_std: float = 0.01,
    reset_optimizer: bool = True,
    learning_rate: float = 0.01,
    drop_first: bool = True,
    dense_activation: str = 'relu',
    verbose: int = 0,
    weight: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
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
    :param lambda_path: Sequence of lambda values. See `prismWImportances`'s docstring --
        if None (default), a fresh path of `n_blocks` values is drawn from
        LogUniform(1e-3, 1e-1) via `rng` at call time.
    :param a_path: Per-stage input-layer penalty values. If None, drawn independently
        from Uniform(0.3, 1) via `rng`, at the resolved lambda_path's length.
    :param n_blocks: Number of BSS blocks/stages to draw when `lambda_path` is None
        (including under `calibrate=True`, where it also controls how many blocks
        the calibration pilot pass averages its gradient ratio over). Incompatible
        with an explicit `lambda_path`/`a_path` -- raises `ValueError` if both are given.
    :param calibrate: If True, runs GRIP2 Eq. 5's gradient-ratio calibration (on the
        post-warmup model) to derive `lambda_path` instead of requiring it; `a_path` is
        still drawn from `Uniform(a_min, a_max)` and the calibration ratio is averaged
        over it. Mutually exclusive with `lambda_path`, `a_path`, `lambda_min`,
        `lambda_max`, `a_min`, `a_max` -- raises `ValueError` if any of those are given
        alongside `calibrate=True`, since calibration determines all of them itself.
    :param lambda_min: Lower bound for `lambda_path`'s `LogUniform` draw when
        `lambda_path` is None and `calibrate=False`. Defaults to this module's
        `_LAMBDA_LOGUNIF_LOW` (1e-3) when omitted. Per GRIP2, `lambda`'s range has no
        universal fixed bound (unlike `a`'s fixed upper bound of 1) -- both endpoints
        are always context-dependent, whether set manually here or derived by `calibrate`.
    :param lambda_max: Upper bound for that same draw; defaults to `_LAMBDA_LOGUNIF_HIGH`
        (1e-1) when omitted.
    :param a_min: Lower bound for `a_path`'s `Uniform` draw when `a_path` is None.
        Defaults to this module's `_A_UNIF_LOW` (0.3) when omitted. GRIP2's own
        recommended default is 0.1; this codebase's historical default is 0.3.
    :param a_max: Upper bound for that same draw; defaults to `_A_UNIF_HIGH` (1.0) when
        omitted -- GRIP2 always fixes `a`'s upper bound at the literal constant 1, so
        this is exposed for override/symmetry with `a_min` rather than because the
        paper ever varies it.
    :param calibrate_rmin: Target lower bound for the gradient-ratio
        `||grad_W R|| / ||grad_W L_pred||` that `calibrate=True` solves for (GRIP2
        Eq. 5's `r_min`). Only consulted when `calibrate=True`.
    :param calibrate_rmax: Target upper bound for that same ratio (`r_max`). Paper
        recommends 0.20 for exact/well-conditioned knockoffs, 1.0 (this function's
        default) for approximate/ill-conditioned ones. Only consulted when `calibrate=True`.
    :param epochs: Full-batch-equivalent training passes. Converted internally to a raw
        step budget (`epochs * ceil(n / effective_batch_size)`), which is what's actually
        distributed -- as evenly as possible in raw-step units, not whole epochs -- across
        lambda stages, so a block can end mid-epoch instead of being quantized to whole
        passes over the data. Ignored if `total_steps` is given.
    :param total_steps: Overrides the `epochs`-derived step budget with an exact step
        count, independent of `n`/`batch_size` (mirrors HuggingFace Trainer's `max_steps`
        overriding `num_train_epochs`). None (default) falls back to the `epochs`-derived
        budget above. Either way, changing `n_blocks`/`lambda_path` length only changes how
        finely this fixed total budget is sliced across BSS blocks, never the total itself.
    :param rng: Seeds the default lambda/a-path draw and torch's global RNG for full
        run-to-run reproducibility. Unseeded if omitted.
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

    lambda_path, a_path = _resolve_lambda_a_path(
        lambda_path, a_path, rng, n_blocks,
        calibrate = calibrate, lambda_min = lambda_min, lambda_max = lambda_max,
        a_min = a_min, a_max = a_max,
    )
    _resolved_n_blocks = n_blocks if n_blocks is not None else _DEFAULT_N_BLOCKS
    _resolved_a_min    = a_min if a_min is not None else _A_UNIF_LOW
    _resolved_a_max    = a_max if a_max is not None else _A_UNIF_HIGH

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
        rng = rng,
    )

    snapshots: list[ np.ndarray ] = predictionModel.fit(
        X = X_all_np,
        y = y_np,
        groups = groups,
        lambda_path = lambda_path,
        a_path = a_path,
        calibrate = calibrate,
        n_blocks = _resolved_n_blocks,
        a_min = _resolved_a_min,
        a_max = _resolved_a_max,
        calibrate_rmin = calibrate_rmin,
        calibrate_rmax = calibrate_rmax,
        batch_size = batch_size,
        total_steps = total_steps,
        weight = weight,
    )

    return np.mean( snapshots, axis = 0 )
#/def prismWImportancesPerOHE


def prismGImportances(
    X: DataFrameLike,
    Xk: DataFrameLike,
    y: SeriesOrDataFrameLike,
    layers: Sequence[ int ],
    outcome_type: Literal['continuous','count','categorical',] | None = None,
    local_grad_method: Literal['auto_diff','bandwidth'] = 'bandwidth',
    lambda_path: Sequence[ float ] | None = None,
    a_path: Iterable[ float ] | None = None,
    n_blocks: int | None = None,
    calibrate: bool = False,
    lambda_min: float | None = None,
    lambda_max: float | None = None,
    a_min: float | None = None,
    a_max: float | None = None,
    calibrate_rmin: float = 0.01,
    calibrate_rmax: float = 1.0,
    batch_size: int | None = None,
    epochs: int = 500,
    total_steps: int | None = None,
    bandwidth: float | None = 1.0,
    exponent: float = 1.0,
    model_type: str = 'mlp',
    n_warmup: int = 5000,
    vertical_prefit: bool = False,
    prefit_noise_std: float = 0.01,
    reset_optimizer: bool = True,
    learning_rate: float = 0.01,
    drop_first: bool = True,
    dense_activation: str = 'relu',
    verbose: int = 0,
    weight: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
    ) -> np.ndarray:
    """
    PRISM-G importances: average of PRISM local-gradient snapshots over a lambda path.

    Same training procedure as prismWImportances; at the end of each lambda stage the
    PRISM importances (auto_diff or bandwidth) of the current model are recorded.
    Delegates snapshot computation to _prismImportances_t.

    :param model_type: see torchImportances.PRISMPredictionModel docstring for the full
        list ('mlp', 'pairwise', 'additive').
    :param local_grad_method: 'auto_diff' (exact autodiff gradient) or 'bandwidth'
        (finite difference). Default 'bandwidth', matching the proposal's central
        difference statistic exactly once combined with `bandwidth=1.0` below -- X
        is already standardized to unit variance before this step, so a bandwidth of
        1.0 IS the proposal's "central difference at +/-1."
    :param lambda_path: Sequence of lambda values. See `prismWImportances`'s docstring --
        if None (default), a fresh path of `n_blocks` values is drawn from
        LogUniform(1e-3, 1e-1) via `rng` at call time.
    :param a_path: Per-stage input-layer penalty values. If None, drawn independently
        from Uniform(0.3, 1) via `rng`, at the resolved lambda_path's length.
    :param n_blocks: Number of BSS blocks/stages to draw when `lambda_path` is None
        (including under `calibrate=True`, where it also controls how many blocks
        the calibration pilot pass averages its gradient ratio over). Incompatible
        with an explicit `lambda_path`/`a_path` -- raises `ValueError` if both are given.
    :param calibrate: If True, runs GRIP2 Eq. 5's gradient-ratio calibration (on the
        post-warmup model) to derive `lambda_path` instead of requiring it; `a_path` is
        still drawn from `Uniform(a_min, a_max)` and the calibration ratio is averaged
        over it. Mutually exclusive with `lambda_path`, `a_path`, `lambda_min`,
        `lambda_max`, `a_min`, `a_max` -- raises `ValueError` if any of those are given
        alongside `calibrate=True`, since calibration determines all of them itself.
    :param lambda_min: Lower bound for `lambda_path`'s `LogUniform` draw when
        `lambda_path` is None and `calibrate=False`. Defaults to this module's
        `_LAMBDA_LOGUNIF_LOW` (1e-3) when omitted. Per GRIP2, `lambda`'s range has no
        universal fixed bound (unlike `a`'s fixed upper bound of 1) -- both endpoints
        are always context-dependent, whether set manually here or derived by `calibrate`.
    :param lambda_max: Upper bound for that same draw; defaults to `_LAMBDA_LOGUNIF_HIGH`
        (1e-1) when omitted.
    :param a_min: Lower bound for `a_path`'s `Uniform` draw when `a_path` is None.
        Defaults to this module's `_A_UNIF_LOW` (0.3) when omitted. GRIP2's own
        recommended default is 0.1; this codebase's historical default is 0.3.
    :param a_max: Upper bound for that same draw; defaults to `_A_UNIF_HIGH` (1.0) when
        omitted -- GRIP2 always fixes `a`'s upper bound at the literal constant 1, so
        this is exposed for override/symmetry with `a_min` rather than because the
        paper ever varies it.
    :param calibrate_rmin: Target lower bound for the gradient-ratio
        `||grad_W R|| / ||grad_W L_pred||` that `calibrate=True` solves for (GRIP2
        Eq. 5's `r_min`). Only consulted when `calibrate=True`.
    :param calibrate_rmax: Target upper bound for that same ratio (`r_max`). Paper
        recommends 0.20 for exact/well-conditioned knockoffs, 1.0 (this function's
        default) for approximate/ill-conditioned ones. Only consulted when `calibrate=True`.
    :param epochs: Full-batch-equivalent training passes. Converted internally to a raw
        step budget (`epochs * ceil(n / effective_batch_size)`), which is what's actually
        distributed -- as evenly as possible in raw-step units, not whole epochs -- across
        lambda stages, so a block can end mid-epoch instead of being quantized to whole
        passes over the data. Ignored if `total_steps` is given.
    :param total_steps: Overrides the `epochs`-derived step budget with an exact step
        count, independent of `n`/`batch_size` (mirrors HuggingFace Trainer's `max_steps`
        overriding `num_train_epochs`). None (default) falls back to the `epochs`-derived
        budget above. Either way, changing `n_blocks`/`lambda_path` length only changes how
        finely this fixed total budget is sliced across BSS blocks, never the total itself.
    :param bandwidth: Bandwidth for the finite-difference approximation when
        `local_grad_method='bandwidth'`. Used exactly as given -- there is no
        auto-scaling from `n` or column std on top of it. Default `1.0`.
    :param exponent: Power applied to each local gradient value before averaging.
    :param rng: Seeds the default lambda/a-path draw and torch's global RNG for full
        run-to-run reproducibility. Unseeded if omitted.
    :returns: Array of shape (2*p,).
    """
    from . import torchImportances

    lambda_path, a_path = _resolve_lambda_a_path(
        lambda_path, a_path, rng, n_blocks,
        calibrate = calibrate, lambda_min = lambda_min, lambda_max = lambda_max,
        a_min = a_min, a_max = a_max,
    )
    _resolved_n_blocks = n_blocks if n_blocks is not None else _DEFAULT_N_BLOCKS
    _resolved_a_min    = a_min if a_min is not None else _A_UNIF_LOW
    _resolved_a_max    = a_max if a_max is not None else _A_UNIF_HIGH

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
        rng = rng,
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
        calibrate = calibrate,
        n_blocks = _resolved_n_blocks,
        a_min = _resolved_a_min,
        a_max = _resolved_a_max,
        calibrate_rmin = calibrate_rmin,
        calibrate_rmax = calibrate_rmax,
        batch_size = batch_size,
        total_steps = total_steps,
        snapshot_fn = snapshot_fn,
        weight = weight,
    )

    return np.mean( snapshots, axis = 0 )
#/def prismGImportances


def prismGWImportances(
    X: DataFrameLike,
    Xk: DataFrameLike,
    y: SeriesOrDataFrameLike,
    layers: Sequence[ int ],
    outcome_type: Literal['continuous','count','categorical',] | None = None,
    local_grad_method: Literal['auto_diff','bandwidth'] = 'bandwidth',
    lambda_path: Sequence[ float ] | None = None,
    a_path: Iterable[ float ] | None = None,
    n_blocks: int | None = None,
    calibrate: bool = False,
    lambda_min: float | None = None,
    lambda_max: float | None = None,
    a_min: float | None = None,
    a_max: float | None = None,
    calibrate_rmin: float = 0.01,
    calibrate_rmax: float = 1.0,
    batch_size: int | None = None,
    epochs: int = 500,
    total_steps: int | None = None,
    bandwidth: float | None = 1.0,
    exponent: float = 1.0,
    model_type: str = 'mlp',
    n_warmup: int = 5000,
    vertical_prefit: bool = False,
    prefit_noise_std: float = 0.01,
    reset_optimizer: bool = True,
    learning_rate: float = 0.01,
    drop_first: bool = True,
    dense_activation: str = 'relu',
    verbose: int = 0,
    weight: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
    categorical_collapse_method: Literal['l2_norm','range'] = 'l2_norm',
    ) -> tuple[ np.ndarray, np.ndarray ]:
    """
    PRISM-G and PRISM-W importances from a single training pass.

    Identical hyperparameters and model to prismGImportances / prismWImportances.
    At each lambda stage the snapshot_fn records PRISM-W group norms as a side
    effect while returning PRISM-G local-gradient importances as the primary snapshot.

    :param model_type: see torchImportances.PRISMPredictionModel docstring for the full
        list ('mlp', 'pairwise', 'additive').
    :param categorical_collapse_method: See `prismWImportances` -- applies only to
        the PRISM-W side snapshots (`w_snapshots`); PRISM-G's own categorical
        handling is unaffected.
    :param local_grad_method: See `prismGImportances`. Default 'bandwidth'.
    :param bandwidth: Used exactly as given, no auto-scaling from `n`. Default `1.0`,
        matching the proposal's central difference at +/-1 on standardized X.
    :param lambda_path: See `prismWImportances` -- if None, drawn from
        LogUniform(1e-3, 1e-1) via `rng`.
    :param a_path: If None, drawn independently from Uniform(0.3, 1) via `rng`.
    :param n_blocks: Number of BSS blocks/stages to draw when `lambda_path` is None
        (including under `calibrate=True`, where it also controls how many blocks
        the calibration pilot pass averages its gradient ratio over). Incompatible
        with an explicit `lambda_path`/`a_path` -- raises `ValueError` if both are given.
    :param calibrate: If True, runs GRIP2 Eq. 5's gradient-ratio calibration (on the
        post-warmup model) to derive `lambda_path` instead of requiring it; `a_path` is
        still drawn from `Uniform(a_min, a_max)` and the calibration ratio is averaged
        over it. Mutually exclusive with `lambda_path`, `a_path`, `lambda_min`,
        `lambda_max`, `a_min`, `a_max` -- raises `ValueError` if any of those are given
        alongside `calibrate=True`, since calibration determines all of them itself.
    :param lambda_min: Lower bound for `lambda_path`'s `LogUniform` draw when
        `lambda_path` is None and `calibrate=False`. Defaults to this module's
        `_LAMBDA_LOGUNIF_LOW` (1e-3) when omitted. Per GRIP2, `lambda`'s range has no
        universal fixed bound (unlike `a`'s fixed upper bound of 1) -- both endpoints
        are always context-dependent, whether set manually here or derived by `calibrate`.
    :param lambda_max: Upper bound for that same draw; defaults to `_LAMBDA_LOGUNIF_HIGH`
        (1e-1) when omitted.
    :param a_min: Lower bound for `a_path`'s `Uniform` draw when `a_path` is None.
        Defaults to this module's `_A_UNIF_LOW` (0.3) when omitted. GRIP2's own
        recommended default is 0.1; this codebase's historical default is 0.3.
    :param a_max: Upper bound for that same draw; defaults to `_A_UNIF_HIGH` (1.0) when
        omitted -- GRIP2 always fixes `a`'s upper bound at the literal constant 1, so
        this is exposed for override/symmetry with `a_min` rather than because the
        paper ever varies it.
    :param calibrate_rmin: Target lower bound for the gradient-ratio
        `||grad_W R|| / ||grad_W L_pred||` that `calibrate=True` solves for (GRIP2
        Eq. 5's `r_min`). Only consulted when `calibrate=True`.
    :param calibrate_rmax: Target upper bound for that same ratio (`r_max`). Paper
        recommends 0.20 for exact/well-conditioned knockoffs, 1.0 (this function's
        default) for approximate/ill-conditioned ones. Only consulted when `calibrate=True`.
    :param epochs: See `prismWImportances`'s docstring -- converted to a raw step budget,
        distributed evenly across lambda stages in raw-step units. Ignored if `total_steps`
        is given.
    :param total_steps: See `prismWImportances`'s docstring -- overrides the `epochs`-derived
        step budget with an exact step count.
    :param rng: Seeds the default lambda/a-path draw and torch's global RNG.
    :returns: (g_importances, w_importances) both of shape (2*p,).
    """
    from . import torchImportances

    lambda_path, a_path = _resolve_lambda_a_path(
        lambda_path, a_path, rng, n_blocks,
        calibrate = calibrate, lambda_min = lambda_min, lambda_max = lambda_max,
        a_min = a_min, a_max = a_max,
    )
    _resolved_n_blocks = n_blocks if n_blocks is not None else _DEFAULT_N_BLOCKS
    _resolved_a_min    = a_min if a_min is not None else _A_UNIF_LOW
    _resolved_a_max    = a_max if a_max is not None else _A_UNIF_HIGH

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
        rng = rng,
    )

    w_snapshots: list[ np.ndarray ] = []

    if outcomeDescriptor.outcome_type == 'categorical':
        def snapshot_fn( model: torchImportances.PRISMPredictionModel, X_t: torch.Tensor ) -> np.ndarray:
            w_snapshots.append( model.get_group_importances( groups, categorical_collapse_method ) )
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
                ).cpu().numpy()
            #
        #/def snapshot_fn
    else:
        def snapshot_fn( model: torchImportances.PRISMPredictionModel, X_t: torch.Tensor ) -> np.ndarray:
            w_snapshots.append( model.get_group_importances( groups, categorical_collapse_method ) )
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
        calibrate = calibrate,
        n_blocks = _resolved_n_blocks,
        a_min = _resolved_a_min,
        a_max = _resolved_a_max,
        calibrate_rmin = calibrate_rmin,
        calibrate_rmax = calibrate_rmax,
        batch_size = batch_size,
        total_steps = total_steps,
        snapshot_fn = snapshot_fn,
        weight = weight,
    )

    return np.mean( g_snapshots, axis=0 ), np.mean( w_snapshots, axis=0 )
#/def prismGWImportances


def _get_localGrad_ohe_matrix_t(
    model:             'object',
    X_all_t:           torch.Tensor,
    x_oheDict:         dict,
    local_grad_method: str,
    bandwidth:         float | None,
    output_dimension:  int = 1,
    cat_ohe_vals:      dict[ int, tuple[ float, float ] ] | None = None,
    ) -> torch.Tensor:
    """
    Per-sample local gradient matrix for X-only features (not Xk).

    For numeric variables: bandwidth finite-diff or auto_diff gradient (one column each).
    For categorical variables: model-prediction contrast vs. reference category (drop_first=True
    convention, so c-1 columns per variable; category 0 is the reference).

    x_oheDict: oheDict filtered to X columns only (keys without '~').
    Returns tensor of shape (n, p_ohe_x) where p_ohe_x = p_numeric + sum(c_k - 1).

    `bandwidth` is used exactly as given -- no auto-scaling from `n`, matching
    `_prismImportances_t`.

    `cat_ohe_vals`: per-OHE-column-index (0.0-value, 1.0-value) pair, standardized
    the same way as `prismGImportances`'s `cat_ohe_vals` -- required whenever
    `X_all_t` has been standardized (its 0/1 dummy encoding no longer literally
    means 0.0/1.0), so the categorical branch below evaluates the reference/active
    states at the correct standardized values instead of raw 0.0/1.0. `None` keeps
    the legacy raw-0.0/1.0 behavior, for callers that pass unstandardized input.

    Only `output_dimension == 1` (continuous/count outcomes) is supported: for a
    multiclass outcome, `model.predict_t` returns (n, k) logits per sample, and
    there is no established single-column reduction of that into this function's
    (n, p_ohe_x) per-sample-scalar-gradient contract (unlike prismGImportances,
    which aggregates via a Mahalanobis distance into one importance number).
    """
    if output_dimension != 1:
        raise NotImplementedError(
            "_get_localGrad_ohe_matrix_t only supports output_dimension=1 "
            "(continuous/count outcomes); got output_dimension={}. Categorical "
            "outcomes have no established per-sample scalar-gradient reduction "
            "here.".format( output_dimension )
        )
    #

    n = X_all_t.shape[0]
    p_ohe = sum( 1 if isinstance( v, int ) else len( v ) for v in x_oheDict.values() )

    if local_grad_method == 'auto_diff':
        auto_diff_full_t: torch.Tensor = model.auto_diff_t( X_all_t )  # (n, p_all_ohe)
    elif local_grad_method == 'bandwidth':
        if bandwidth is None:
            raise ValueError(
                "local_grad_method='bandwidth' requires an explicit bandwidth "
                "(no auto-scaling from n is applied)."
            )
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
            _ref_val = ( lambda k: cat_ohe_vals[ k ][ 0 ] ) if cat_ohe_vals is not None else ( lambda k: 0.0 )
            _act_val = ( lambda k: cat_ohe_vals[ k ][ 1 ] ) if cat_ohe_vals is not None else ( lambda k: 1.0 )
            # reference: set all OHE bits for this variable to their "0" value (implicit category 0)
            X_ref = X_all_t.clone()
            for k in cat_indices:
                X_ref[ :, k ] = _ref_val( k )
            pred_ref = model.predict_t( X_ref ).reshape( n )  # (n,)
            for ohe_col in cat_indices:
                X_j = X_all_t.clone()
                for k in cat_indices:
                    X_j[ :, k ] = _ref_val( k )
                X_j[ :, ohe_col ] = _act_val( ohe_col )
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
    n_blocks:          int | None = None,
    calibrate:         bool = False,
    lambda_min:        float | None = None,
    lambda_max:        float | None = None,
    a_min:             float | None = None,
    a_max:             float | None = None,
    calibrate_rmin:    float = 0.01,
    calibrate_rmax:    float = 1.0,
    batch_size:        int | None = None,
    epochs:            int = 500,
    total_steps:       int | None = None,
    bandwidth:         float | None = 1.0,
    model_type:        str = 'mlp',
    n_warmup:          int = 5000,
    vertical_prefit:   bool = False,
    prefit_noise_std:  float = 0.01,
    reset_optimizer:   bool = True,
    learning_rate:     float = 0.01,
    drop_first:        bool = True,
    dense_activation:  str = 'relu',
    verbose:           int = 0,
    weight:            np.ndarray | None = None,
    rng:               np.random.Generator | None = None,
    ) -> np.ndarray:
    """
    Train a PRISM-G network on (X, Xk, y) and return the per-sample local gradient
    matrix for X only.

    Returns array of shape (n, p_ohe_x) where
      p_ohe_x = p_numeric + sum(c_k - 1 for each categorical variable in X).
    Numeric columns: bandwidth or auto_diff gradient.
    Categorical columns (c-1 per variable): model-prediction contrast vs. category 0
      (drop_first=True convention — category 0 is the implicit reference).

    Only continuous/count outcomes are supported (see `_get_localGrad_ohe_matrix_t`);
    a `categorical` outcome_type raises `NotImplementedError` before training, since
    there's no established reduction of a multiclass model's (n, k) logits into this
    function's (n, p_ohe_x) per-sample-scalar-gradient contract.

    `lambda_path`/`a_path`/`bandwidth`/`rng` follow the same conventions as
    `prismGImportances` -- see that docstring. X is standardized the same way as
    `prismGImportances`/`prismGWImportances` before the local-gradient step.

    `epochs`/`total_steps` also follow `prismWImportances`'s docstring: `epochs` is
    converted to a raw step budget distributed evenly (in raw-step units) across lambda
    stages; `total_steps`, if given, overrides that budget with an exact step count.

    `n_blocks`/`calibrate`/`lambda_min`/`lambda_max`/`a_min`/`a_max`/`calibrate_rmin`/
    `calibrate_rmax` also follow `prismWImportances`'s docstring: `calibrate=True`
    derives `lambda_path` via GRIP2 Eq. 5's gradient-ratio calibration and is mutually
    exclusive with `lambda_path`/`a_path`/`lambda_min`/`lambda_max`/`a_min`/`a_max`
    (raises `ValueError` if any are given alongside it); `n_blocks` is mutually
    exclusive with an explicit `lambda_path`/`a_path` regardless of `calibrate`.
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

    if outcomeDescriptor.outcome_type == 'categorical':
        raise NotImplementedError(
            "prismGLocalGradients does not support categorical outcomes -- "
            "see _get_localGrad_ohe_matrix_t's docstring."
        )
    #

    lambda_path, a_path = _resolve_lambda_a_path(
        lambda_path, a_path, rng, n_blocks,
        calibrate = calibrate, lambda_min = lambda_min, lambda_max = lambda_max,
        a_min = a_min, a_max = a_max,
    )
    _resolved_n_blocks = n_blocks if n_blocks is not None else _DEFAULT_N_BLOCKS
    _resolved_a_min    = a_min if a_min is not None else _A_UNIF_LOW
    _resolved_a_max    = a_max if a_max is not None else _A_UNIF_HIGH

    _mu = X_all_np.mean( axis=0 )
    _sd = np.maximum( X_all_np.std( axis=0 ), 1e-8 )
    X_all_np = ( X_all_np - _mu ) / _sd

    # Standardized reference/active values for each categorical OHE column, so the
    # categorical branch of _get_localGrad_ohe_matrix_t evaluates at the correct
    # (standardized) 0/1 states instead of raw 0.0/1.0 -- same construction as
    # prismGImportances/prismGWImportances.
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
        rng              = rng,
    )

    predictionModel.fit(
        X           = X_all_np,
        y           = y_np,
        groups      = groups,
        lambda_path = lambda_path,
        a_path      = a_path,
        calibrate      = calibrate,
        n_blocks       = _resolved_n_blocks,
        a_min          = _resolved_a_min,
        a_max          = _resolved_a_max,
        calibrate_rmin = calibrate_rmin,
        calibrate_rmax = calibrate_rmax,
        batch_size  = batch_size,
        total_steps = total_steps,
        weight      = weight,
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
        output_dimension  = output_dimension,
        cat_ohe_vals      = cat_ohe_vals,
    )

    return grad_t.cpu().numpy()
#/def prismGLocalGradients
