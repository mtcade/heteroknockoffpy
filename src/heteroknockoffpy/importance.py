from . import utilities
from .utilities import OutcomeDescriptor, DataFrameLike, SeriesOrDataFrameLike, _resolve_df, _resolve_y

import numpy as np
import polars as pl

from typing import Iterable, Literal, Sequence

# The PRISM-torch family (prismWImportances, prismWImportancesPerOHE, prismGImportances,
# prismGWImportances, prismGLocalGradients) is implemented in heteroknockofftorch.prismImportances
# so that importing this module never triggers `import torch` -- torch is only loaded when one
# of these functions is actually called. See heteroknockofftorch/prismImportances.py for the
# real implementations; the stubs below just forward with identical signatures/behavior.

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
    weight: np.ndarray | None = None,
    ) -> np.ndarray:
    """
    PRISM-W importances: average of group-norm snapshots over a lambda regularization path.

    Trains a single MLP on [X, Xk] → y with an adaptive proximal penalty on the input layer.
    At the end of each lambda stage the group norms ||w[:, group_j]||_F are recorded;
    the final importances are the mean over all snapshots.

    :param X: Original data (numeric + `pl.Categorical` columns).
    :param Xk: Knockoffs of `X`, same schema.
    :param y: Outcome; scalar (continuous/count) or categorical Series/DataFrame.
    :param layers: Hidden-layer widths for the MLP. A common construction is
        `[round(p * first_layer_ratio)]` then further entries each
        `round(prev * layer_ratio)`, where `p` is the OHE-expanded width of
        [X, Xk] combined, though any explicit sequence of widths works.
    :param model_type: see heteroknockofftorch.torchImportances.PRISMPredictionModel docstring
        for the full list ('mlp', 'pairwise', 'additive'). Default `'pairwise'`.
    :param outcome_type: 'continuous'/'count'/'categorical'; inferred from `y` if omitted.
    :param lambda_path: Sequence of lambda values. Defaults to logspace(1,-2,50) if omitted.
    :param a_path: Per-stage input-layer penalty values. If None, uses lambda_path values.
    :param batch_size: Minibatch size; `None` (the default) uses full-batch training.
    :param epochs: Total training epochs, distributed as evenly as possible
        across lambda stages. Default `500`.
    :param n_warmup: Epochs of warmup training before entering the lambda-path
        schedule. Default `0`.
    :param vertical_prefit: Whether to prefit a smaller model and transfer its
        weights vertically into the full model before the lambda-path loop.
    :param prefit_noise_std: Std of noise added when duplicating prefit weights
        into the larger model (only relevant if `vertical_prefit=True`).
    :param reset_optimizer: Whether to reset the Adam optimizer state at each
        lambda-path stage transition.
    :param learning_rate: Adam learning rate. Default `0.01`.
    :param drop_first: Whether categorical columns are OHE'd with the first
        category dropped (standard identifiability convention).
    :param dense_activation: Activation function name for the MLP's hidden layers.
    :param verbose: Verbosity level (0 = silent).
    :returns: Array of shape (2*p,) — first p entries for X, last p for Xk.
    """
    from . import _processIsolation
    return _processIsolation.run_isolated_if_loaded(
        'heteroknockoffpy.heteroknockofftorch.prismImportances',
        'prismWImportances',
        X = X, Xk = Xk, y = y,
        layers = layers,
        outcome_type = outcome_type,
        lambda_path = lambda_path,
        a_path = a_path,
        batch_size = batch_size,
        epochs = epochs,
        model_type = model_type,
        n_warmup = n_warmup,
        vertical_prefit = vertical_prefit,
        prefit_noise_std = prefit_noise_std,
        reset_optimizer = reset_optimizer,
        learning_rate = learning_rate,
        drop_first = drop_first,
        dense_activation = dense_activation,
        verbose = verbose,
        weight = weight,
    )
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
    weight: np.ndarray | None = None,
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

    See `prismWImportances`'s docstring for the shared parameter meanings
    (`layers` construction, `lambda_path`/`a_path`, etc.) -- all apply
    identically here.

    :param X: Original data (numeric + `pl.Categorical` columns).
    :param Xk: Knockoffs of `X`, same schema.
    :param y: Outcome; scalar (continuous/count) or categorical Series/DataFrame.
    :param layers: Hidden-layer widths for the MLP; see `prismWImportances`.
    :param outcome_type: 'continuous'/'count'/'categorical'; inferred from `y` if omitted.
    :param model_type: 'mlp' or 'pairwise' only.
    :param lambda_path: Sequence of lambda values. Defaults to logspace(1,-2,50).
    :param a_path: Per-stage input-layer penalty values. If None, uses lambda_path values.
    :param batch_size: Minibatch size; `None` uses full-batch training.
    :param epochs: Total training epochs, distributed as evenly as possible across lambda stages.
    :param n_warmup: Epochs of warmup training before entering the lambda-path schedule.
    :param vertical_prefit: Whether to prefit a smaller model and transfer its
        weights vertically into the full model before the lambda-path loop.
    :param prefit_noise_std: Std of noise added when duplicating prefit weights
        into the larger model (only relevant if `vertical_prefit=True`).
    :param reset_optimizer: Whether to reset the Adam optimizer state at each
        lambda-path stage transition.
    :param learning_rate: Adam learning rate.
    :param drop_first: Whether categorical columns are OHE'd with the first
        category dropped (standard identifiability convention).
    :param dense_activation: Activation function name for the MLP's hidden layers.
    :param verbose: Verbosity level (0 = silent).
    :returns: Array of shape (2*p_ohe,) — first p_ohe entries for X's OHE-expanded columns,
        last p_ohe for Xk's. p_ohe is the total OHE-expanded width per side (numeric columns
        contribute 1 entry each, a K-category column contributes K-1 entries under
        drop_first=True), NOT the original variable count.
    """
    from . import _processIsolation
    return _processIsolation.run_isolated_if_loaded(
        'heteroknockoffpy.heteroknockofftorch.prismImportances',
        'prismWImportancesPerOHE',
        X = X, Xk = Xk, y = y,
        layers = layers,
        outcome_type = outcome_type,
        lambda_path = lambda_path,
        a_path = a_path,
        batch_size = batch_size,
        epochs = epochs,
        model_type = model_type,
        n_warmup = n_warmup,
        vertical_prefit = vertical_prefit,
        prefit_noise_std = prefit_noise_std,
        reset_optimizer = reset_optimizer,
        learning_rate = learning_rate,
        drop_first = drop_first,
        dense_activation = dense_activation,
        verbose = verbose,
        weight = weight,
    )
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
    weight: np.ndarray | None = None,
    ) -> np.ndarray:
    """
    PRISM-G importances: average of PRISM local-gradient snapshots over a lambda path.

    Same training procedure as prismWImportances; at the end of each lambda stage the
    PRISM importances (auto_diff or bandwidth) of the current model are recorded.

    Shares `prismWImportances`'s training-loop parameters -- see that docstring
    for details. `local_grad_method`/`bandwidth`/`exponent`/`bandwidth_exponent`
    below are specific to the PRISM-G local-gradient step.

    :param X: Original data (numeric + `pl.Categorical` columns).
    :param Xk: Knockoffs of `X`, same schema.
    :param y: Outcome; scalar (continuous/count) or categorical Series/DataFrame.
    :param layers: Hidden-layer widths for the MLP; see `prismWImportances`.
    :param outcome_type: 'continuous'/'count'/'categorical'; inferred from `y` if omitted.
    :param model_type: see heteroknockofftorch.torchImportances.PRISMPredictionModel docstring
        for the full list ('mlp', 'pairwise', 'additive'). Default `'pairwise'`.
    :param local_grad_method: 'auto_diff' (exact) or 'bandwidth' (finite
        difference). Default `'auto_diff'`.
    :param lambda_path: Sequence of lambda values. Defaults to logspace(1,-2,50).
    :param a_path: Per-stage input-layer penalty values. If None, uses lambda_path values.
    :param batch_size: Minibatch size; `None` uses full-batch training.
    :param epochs: Total training epochs, distributed as evenly as possible across lambda stages.
    :param bandwidth: Bandwidth for finite-difference approximation (auto-set if
        None). Only relevant when `local_grad_method='bandwidth'`.
    :param exponent: Power applied to each local gradient value before
        averaging. Default `1.0`.
    :param n_warmup: Epochs of warmup training before entering the lambda-path schedule.
    :param vertical_prefit: Whether to prefit a smaller model and transfer its
        weights vertically into the full model before the lambda-path loop.
    :param prefit_noise_std: Std of noise added when duplicating prefit weights
        into the larger model (only relevant if `vertical_prefit=True`).
    :param reset_optimizer: Whether to reset the Adam optimizer state at each
        lambda-path stage transition.
    :param learning_rate: Adam learning rate.
    :param drop_first: Whether categorical columns are OHE'd with the first
        category dropped (standard identifiability convention).
    :param dense_activation: Activation function name for the MLP's hidden layers.
    :param verbose: Verbosity level (0 = silent).
    :param bandwidth_exponent: Exponent used for the auto-set bandwidth (n ** -bandwidth_exponent)
        when bandwidth is None. Ignored if bandwidth is given explicitly. Default `0.2`.
    :returns: Array of shape (2*p,).
    """
    from . import _processIsolation
    return _processIsolation.run_isolated_if_loaded(
        'heteroknockoffpy.heteroknockofftorch.prismImportances',
        'prismGImportances',
        X = X, Xk = Xk, y = y,
        layers = layers,
        outcome_type = outcome_type,
        local_grad_method = local_grad_method,
        lambda_path = lambda_path,
        a_path = a_path,
        batch_size = batch_size,
        epochs = epochs,
        bandwidth = bandwidth,
        exponent = exponent,
        model_type = model_type,
        n_warmup = n_warmup,
        vertical_prefit = vertical_prefit,
        prefit_noise_std = prefit_noise_std,
        reset_optimizer = reset_optimizer,
        learning_rate = learning_rate,
        drop_first = drop_first,
        dense_activation = dense_activation,
        verbose = verbose,
        bandwidth_exponent = bandwidth_exponent,
        weight = weight,
    )
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
    weight: np.ndarray | None = None,
    ) -> tuple[ np.ndarray, np.ndarray ]:
    """
    PRISM-G and PRISM-W importances from a single training pass.

    Identical hyperparameters and model to prismGImportances / prismWImportances.
    At each lambda stage the snapshot_fn records PRISM-W group norms as a side
    effect while returning PRISM-G local-gradient importances as the primary snapshot.

    :param X: Original data (numeric + `pl.Categorical` columns).
    :param Xk: Knockoffs of `X`, same schema.
    :param y: Outcome; scalar (continuous/count) or categorical Series/DataFrame.
    :param layers: Hidden-layer widths for the MLP. A common construction is
        `[round(p * first_layer_ratio)]` then further entries each
        `round(prev * layer_ratio)`, though any explicit sequence of widths works.
    :param outcome_type: 'continuous'/'count'/'categorical'; inferred from `y` if omitted.
    :param model_type: see heteroknockofftorch.torchImportances.PRISMPredictionModel docstring
        for the full list ('mlp', 'pairwise', 'additive'). Default `'pairwise'`.
    :param local_grad_method: 'auto_diff' (exact) or 'bandwidth' (finite
        difference). Default `'auto_diff'`.
    :param lambda_path: Sequence of lambda values. Defaults to logspace(1,-2,50) if omitted.
    :param a_path: Per-stage input-layer penalty values. If None, uses lambda_path values.
    :param batch_size: Minibatch size; `None` (the default) uses full-batch training.
    :param epochs: Total training epochs, distributed as evenly as possible
        across lambda stages. Default `500`.
    :param bandwidth: Bandwidth for finite-difference approximation (auto-set if
        None). Only relevant when `local_grad_method='bandwidth'`.
    :param exponent: Power applied to each local gradient value before
        averaging. Default `1.0`.
    :param n_warmup: Epochs of warmup training before entering the lambda-path
        schedule. Default `0`.
    :param vertical_prefit: Whether to prefit a smaller model and transfer its
        weights vertically into the full model before the lambda-path loop.
    :param prefit_noise_std: Std of noise added when duplicating prefit weights
        into the larger model (only relevant if `vertical_prefit=True`).
    :param reset_optimizer: Whether to reset the Adam optimizer state at each
        lambda-path stage transition.
    :param learning_rate: Adam learning rate. Default `0.01`.
    :param drop_first: Whether categorical columns are OHE'd with the first
        category dropped (standard identifiability convention).
    :param dense_activation: Activation function name for the MLP's hidden layers.
    :param verbose: Verbosity level (0 = silent).
    :param bandwidth_exponent: Exponent used for the auto-set bandwidth (n ** -bandwidth_exponent)
        when bandwidth is None. Ignored if bandwidth is given explicitly. Default `0.2`.
    :returns: (g_importances, w_importances) both of shape (2*p,).
    """
    from . import _processIsolation
    return _processIsolation.run_isolated_if_loaded(
        'heteroknockoffpy.heteroknockofftorch.prismImportances',
        'prismGWImportances',
        X = X, Xk = Xk, y = y,
        layers = layers,
        outcome_type = outcome_type,
        local_grad_method = local_grad_method,
        lambda_path = lambda_path,
        a_path = a_path,
        batch_size = batch_size,
        epochs = epochs,
        bandwidth = bandwidth,
        exponent = exponent,
        model_type = model_type,
        n_warmup = n_warmup,
        vertical_prefit = vertical_prefit,
        prefit_noise_std = prefit_noise_std,
        reset_optimizer = reset_optimizer,
        learning_rate = learning_rate,
        drop_first = drop_first,
        dense_activation = dense_activation,
        verbose = verbose,
        bandwidth_exponent = bandwidth_exponent,
        weight = weight,
    )
#/def prismGWImportances


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
    weight:            np.ndarray | None = None,
    ) -> np.ndarray:
    """
    Train a PRISM-G network on (X, Xk, y) and return the per-sample local gradient
    matrix for X only.

    Returns array of shape (n, p_ohe_x) where
      p_ohe_x = p_numeric + sum(c_k - 1 for each categorical variable in X).
    Numeric columns: bandwidth or auto_diff gradient.
    Categorical columns (c-1 per variable): model-prediction contrast vs. category 0
      (drop_first=True convention — category 0 is the implicit reference).

    Shares `prismGImportances`'s training-loop parameters -- see that docstring
    for details; this function differs only in defaulting `local_grad_method`
    to `'bandwidth'` and returning the raw per-sample gradient matrix (for X
    only) instead of the lambda-path-averaged scalar importances.

    :param X: Original data (numeric + `pl.Categorical` columns).
    :param Xk: Knockoffs of `X`, same schema.
    :param y: Outcome; scalar (continuous/count) or categorical Series/DataFrame.
    :param layers: Hidden-layer widths for the MLP; see `prismGImportances`.
    :param outcome_type: 'continuous'/'count'/'categorical'; inferred from `y` if omitted.
    :param local_grad_method: 'auto_diff' (exact) or 'bandwidth' (finite
        difference, the default here -- opposite of `prismGImportances`'s default).
    :param lambda_path: Sequence of lambda values.
    :param a_path: Per-stage input-layer penalty values. If None, uses lambda_path values.
    :param batch_size: Minibatch size; `None` uses full-batch training.
    :param epochs: Total training epochs, distributed as evenly as possible across lambda stages.
    :param bandwidth: Bandwidth for finite-difference approximation (auto-set if
        None). Relevant since `local_grad_method` defaults to `'bandwidth'` here.
    :param model_type: see heteroknockofftorch.torchImportances.PRISMPredictionModel docstring
        for the full list ('mlp', 'pairwise', 'additive').
    :param n_warmup: Epochs of warmup training before entering the lambda-path schedule.
    :param vertical_prefit: Whether to prefit a smaller model and transfer its
        weights vertically into the full model before the lambda-path loop.
    :param prefit_noise_std: Std of noise added when duplicating prefit weights
        into the larger model (only relevant if `vertical_prefit=True`).
    :param reset_optimizer: Whether to reset the Adam optimizer state at each
        lambda-path stage transition.
    :param learning_rate: Adam learning rate.
    :param drop_first: Whether categorical columns are OHE'd with the first
        category dropped (standard identifiability convention).
    :param dense_activation: Activation function name for the MLP's hidden layers.
    :param verbose: Verbosity level (0 = silent).
    """
    from . import _processIsolation
    return _processIsolation.run_isolated_if_loaded(
        'heteroknockoffpy.heteroknockofftorch.prismImportances',
        'prismGLocalGradients',
        X = X, Xk = Xk, y = y,
        layers = layers,
        outcome_type = outcome_type,
        local_grad_method = local_grad_method,
        lambda_path = lambda_path,
        a_path = a_path,
        batch_size = batch_size,
        epochs = epochs,
        bandwidth = bandwidth,
        model_type = model_type,
        n_warmup = n_warmup,
        vertical_prefit = vertical_prefit,
        prefit_noise_std = prefit_noise_std,
        reset_optimizer = reset_optimizer,
        learning_rate = learning_rate,
        drop_first = drop_first,
        dense_activation = dense_activation,
        verbose = verbose,
        weight = weight,
    )
#/def prismGLocalGradients


def rangerGiniImportances(
    X: DataFrameLike,
    Xk: DataFrameLike,
    y: SeriesOrDataFrameLike,
    outcome_type: Literal['continuous','count','categorical',] | None = None,
    verbose: int = 0,
    weight: np.ndarray | None = None,
    **kwargs,
    ) -> np.ndarray:
    """
        Gini-impurity/variance-reduction importances from a single `ranger::ranger`
        random forest fit on [X, Xk] → y (`rbridge.rangerGiniImportances`).

        :param X: Original data (numeric + `pl.Categorical` columns).
        :param Xk: Knockoffs of `X`, same schema.
        :param y: Outcome; scalar (continuous/count) or categorical Series/DataFrame.
        :param outcome_type: 'continuous'/'count'/'categorical'; inferred from `y` if omitted.
        :param verbose: Verbosity level (0 = silent).
        :param kwargs: Forwarded to `ranger::ranger` via
            `rbridge.rangerGiniImportances`/`stat.forest.hetero_gini.R`. Same
            kwarg set as `knockoff.get_rangerSCIP` forwards, all optional
            (omitted values fall back to ranger's own defaults):
              - `num_trees` (int), `mtry` (int), `min_node_size` (int),
                `max_depth` (int), `sample_fraction` (float), `num_threads` (int).
              - `respect_unordered_factors` (str, e.g. `'partition'`): how ranger
                splits unordered categorical predictors.
        :param weight: Optional length-n sample weight, forwarded to
            ranger::ranger's case.weights (resampling-probability weighting,
            not a loss multiplier). None (default) fits unweighted.
        :returns: Array of shape (2*p,) — first p entries for X, last p for Xk.
    """
    from . import _processIsolation
    return _processIsolation.run_isolated_if_loaded(
        'heteroknockoffpy.rbridge',
        'rangerGiniImportances',
        X = X,
        Xk = Xk,
        y = y,
        outcome_type = outcome_type,
        verbose = verbose,
        weight = weight,
        **kwargs,
    )
#/def rangerGiniImportances


def rangerPrismImportances(
    X: DataFrameLike,
    Xk: DataFrameLike,
    y: SeriesOrDataFrameLike,
    outcome_type: Literal['continuous','count','categorical',] | None = None,
    verbose: int = 0,
    weight: np.ndarray | None = None,
    **kwargs,
    ) -> np.ndarray:
    """
        PRISM local-gradient importances using a single `ranger::ranger` forest
        fit on [X, Xk] → y (`rbridge.rangerPrismImportances`/
        `stat.forest.prism_{continuous,count,categorical}.R`) -- the ranger
        analogue of `xgbPrismImportances`; see that docstring for the algorithm
        description (numeric = bandwidth finite-difference, categorical =
        max-minus-min level sweep, categorical outcome = Mahalanobis norm of
        log-probability contrasts).

        :param X: Original data (numeric + `pl.Categorical` columns).
        :param Xk: Knockoffs of `X`, same schema.
        :param y: Outcome; scalar (continuous/count) or categorical Series/DataFrame.
        :param outcome_type: 'continuous'/'count'/'categorical'; inferred from `y` if omitted.
        :param verbose: Verbosity level (0 = silent).
        :param kwargs: Forwarded to `ranger::ranger` via the R PRISM scripts --
            same ranger kwarg set as `rangerGiniImportances`/`get_rangerSCIP`
            (`num_trees`, `mtry`, `min_node_size`, `max_depth`, `sample_fraction`,
            `num_threads`, `respect_unordered_factors`), plus the PRISM-specific
            `bandwidth`/`bandwidth_exponent`/`exponent` accepted by the R script
            itself (mirroring `xgbPrismImportances`'s parameters of the same
            name, forwarded here as plain kwargs rather than named parameters).
        :param weight: Optional length-n sample weight, forwarded to
            ranger::ranger's case.weights (resampling-probability weighting,
            not a loss multiplier). None (default) fits unweighted.
        :returns: Array of shape (2*p,) — first p entries for X, last p for Xk.
    """
    from . import _processIsolation
    return _processIsolation.run_isolated_if_loaded(
        'heteroknockoffpy.rbridge',
        'rangerPrismImportances',
        X = X,
        Xk = Xk,
        y = y,
        outcome_type = outcome_type,
        verbose = verbose,
        weight = weight,
        **kwargs,
    )
#/def rangerPrismImportances


def xgbImportances(
    X: DataFrameLike,
    Xk: DataFrameLike,
    y: SeriesOrDataFrameLike,
    outcome_type: Literal['continuous','count','categorical',] | None = None,
    importance_type: Literal[ 'weight','gain','cover','total_gain','total_cover'] = 'gain',
    verbose: int = 0,
    rng: np.random.Generator | None = None,
    weight: np.ndarray | None = None,
    **kwargs,
    ) -> np.ndarray:
    """
        Split-based importances (weight/gain/cover/...) from a single xgboost model
        fit on [X, Xk]. Categorical columns are handled natively by xgboost
        (tree_method='hist', enable_categorical=True), not one-hot encoded.

        :param X: Original data (numeric + `pl.Categorical` columns).
        :param Xk: Knockoffs of `X`, same schema.
        :param y: Outcome; scalar (continuous/count) or categorical Series/DataFrame.
        :param outcome_type: 'continuous'/'count'/'categorical'; inferred from `y` if omitted.
        :param importance_type: Which `Booster.get_score()` importance type to
            report -- `'weight'` (split count), `'gain'` (avg. loss reduction
            per split), `'cover'` (avg. samples affected per split),
            `'total_gain'`, `'total_cover'`. Default `'gain'`.
        :param verbose: Verbosity level (0 = silent).
        :param kwargs: `model_kwargs` (dict, forwarded to the
            `XGBRegressor`/`XGBClassifier` constructor -- e.g. `max_depth`,
            `learning_rate`, `min_child_weight`, `subsample`, `colsample_bytree`,
            `reg_alpha`, `reg_lambda`, `gamma`, `n_estimators`), plus anything
            else forwarded to `XGBRegressor`/`XGBClassifier.fit`. See
            https://xgboost.readthedocs.io/en/latest/python/python_api.html for
            the full parameter reference and
            https://xgboost.readthedocs.io/en/latest/tutorials/categorical.html
            for the native-categorical-support requirements
            (`enable_categorical=True` + `tree_method='hist'`/`'approx'`).
        :param rng: If given, seeds the XGBoost fit (random_state=rng) for
            reproducibility. Unseeded if omitted.
        :param weight: Optional length-n sample weight. None (default) fits
            unweighted, xgboost's own default.
        :returns: Array of shape (2*p,) — first p entries for X, last p for Xk.
    """
    from . import _processIsolation
    return _processIsolation.run_isolated_if_loaded(
        'heteroknockoffpy.xgbImportances',
        'score_importances',
        X = X,
        Xk = Xk,
        y = y,
        outcome_type = outcome_type,
        importance_type = importance_type,
        verbose = verbose,
        rng = rng,
        weight = weight,
        **kwargs,
    )
#/def xgbImportances

def xgbPrismImportances(
    X: DataFrameLike,
    Xk: DataFrameLike,
    y: SeriesOrDataFrameLike,
    outcome_type: Literal['continuous','count','categorical',] | None = None,
    verbose: int = 0,
    rng: np.random.Generator | None = None,
    weight: np.ndarray | None = None,
    **kwargs,
    ) -> np.ndarray:
    """
        PRISM local-gradient importance using a single xgboost model on [X, Xk].
        Mirrors the R stat.forest.prism_{continuous,count,categorical}.R scripts,
        using xgboost's native categorical handling instead of one-hot encoding.

        :param X: Original data (numeric + `pl.Categorical` columns).
        :param Xk: Knockoffs of `X`, same schema.
        :param y: Outcome; scalar (continuous/count) or categorical Series/DataFrame.
        :param outcome_type: 'continuous'/'count'/'categorical'; inferred from `y` if omitted.
        :param verbose: Verbosity level (0 = silent).
        :param kwargs: `bandwidth` (float, scale multiplier for the numeric
            finite-difference step, `sd(col) * bandwidth / n ** bandwidth_exponent`;
            default `1.0`), `bandwidth_exponent` (float, sample-size exponent in
            that denominator; default `0.2`), `exponent` (float, power applied
            to each pointwise importance before averaging; default `1.0`), and
            `model_kwargs` (dict, forwarded to the `XGBRegressor`/`XGBClassifier`
            constructor -- e.g. `max_depth`, `learning_rate`, `min_child_weight`,
            `subsample`, `colsample_bytree`, `reg_alpha`, `reg_lambda`, `gamma`,
            `n_estimators`), plus anything else forwarded to
            `XGBRegressor`/`XGBClassifier.fit`. See
            https://xgboost.readthedocs.io/en/latest/python/python_api.html
            for the full `model_kwargs` reference.
        :param rng: If given, seeds the XGBoost fit (random_state=rng) for
            reproducibility. Unseeded if omitted.
        :param weight: Optional length-n sample weight. None (default) fits
            unweighted, xgboost's own default.
        :returns: Array of shape (2*p,) — first p entries for X, last p for Xk.
    """
    from . import _processIsolation
    return _processIsolation.run_isolated_if_loaded(
        'heteroknockoffpy.xgbImportances',
        'prism_importances',
        X = X,
        Xk = Xk,
        y = y,
        outcome_type = outcome_type,
        verbose = verbose,
        rng = rng,
        weight = weight,
        **kwargs,
    )
#/def xgbPrismImportances

def xgbShapImportances(
    X: DataFrameLike,
    Xk: DataFrameLike,
    y: SeriesOrDataFrameLike,
    outcome_type: Literal['continuous','count','categorical',] | None = None,
    verbose: int = 0,
    rng: np.random.Generator | None = None,
    weight: np.ndarray | None = None,
    **kwargs,
    ) -> np.ndarray:
    """
        Shap contribution importances using xgboost and shap.TreeExplainer.

        Requires the optional `shap` dependency: pip install heteroknockoffpy[shap]

        :param X: Original data (numeric + `pl.Categorical` columns).
        :param Xk: Knockoffs of `X`, same schema.
        :param y: Outcome; scalar (continuous/count) or categorical Series/DataFrame.
        :param outcome_type: 'continuous'/'count'/'categorical'; inferred from `y` if omitted.
        :param verbose: Verbosity level (0 = silent).
        :param kwargs: `model_kwargs` (dict, forwarded to the
            `XGBRegressor`/`XGBClassifier` constructor -- e.g. `max_depth`,
            `learning_rate`, `min_child_weight`, `subsample`, `colsample_bytree`,
            `reg_alpha`, `reg_lambda`, `gamma`, `n_estimators`), plus anything
            else forwarded to `XGBRegressor`/`XGBClassifier.fit`. See
            https://xgboost.readthedocs.io/en/latest/python/python_api.html for
            the full parameter reference.
        :param rng: If given, seeds the XGBoost fit (random_state=rng) for
            reproducibility. Unseeded if omitted.
        :param weight: Optional length-n sample weight. None (default) fits
            unweighted, xgboost's own default.
        :returns: Array of shape (2*p,) — first p entries for X, last p for Xk.
    """
    from . import _processIsolation
    return _processIsolation.run_isolated_if_loaded(
        'heteroknockoffpy.xgbImportances',
        'shap_importances',
        X = X,
        Xk = Xk,
        y = y,
        outcome_type = outcome_type,
        verbose = verbose,
        rng = rng,
        weight = weight,
        **kwargs,
    )
#/def xgbShapImportances

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
#/def _collapse_cat_importance


def lassoImportances(
    X: DataFrameLike,
    Xk: DataFrameLike,
    y: SeriesOrDataFrameLike,
    outcome_type: Literal['continuous','count','categorical',] | None = None,
    fit_intercept: bool = True,
    exponent: float = 1.0,
    verbose: int = 0,
    weight: np.ndarray | None = None,
    **kwargs,
    ) -> np.ndarray:
    """
        L1-penalised (LASSO) linear/GLM importance measures on the one-hot-encoded
        design matrix [X, Xk]:
          - continuous: sklearn `LassoCV`
          - count:      `PoissonLassoCV` (statsmodels-backed Poisson GLM with an
                         L1 path, cross-validated over `alphas`)
          - categorical: sklearn `LogisticRegressionCV` with `l1_ratios=(1,)`,
                         `solver='saga'` (multi-class: Mahalanobis distance on
                         contrasted coefficients, mirroring the OHE-collapse
                         convention used by `lassoImportances`'s sibling functions)
        :param X: Original data (numeric + `pl.Categorical` columns).
        :param Xk: Knockoffs of `X`, same schema.
        :param y: Outcome; scalar (continuous/count) or categorical Series/DataFrame.
        :param outcome_type: 'continuous'/'count'/'categorical'; inferred from `y` if omitted.
        :param fit_intercept: Whether the underlying sklearn/statsmodels model
            fits an intercept term.
        :param exponent: Power applied to each (OHE-collapsed) coefficient
            magnitude before returning.
        :param verbose: Verbosity level (0 = silent).
        :param kwargs:
            - `n_splits` (int, default `5`): CV fold count, forwarded to
              `LassoCV(cv=...)`/`PoissonLassoCV(n_splits=...)`/
              `LogisticRegressionCV(cv=...)`.
            - `max_iter` (int): solver iteration cap. Default `200` for `count`
              (statsmodels Poisson IRLS converges quickly), `4000` for
              continuous/categorical (sklearn's coordinate-descent/SAGA solvers).
            - `alphas` (array-like): only consulted for `count` outcomes
              (`PoissonLassoCV`'s own regularization path) -- default
              `np.logspace(-4, 2, 10)`. `continuous`/`categorical` use
              `LassoCV`/`LogisticRegressionCV`'s own internal alpha search
              instead and don't consult this kwarg.
        :returns: Array of shape (2*p,) — one importance per column of X, then
            per column of Xk (categorical columns' OHE dummy coefficients are
            collapsed back to a single value via `_collapse_cat_importance`).
    """
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
            sample_weight = weight,
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
            weight = weight,
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
            sample_weight = weight,
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
    weight: np.ndarray | None = None,
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

    :param X: Original data (numeric + `pl.Categorical` columns).
    :param Xk: Knockoffs of `X`, same schema.
    :param y: Outcome; scalar (continuous/count) or categorical Series/DataFrame.
    :param outcome_type: 'continuous'/'count'/'categorical'; inferred from `y` if omitted.
    :param fit_intercept: Whether the underlying sklearn model fits an intercept term.
    :param exponent: Power applied to each (OHE-collapsed) coefficient magnitude
        before returning.
    :param verbose: Verbosity level (0 = silent).
    :param kwargs:
        - `n_splits` (int, default `5`): CV fold count, forwarded to
          `RidgeCV(cv=...)`/`GridSearchCV(cv=...)`/`LogisticRegressionCV(cv=...)`.
        - `max_iter` (int, default `4000`): forwarded to `PoissonRegressor`/
          `LogisticRegressionCV`'s solver iteration cap (`RidgeCV` doesn't take one).
        - `alphas` (array-like): the L2 penalty grid searched. Default
          `np.logspace(-4, 4, 13)` for `continuous` (`RidgeCV`), default
          `np.logspace(-4, 2, 10)` for `count` (`GridSearchCV` over
          `PoissonRegressor(alpha=...)`). `categorical` doesn't consult this
          kwarg (`LogisticRegressionCV`'s own internal `Cs` search is used instead).
    :returns: Array of shape (2*p,) — one importance per column of X, then per
        column of Xk (categorical columns' OHE dummy coefficients collapsed
        back to a single value via `_collapse_cat_importance`).
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
        ridgeModel.fit( X = X_ohe, y = y, sample_weight = weight )
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
            poissonModel.fit( X = X_ohe_scaled, y = y, sample_weight = weight )
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
            sample_weight = weight,
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
    weight: np.ndarray | None = None,
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

    :param X: Original data (numeric + `pl.Categorical` columns).
    :param Xk: Knockoffs of `X`, same schema.
    :param y: Outcome; scalar (continuous/count) or categorical Series/DataFrame.
    :param outcome_type: 'continuous'/'count'/'categorical'; inferred from `y` if omitted.
    :param l1_ratio: Mixing weight between L1 (LASSO) and L2 (ridge)
        regularization -- `1.0` = pure LASSO (delegates to `lassoImportances`),
        `0.0` = pure ridge (delegates to `ridgeImportances`), values in between
        fit an actual elastic-net model. Default `0.5` is an even L1/L2 mix.
    :param fit_intercept: Whether the underlying sklearn/statsmodels model fits
        an intercept term.
    :param exponent: Power applied to each (OHE-collapsed) coefficient
        magnitude before returning.
    :param verbose: Verbosity level (0 = silent).
    :param kwargs: Same `n_splits`/`max_iter`/`alphas` kwargs as
        `lassoImportances`/`ridgeImportances` (whichever applies depends on
        `outcome_type` and which delegate, if any, `l1_ratio` triggers):
        `n_splits` default `5`; `max_iter` default `200` for `count`, `4000`
        otherwise; `alphas` (only consulted for `count`, via `PoissonLassoCV`)
        default `np.logspace(-4, 2, 10)`.
    :returns: Array of shape (2*p,) — one importance per column of X, then per
        column of Xk (categorical columns' OHE dummy coefficients collapsed
        back to a single value via `_collapse_cat_importance`).
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
            weight = weight,
            **kwargs,
        )
    if l1_ratio == 0.0:
        return ridgeImportances(
            X = X, Xk = Xk, y = y,
            outcome_type = outcome_type,
            fit_intercept = fit_intercept,
            exponent = exponent,
            verbose = verbose,
            weight = weight,
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
        elasticModel.fit( X = X_ohe, y = y, sample_weight = weight )
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
        glmModel.fit( X = X_ohe, y = y, weight = weight )
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
            sample_weight = weight,
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
