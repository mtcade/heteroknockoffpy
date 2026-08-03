#
#//  knockoffs.py
#//  heteroknockoffpy
#//
#//  Created by Evan Mason on 2/13/26.
#//
"""
    Interface for all knockoff creation
"""

from . import utilities
from .utilities import DataFrameLike, _resolve_df

import polars as pl
import numpy as np

from typing import Callable, Literal

_XGB_MODEL_KWARGS: tuple[ str,... ] = (
    'max_depth', 'learning_rate', 'min_child_weight', 'subsample',
    'colsample_bytree', 'reg_alpha', 'reg_lambda', 'gamma', 'n_estimators',
)

def _pluck_xgb_kwargs( kwargs: dict ) -> dict:
    """
        Recognized xgboost hyperparameters, pulled out of a pooled kwargs dict
        that may also contain unrelated method-level kwargs (e.g.
        second_order's `shrink`) -- same names xgbImportances/xgbScip use for
        their own `model_kwargs`, so one set of hyperparameters can be reused
        across knockoff generation (categorical_method='xgb') and importance
        scoring (e.g. xgbPrismImportances) without collision.
    """
    return { k: kwargs[k] for k in _XGB_MODEL_KWARGS if k in kwargs }
#/def _pluck_xgb_kwargs

def _drop_xgb_kwargs( kwargs: dict ) -> dict:
    """
        Complement of `_pluck_xgb_kwargs`: strips the recognized xgboost
        hyperparameters out of a pooled kwargs dict before forwarding the rest
        elsewhere (e.g. to R's `create.second_order`, which would raise on an
        unrecognized argument like `max_depth`/`n_estimators` meant for
        categorical_method='xgb' instead).
    """
    return { k: v for k, v in kwargs.items() if k not in _XGB_MODEL_KWARGS }
#/def _drop_xgb_kwargs

def get_withCallable(
    X: DataFrameLike,
    rng: np.random.Generator,
    categorical_method: Literal['ranger','linear','ohe','xgb','ranger_scip','xgb_scip'],
    knockoffCallable: Callable[ [ np.ndarray ], np.ndarray ],
    conditional_expectations: pl.DataFrame | None = None,
    verbose: int = 0,
    verbose_prefix: str = '',
    weight: np.ndarray | None = None,
    **kwargs,
    ) -> np.ndarray:
    """
        Shared plumbing behind `get_second_order`/`get_torchGAN`: handles the
        `categorical_method` switch (numeric-residual/SCIP path vs. one-hot-encode
        path), then hands the resulting numeric array to `knockoffCallable` to
        produce the actual knockoff draw, and converts the result back to a
        `pl.DataFrame` matching `X`'s schema.

        :param X: Original data (numeric + `pl.Categorical` columns).
        :param rng: Used to seed the R/xgboost/OHE-sampling randomness downstream
            (in the SCIP conditional-expectation/categorical-sampling steps, and
            in `utilities.collapse_ohe`'s softmax sampling for the OHE-based
            categorical_methods).
        :param categorical_method:
            - 'ranger': Get logit probabilities with R ranger random forests
            - 'xgb': Same as 'ranger', but the per-column probability model is
              an xgboost.XGBClassifier instead of an R ranger forest -- no
              R/rpy2 dependency. Recognized xgboost hyperparameters (see
              `_pluck_xgb_kwargs`: max_depth, learning_rate, min_child_weight,
              subsample, colsample_bytree, reg_alpha, reg_lambda, gamma,
              n_estimators -- same names as `xgbImportances`'s `model_kwargs`)
              are plucked out of `kwargs`.
            - 'linear': Get logit probabilities with logistic regression
            - 'ohe': One hot encode as a float
            - 'ranger_scip': With conditional residuals for numeric data, use ranger forest SCIP for categorical
            - 'xgb_scip': Same as 'ranger_scip', but conditional expectations/categorical SCIP are computed with sequential xgboost models instead of R ranger
        :param knockoffCallable: Closure to convert either conditional residuals or one-hot-encoded data to knockoffs of the same format. Make sure it has the desired parameters based on whether you are using ranger_scip/xgb_scip, or another method
        :param conditional_expectations: Numeric conditional expectations. If not provided, uses `rbridge.get_forest_conditional_expectations` (for `categorical_method='ranger_scip'`) or `xgbScip.get_forest_conditional_expectations` (for `categorical_method='xgb_scip'`) to calculate
        :param verbose: Verbosity level (0 = silent), forwarded to whichever
            backend (`rbridge`/`xgbScip`) is doing the categorical-preprocessing work.
        :param verbose_prefix: String prepended to any verbose print output, for
            nesting context when called from a higher-level loop.
        :param kwargs: Currently unused here for the `'ranger_scip'`/`'xgb_scip'`
            branch (deliberately *not* forwarded to the conditional-expectation
            calls -- those take backend-specific kwargs like ranger's `num_trees`/
            `mtry` that would collide in meaning with kwargs meant for
            `knockoffCallable`'s own call, e.g. `second_order`'s `shrink`); also
            unused for the `'ranger'`/`'linear'`/`'ohe'` branch, EXCEPT `'xgb'`,
            which pulls its recognized xgboost hyperparameters out of this same
            pooled dict via `_pluck_xgb_kwargs` (safe precisely because it only
            takes the keys it recognizes, ignoring anything meant for
            `knockoffCallable`). Otherwise reserved for future per-categorical_method
            tuning.
        :param weight: Optional length-n sample weight, forwarded to whichever
            categorical_method branch actually fits something ('ranger'/'xgb'/
            'linear'/'ranger_scip'/'xgb_scip'); unused for 'ohe' (no fit on
            that path). None (default) fits unweighted.
        :returns: `pl.DataFrame` of knockoffs with the same schema as `X`.
    """
    from . import _processIsolation

    X = _resolve_df(X)
    if categorical_method in ( 'ranger_scip', 'xgb_scip' ):
        _scipModule = (
            'heteroknockoffpy.rbridge' if categorical_method == 'ranger_scip'
            else 'heteroknockoffpy.xgbScip'
        )

        # Conditional residuals knockoffs. Routed through _processIsolation (rather
        # than a direct top-level import + call, as this used to do) because both
        # backends are _GUARDED_MODULES entries -- rpy2/xgboost crash on macOS if
        # loaded into the same process as torch (e.g. after any prior PRISM-torch
        # or Deep-Knockoffs-GAN call), and a direct import here bypassed that guard.
        if conditional_expectations is None:
            conditional_expectations: pl.DataFrame = _processIsolation.run_isolated_if_loaded(
                _scipModule,
                'get_forest_conditional_expectations',
                X = X,
                verbose = verbose,
                verbose_prefix = verbose_prefix,
                weight = weight,
                #**kwargs,
            )
        #

        ce_np: np.ndarray = conditional_expectations.to_numpy()

        Xk_residuals: np.ndarray = knockoffCallable(
            X.select(
                conditional_expectations.columns
            ).to_numpy() - ce_np
        )

        return _processIsolation.run_isolated_if_loaded(
            _scipModule,
            'get_knockoffs_with_Xk_numeric',
            X = X,
            Xk_numeric = ce_np + Xk_residuals,
            rng = rng,
            verbose = verbose,
            verbose_prefix = verbose_prefix,
            weight = weight,
            #**kwargs,
        )
    #/if categorical_method in ( 'ranger_scip', 'xgb_scip' )
    else:
        oheMethod: Literal['softmax','max']
        logit: bool
        X_ohe_np: np.ndarray

        if categorical_method == 'ranger':
            oheMethod = 'softmax'
            logit = True

            # Routed through _processIsolation for the same reason as the scip
            # branch above: heteroknockoffpy.rbridge is a _GUARDED_MODULES entry
            # (rpy2), and a direct top-level `from . import rbridge` here bypassed
            # that guard.
            X_ohe_np = _processIsolation.run_isolated_if_loaded(
                'heteroknockoffpy.rbridge',
                'get_ohe_forest_probabilities_np',
                X = X,
                logit = logit,
                drop_first = True,
                verbose = verbose,
                verbose_prefix = verbose_prefix,
                weight = weight,
                #**kwargs,
            )
        #
        elif categorical_method == 'xgb':
            oheMethod = 'softmax'
            logit = True

            # Routed through _processIsolation: heteroknockoffpy.xgbScip is a
            # _GUARDED_MODULES entry (xgboost), and a direct top-level
            # `from . import xgbScip` here bypassed that guard -- this was the
            # actual segfault reproduced with torch already loaded in-process.
            X_ohe_np = _processIsolation.run_isolated_if_loaded(
                'heteroknockoffpy.xgbScip',
                'get_ohe_forest_probabilities_np',
                X = X,
                logit = logit,
                drop_first = True,
                verbose = verbose,
                verbose_prefix = verbose_prefix,
                rng = rng,
                weight = weight,
                **_pluck_xgb_kwargs( kwargs ),
            )
        #
        elif categorical_method == 'linear':
            oheMethod = 'softmax'
            logit = True
            
            X_ohe_np = utilities.get_ohe_linear_probabilities_np(
                X = X,
                logit = logit,
                drop_first = True,
                verbose = verbose,
                verbose_prefix = verbose_prefix,
                weight = weight,
                #**kwargs,
            )
        #
        elif categorical_method == 'ohe':
            oheMethod = 'max'
            logit = False
            
            X_ohe_np = utilities.get_ohe_np(
                X = X,
                drop_first = True,
            )
        #
        else:
            raise ValueError(
                "Unrecognized categorical_method={}".format(categorical_method)
            )
        #/switch categorical_method
        
        Xk_ohe_np: np.ndarray = knockoffCallable(
            X_ohe_np,
        )
        
        return utilities.collapse_ohe(
            X = X,
            X_ohe = Xk_ohe_np,
            method = oheMethod,
            logit = logit,
            drop_first = True,
            rng = rng,
        )
    #/switch categorical_method
    # EARLY RETURN/
#/def get_withCallable

def get_second_order(
    X: DataFrameLike,
    rng: np.random.Generator,
    categorical_method: Literal['ranger','linear','ohe','xgb','ranger_scip','xgb_scip',],
    conditional_expectations: pl.DataFrame | None = None,
    verbose: int = 0,
    verbose_prefix: str = '',
    weight: np.ndarray | None = None,
    **kwargs,
    ) -> pl.DataFrame:
    """
        Second-order (Gaussian-moment-matching) knockoffs via R `knockoff::create.second_order`
        (`rbridge.get_knockoffs_second_order_np`), the fastest/closed-form method:
        matches the first two moments (mean, covariance) of the (optionally
        OHE'd/SCIP-residual) numeric array. Categorical columns are handled by
        `get_withCallable` per `categorical_method` before this ever touches R.

        :param X: Original data (numeric + `pl.Categorical` columns).
        :param rng: Seeds both `categorical_method`'s randomness (see
            `get_withCallable`) and R's RNG (via `set.seed`) for the
            `create_second_order` draw itself.
        :param categorical_method: See `get_withCallable`'s docstring for the
            full list ('ranger'/'linear'/'ohe'/'xgb'/'ranger_scip'/'xgb_scip'); 'ranger_scip'/'xgb_scip'
            route numeric columns through conditional-residual knockoffs (this
            function's second-order draw operates on the *residuals*, not the raw
            columns) and categorical columns through backend-specific SCIP.
        :param conditional_expectations: Numeric conditional expectations. If not provided, uses `rbridge.get_forest_conditional_expectations`/`xgbScip.get_forest_conditional_expectations` to calculate, if `categorical_method` in ('ranger_scip', 'xgb_scip')
        :param verbose: Verbosity level (0 = silent), forwarded through to the R call.
        :param verbose_prefix: String prepended to any verbose print output.
        :param kwargs: Forwarded to `rbridge.get_knockoffs_second_order_np` (minus
            any recognized xgboost hyperparameters -- see `_drop_xgb_kwargs` --
            which are meant for `categorical_method='xgb'` instead and would
            otherwise reach R as unrecognized arguments), which forwards its own
            remaining kwargs to R `knockoff::create.second_order`. The one kwarg
            it specifically recognizes:
              - `shrink` (bool, default `True`): whether `create.second_order`
                shrinks the estimated covariance matrix before drawing knockoffs
                (recommended when `X`'s column count approaches or exceeds `X`'s
                row count, where the raw sample covariance is ill-conditioned).
                `False` uses the raw sample covariance directly; changing it
                produces materially different knockoffs (confirmed empirically),
                so it's worth setting explicitly when `p` is large relative to `n`.
            Also forwarded down to `get_withCallable`'s own `**kwargs` (used only
            by the `'xgb'` branch there via `_pluck_xgb_kwargs` -- see that
            docstring).
        :returns: `pl.DataFrame` of knockoffs with the same schema as `X`.
    """
    from . import _processIsolation

    def knockoffCallable( x: np.ndarray ) -> np.ndarray:
        return _processIsolation.run_isolated_if_loaded(
            'heteroknockoffpy.rbridge',
            'get_knockoffs_second_order_np',
            X = x,
            verbose = verbose,
            verbose_prefix = verbose_prefix,
            rng = rng,
            weight = weight,
            **_drop_xgb_kwargs( kwargs ),
        )
    #/def knockoffCallable

    return get_withCallable(
        X = X,
        rng = rng,
        categorical_method = categorical_method,
        knockoffCallable = knockoffCallable,
        conditional_expectations = conditional_expectations,
        verbose = verbose,
        verbose_prefix = verbose_prefix,
        weight = weight,
        **kwargs,
    )
#/def get_second_order

def get_torchGAN(
    X: DataFrameLike,
    rng: np.random.Generator,
    categorical_method: Literal['ranger','linear','ohe','xgb','ranger_scip','xgb_scip',],
    conditional_expectations: pl.DataFrame | None = None,
    verbose: int = 0,
    verbose_prefix: str = '',
    weight: np.ndarray | None = None,
    **kwargs,
    ) -> pl.DataFrame:
    """
        GAN-based knockoffs: trains a `heteroknockofftorch.torchKnockoffs.TorchGAN`
        (generator + discriminator + WGAN-critic + MINE mutual-information
        critic) on the (optionally OHE'd/SCIP-residual) numeric array and uses
        the trained generator to produce knockoffs. Slower than `second_order`
        but can capture non-Gaussian/non-linear dependence structure. Runs in an
        isolated subprocess whenever `xgboost`/`rpy2` are already loaded in this
        process (see `_processIsolation`), since torch's bundled `libomp` can
        crash alongside them.

        Parameter meanings below are sourced from `TorchGAN.__init__`/
        `.fit_predict` in `heteroknockofftorch/torchKnockoffs.py`; the values
        noted are this function's own signature defaults.

        :param X: Original data (numeric + `pl.Categorical` columns).
        :param rng: Seeds `categorical_method`'s randomness (see
            `get_withCallable`) AND torch's global RNG for the GAN training loop:
            each `knockoffCallable` invocation draws a fresh seed from `rng` and
            passes it to `fit_predict`, which calls `torch.manual_seed` before
            constructing `TorchGAN` -- covering parameter init and minibatch
            sampling for full run-to-run reproducibility.
        :param categorical_method: See `get_withCallable`'s docstring for the
            full list ('ranger'/'linear'/'ohe'/'xgb'/'ranger_scip'/'xgb_scip').
        :param conditional_expectations: Numeric conditional expectations. If not
            provided, uses `rbridge.get_forest_conditional_expectations`/
            `xgbScip.get_forest_conditional_expectations` to calculate, if
            `categorical_method` in ('ranger_scip', 'xgb_scip').
        :param verbose: Verbosity level (0 = silent). Currently unused by this
            function's own body (not forwarded into `fit_predict`, which has no
            verbose output), but still threaded through to `get_withCallable`.
        :param verbose_prefix: String prepended to any verbose print output.
        :param kwargs: Named GAN hyperparameters, each individually plucked out
            (via `kwargs.get(name, default)`, so unrecognized extra kwargs are
            silently ignored rather than erroring) and forwarded to
            `heteroknockofftorch.torchKnockoffs.fit_predict`/`TorchGAN`:
              - `x_name` (str, default `'Normal'`): label for the input
                distribution family, passed straight through to `TorchGAN`
                (used for logging/bookkeeping, not the training math itself).
              - `lamda` (float, default `1`): weight on the MINE mutual-information
                loss term in the generator's loss (`generator_loss = -D_loss +
                mu * -WD_fake.mean() + lamda * M_loss`) -- discourages the
                knockoffs from encoding excess information about which columns
                are "real" vs. "knockoff".
              - `mu` (float, default `1`): weight on the WGAN critic loss term in
                the same generator loss expression above.
              - `lam` (float, default `10`): WGAN gradient-penalty coefficient
                (`lam * ((grad_norm - 1) ** 2).mean()`), the standard WGAN-GP
                Lipschitz-constraint weight.
              - `lr` (float, default `1e-4`): Adam learning rate shared by the
                generator and all three critics (discriminator, WGAN-discriminator,
                MINE).
              - `mb_size` (int, default `128`): minibatch size for training.
              - `niter` (int, default `2000`): number of training iterations.
              - `combined_inner` (bool, default `False`): if `True`, the
                discriminator + WGAN-discriminator + MINE critics share a single
                Adam optimizer instead of three separate ones (fewer optimizer
                objects, coupled step sizes across the three critic losses).
        :param weight: Optional length-n sample weight. NOT used by the GAN
            training itself -- TorchGAN is unsupervised (no y-target), so
            there's no loss to multiply a per-row weight against; accepted
            here only for signature consistency with the other knockoff
            methods and documented as a no-op (per design decision -- a real
            weighted-training scheme would mean weighted minibatch sampling,
            a materially different mechanism, and was decided out of scope).
            Still forwarded to `categorical_method`'s own fit (which DOES use
            it, when 'ranger'/'xgb'/'linear'/'ranger_scip'/'xgb_scip').
        :returns: `pl.DataFrame` of knockoffs with the same schema as `X`.
    """
    from . import _processIsolation

    def knockoffCallable( x: np.ndarray ) -> np.ndarray:
        # Fresh draw per invocation (knockoffCallable may be called more than once,
        # e.g. once per categorical group) so each GAN fit gets a decorrelated but
        # run-to-run-reproducible seed from the same rng stream.
        torch_seed = int( rng.integers( 0, 2**63 ) )
        return _processIsolation.run_isolated_if_loaded(
            'heteroknockoffpy.heteroknockofftorch.torchKnockoffs',
            'fit_predict',
            x              = x,
            x_name         = kwargs.get( 'x_name',         'Normal' ),
            lamda          = kwargs.get( 'lamda',           1        ),
            mu             = kwargs.get( 'mu',              1        ),
            lam            = kwargs.get( 'lam',             10       ),
            lr             = kwargs.get( 'lr',              1e-4     ),
            mb_size        = kwargs.get( 'mb_size',         128      ),
            niter          = kwargs.get( 'niter',           2000     ),
            combined_inner = kwargs.get( 'combined_inner',  False    ),
            torch_seed     = torch_seed,
        )
    #/def knockoffCallable
    
    return get_withCallable(
        X = X,
        rng = rng,
        categorical_method = categorical_method,
        knockoffCallable = knockoffCallable,
        conditional_expectations = conditional_expectations,
        verbose = verbose,
        verbose_prefix = verbose_prefix,
        weight = weight,
        **kwargs,
    )
#/def get_torchGAN

def get_rangerSCIP(
    X: DataFrameLike,
    rng: np.random.Generator,
    residuals_method: Literal['normal','permute',] = 'normal',
    verbose: int = 0,
    verbose_prefix: str = '',
    weight: np.ndarray | None = None,
    **kwargs,
    ) -> pl.DataFrame:
    """
        Full sequential SCIP knockoff generation via R ranger (rbridge).
        Each column's knockoff conditions on all original columns plus all
        previously generated knockoffs; categorical columns are handled natively
        (probability-forest sampling), so this needs no separate
        `categorical_method` -- see `xgbScip`'s module docstring / `scip.knockoffs.R`
        for the full algorithm description (`get_xgbSCIP` is the xgboost-based
        drop-in alternative with an identical call shape).

        :param X: Original data (numeric + `pl.Categorical` columns).
        :param rng: Seeds R's RNG (`set.seed`) for both the categorical sampling
            draw and the numeric residual draw.
        :param residuals_method: "normal" (default) -- numeric knockoff residual
            drawn from `N(0, sd(residuals, ddof=1))` -- or "permute" -- residual
            is a random permutation of the observed residuals.
        :param verbose: Verbosity level (0 = silent), forwarded to the R call.
        :param verbose_prefix: String prepended to any verbose print output.
        :param kwargs: Forwarded to `rbridge.get_knockoffs_SCIP`, which forwards
            its own remaining kwargs to `ranger::ranger` for both the per-column
            probability forests (categorical) and regression forests (numeric).
            All optional and omitted (letting ranger use its own default) when absent:
              - `num_trees` (int): number of trees in the forest.
              - `mtry` (int): number of variables randomly sampled as candidates
                at each split.
              - `min_node_size` (int): minimum number of observations in a
                terminal node.
              - `max_depth` (int): maximum tree depth (0 = unlimited, ranger's default).
              - `sample_fraction` (float): fraction of observations sampled per tree.
              - `num_threads` (int): number of threads for ranger to use.
              - `respect_unordered_factors` (str, e.g. `'partition'`): how ranger
                splits unordered categorical predictors; `scip.knockoffs.R`'s
                `.scip.fit_probability_forest`/`.scip.fit_regression_forest`
                default to `'partition'` when this kwarg is omitted.
        :returns: `pl.DataFrame` of knockoffs with the same schema as `X`.
    """
    from . import _processIsolation

    return _processIsolation.run_isolated_if_loaded(
        'heteroknockoffpy.rbridge',
        'get_knockoffs_SCIP',
        X = X,
        rng = rng,
        residuals_method = residuals_method,
        verbose = verbose,
        verbose_prefix = verbose_prefix,
        weight = weight,
        **kwargs,
    )
#/def get_rangerSCIP

def get_xgbSCIP(
    X: DataFrameLike,
    rng: np.random.Generator,
    residuals_method: Literal['normal','permute',] = 'normal',
    verbose: int = 0,
    verbose_prefix: str = '',
    weight: np.ndarray | None = None,
    **kwargs,
    ) -> pl.DataFrame:
    """
        Full sequential SCIP knockoff generation via sequential xgboost models
        (xgbScip) -- a drop-in alternative to `get_rangerSCIP` that doesn't
        require R/rpy2. Same algorithm/call shape as `get_rangerSCIP`; see
        `xgbScip`'s module docstring for the full sequential-SCIP description.

        :param X: Original data (numeric + `pl.Categorical` columns).
        :param rng: Used directly (no separate R/native seed dance) for both the
            categorical sampling draw (`utilities.choices_from_weights`) and the
            numeric residual draw (`rng.normal`/`rng.permutation`).
        :param residuals_method: "normal" (default) -- numeric knockoff residual
            drawn from `N(0, sd(residuals, ddof=1))` via `rng.normal` -- or
            "permute" -- residual is `rng.permutation(residuals)`.
        :param verbose: Verbosity level (0 = silent).
        :param verbose_prefix: String prepended to any verbose print output.
        :param kwargs: Forwarded to xgboost.XGBRegressor/XGBClassifier -- see
            `xgbScip`'s module docstring for the relevant kwargs (max_depth,
            learning_rate, min_child_weight, subsample, colsample_bytree,
            reg_alpha, reg_lambda, gamma, n_estimators), and
            https://xgboost.readthedocs.io/en/latest/python/python_api.html
            for the full reference.
    """
    from . import _processIsolation

    return _processIsolation.run_isolated_if_loaded(
        'heteroknockoffpy.xgbScip',
        'get_knockoffs_SCIP',
        X = X,
        rng = rng,
        residuals_method = residuals_method,
        verbose = verbose,
        verbose_prefix = verbose_prefix,
        weight = weight,
        **kwargs,
    )
#/def get_xgbSCIP

def get_knockoffs(
    X: DataFrameLike,
    method: Literal[
        "second_order",
        "torch_GAN",
        "ranger_SCIP",
        "xgb_SCIP",
    ],
    rng: np.random.Generator,
    conditional_expectations: pl.DataFrame | None = None,
    verbose: int = 0,
    verbose_prefix: str = '',
    weight: np.ndarray | None = None,
    **kwargs,
    ) -> pl.DataFrame | tuple[ pl.DataFrame, pl.DataFrame ]:
    """
        Interface to name the knockoff method by string

        :param conditional_expectations: Necessary if kwargs['categorical_method'] in ("ranger_scip", "xgb_scip")
        :param weight: Optional length-n sample weight, forwarded to whichever
            method is selected. None (default) fits unweighted. See each
            get_*'s own docstring for exactly how weight is used
            (second_order: real weighted mean/covariance; torch_GAN:
            documented no-op for the GAN itself, still used by
            categorical_method's own fit; ranger_SCIP/xgb_SCIP: native
            case.weights/sample_weight).
    """
    numeric_columns: tuple[ str,... ] = tuple(
        col for col, dtype in X.schema.items() if dtype != pl.Categorical
    )
    
    Xk: pl.DataFrame
    
    if verbose > 0:
        print( verbose_prefix + 'Making Knockoffs')
        print( verbose_prefix +
            '  method={}'.format( method )
        )
        for key, val in kwargs.items():
            print(
                verbose_prefix +\
                    '  {}={}'.format( key, val )
                #/
            )
        #
    #
    
    if method == "second_order":
        # kwargs should have 'categorical_method'
        if kwargs['categorical_method'] in ( 'ranger_scip', 'xgb_scip' ):
            assert set( numeric_columns ) == set( conditional_expectations.columns )
        #
        else:
            conditional_expectations = None
        #

        Xk = get_second_order(
            X = X,
            rng = rng,
            conditional_expectations = conditional_expectations,
            verbose = verbose,
            verbose_prefix = verbose_prefix,
            weight = weight,
            **kwargs,
        )
    #
    elif method == "torch_GAN":
        # kwargs should have 'categorical_method'
        if kwargs['categorical_method'] in ( 'ranger_scip', 'xgb_scip' ):
            assert set( numeric_columns ) == set( conditional_expectations.columns )
        #
        else:
            conditional_expectations = None
        #

        Xk = get_torchGAN(
            X = X,
            rng = rng,
            conditional_expectations = conditional_expectations,
            verbose = verbose,
            verbose_prefix = verbose_prefix,
            weight = weight,
            **kwargs,
        )
    #
    elif method == "ranger_SCIP":
        # kwargs may have 'residuals_method'
        assert kwargs['residuals_method'] in ("permute","normal",)
        Xk = get_rangerSCIP(
            X = X,
            rng = rng,
            verbose = verbose,
            verbose_prefix = verbose_prefix,
            weight = weight,
            **kwargs,
        )
    #
    elif method == "xgb_SCIP":
        # kwargs may have 'residuals_method'
        assert kwargs['residuals_method'] in ("permute","normal",)
        Xk = get_xgbSCIP(
            X = X,
            rng = rng,
            verbose = verbose,
            verbose_prefix = verbose_prefix,
            weight = weight,
            **kwargs,
        )
    #
    else:
        raise ValueError("Unrecognized method={}".format(method))
    #
    
    return Xk
#/def get_knockoffs
