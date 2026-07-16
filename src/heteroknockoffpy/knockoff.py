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

def get_withCallable(
    X: DataFrameLike,
    rng: np.random.Generator,
    categorical_method: Literal['forest','linear','ohe','ranger_scip','xgb_scip'],
    knockoffCallable: Callable[ [ np.ndarray ], np.ndarray ],
    conditional_expectations: pl.DataFrame | None = None,
    verbose: int = 0,
    verbose_prefix: str = '',
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
            - 'forest': Get logit probabilities with random forests
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
            unused for the `'forest'`/`'linear'`/`'ohe'` branch. Reserved for
            future per-categorical_method tuning.
        :returns: `pl.DataFrame` of knockoffs with the same schema as `X`.
    """
    X = _resolve_df(X)
    if categorical_method in ( 'ranger_scip', 'xgb_scip' ):
        if categorical_method == 'ranger_scip':
            from . import rbridge as _scipBackend
        else:
            from . import xgbScip as _scipBackend
        #/if categorical_method == 'ranger_scip'/else

        # Conditional residuals knockoffs
        if conditional_expectations is None:
            conditional_expectations: pl.DataFrame = _scipBackend.get_forest_conditional_expectations(
                X = X,
                verbose = verbose,
                verbose_prefix = verbose_prefix,
                #**kwargs,
            )
        #

        ce_np: np.ndarray = conditional_expectations.to_numpy()

        Xk_residuals: np.ndarray = knockoffCallable(
            X.select(
                conditional_expectations.columns
            ).to_numpy() - ce_np
        )

        return _scipBackend.get_knockoffs_with_Xk_numeric(
            X = X,
            Xk_numeric = ce_np + Xk_residuals,
            rng = rng,
            verbose = verbose,
            verbose_prefix = verbose_prefix,
            #**kwargs,
        )
    #/if categorical_method in ( 'ranger_scip', 'xgb_scip' )
    else:
        oheMethod: Literal['softmax','max']
        logit: bool
        X_ohe_np: np.ndarray
        
        if categorical_method == 'forest':
            from . import rbridge
            
            oheMethod = 'softmax'
            logit = True
            
            X_ohe_np = rbridge.get_ohe_forest_probabilities_np(
                X = X,
                logit = logit,
                drop_first = True,
                verbose = verbose,
                verbose_prefix = verbose_prefix,
                #**kwargs,
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
    categorical_method: Literal['forest','linear','ohe','ranger_scip','xgb_scip',],
    conditional_expectations: pl.DataFrame | None = None,
    verbose: int = 0,
    verbose_prefix: str = '',
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
            full list ('forest'/'linear'/'ohe'/'ranger_scip'/'xgb_scip'); 'ranger_scip'/'xgb_scip'
            route numeric columns through conditional-residual knockoffs (this
            function's second-order draw operates on the *residuals*, not the raw
            columns) and categorical columns through backend-specific SCIP.
        :param conditional_expectations: Numeric conditional expectations. If not provided, uses `rbridge.get_forest_conditional_expectations`/`xgbScip.get_forest_conditional_expectations` to calculate, if `categorical_method` in ('ranger_scip', 'xgb_scip')
        :param verbose: Verbosity level (0 = silent), forwarded through to the R call.
        :param verbose_prefix: String prepended to any verbose print output.
        :param kwargs: Forwarded to `rbridge.get_knockoffs_second_order_np`, which
            forwards its own remaining kwargs to R `knockoff::create.second_order`.
            The one kwarg it specifically recognizes:
              - `shrink` (bool, default `True`): whether `create.second_order`
                shrinks the estimated covariance matrix before drawing knockoffs
                (recommended when `X`'s column count approaches or exceeds `X`'s
                row count, where the raw sample covariance is ill-conditioned).
                `False` uses the raw sample covariance directly. Not tuned
                anywhere in `silverknockoff` today (always left at the default);
                changing it produces materially different knockoffs (confirmed
                empirically -- large deviation on a small correlated-Gaussian
                test array), so it's worth setting explicitly when `p` is large
                relative to `n`.
            Also forwarded down to `get_withCallable`'s own `**kwargs` (currently
            unused there for either branch -- see that docstring).
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
            **kwargs,
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
        **kwargs,
    )
#/def get_second_order

def get_torchGAN(
    X: DataFrameLike,
    rng: np.random.Generator,
    categorical_method: Literal['forest','linear','ohe','ranger_scip','xgb_scip',],
    conditional_expectations: pl.DataFrame | None = None,
    verbose: int = 0,
    verbose_prefix: str = '',
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

        Not currently wired up / tuned anywhere in `silverknockoff` (no
        `synth_sweep_*` bundle exercises this method), so unlike `get_xgbSCIP`/
        `xgbImportances` there's no real-world "commonly tuned" value set to cite
        here -- the parameter meanings below (from `TorchGAN.__init__`/`.fit_predict`
        in `heteroknockofftorch/torchKnockoffs.py`) and their function-signature
        defaults are what's actually exercised in this codebase today.

        :param X: Original data (numeric + `pl.Categorical` columns).
        :param rng: Seeds `categorical_method`'s randomness (see
            `get_withCallable`); NOT used to seed torch's own RNG (the GAN
            training loop uses torch's ambient/global RNG state, unseeded).
        :param categorical_method: See `get_withCallable`'s docstring for the
            full list ('forest'/'linear'/'ohe'/'ranger_scip'/'xgb_scip').
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
        :returns: `pl.DataFrame` of knockoffs with the same schema as `X`.
    """
    from . import _processIsolation

    def knockoffCallable( x: np.ndarray ) -> np.ndarray:
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
        **kwargs,
    )
#/def get_torchGAN

def get_rangerSCIP(
    X: DataFrameLike,
    rng: np.random.Generator,
    residuals_method: Literal['normal','permute',] = 'normal',
    verbose: int = 0,
    verbose_prefix: str = '',
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
            The kwargs `silverknockoff` tunes/forwards for ranger-based importance
            methods (`_ranger_kwargs_from_params` in
            `silverknockoff/src/silverknockoff/cellOps/calculatorOps.py`), all
            optional and omitted (letting ranger use its own default) when absent:
              - `num_trees` (int): number of trees in the forest.
              - `mtry` (int): number of variables randomly sampled as candidates
                at each split.
              - `min_node_size` (int): minimum number of observations in a
                terminal node.
              - `max_depth` (int): maximum tree depth (0 = unlimited, ranger's default).
              - `sample_fraction` (float): fraction of observations sampled per tree.
              - `num_threads` (int): number of threads for ranger to use.
              - `respect_unordered_factors` (str, e.g. `'partition'`): how ranger
                splits unordered categorical predictors -- `'partition'` is the
                only value actually tuned/used across the `synth_sweep_*_3`
                bundles (both `ranger_gini` importances and the SCIP scripts
                default to it too, per `scip.knockoffs.R`'s
                `.scip.fit_probability_forest`/`.scip.fit_regression_forest`).
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
        **kwargs,
    )
#/def get_rangerSCIP

def get_xgbSCIP(
    X: DataFrameLike,
    rng: np.random.Generator,
    residuals_method: Literal['normal','permute',] = 'normal',
    verbose: int = 0,
    verbose_prefix: str = '',
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
            reg_alpha, reg_lambda, gamma, n_estimators) and their tuned
            values, and
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
    **kwargs,
    ) -> pl.DataFrame | tuple[ pl.DataFrame, pl.DataFrame ]:
    """
        Interface to name the knockoff method by string

        :param conditional_expectations: Necessary if kwargs['categorical_method'] in ("ranger_scip", "xgb_scip")
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
            **kwargs,
        )
    #
    else:
        raise ValueError("Unrecognized method={}".format(method))
    #
    
    return Xk
#/def get_knockoffs
