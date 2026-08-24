#
#//  glmScip.py
#//  heteroknockoffpy
#//
"""
    Zero-inflated Sequential Conditional Independence Procedure (SCIP)
    knockoff generation using simple linear GLMs -- a lighter-weight
    alternative backend to `xgbScip`'s tree-based `residuals_method=
    'zero_inflated'` for sparse numeric columns (see `xgbScip.py`'s module
    docstring for the general SCIP algorithm).

    Each sparse numeric column is modeled as a two-part hurdle model:
      1. L2-regularized logistic regression for P(col != 0 | rest).
      2. A nonzero-part regression for E[col | col != 0, rest], fit on the
         nonzero subset only -- either:
           - 'gamma' (default): L2-regularized Gamma GLM, log link
             (`sklearn.linear_model.GammaRegressor`). Assumes Var[y] scales
             with mean^2 (constant coefficient of variation).
           - 'exponential': Ridge regression of log(y) on the (standardized)
             other columns, i.e. a log-normal model -- `cond_exp = exp(raw
             linear prediction)`. Assumes constant variance *on the log
             scale* rather than Gamma's constant-CV assumption; matches the
             'exponential' family / 'log_diff' loss convention already used
             elsewhere in silverknockoff's `cellOps/zeroInflatedOps.py` for
             the torch hurdle model.

    Both parts are linear in the (standardized) other columns -- unlike
    `xgbScip`'s tree-based forests, which can capture nonlinear/interaction
    structure but cost more to fit per column and are harder to diagnose.
    Regularized linear models are used deliberately here (rather than plain
    unregularized MLE, e.g. `statsmodels.GLM`) because the explanatory
    columns are compositional (element stoichiometric proportions summing to
    ~1 per row) and thus strongly collinear -- unregularized IRLS on the
    Gamma log link overflows on this data (verified).

    The knockoff draw itself samples directly from the fitted distribution
    rather than perturbing a residual: a Bernoulli(prob_nonzero) gate decides
    zero vs nonzero, and nonzero rows draw from either Gamma(mean=cond_exp,
    dispersion=phi) or LogNormal(via exp(log_cond_exp + N(0, sigma_log))) --
    both keep the draw strictly positive, unlike an additive normal residual
    on the raw scale (as in `xgbScip`'s `_draw_zero_inflated_knockoff`).
"""

from .utilities import DataFrameLike, _resolve_df

import numpy as np
import polars as pl
import pandas as pd

from sklearn.linear_model import LogisticRegression, GammaRegressor, Ridge
from sklearn.preprocessing import StandardScaler

from typing import Any, Literal


def _standardized_design(
    scip_pd: pd.DataFrame,
    col: str,
    ) -> np.ndarray:
    """
        Other columns of `scip_pd`, zero-variance columns dropped, then
        standardized (zero mean, unit variance) -- shared design matrix for
        both the logistic and nonzero-part fits, since both estimators' L2
        penalties are scale-sensitive.
    """
    X_expl: np.ndarray = scip_pd.drop( columns=[ col ] ).to_numpy().astype( np.float64 )
    keep: np.ndarray = X_expl.var( axis=0 ) > 1e-12
    X_expl = X_expl[ :, keep ]

    scaler = StandardScaler()
    return scaler.fit_transform( X_expl ) if X_expl.shape[1] > 0 else X_expl
#/def _standardized_design


def _fit_nonzero_probability(
    X_std: np.ndarray,
    nonzero_mask: np.ndarray,
    logistic_kwargs: dict | None = None,
    ) -> np.ndarray:
    """
        LogisticRegression: P(col != 0 | rest). Degenerate cases (col
        all-zero or all-nonzero) skip the fit and use a constant probability.
    """
    n: int = nonzero_mask.shape[0]

    if nonzero_mask.all():
        return np.ones( n )
    #
    if not nonzero_mask.any():
        return np.zeros( n )
    #

    logit = LogisticRegression( **( logistic_kwargs or {} ) )
    logit.fit( X_std, nonzero_mask.astype( np.int64 ) )
    return logit.predict_proba( X_std )[ :, 1 ]
#/def _fit_nonzero_probability


def _fit_zero_inflated_glm(
    scip_pd: pd.DataFrame,
    col: str,
    rng: np.random.Generator | None,
    weight: np.ndarray | None = None,
    logistic_kwargs: dict | None = None,
    gamma_kwargs: dict | None = None,
    ) -> tuple[ np.ndarray, np.ndarray, float ]:
    """
        Two-part linear model for a sparse/zero-inflated numeric column
        `col` (Gamma nonzero part) -- see the module docstring.

        :param weight: Currently unused (LogisticRegression/GammaRegressor
            both accept sample_weight, but the zero-inflated draw path here
            has not been exercised with weighted data yet) -- reserved for
            interface parity with `xgbScip`'s SCIP functions.
        :returns:
            - prob_nonzero: length-n predicted P(col != 0), one per row.
            - cond_exp: length-n conditional expectation from the
                nonzero-fit Gamma regressor, predicted for every row (only
                used where a knockoff draw comes out nonzero).
            - dispersion: method-of-moments Gamma dispersion estimate phi,
                s.t. Var[col | col != 0] ~= phi * cond_exp^2 -- 0.0 when
                there are fewer than 2 nonzero rows (no signal to estimate
                from).
    """
    y: np.ndarray = scip_pd[ col ].to_numpy().astype( np.float64 )
    nonzero_mask: np.ndarray = ( y != 0.0 )
    n: int = y.shape[0]

    X_std: np.ndarray = _standardized_design( scip_pd, col )
    prob_nonzero: np.ndarray = _fit_nonzero_probability( X_std, nonzero_mask, logistic_kwargs )

    n_nonzero: int = int( nonzero_mask.sum() )
    if n_nonzero < 2:
        return prob_nonzero, np.zeros( n ), 0.0
    #

    gamma = GammaRegressor( **( gamma_kwargs or {} ) )
    gamma.fit( X_std[ nonzero_mask ], y[ nonzero_mask ] )

    cond_exp: np.ndarray = gamma.predict( X_std )
    mu_nonzero: np.ndarray = cond_exp[ nonzero_mask ]
    # Method-of-moments Gamma dispersion: Var[y] = phi * mu^2, so
    # phi_hat = mean( ((y - mu) / mu)^2 ) (a.k.a. mean squared Pearson residual).
    dispersion: float = float( np.mean( ( ( y[ nonzero_mask ] - mu_nonzero ) / mu_nonzero ) ** 2 ) )

    return prob_nonzero, cond_exp, dispersion
#/def _fit_zero_inflated_glm


def _fit_zero_inflated_exponential(
    scip_pd: pd.DataFrame,
    col: str,
    rng: np.random.Generator | None,
    weight: np.ndarray | None = None,
    logistic_kwargs: dict | None = None,
    ridge_kwargs: dict | None = None,
    clipped: bool = True,
    ) -> tuple[ np.ndarray, np.ndarray, float, int ]:
    """
        Two-part linear model for a sparse/zero-inflated numeric column
        `col` (log-normal / 'exponential' nonzero part) -- see the module
        docstring. The nonzero part is Ridge regression of log(y) on the
        (standardized) other columns; `cond_exp` here is on the LOG scale
        (`log_cond_exp`, not `E[y]` directly) -- exponentiate it (as
        `_draw_zero_inflated_lognormal_knockoff` does) to get a point
        prediction of y itself.

        The Ridge-on-log(y) fit extrapolates badly on rows whose other
        columns are underrepresented in the training data -- several
        correlated predictors lining up can push the linear predictor far
        outside the training range, and `exp()` then amplifies that into a
        wildly implausible prediction (verified: a held-out Cu row predicted
        ~40,000 against an observed max of 70).

        :param clipped: True (default) -- `log_cond_exp` is clipped to
            `[log_y_nonzero.min(), log_y_nonzero.max()]`, the range actually
            observed in the training nonzero subset, which bounds
            `exp(log_cond_exp)` to the training data's own [min, max]
            nonzero value at the cost of not extrapolating beyond it. False
            leaves the raw (potentially blown-up) prediction as-is --
            `n_clipped` is still computed either way so callers can tell how
            many rows *would* have been clipped.
        :returns:
            - prob_nonzero: length-n predicted P(col != 0), one per row.
            - log_cond_exp: length-n predicted E[log(col) | col != 0, rest],
                clipped to the training nonzero log-range iff `clipped`,
                predicted for every row (only used where a knockoff draw
                comes out nonzero).
            - sigma_log: residual standard deviation of log(y) - log_cond_exp
                on the nonzero subset (ddof=1, computed pre-clipping) -- 0.0
                when there are fewer than 2 nonzero rows.
            - n_clipped: number of rows (out of n) whose raw prediction fell
                outside the training range (regardless of whether `clipped`
                actually applied it).
    """
    y: np.ndarray = scip_pd[ col ].to_numpy().astype( np.float64 )
    nonzero_mask: np.ndarray = ( y != 0.0 )
    n: int = y.shape[0]

    X_std: np.ndarray = _standardized_design( scip_pd, col )
    prob_nonzero: np.ndarray = _fit_nonzero_probability( X_std, nonzero_mask, logistic_kwargs )

    n_nonzero: int = int( nonzero_mask.sum() )
    if n_nonzero < 2:
        return prob_nonzero, np.zeros( n ), 0.0, 0
    #

    log_y_nonzero: np.ndarray = np.log( y[ nonzero_mask ] )

    ridge = Ridge( **( ridge_kwargs or { 'alpha': 1.0 } ) )
    ridge.fit( X_std[ nonzero_mask ], log_y_nonzero )

    log_cond_exp_raw: np.ndarray = ridge.predict( X_std )
    residuals: np.ndarray = log_y_nonzero - log_cond_exp_raw[ nonzero_mask ]
    sigma_log: float = float( residuals.std( ddof=1 ) ) if n_nonzero > 1 else 0.0

    lo: float = float( log_y_nonzero.min() )
    hi: float = float( log_y_nonzero.max() )
    n_clipped: int = int( np.sum( ( log_cond_exp_raw < lo ) | ( log_cond_exp_raw > hi ) ) )
    log_cond_exp: np.ndarray = np.clip( log_cond_exp_raw, lo, hi ) if clipped else log_cond_exp_raw

    return prob_nonzero, log_cond_exp, sigma_log, n_clipped
#/def _fit_zero_inflated_exponential


def _draw_zero_inflated_gamma_knockoff(
    cond_exp: np.ndarray,
    prob_nonzero: np.ndarray,
    dispersion: float,
    rng: np.random.Generator,
    ) -> np.ndarray:
    """
        Draw a knockoff column from a fitted zero-inflated Gamma GLM
        (`_fit_zero_inflated_glm`): a Bernoulli(prob_nonzero) draw decides
        whether each row is zero or nonzero; nonzero rows draw from
        Gamma(mean=cond_exp, dispersion=phi) via `rng.gamma(shape, scale)`
        with shape=1/phi, scale=cond_exp*phi (mean = shape*scale = cond_exp).

        dispersion <= 0 (degenerate fit, e.g. < 2 nonzero rows observed)
        makes every draw zero, matching `_fit_zero_inflated_glm`'s all-zero
        `cond_exp` in that case.
    """
    n: int = cond_exp.shape[0]
    is_nonzero: np.ndarray = rng.random( n ) < prob_nonzero

    if dispersion <= 0.0:
        return np.zeros( n )
    #

    shape: float = 1.0 / dispersion
    scale: np.ndarray = np.clip( cond_exp, 1e-12, None ) * dispersion
    draw: np.ndarray = rng.gamma( shape, scale )

    return np.where( is_nonzero, draw, 0.0 )
#/def _draw_zero_inflated_gamma_knockoff


def _draw_zero_inflated_lognormal_knockoff(
    log_cond_exp: np.ndarray,
    prob_nonzero: np.ndarray,
    sigma_log: float,
    rng: np.random.Generator,
    ) -> np.ndarray:
    """
        Draw a knockoff column from a fitted zero-inflated log-normal model
        (`_fit_zero_inflated_exponential`): a Bernoulli(prob_nonzero) draw
        decides whether each row is zero or nonzero; nonzero rows draw
        `exp(log_cond_exp + N(0, sigma_log))`.

        sigma_log <= 0 (degenerate fit) makes every draw zero, matching
        `_fit_zero_inflated_exponential`'s all-zero `log_cond_exp` in that case.
    """
    n: int = log_cond_exp.shape[0]
    is_nonzero: np.ndarray = rng.random( n ) < prob_nonzero

    if sigma_log <= 0.0:
        return np.zeros( n )
    #

    draw: np.ndarray = np.exp( log_cond_exp + rng.normal( 0, sigma_log, size=n ) )

    return np.where( is_nonzero, draw, 0.0 )
#/def _draw_zero_inflated_lognormal_knockoff


def get_knockoffs_SCIP(
    X: DataFrameLike,
    rng: np.random.Generator,
    nonzero_model: Literal[ 'gamma', 'exponential' ] = 'gamma',
    clipped: bool = True,
    verbose: int = 0,
    verbose_prefix: str = '',
    weight: np.ndarray | None = None,
    logistic_kwargs: dict | None = None,
    nonzero_kwargs: dict | None = None,
    ) -> pl.DataFrame:
    """
        Full sequential SCIP knockoff generation for all-numeric, sparse/
        zero-inflated X, using a linear zero-inflated model in place of
        `xgbScip`'s tree-based forests. Each column's knockoff conditions on
        all original columns plus all previously generated knockoffs, same
        chaining convention as `xgbScip.get_knockoffs_SCIP`.

        Categorical (`pl.Categorical`) columns are not supported -- this
        module targets superconduct_1-style all-numeric compositional data;
        raises `ValueError` if X has any.

        :param nonzero_model: 'gamma' (default) -- L2-regularized Gamma GLM,
            log link (`_fit_zero_inflated_glm` /
            `_draw_zero_inflated_gamma_knockoff`) -- or 'exponential' --
            log-normal (Ridge on log(y)) (`_fit_zero_inflated_exponential` /
            `_draw_zero_inflated_lognormal_knockoff`). See the module
            docstring for the modeling difference.
        :param clipped: `nonzero_model='exponential'` only (ignored for
            'gamma', which has no equivalent blowup failure mode -- see
            `_fit_zero_inflated_glm`'s dispersion-based draw). True (default)
            clips each column's log-scale prediction to its own training
            nonzero range before exponentiating -- see
            `_fit_zero_inflated_exponential`'s docstring for why (unclipped,
            a handful of out-of-distribution rows can blow up to values
            orders of magnitude past the observed range).
        :param logistic_kwargs: Forwarded to `sklearn.linear_model.
            LogisticRegression` (e.g. `C` for the inverse regularization
            strength).
        :param nonzero_kwargs: Forwarded to the nonzero part's estimator --
            `sklearn.linear_model.GammaRegressor` (`nonzero_model='gamma'`)
            or `sklearn.linear_model.Ridge` (`nonzero_model='exponential'`).
    """
    X = _resolve_df( X )

    if any( dtype == pl.Categorical for dtype in X.schema.values() ):
        raise ValueError( "glmScip.get_knockoffs_SCIP does not support categorical columns" )
    #
    if nonzero_model not in ( 'gamma', 'exponential' ):
        raise ValueError( "Unrecognized nonzero_model={}".format( nonzero_model ) )
    #

    scip_pd: pd.DataFrame = X.to_pandas()

    for col in X.columns:
        ko: str = col + '~'

        if verbose > 0:
            print( verbose_prefix + 'glm SCIP column: {}'.format( col ) )
        #

        if nonzero_model == 'gamma':
            prob_nonzero, cond_exp, dispersion = _fit_zero_inflated_glm(
                scip_pd, col, rng, weight=weight,
                logistic_kwargs=logistic_kwargs, gamma_kwargs=nonzero_kwargs,
            )
            scip_pd[ ko ] = _draw_zero_inflated_gamma_knockoff(
                cond_exp, prob_nonzero, dispersion, rng,
            )
        #
        else:
            prob_nonzero, log_cond_exp, sigma_log, n_clipped = _fit_zero_inflated_exponential(
                scip_pd, col, rng, weight=weight,
                logistic_kwargs=logistic_kwargs, ridge_kwargs=nonzero_kwargs,
                clipped=clipped,
            )
            if verbose > 0 and n_clipped > 0:
                print( verbose_prefix + '  clipped {} / {} predictions to training range'.format(
                    n_clipped, len( log_cond_exp )
                ) )
            #
            scip_pd[ ko ] = _draw_zero_inflated_lognormal_knockoff(
                log_cond_exp, prob_nonzero, sigma_log, rng,
            )
        #/if nonzero_model == 'gamma'/else
    #/for col in X.columns

    Xk_pd: pd.DataFrame = scip_pd[ [ col + '~' for col in X.columns ] ]
    Xk_pd.columns = list( X.columns )

    return pl.from_pandas( Xk_pd ).cast(
        { col: dtype for col, dtype in X.schema.items() }
    )
#/def get_knockoffs_SCIP
