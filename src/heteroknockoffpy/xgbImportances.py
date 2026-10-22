#
#//  xgbImportances.py
#//  heteroknockoffpy
#//

from .utilities import DataFrameLike, SeriesOrDataFrameLike, OutcomeDescriptor, _resolve_df, _resolve_y

import numpy as np
import polars as pl
import pandas as pd
import xgboost

from typing import Literal


def _concat_X_Xk(X: pl.DataFrame, Xk: pl.DataFrame) -> pl.DataFrame:
    return pl.concat(
        (
            X,
            Xk.rename( { col: col + '~' for col in Xk.columns } ),
        ),
        how = 'horizontal_extend',
    )
#/def _concat_X_Xk


def _any_categorical(X_all: pl.DataFrame) -> bool:
    return any( dtype == pl.Categorical for dtype in X_all.dtypes )
#/def _any_categorical


def _make_model(
    outcome_type: Literal['continuous','count','categorical',],
    enable_categorical: bool,
    rng: np.random.Generator | None = None,
    **model_kwargs,
    ) -> 'xgboost.XGBRegressor | xgboost.XGBClassifier':
    if rng is not None:
        model_kwargs = dict( model_kwargs )
        model_kwargs.setdefault( 'random_state', rng )
    #
    if outcome_type == 'continuous':
        return xgboost.XGBRegressor(
            objective = 'reg:squarederror',
            tree_method = 'hist',
            enable_categorical = enable_categorical,
            **model_kwargs,
        )
    #
    elif outcome_type == 'count':
        return xgboost.XGBRegressor(
            objective = 'count:poisson',
            tree_method = 'hist',
            enable_categorical = enable_categorical,
            **model_kwargs,
        )
    #
    elif outcome_type == 'categorical':
        # No explicit objective: the sklearn wrapper auto-selects
        # binary:logistic / multi:softprob (with num_class) from y's classes.
        return xgboost.XGBClassifier(
            tree_method = 'hist',
            enable_categorical = enable_categorical,
            **model_kwargs,
        )
    #
    else:
        raise ValueError( "Unrecognized outcome_type={}".format( outcome_type ) )
    #/switch outcome_type
#/def _make_model


def _resolve_y_for_fit(
    y: pl.Series | pl.DataFrame,
    outcome_type: Literal['continuous','count','categorical',],
    ) -> np.ndarray:
    if outcome_type == 'categorical':
        _y_series: pl.Series = y.to_series() if isinstance( y, pl.DataFrame ) else y
        # xgboost's sklearn API requires contiguous integer class labels (0..k-1);
        # it no longer label-encodes arbitrary string/categorical targets itself.
        _, codes = np.unique( _y_series.cast( pl.Utf8 ).to_numpy(), return_inverse=True )
        return codes.astype( np.int64 )
    #
    _y_arr: np.ndarray = (
        y.to_numpy() if isinstance( y, pl.Series )
        else y.to_numpy().squeeze()
    )
    return _y_arr.astype( np.float64 )
#/def _resolve_y_for_fit


def _fit_model(
    X: DataFrameLike,
    Xk: DataFrameLike,
    y: SeriesOrDataFrameLike,
    outcome_type: Literal['continuous','count','categorical',] | None,
    verbose: int,
    model_kwargs: dict | None = None,
    rng: np.random.Generator | None = None,
    weight: np.ndarray | None = None,
    **fit_kwargs,
    ) -> tuple[
        xgboost.XGBRegressor | xgboost.XGBClassifier,
        pl.DataFrame,
        pd.DataFrame,
        OutcomeDescriptor,
    ]:
    """
    :param weight: Optional length-n sample weight, forwarded to
        XGBRegressor/XGBClassifier.fit(sample_weight=weight). None (default)
        fits unweighted, xgboost's own default.
    """
    X = _resolve_df( X )
    Xk = _resolve_df( Xk )
    y = _resolve_y( y )

    outcomeDescriptor: OutcomeDescriptor = OutcomeDescriptor.infer(
        y = y,
        outcome_type = outcome_type,
    )
    if outcomeDescriptor.outcome_dimension != 'single':
        raise TypeError( "Joint outcomes unavailable" )
    #

    X_all: pl.DataFrame = _concat_X_Xk( X, Xk )
    enable_categorical: bool = _any_categorical( X_all )
    X_all_pd: pd.DataFrame = X_all.to_pandas()

    model = _make_model( outcomeDescriptor.outcome_type, enable_categorical, rng=rng, **( model_kwargs or {} ) )
    y_fit: np.ndarray = _resolve_y_for_fit( y, outcomeDescriptor.outcome_type )

    if verbose > 0:
        print( "Fitting {}:".format( type( model ).__name__ ) )
        print( "  enable_categorical={}".format( enable_categorical ) )
        print( "  outcome_type={}".format( outcomeDescriptor.outcome_type ) )
    #

    if weight is not None:
        fit_kwargs = dict( fit_kwargs, sample_weight=weight )
    model.fit( X_all_pd, y_fit, **fit_kwargs )

    return model, X_all, X_all_pd, outcomeDescriptor
#/def _fit_model


def score_importances(
    X: DataFrameLike,
    Xk: DataFrameLike,
    y: SeriesOrDataFrameLike,
    outcome_type: Literal['continuous','count','categorical',] | None = None,
    importance_type: Literal['weight','gain','cover','total_gain','total_cover',] = 'gain',
    verbose: int = 0,
    model_kwargs: dict | None = None,
    rng: np.random.Generator | None = None,
    weight: np.ndarray | None = None,
    **fit_kwargs,
    ) -> np.ndarray:
    """
        Fits a single xgboost model on [X, Xk] and returns split-based importances
        from Booster.get_score(importance_type=importance_type), one per OHE-free
        column (numeric columns and native categorical columns alike).

        :param model_kwargs: Forwarded to the XGBRegressor/XGBClassifier constructor
            (e.g. max_depth, n_estimators, learning_rate, subsample, reg_alpha, ...).
        :param rng: If given, seeds the XGBoost fit (random_state=rng) for
            reproducibility -- xgboost's sklearn API accepts a np.random.Generator
            directly. Unseeded (xgboost's own default) if omitted.
        :param weight: Optional length-n sample weight. None (default) fits
            unweighted, xgboost's own default.
        :param fit_kwargs: Forwarded to XGBRegressor/XGBClassifier.fit (e.g.
            eval_set, early_stopping_rounds).
    """
    model, X_all, X_all_pd, _ = _fit_model( X, Xk, y, outcome_type, verbose, model_kwargs=model_kwargs, rng=rng, weight=weight, **fit_kwargs )

    score_dict: dict[ str, float ] = model.get_booster().get_score( importance_type = importance_type )

    importances: np.ndarray = np.zeros( X_all.shape[1] )
    for i, col in enumerate( X_all.columns ):
        importances[ i ] = score_dict.get( col, 0.0 )
    #

    return importances
#/def score_importances


def _unzero(vec: np.ndarray) -> np.ndarray:
    # Replace non-positive predicted values with the minimum positive value
    # before taking logs, to avoid -inf. Mirrors .prism_count.unzero in the R scripts.
    pos = vec[ vec > 0 ]
    if pos.size == 0:
        return np.full_like( vec, 1e-10 )
    #
    min_pos = pos.min()
    out = vec.copy()
    out[ out <= 0 ] = min_pos
    return out
#/def _unzero


def _unzero_normalize(mat: np.ndarray) -> np.ndarray:
    # Replace zero cells with the minimum nonzero value, then row-normalize.
    # Mirrors .prism_cat.unzero_normalize in the R scripts.
    min_nonzero = mat[ mat > 0 ].min()
    out = mat.copy()
    out[ out == 0 ] = min_nonzero
    return out / out.sum( axis=1, keepdims=True )
#/def _unzero_normalize


def _ridge_inv_cov( cov: np.ndarray ) -> np.ndarray:
    """Inverse of a (k, k) contrast covariance with a trace-relative ridge,
    cov + max(1e-6 * tr(cov) / k, 1e-12) * I -- the same rule as
    heteroknockofftorch.prismImportances._ridge_inv_cov_t."""
    k = cov.shape[ 0 ]
    eps = max( 1e-6 * float( np.trace( cov ) ) / k, 1e-12 )
    return np.linalg.inv( cov + eps * np.eye( k ) )
#/def _ridge_inv_cov


def _mahalanobis_norm( d: np.ndarray, VI: np.ndarray ) -> np.ndarray:
    """Row-wise sqrt(d_i' VI d_i) of an (n, k) array, clamped at 0 before the root."""
    return np.sqrt( np.maximum( np.einsum( 'ij,jk,ik->i', d, VI, d ), 0.0 ) )
#/def _mahalanobis_norm


def prism_importances(
    X: DataFrameLike,
    Xk: DataFrameLike,
    y: SeriesOrDataFrameLike,
    outcome_type: Literal['continuous','count','categorical',] | None = None,
    bandwidth: float = 1.0,
    # 2026-10-20: default 0.2 -> 0.0 -- the step is now `bandwidth` on the
    # standardized scale, used as given, matching the torch PRISM convention.
    bandwidth_exponent: float = 0.0,
    exponent: float = 1.0,
    verbose: int = 0,
    model_kwargs: dict | None = None,
    rng: np.random.Generator | None = None,
    weight: np.ndarray | None = None,
    **fit_kwargs,
    ) -> np.ndarray:
    """
        PRISM local-gradient importance using a single xgboost model on [X, Xk],
        mirroring stat.forest.prism_{continuous,count,categorical}.R:

        Numeric columns   — central finite difference on the standardized scale:
            (g(x + h*sd) - g(x - h*sd)) / (2h), h = bandwidth / n**bandwidth_exponent
            (h = bandwidth by default), i.e. a per-SD derivative -- the same
            convention as the torch PRISM networks, which standardize X and use
            +/- bandwidth directly.
        Categorical columns — sweep across category levels: the frequency-weighted
            SD of the level predictions, sqrt( sum_k pi_k ( g_k - gbar )^2 ), for a
            scalar response, and its Omega-norm analog
            sqrt( sum_k pi_k ||g_k - gbar||_Omega^2 ) for a categorical response;
            pi_k pools the level frequencies of the predictor and its knockoff.

        continuous: raw prediction. count: log(unzero(prediction)).
        categorical: log-odds contrasts (vs. first class) of predict_proba,
        reduced via the Mahalanobis norm sqrt(d' Omega d), Omega the
        trace-ridge-regularized inverse covariance of the base contrasts
        (same rule as the torch networks' `_ridge_inv_cov_t`).

        :param model_kwargs: Forwarded to the XGBRegressor/XGBClassifier constructor
            (e.g. max_depth, n_estimators, learning_rate, subsample, reg_alpha, ...).
        :param rng: If given, seeds the XGBoost fit (random_state=rng) for
            reproducibility -- xgboost's sklearn API accepts a np.random.Generator
            directly. Unseeded (xgboost's own default) if omitted.
        :param weight: Optional length-n sample weight. None (default) fits
            unweighted, xgboost's own default.
        :param fit_kwargs: Forwarded to XGBRegressor/XGBClassifier.fit.
    """
    model, X_all, X_all_pd, outcomeDescriptor = _fit_model( X, Xk, y, outcome_type, verbose, model_kwargs=model_kwargs, rng=rng, weight=weight, **fit_kwargs )
    ot = outcomeDescriptor.outcome_type

    n: int = X_all_pd.shape[0]
    p_all: int = X_all_pd.shape[1]
    # 2026-10-20: forward difference with raw step sd*bandwidth/n**bandwidth_exponent
    # replaced by a central difference with standardized step h (see docstring).
    if False:
        n_factor: float = n ** bandwidth_exponent
    #
    h_std: float = bandwidth / ( n ** bandwidth_exponent )

    VI: np.ndarray | None = None
    base_contrasts: np.ndarray | None = None
    base_log: np.ndarray | None = None
    base_preds: np.ndarray | None = None

    if ot == 'categorical':
        base_probs: np.ndarray = model.predict_proba( X_all_pd )
        base_log = np.log( _unzero_normalize( base_probs ) )
        k_y: int = base_log.shape[1]
        if k_y < 2:
            import warnings
            warnings.warn( "xgbPrismImportances: only one class predicted; returning zero importances" )
            return np.zeros( p_all )
        #
        base_contrasts = base_log[ :, 1: ] - base_log[ :, 0:1 ]
        # np.cov collapses to a 0-d scalar for a single-column input (binary
        # outcome, k_y=2); atleast_2d restores the 1x1 matrix shape inv() needs.
        # 2026-10-20: same trace-relative ridge as the torch networks, instead of
        # an unregularized inverse.
        if False:
            VI = np.linalg.inv( np.atleast_2d( np.cov( base_contrasts, rowvar=False ) ) )
        #
        VI = _ridge_inv_cov( np.atleast_2d( np.cov( base_contrasts, rowvar=False ) ) )
    #
    elif ot == 'count':
        base_preds = np.log( _unzero( model.predict( X_all_pd ) ) )
    #
    else:
        base_preds = model.predict( X_all_pd )
    #/switch ot

    importances_pointwise: np.ndarray = np.zeros( ( n, p_all ) )

    for j, col in enumerate( X_all_pd.columns ):
        if verbose > 0 and (j + 1) % 10 == 0:
            print( "xgbPrismImportances: column {} / {}".format( j + 1, p_all ) )
        #

        is_categorical: bool = isinstance( X_all_pd[ col ].dtype, pd.CategoricalDtype )

        if is_categorical:
            levels = X_all_pd[ col ].cat.categories
            if len( levels ) == 0:
                continue
            #
            # 2026-10-22: level frequencies pooled over the predictor and its
            # knockoff (col <-> col~), so T_j and T~_j use the same weights.
            partner = col[ :-1 ] if col.endswith( '~' ) else col + '~'
            pooled_vals = pd.concat( ( X_all_pd[ col ].astype( object ), X_all_pd[ partner ].astype( object ) ) )
            pi = pooled_vals.value_counts( normalize=True ).reindex( list( levels ), fill_value=0.0 ).to_numpy( dtype=float )
            pi = pi / pi.sum()

            if ot == 'categorical':
                logodds_per_level: np.ndarray = np.zeros( ( n, VI.shape[0], len( levels ) ) )
                for ki, lev in enumerate( levels ):
                    X_test = X_all_pd.copy()
                    X_test[ col ] = pd.Categorical( [ lev ] * n, dtype = X_all_pd[ col ].dtype )
                    log_p = np.log( _unzero_normalize( model.predict_proba( X_test ) ) )
                    logodds_per_level[ :, :, ki ] = log_p[ :, 1: ] - log_p[ :, 0:1 ]
                #
                # 2026-10-20: per-coordinate range replaced by the Omega-diameter
                # over pairs of levels (invariant to the reference class and to
                # level labels; reduces to range/sd for a binary response).
                if False:
                    contrasts = logodds_per_level.max( axis=2 ) - logodds_per_level.min( axis=2 )
                    importances_pointwise[ :, j ] = np.sqrt( np.einsum( 'ij,jk,ik->i', contrasts, VI, contrasts ) )
                #
                # 2026-10-22: Omega-diameter replaced by the frequency-weighted
                # Omega-SD sqrt( sum_k pi_k ||g_k - gbar||_Omega^2 ), matching the
                # torch networks' _localGrad_forCategories_t.
                if False:
                    diameter = np.zeros( n )
                    for ka in range( len( levels ) ):
                        for kb in range( ka + 1, len( levels ) ):
                            diameter = np.maximum(
                                diameter,
                                _mahalanobis_norm( logodds_per_level[ :, :, ka ] - logodds_per_level[ :, :, kb ], VI ),
                            )
                        #
                    #
                    importances_pointwise[ :, j ] = diameter
                #
                dev = logodds_per_level - ( logodds_per_level * pi ).sum( axis=2, keepdims=True )
                sq = np.einsum( 'nak,ab,nbk->nk', dev, VI, dev )
                importances_pointwise[ :, j ] = np.sqrt( np.clip( ( sq * pi ).sum( axis=1 ), 0.0, None ) )
            #
            else:
                preds_mat: np.ndarray = np.zeros( ( n, len( levels ) ) )
                for ki, lev in enumerate( levels ):
                    X_test = X_all_pd.copy()
                    X_test[ col ] = pd.Categorical( [ lev ] * n, dtype = X_all_pd[ col ].dtype )
                    preds = model.predict( X_test )
                    preds_mat[ :, ki ] = np.log( _unzero( preds ) ) if ot == 'count' else preds
                #
                # 2026-10-22: prediction range replaced by the frequency-weighted
                # SD of the level predictions (same scale as the per-SD
                # continuous score; see _localGrad_forCategories_t).
                if False:
                    importances_pointwise[ :, j ] = preds_mat.max( axis=1 ) - preds_mat.min( axis=1 )
                #
                dev = preds_mat - ( preds_mat @ pi )[ :, None ]
                importances_pointwise[ :, j ] = np.sqrt( np.clip( ( dev ** 2 ) @ pi, 0.0, None ) )
            #/if ot == 'categorical'/else
        #
        else:
            # 2026-10-20: forward difference in raw units replaced by a central
            # difference on the standardized scale (per-SD derivative), matching
            # the torch PRISM networks.
            if False:
                col_std = X_all_pd[ col ].std()
                bw = float( col_std ) * bandwidth / n_factor
                if bw == 0:
                    continue
                #

                X_test = X_all_pd.copy()
                X_test[ col ] = X_all_pd[ col ] + bw

                if ot == 'categorical':
                    log_p = np.log( _unzero_normalize( model.predict_proba( X_test ) ) )
                    grad_mat = ( log_p - base_log ) / bw
                    contrasts = grad_mat[ :, 1: ] - grad_mat[ :, 0:1 ]
                    importances_pointwise[ :, j ] = np.sqrt( np.einsum( 'ij,jk,ik->i', contrasts, VI, contrasts ) )
                #
                elif ot == 'count':
                    mod_log = np.log( _unzero( model.predict( X_test ) ) )
                    importances_pointwise[ :, j ] = ( mod_log - base_preds ) / bw
                #
                else:
                    mod_preds = model.predict( X_test )
                    importances_pointwise[ :, j ] = ( mod_preds - base_preds ) / bw
            #
            col_std = float( X_all_pd[ col ].std() )
            if col_std == 0 or h_std == 0:
                continue
            #
            step = col_std * h_std

            X_plus = X_all_pd.copy()
            X_plus[ col ] = X_all_pd[ col ] + step
            X_minus = X_all_pd.copy()
            X_minus[ col ] = X_all_pd[ col ] - step

            if ot == 'categorical':
                log_p_plus  = np.log( _unzero_normalize( model.predict_proba( X_plus ) ) )
                log_p_minus = np.log( _unzero_normalize( model.predict_proba( X_minus ) ) )
                grad_mat = ( log_p_plus - log_p_minus ) / ( 2.0 * h_std )
                contrasts = grad_mat[ :, 1: ] - grad_mat[ :, 0:1 ]
                importances_pointwise[ :, j ] = _mahalanobis_norm( contrasts, VI )
            #
            elif ot == 'count':
                importances_pointwise[ :, j ] = (
                    np.log( _unzero( model.predict( X_plus ) ) ) - np.log( _unzero( model.predict( X_minus ) ) )
                ) / ( 2.0 * h_std )
            #
            else:
                importances_pointwise[ :, j ] = (
                    model.predict( X_plus ) - model.predict( X_minus )
                ) / ( 2.0 * h_std )
            #/if ot == 'categorical'/elif 'count'/else
        #/if is_categorical/else
    #/for j, col

    if ot == 'categorical':
        return np.mean( importances_pointwise ** exponent, axis=0 )
    #
    return np.mean( np.abs( importances_pointwise ) ** exponent, axis=0 )
#/def prism_importances


def shap_importances(
    X: DataFrameLike,
    Xk: DataFrameLike,
    y: SeriesOrDataFrameLike,
    outcome_type: Literal['continuous','count','categorical',] | None = None,
    verbose: int = 0,
    model_kwargs: dict | None = None,
    rng: np.random.Generator | None = None,
    weight: np.ndarray | None = None,
    **fit_kwargs,
    ) -> np.ndarray:
    """
        SHAP contribution importances using xgboost.TreeExplainer.

        Requires the optional `shap` dependency: pip install heteroknockoffpy[shap]

        :param model_kwargs: Forwarded to the XGBRegressor/XGBClassifier constructor
            (e.g. max_depth, n_estimators, learning_rate, subsample, reg_alpha, ...).
        :param rng: If given, seeds the XGBoost fit (random_state=rng) for
            reproducibility -- xgboost's sklearn API accepts a np.random.Generator
            directly. Unseeded (xgboost's own default) if omitted.
        :param weight: Optional length-n sample weight. None (default) fits
            unweighted, xgboost's own default.
        :param fit_kwargs: Forwarded to XGBRegressor/XGBClassifier.fit.
    """
    import shap

    model, X_all, X_all_pd, _ = _fit_model( X, Xk, y, outcome_type, verbose, model_kwargs=model_kwargs, rng=rng, weight=weight, **fit_kwargs )

    explainer = shap.TreeExplainer( model )
    shap_values = explainer.shap_values( X_all_pd )

    shap_arr: np.ndarray = np.asarray( shap_values )
    if shap_arr.ndim == 3:
        # (n, p, k) or (k, n, p) depending on shap version -- normalize to (n, p, k)
        if shap_arr.shape[1] == X_all.shape[1]:
            pass
        else:
            shap_arr = np.moveaxis( shap_arr, 0, -1 )
        #
        return np.abs( shap_arr ).mean( axis=( 0, 2 ) )
    #

    return np.abs( shap_arr ).mean( axis=0 )
#/def shap_importances
