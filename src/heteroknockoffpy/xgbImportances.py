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
        how = 'horizontal',
    )
#/def _concat_X_Xk


def _any_categorical(X_all: pl.DataFrame) -> bool:
    return any( dtype == pl.Categorical for dtype in X_all.dtypes )
#/def _any_categorical


def _make_model(
    outcome_type: Literal['continuous','count','categorical',],
    enable_categorical: bool,
    ) -> 'xgboost.XGBRegressor | xgboost.XGBClassifier':
    if outcome_type == 'continuous':
        return xgboost.XGBRegressor(
            objective = 'reg:squarederror',
            tree_method = 'hist',
            enable_categorical = enable_categorical,
        )
    #
    elif outcome_type == 'count':
        return xgboost.XGBRegressor(
            objective = 'count:poisson',
            tree_method = 'hist',
            enable_categorical = enable_categorical,
        )
    #
    elif outcome_type == 'categorical':
        # No explicit objective: the sklearn wrapper auto-selects
        # binary:logistic / multi:softprob (with num_class) from y's classes.
        return xgboost.XGBClassifier(
            tree_method = 'hist',
            enable_categorical = enable_categorical,
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
    **fit_kwargs,
    ) -> tuple[ 'xgboost.XGBRegressor | xgboost.XGBClassifier', pl.DataFrame, pd.DataFrame, OutcomeDescriptor ]:
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

    model = _make_model( outcomeDescriptor.outcome_type, enable_categorical )
    y_fit: np.ndarray = _resolve_y_for_fit( y, outcomeDescriptor.outcome_type )

    if verbose > 0:
        print( "Fitting {}:".format( type( model ).__name__ ) )
        print( "  enable_categorical={}".format( enable_categorical ) )
        print( "  outcome_type={}".format( outcomeDescriptor.outcome_type ) )
    #

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
    **fit_kwargs,
    ) -> np.ndarray:
    """
        Fits a single xgboost model on [X, Xk] and returns split-based importances
        from Booster.get_score(importance_type=importance_type), one per OHE-free
        column (numeric columns and native categorical columns alike).

        :param fit_kwargs: Forwarded to XGBRegressor/XGBClassifier.fit (e.g.
            sample_weight, eval_set, early_stopping_rounds).
    """
    model, X_all, X_all_pd, _ = _fit_model( X, Xk, y, outcome_type, verbose, **fit_kwargs )

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


def prism_importances(
    X: DataFrameLike,
    Xk: DataFrameLike,
    y: SeriesOrDataFrameLike,
    outcome_type: Literal['continuous','count','categorical',] | None = None,
    bandwidth: float = 1.0,
    bandwidth_exponent: float = 0.2,
    exponent: float = 1.0,
    verbose: int = 0,
    **fit_kwargs,
    ) -> np.ndarray:
    """
        PRISM local-gradient importance using a single xgboost model on [X, Xk],
        mirroring stat.forest.prism_{continuous,count,categorical}.R:

        Numeric columns   — bandwidth finite-difference of the prediction.
        Categorical columns — max-minus-min sweep across category levels.

        continuous: raw prediction. count: log(unzero(prediction)).
        categorical: log-odds contrasts (vs. first class) of predict_proba,
        reduced via the Mahalanobis norm using the inverse covariance of the
        base contrasts.

        :param fit_kwargs: Forwarded to XGBRegressor/XGBClassifier.fit.
    """
    model, X_all, X_all_pd, outcomeDescriptor = _fit_model( X, Xk, y, outcome_type, verbose, **fit_kwargs )
    ot = outcomeDescriptor.outcome_type

    n: int = X_all_pd.shape[0]
    p_all: int = X_all_pd.shape[1]
    n_factor: float = n ** bandwidth_exponent

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
        VI = np.linalg.inv( np.atleast_2d( np.cov( base_contrasts, rowvar=False ) ) )
    #
    elif ot == 'count':
        base_preds = np.log( _unzero( model.predict( X_all_pd ) ) )
    #
    else:
        base_preds = model.predict( X_all_pd )
    #/switch ot

    importances_pointwise: np.ndarray = np.zeros( ( n, p_all ) )

    for j, col in enumerate( X_all_pd.columns ):
        if verbose > 0 and (j + 1) % verbose == 0:
            print( "xgbPrismImportances: column {} / {}".format( j + 1, p_all ) )
        #

        is_categorical: bool = isinstance( X_all_pd[ col ].dtype, pd.CategoricalDtype )

        if is_categorical:
            levels = X_all_pd[ col ].cat.categories
            if len( levels ) == 0:
                continue
            #

            if ot == 'categorical':
                logodds_per_level: np.ndarray = np.zeros( ( n, VI.shape[0], len( levels ) ) )
                for ki, lev in enumerate( levels ):
                    X_test = X_all_pd.copy()
                    X_test[ col ] = pd.Categorical( [ lev ] * n, dtype = X_all_pd[ col ].dtype )
                    log_p = np.log( _unzero_normalize( model.predict_proba( X_test ) ) )
                    logodds_per_level[ :, :, ki ] = log_p[ :, 1: ] - log_p[ :, 0:1 ]
                #
                contrasts = logodds_per_level.max( axis=2 ) - logodds_per_level.min( axis=2 )
                importances_pointwise[ :, j ] = np.sqrt( np.einsum( 'ij,jk,ik->i', contrasts, VI, contrasts ) )
            #
            else:
                preds_mat: np.ndarray = np.zeros( ( n, len( levels ) ) )
                for ki, lev in enumerate( levels ):
                    X_test = X_all_pd.copy()
                    X_test[ col ] = pd.Categorical( [ lev ] * n, dtype = X_all_pd[ col ].dtype )
                    preds = model.predict( X_test )
                    preds_mat[ :, ki ] = np.log( _unzero( preds ) ) if ot == 'count' else preds
                #
                importances_pointwise[ :, j ] = preds_mat.max( axis=1 ) - preds_mat.min( axis=1 )
            #/if ot == 'categorical'/else
        #
        else:
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
    **fit_kwargs,
    ) -> np.ndarray:
    """
        SHAP contribution importances using xgboost.TreeExplainer.

        Requires the optional `shap` dependency: pip install heteroknockoffpy[shap]

        :param fit_kwargs: Forwarded to XGBRegressor/XGBClassifier.fit.
    """
    import shap

    model, X_all, X_all_pd, _ = _fit_model( X, Xk, y, outcome_type, verbose, **fit_kwargs )

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
