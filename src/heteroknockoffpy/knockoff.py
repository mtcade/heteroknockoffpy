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
        :param categorical_method:
            - 'forest': Get logit probabilities with random forests
            - 'linear': Get logit probabilities with logistic regression
            - 'ohe': One hot encode as a float
            - 'ranger_scip': With conditional residuals for numeric data, use ranger forest SCIP for categorical
            - 'xgb_scip': Same as 'ranger_scip', but conditional expectations/categorical SCIP are computed with sequential xgboost models instead of R ranger
        :param knockoffCallable: Closure to convert either conditional residuals or one-hot-encoded data to knockoffs of the same format. Make sure it has the desired parameters based on whether you are using ranger_scip/xgb_scip, or another method
        :param conditional_expectations: Numeric conditional expectations. If not provided, uses `rbridge.get_forest_conditional_expectations` (for `categorical_method='ranger_scip'`) or `xgbScip.get_forest_conditional_expectations` (for `categorical_method='xgb_scip'`) to calculate
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
        Second order method from rbridge, using r knockoff
        
        :param conditional_expectations: Numeric conditional expectations. If not provided, uses `rbridge.get_forest_conditional_expectations`/`xgbScip.get_forest_conditional_expectations` to calculate, if `categorical_method` in ('ranger_scip', 'xgb_scip')
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
            #*kwargs
        )
    #/def knockoffCallable
    
    # TODO: #**kwargs
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

        :param SCIP_method: Passed to `rangerKnockoff::create.forest.SCIP` as `method` parameter
        :param kwargs:
            - `rangerKnockoff::create.forest.SCIP` for creating numeric knockoffs
            - Others: Passed to `ranger::ranger`
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
        require R/rpy2.

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
