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

import polars as pl
import numpy as np

from typing import Callable, Literal

def get_withCallable(
    X: pl.DataFrame,
    rng: np.random.Generator,
    categorical_method: Literal['forest','linear','ohe','scip'],
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
            - 'scip': With conditional residuals for numeric data, use fores scip for categorical
        :param knockoffCallable: Closure to convert either conditional residuals or one-hot-encoded data to knockoffs of the same format. Make sure it has the desired parameters based on whether you are using scip, or another method
        :param conditional_expectations: Numeric conditional expectations. If not provided, uses `rbridge.get_forest_conditional_expectations` to calculate, if `categorical_method='scip'`
    """
    if categorical_method == 'scip':
        from . import rbridge
        
        # Conditional residuals knockoffs
        if conditional_expectations is None:
            conditional_expectations: pl.DataFrame = rbridge.get_forest_conditional_expectations(
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
        
        return rbridge.get_knockoffs_with_Xk_numeric(
            X = X,
            Xk_numeric = ce_np + Xk_residuals,
            rng = rng,
            verbose = verbose,
            verbose_prefix = verbose_prefix,
            #**kwargs,
        )
    #/if categorical_method == 'scip'
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
    X: pl.DataFrame,
    rng: np.random.Generator,
    categorical_method: Literal['forest','linear','ohe','scip',],
    conditional_expectations: pl.DataFrame | None = None,
    verbose: int = 0,
    verbose_prefix: str = '',
    **kwargs,
    ) -> pl.DataFrame:
    """
        Second order method from rbridge, using r knockoff
        
        :param conditional_expectations: Numeric conditional expectations. If not provided, uses `rbridge.get_forest_conditional_expectations` to calculate, if `categorical_method='scip'`
    """
    from . import rbridge
    
    knockoffCallable: Callable[ [np.ndarray], np.ndarray ] = lambda x:\
        rbridge.get_knockoffs_second_order_np(
            X = x,
            verbose = verbose,
            verbose_prefix = verbose_prefix,
            #*kwargs
        )
    #/
    
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

def get_GAN(
    X: pl.DataFrame,
    rng: np.random.Generator,
    categorical_method: Literal['forest','linear','ohe','scip',],
    conditional_expectations: pl.DataFrame | None = None,
    verbose: int = 0,
    verbose_prefix: str = '',
    **kwargs,
    ) -> pl.DataFrame:
    """
        Uses GAN for conditional residual or ohe knockoffs
        
        :param conditional_expectations: Numeric conditional expectations. If not provided, uses `rbridge.get_forest_conditional_expectations` to calculate, if `categorical_method='scip'`
    """
    from . import KnockoffGAN
    
    knockoffCallable: Callable[ [np.ndarray], np.ndarray ] = lambda x:\
        KnockoffGAN.KnockoffGAN(
            x_train = x,
            **{
                "x_name": 'Normal',
                "lamda": 1,
                "mu": 1,
                "mb_size": 128,
                "niter": 2000,
            } | {
                key: val\
                    for key, val in kwargs.items()\
                    if key in ("x_name","lamda","mu","mb_size","niter",)
                #/
            }
        )
    #/
    
    # TODO: #**kwargs
    return get_withCallable(
        X = X,
        rng = rng,
        categorical_method = categorical_method,
        knockoffCallable = knockoffCallable,
        conditional_expectations = conditional_expectations,
        verbose = verbose,
        verbose_prefix = verbose_prefix,
        **{
            key: val\
                for key, val in kwargs.items()\
                if key not in ("x_name","lamda","mu","mb_size","niter",)
            #/
        },
    )
#/def get_GAN

def get_torchGAN(
    X: pl.DataFrame,
    rng: np.random.Generator,
    categorical_method: Literal['forest','linear','ohe','scip',],
    conditional_expectations: pl.DataFrame | None = None,
    verbose: int = 0,
    verbose_prefix: str = '',
    **kwargs,
    ) -> pl.DataFrame:
    
    from . import torchGAN
    
    def knockoffCallable( x: np.ndarray ) -> np.ndarray:
        model = torchGAN.TorchGAN(
            shape          = x.shape,
            x_name         = kwargs.get( 'x_name',         'Normal' ),
            lamda          = kwargs.get( 'lamda',           1        ),
            mu             = kwargs.get( 'mu',              1        ),
            lam            = kwargs.get( 'lam',             10       ),
            lr             = kwargs.get( 'lr',              1e-4     ),
            mb_size        = kwargs.get( 'mb_size',         128      ),
            niter          = kwargs.get( 'niter',           2000     ),
            combined_inner = kwargs.get( 'combined_inner',  False    ),
        )
        model.fit( x )
        return model( x )
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

def get_SCIP(
    X: pl.DataFrame,
    rng: np.random.Generator,
    residuals_method: Literal['normal','permute',] = 'normal',
    verbose: int = 0,
    verbose_prefix: str = '',
    **kwargs,
    ) -> pl.DataFrame:
    """
        :param SCIP_method: Passed to `rangerKnockoff::create.forest.SCIP` as `method` parameter
        :param kwargs:
            - `rangerKnockoff::create.forest.SCIP` for creating numeric knockoffs
            - Others: Passed to `ranger::ranger`
    """
    from . import rbridge
    
    return rbridge.get_knockoffs_SCIP(
        X = X,
        rng = rng,
        residuals_method = residuals_method,
        verbose = verbose,
        verbose_prefix = verbose_prefix,
        **kwargs,
    )
#/def get_SCIP

def get_knockoffs(
    X: pl.DataFrame,
    method: Literal[
        "second_order",
        "GAN",
        "GAN_torch",
        "SCIP",
    ],
    rng: np.random.Generator,
    conditional_expectations: pl.DataFrame | None = None,
    verbose: int = 0,
    verbose_prefix: str = '',
    **kwargs,
    ) -> pl.DataFrame:
    """
        Interface to name the knockoff method by string
        
        :param conditional_expectations: Necessary if kwargs['categorical_method'] == "scip"
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
        if kwargs['categorical_method'] == 'scip':
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
    elif method == "GAN":
        # kwargs should have 'categorical_method'
        if kwargs['categorical_method'] == 'scip':
            assert set( numeric_columns ) == set( conditional_expectations.columns )
        #
        else:
            conditional_expectations = None
        #
        
        Xk = get_GAN(
            X = X,
            rng = rng,
            conditional_expectations = conditional_expectations,
            verbose = verbose,
            verbose_prefix = verbose_prefix,
            **kwargs,
        )
    #
    elif method == "GAN_torch":
        # kwargs should have 'categorical_method'
        if kwargs['categorical_method'] == 'scip':
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
    elif method == "SCIP":
        # kwargs may have 'residuals_method'
        assert kwargs['residuals_method'] in ("permute","normal",)
        Xk = get_SCIP(
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
