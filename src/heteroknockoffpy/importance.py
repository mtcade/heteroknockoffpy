"""
    Interface to calculate the MALD Importance and W Statistics given a `PredictionModel`, either your own or created through ``maldImportance.superBasicNetworks``; see :doc:`superBasicNetworks`.
    
    If you do wish to use a ``maldImportance.superBasicNetworks.SimpleNN`` then you can use the ``maldImportance.nnImportance`` (:doc:`nnImportance`) module for simplicity.
"""
from . import utilities

import numpy as np
import polars as pl

from typing import Callable, Literal, Protocol, Self, Sequence

class PredictionModel( Protocol ):
    """
        Interface for prediction models, including ``hexathello.superBasicNetworks.SimpleNN``, as well as most tensorflow keras network predictors.
    """

    def fit( self: Self, X, y, **kwargs ) -> None:
        """
            :param X: Explanatory data, likely concatenated with knockoffs
            :param y: Outcome data
            :param kwargs: Other arguments
        """
        ...
    #

    def predict( self: Self, X ) -> np.ndarray:
        """
            :param X: Explanatory data, likely concatenated with knockoffs
            :returns: Predictions, one for each row of `X`
            :rtype: np.ndarray
        """
        ...
    #
    
    def auto_diff( self: Self, X: np.ndarray, ) -> np.ndarray:
        """
            :param X: Explanatory data, likely concatenated with knockoffs
            
            :returns: Matrix of gradients, same dimension as X
        """
        ...
    #
#/class PredictionModel( Protocol )

def auto_diff(
    model: PredictionModel,
    X: np.ndarray
    ) -> np.ndarray:
    """
        :param PredictionModel model: Predictor which can use autodifferentiation
        :param np.ndarray X: Explanatory data, likely including the knockoffs
        :returns: Array of partial derivatives for each variable at each data point. Same dimensions as `X`
        :rtype: np.ndarray
        
        Uses the auto differentiating capabilities of our `model` to get the exact MALD values.
    """
    return model.auto_diff( X )
#/def auto_diff

def _localGrad_forNumeric(
    j: int,
    X: np.ndarray,
    y_hat: np.ndarray,
    model: PredictionModel,
    bandwidth: float
    ) -> np.ndarray:
    """
        Get the bandwidth local gradient approximation for variable j
        X: all data
        y_hat: The base prediction, result of `model.predict(X)`
        model: PredictionModel already fit and trained
        bandwidth: Exact literal value
    """
    # Set the X + bandwidth matrix
    X_epsilon: np.ndarray = np.copy( X )
    X_epsilon[:, j ] += bandwidth
    
    # Get the approximation of local gradient via the definition of the derivative
    return ( model.predict(X_epsilon) - y_hat )/bandwidth
#/def _localGrad_forNumeric

def _localGrad_forCategories(
    j: list[ int ],
    X: np.ndarray,
    model: PredictionModel,
    drop_first: bool
    ) -> np.ndarray:
    """
        Get the change in prediction for a group of category columns by changing each of the values to. 1, and the others to 0
        
        If drop_first, it gets compared with setting all to 0.
    """
    _X: np.ndarray = np.copy( X )
    
    y_out_dim: int = len( j )
    if drop_first: y_out_dim += 1
    
    y_out: np.ndarray = np.zeros( shape = (X.shape[0], y_out_dim ) )
    
    # Get the predicted values at each test category
    for h in range( len(j) ):
        # Reset all to 0, set one to 1
        _X[ :, j ] = 0
        _X[ :, j[h] ] = 1
        
        y_out[ :, h ] = model.predict( _X )
    #
    
    # Use last index setting all to 0
    if drop_first:
        _X[ :, j ] = 0
        y_out[ :, -1 ] = model.predict( _X )
    #
    
    return np.max( y_out, axis = 1 ) - np.min( y_out, axis = 1 )
#/def _localGrad_forCategories

def maldImportancesFromModel(
    model: PredictionModel,
    X: np.ndarray | pl.DataFrame,
    Xk: np.ndarray | pl.DataFrame,
    y: np.ndarray | pl.DataFrame | pl.Series,
    local_grad_method: Literal['auto_diff','bandwidth'] = 'auto_diff',
    fit: bool = True,
    bandwidth: float | None = None,
    exponent: float = 1.0,
    drop_first: bool = True,
    verbose: int = 0,
    ) -> np.ndarray:
    """
        :param PredictionModel model: Predictor which can use autodifferentiation
        :param np.ndarray|pl.DataFrame X: Explanatory data
        :param np.ndarray|pl.DataFrame Xk: Knockoff explanatory data
        :param np.ndarray|pl.Series|pl.DataFrame y: Outcome data
        :param Literal['auto_diff','bandwidth'] local_grad_method: Method of MALD. Defaults to `'auto_diff'` for exact autodifferentiation. `'bandwidth'` uses the bandwidth approximation when auto differentiation is not available.
        :param bool fit: Whether to fit `model`. Set to `False` if you have already trained it to your satisfaction on the combined explanatory and knockoff data together.
        :param float|None bandwidth: Width if `local_grad_method = 'bandwidth'`
        :param float exponent: Power to take of each MALD value. 1.0 and 2.0 both work reasonably well.
        :param bool drop_first: How to handle one-hot-encoding categorical variables. If `True` the number of associated columns is the number of categories minus 1.
        :param int verbose: How much to print out, for mostly for debugging.
        :returns: Array of importances, with length equal to twice the width of `X`
        :rtype: np.ndarray
        
        Takes an initialized PredictionModel, likely fits it, and gets the MALD importances for each `X` and `Xk` variable
        
    """
    assert X.shape == Xk.shape
    
    # Check for categorical variables, one hot encode variables if necessary
    
    X_all: pl.DataFrame = pl.concat(
        (
            X,
            Xk.rename({
                col: col + '~' for col in Xk.columns
            }),
        ),
        how = 'horizontal',
    )
    
    oheDict: dict[ str, int | tuple[ int,... ] ] = utilities.get_oheDict(
        X_all,
        drop_first = drop_first,
    )
    
    X_all_np: np.ndarray
    if isinstance( X, pl.DataFrame ):
        
        assert all( X.schema[col] == Xk.schema[col] for col in X.columns )
        
        X_all_np = utilities.get_ohe_np(
            X = X_all,
            drop_first = drop_first,
        )
    #/if isinstance( X, pl.DataFrame )
    else:
        X_all_np = np.concatenate(
            ( X, Xk ),
            axis = 1
        )
    #/if isinstance( X, pl.DataFrame )/else
    
    y_np: np.ndarray
    if isinstance( y, pl.Series | pl.DataFrame ):
        y_np = y.to_numpy()
    #
    else:
        y_np = y
    #/if isinstance( y, pl.Series )/else
    y_np = np.reshape( y_np, ( X_all_np.shape[0], ) )
    
    # Fit if necessary; we can have a model already trained
    #   by setting to False
    if fit:
        model.fit( X_all_np, y_np, )
    #/if fit
    
    auto_diff_matrix: np.ndarray | None
    
    y_hat: np.ndarray | None
    if local_grad_method == 'auto_diff':
        auto_diff_matrix: np.ndarray = auto_diff(
            model,
            X_all_np
        )
        y_hat = None
    #
    elif local_grad_method == 'bandwidth':
        auto_diff_matrix = None
        y_hat = model.predict( X_all_np )
        
        if bandwidth is None:
            bandwidth = X_all_np.shape[0]**(-0.2)
        #/if bandwidth is None
    #
    else:
        raise ValueError("Unrecognized local_grad_method='{}'".format(local_grad_method))
    #/switch local_grad_method

    p_out: int = X.shape[1] + Xk.shape[1]
    localGrad_matrix: np.ndarray = np.zeros(
        shape = ( X.shape[0], p_out )
    )
    j: int = 0
    for col in tuple( oheDict ):
        if isinstance( oheDict[col], int ):
            # numeric
            if local_grad_method == 'auto_diff':
                column_grad = auto_diff_matrix[:, oheDict[col] ]
                localGrad_matrix[:,j] = column_grad
            #
            elif local_grad_method == 'bandwidth':
                column_grad = _localGrad_forNumeric(
                    j = oheDict[col],
                    X = X_all_np,
                    y_hat = y_hat,
                    model = model,
                    bandwidth = bandwidth
                )
                
                localGrad_matrix[ :, j ] = column_grad
            #
            else:
                raise ValueError("Unrecognized local_grad_method='{}'".format(local_grad_method))
            #/switch local_grad_method
        #
        else:
            # category
            column_grad = _localGrad_forCategories(
                j = oheDict[ col ],
                X = X_all_np,
                model = model,
                drop_first = drop_first
            )
            localGrad_matrix[:,j] = column_grad
        #/if isinstance( oheDict[j], int )/else
        j += 1
    #/for j in range( p_out )
    
    importances: np.ndarray = np.mean(
        np.abs( localGrad_matrix )**exponent,
        axis = 0
    )
    
    # If you get all zero importances, likely something went wrong
    if np.allclose(
        importances, 0
    ):
        raise Exception("Got zeros importances")
    #
    
    return importances
#/def importancesFromModel

def torchMaldImportances(
    X: pl.DataFrame,
    Xk: pl.DataFrame,
    y: pl.Series | pl.DataFrame,
    layers: Sequence[ int ],
    learning_rate: float = 0.01,
    epochs: int = 500,
    local_grad_method: Literal['auto_diff','bandwidth'] = 'auto_diff',
    bandwidth: float | None = None,
    exponent: float = 1.0,
    drop_first: bool = True,
    dense_activation: str = 'relu',
    verbose: int = 0,
    ) -> np.ndarray:
    
    from . import torchNetworks
    
    X_all: pl.DataFrame = pl.concat(
        (
            X,
            Xk.rename({
                col: col + '~' for col in Xk.columns
            }),
        ),
        how = 'horizontal',
    )
    
    X_all_np: np.ndarray
    oheDict: dict[ str, int | tuple[ int,... ] ]
    if isinstance( X, pl.DataFrame ):
        from . import utilities
        assert all( X.schema[col] == Xk.schema[col] for col in X.columns )
        
        X_all_np = utilities.get_ohe_np(
            X = X_all,
            drop_first = drop_first,
        )
        
        oheDict = utilities.get_oheDict(
            X = X_all,
            drop_first = drop_first,
        )
    #/if isinstance( X, pl.DataFrame )
    else:
        X_all_np = np.concatenate(
            ( X, Xk ),
            axis = 1
        )
    #/if isinstance( X, pl.DataFrame )/else
    
    y_np: np.ndarray
    if isinstance( y, pl.Series | pl.DataFrame ):
        y_np = y.to_numpy()
    #
    else:
        y_np = y
    #/if isinstance( y, pl.Series )/else
    y_np = np.reshape(
        y_np, ( X_all_np.shape[0], )
    )
    
    # -- Make model
    predictionModel_numeric: torchNetworks.PredictionModel_Numeric = torchNetworks.PredictionModel_Numeric(
        input_size = X_all_np.shape[1],
        layers = layers,
        dense_activation = dense_activation,
        learning_rate = learning_rate,
        epochs = epochs,
        verbose = verbose,
    )
    
    predictionModel_numeric.fit(
        X = X_all_np,
        y = y_np,
    )
    
    # -- Get the mald importance
    auto_diff_matrix: np.ndarray | None
    y_hat: np.ndarray | None
    
    if local_grad_method == 'auto_diff':
        auto_diff_matrix: np.ndarray = auto_diff(
            predictionModel_numeric,
            X_all_np
        )
        y_hat = None
    #
    elif local_grad_method == 'bandwidth':
        auto_diff_matrix = None
        y_hat = predictionModel_numeric.predict( X_all_np )
        
        if bandwidth is None:
            bandwidth = X_all_np.shape[0]**(-0.2)
        #/if bandwidth is None
    #
    else:
        raise ValueError("Unrecognized local_grad_method='{}'".format(local_grad_method))
    #/switch local_grad_method
    
    p_out: int = X.shape[1] + Xk.shape[1]
    localGrad_matrix: np.ndarray = np.zeros(
        shape = ( X.shape[0], p_out )
    )
    j: int = 0
    for col in tuple( oheDict ):
        if isinstance( oheDict[col], int ):
            # numeric
            if local_grad_method == 'auto_diff':
                column_grad = auto_diff_matrix[:, oheDict[col] ]
                localGrad_matrix[:,j] = column_grad
            #
            elif local_grad_method == 'bandwidth':
                column_grad = _localGrad_forNumeric(
                    j = oheDict[col],
                    X = X_all_np,
                    y_hat = y_hat,
                    model = predictionModel_numeric,
                    bandwidth = bandwidth
                )
                
                localGrad_matrix[ :, j ] = column_grad
            #
            else:
                raise ValueError("Unrecognized local_grad_method='{}'".format(local_grad_method))
            #/switch local_grad_method
        #
        else:
            # category
            column_grad = _localGrad_forCategories(
                j = oheDict[ col ],
                X = X_all_np,
                model = predictionModel_numeric,
                drop_first = drop_first
            )
            localGrad_matrix[:,j] = column_grad
        #/if isinstance( oheDict[j], int )/else
        j += 1
    #/for j in range( p_out )
    
    importances: np.ndarray = np.mean(
        np.abs( localGrad_matrix )**exponent,
        axis = 0
    )
    
    # If you get all zero importances, likely something went wrong
    if np.allclose(
        importances, 0
    ):
        raise Exception("Got zeros importances")
    #
    
    return importances
#/def torchMaldImportance

def rangerMaldImportances(
    X: pl.DataFrame,
    Xk: pl.DataFrame,
    y: pl.Series | pl.DataFrame,
    bandwidth: float = 1.0,
    bandwidth_exponent: float = 0.2,
    exponent: float = 1.0,
    verbose: int = 0,
    **kwargs,
    ) -> np.ndarray:
    """
        Uses a ranger random forest from R, with bandwidth mald importance
    """
    from . import rbridge
    
    return rbridge.rangerMaldImportances(
        X = X,
        Xk = Xk,
        y = y,
        bandwidth = bandwidth,
        bandwidth_exponent = bandwidth_exponent,
        exponent = exponent,
        verbose = verbose,
        **kwargs,
    )
#/def rangerMaldImportances

def rangerGiniImportances(
    X: pl.DataFrame,
    Xk: pl.DataFrame,
    y: pl.Series | pl.DataFrame,
    verbose: int = 0,
    **kwargs,
    ) -> np.ndarray:
    from . import rbridge
    
    return rbridge.rangerGiniImportances(
        X = X,
        Xk = Xk,
        y = y,
        verbose = verbose,
        **kwargs,
    )
#/def rangerGiniImportances

def lassoImportances(
    X: pl.DataFrame,
    Xk: pl.DataFrame,
    y: pl.Series | pl.DataFrame,
    fit_intercept: bool = True,
    exponent: float = 1.0,
    verbose: int = 0,
    **kwargs,
    ) -> np.ndarray:
    from sklearn.linear_model import LassoCV
    
    max_iter: int = kwargs['max_iter'] if 'max_iter' in kwargs else 4000
    lassoModel: LassoCV = LassoCV(
        max_iter = max_iter,
        fit_intercept = fit_intercept,
    )
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

    if verbose > 0:
        print( 'Fitting LassoCV with max_iter={}'.format(max_iter))
    #

    lassoModel.fit(
        X = utilities.get_ohe_np(
            X = X_all_df,
            drop_first = True,
        ),
        y = y.to_numpy().reshape( (X.shape[0],) ),
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
    
    lasso_importances = lasso_coefficients.reshape( (p,) )

    # Collapse the ohe categories if needed
    importances = np.fromiter(
        (
            np.abs(lasso_importances[oheDict[col]]) if isinstance( oheDict[col], int )\
                else max(
                    np.max(
                        lasso_importances[list( oheDict[col] )]
                    ),
                    0,
                ) - min(
                    np.min(
                        lasso_importances[list( oheDict[col] )]
                    ),
                    0,
                )\
                    for col in X_all_df.columns
                #/
        ),
        dtype = float,
    )**exponent
    
    return importances
#/def lassoImportances

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

def wFromModel(
    model: PredictionModel,
    X: np.ndarray | pl.DataFrame,
    Xk: np.ndarray | pl.DataFrame,
    y: np.ndarray | pl.Series | pl.DataFrame,
    W_method: Literal['difference','signed_max'] = 'difference',
    local_grad_method: Literal['auto_diff','bandwidth'] = 'auto_diff',
    fit: bool = True,
    bandwidth: float | None = None,
    exponent: float = 2.0,
    drop_first: bool = True,
    verbose: int = 0
    ) -> np.ndarray:
    """
        :param PredictionModel model: Predictor which can use autodifferentiation
        :param np.ndarray|pl.DataFrame X: Explanatory data
        :param np.ndarray|pl.DataFrame Xk: Knockoff explanatory data
        :param np.ndarray|pl.Series|pl.DataFrame y: Outcome data
        :param Literal['difference','signed_max'] W_method: How to calculate W statistics from importance measures, given the two most common methods.
        :param Literal['auto_diff','bandwidth'] local_grad_method: Method of MALD. Defaults to `'auto_diff'` for exact autodifferentiation. `'bandwidth'` uses the bandwidth approximation when auto differentiation is not available.
        :param bool fit: Whether to fit `model`. Set to `False` if you have already trained it to your satisfaction on the combined explanatory and knockoff data together.
        :param float|None bandwidth: Width if `local_grad_method = 'bandwidth'`
        :param float exponent: Power to take of each MALD value. 1.0 and 2.0 both work reasonably well.
        :param bool drop_first: How to handle one-hot-encoding categorical variables. If `True` the number of associated columns is the number of categories minus 1.
        :param int verbose: How much to print out, for mostly for debugging.
        :returns: W statistics, half the length of `importances`, the same length as the original number of variables
        :rtype: np.ndarray
        
        A one step method of getting W statistics given a `PredictionModel`, wrapping ``importancesFromModel()`` and ``wFromImportances()``
    """
    importances: np.ndarray = importancesFromModel(
        model = model,
        X = X,
        Xk = Xk,
        y = y,
        local_grad_method = local_grad_method,
        fit = fit,
        bandwidth = bandwidth,
        exponent = exponent,
        drop_first = drop_first,
        verbose = verbose
    )
    
    return wFromImportances(
        importances = importances,
        W_method = W_method
    )
#/def wFromModel

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
    # Maximum threshold such that hat_fdr <= nominal level
    if np.any(hat_fdrs <= fdr):
        T = absW[inds[np.where(hat_fdrs <= fdr)[0].max()]]
        if T == 0:
            T = np.min(W[W > 0])
    else:
        T = np.inf
    return T
#/def selection_threshold
