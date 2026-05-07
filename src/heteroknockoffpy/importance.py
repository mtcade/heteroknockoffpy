"""
    Interface to calculate the MALD Importance and W Statistics given a `PredictionModel`, either your own or created through ``maldImportance.superBasicNetworks``; see :doc:`superBasicNetworks`.
    
    If you do wish to use a ``maldImportance.superBasicNetworks.SimpleNN`` then you can use the ``maldImportance.nnImportance`` (:doc:`nnImportance`) module for simplicity.
"""
from . import utilities
from .utilities import OutcomeDescriptor

import numpy as np
import polars as pl

from typing import Callable, Literal, Protocol, Self, Sequence
from dataclasses import dataclass

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
    bandwidth: float,
    inv_cov: np.ndarray | None = None,
    drop_first_y: bool = True,
    ) -> np.ndarray:
    """
        Get the bandwidth local gradient approximation for variable j
        :param X: all data
        :param y_hat: The base prediction, result of `model.predict(X)`
        :param model: PredictionModel already fit and trained
        :param bandwidth: Exact literal value
        :param inv_cov: Information of y_hat, used to calculate Mahalanobis distance. Used for categorical data and for joint outcomes.
        :param drop_first_y: If we should subtract the first column of predictions from the others; use this for categorical outcomes, but not other joint outcomes
    """
    # Symmetric finite difference: estimate at X - h and X + h
    X_minus: np.ndarray = np.copy( X )
    X_minus[:, j ] -= bandwidth
    X_plus: np.ndarray = np.copy( X )
    X_plus[:, j ] += bandwidth

    # Central difference approximation of local gradient
    local_grad: np.ndarray = ( model.predict(X_plus) - model.predict(X_minus) ) / ( 2 * bandwidth )
    if inv_cov is None:
        assert y_hat.shape[1] == 1
        return local_grad.reshape( -1 )
    #
    else:
        if drop_first_y:
            assert y_hat.shape[1] == inv_cov.shape[0] + 1
            local_grad = local_grad[:,1:] - local_grad[:,0:1]
        #
        else:
            assert y_hat.shape[1] == inv_cov.shape[0]
        #/if drop_first/else

        # local_grad: (n, k-1); mahalanobis distance → (n,)
        return np.einsum(
            'nk,kl,nl->n',
            local_grad,
            inv_cov,
            local_grad,
        )
    #/if inv_cov is None/else
    # EARLY RETURN/
#/def _localGrad_forNumeric

def _localGrad_forCategories(
    j: list[ int ],
    X: np.ndarray,
    model: PredictionModel,
    drop_first: bool,
    inv_cov: np.ndarray | None = None,
    drop_first_y: bool = True,
    ) -> np.ndarray:
    """
        Get the change in prediction for a group of category columns by changing each of the values to. 1, and the others to 0
        
        If drop_first, it gets compared with setting all to 0.
    """
    _X: np.ndarray = np.copy( X )
    
    X_categories: int = len( j )
    if drop_first: X_categories += 1
    
    output_dimension: int
    if inv_cov is not None:
        if drop_first_y:
            output_dimension = inv_cov.shape[0] + 1
        #
        else:
            output_dimension = inv_cov.shape[0]
        #
    #
    else:
        output_dimension = 1
    #/if inv_cov is not None/else
    
    y_out: np.ndarray = np.zeros( shape = (X.shape[0], output_dimension, X_categories ) )
    
    # Get the predicted values at each test category
    for h in range( len(j) ):
        # Reset all to 0, set one to 1
        _X[ :, j ] = 0
        _X[ :, j[h] ] = 1
        
        y_out[ :, :, h ] = model.predict( _X )
    #
    
    # Use last index setting all to 0
    if drop_first:
        _X[ :, j ] = 0
        y_out[ :, :, -1 ] = model.predict( _X )
    #
    
    local_grad: np.ndarray = np.max( y_out, axis = 2 ) - np.min( y_out, axis = 2 )
    # local_grad: (n, output_dimension)
    if inv_cov is None:
        return local_grad.reshape( -1 )
    #
    else:
        if drop_first_y:
            local_grad = local_grad[:, 1:] - local_grad[:, 0:1]
        #/if drop_first_y
        # local_grad: (n, k-1); mahalanobis distance → (n,)
        return np.einsum(
            'nk,kl,nl->n',
            local_grad,
            inv_cov,
            local_grad,
        )
    #/if inv_cov is None/else
    
    # EARLY RETURN/
#/def _localGrad_forCategories

def maldImportancesFromModel(
    model: PredictionModel,
    X: np.ndarray | pl.DataFrame,
    Xk: np.ndarray | pl.DataFrame,
    y: np.ndarray | pl.DataFrame | pl.Series,
    outcome_type: Literal['continuous','count','categorical',] | None = None,
    local_grad_method: Literal['auto_diff','bandwidth'] = 'auto_diff',
    fit: bool = True,
    bandwidth: float | None = None,
    exponent: float = 1.0,
    drop_first: bool = True,
    verbose: int = 0,
    ) -> np.ndarray:
    """
        :param model: Predictor which can use autodifferentiation
        :param X: Explanatory data
        :param Xk: Knockoff explanatory data
        :param y: Outcome data. It may be wider than 1, for a joint outcome.
        :param outcome_type: Type of `y` data; if None, gets inferred from the datatype of y:
        
            - int: Count
            - category: Categorical
            - float: continuous
            
            If you have a categorical outcome, do not one-hot encode it; a multi variate integer outcome will default to count data. Providing one hot encoded data while specifying 'categorical' will assume a multi variate, two category per outcome.
            
            Count data will use the change in log expected value, while categorical data will use the change in log odds ratio, with the first category as base line.
        :param local_grad_method: Method of MALD. Defaults to `'auto_diff'` for exact autodifferentiation. `'bandwidth'` uses the bandwidth approximation when auto differentiation is not available.
        :param fit: Whether to fit `model`. Set to `False` if you have already trained it to your satisfaction on the combined explanatory and knockoff data together.
        :param bandwidth: Width if `local_grad_method = 'bandwidth'`
        :param exponent: Power to take of each MALD value. 1.0 and 2.0 both work reasonably well.
        :param drop_first: How to handle one-hot-encoding categorical variables. If `True` the number of associated columns is the number of categories minus 1.
        :param verbose: How much to print out, for mostly for debugging.
        
        :returns: Array of importances, with length equal to twice the width of `X`
        
        Takes an initialized PredictionModel, likely fits it, and gets the MALD importances for each `X` and `Xk` variable. Throws a warning if importances are 0.
        
    """
    import warnings
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
    y_np = y_np.reshape( X_all_np.shape[0], -1 )
    if y_np.shape[1] == 1:
        y_np = y_np[:, 0]
    
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
    if np.allclose( importances, 0 ):
        warnings.warn(
            "Got zeros importances",
            UserWarning,
        )
    #/if np.allclose( importances, 0 )
    
    return importances
#/def importancesFromModel

def torchMaldImportances(
    X: pl.DataFrame,
    Xk: pl.DataFrame,
    y: pl.Series | pl.DataFrame,
    layers: Sequence[ int ],
    outcome_type: Literal['continuous','count','categorical',] | None = None,
    learning_rate: float = 0.01,
    epochs: int = 500,
    local_grad_method: Literal['auto_diff','bandwidth'] = 'auto_diff',
    bandwidth: float | None = None,
    exponent: float = 1.0,
    drop_first: bool = True,
    dense_activation: str = 'relu',
    verbose: int = 0,
    ) -> np.ndarray:
    """
        Throws a warning if getting zeroes importances
    """
    from . import torchNetworks
    import torch
    import torch.nn as nn
    from torch import Tensor, tensor
    
    import warnings
    
    outcomeDescriptor: OutcomeDescriptor = OutcomeDescriptor.infer(
        y = y,
        outcome_type = outcome_type,
    )
    
    if outcomeDescriptor.outcome_dimension != 'single':
        raise TypeError("Joint outcomes unavailable")
    #
    
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
    
    # -- Make model
    if outcomeDescriptor.outcome_type == 'categorical':
        _y_series: pl.Series = y if isinstance( y, pl.Series ) else y.to_series()
        predictionModel: torchNetworks.PredictionModel_Numeric = torchNetworks.PredictionModel_Numeric(
            input_size = X_all_np.shape[1],
            layers = layers,
            dense_activation = dense_activation,
            loss_func = nn.CrossEntropyLoss(),
            output_dimension = len( _y_series.cat.get_categories() ),
            learning_rate = learning_rate,
            epochs = epochs,
            verbose = verbose,
        )
        
    #
    else:
        if outcomeDescriptor.outcome_type == 'continuous':
            predictionModel: torchNetworks.PredictionModel_Numeric = torchNetworks.PredictionModel_Numeric(
                input_size = X_all_np.shape[1],
                layers = layers,
                dense_activation = dense_activation,
                loss_func = nn.MSELoss(),
                learning_rate = learning_rate,
                epochs = epochs,
                verbose = verbose,
            )
        #
        elif outcomeDescriptor.outcome_type == 'count':
            # Uses a log prediction of expected value, due to nn.PoissonNLLLoss( log_input=True)
            # Raw gradient will still work for numeric data
            predictionModel: torchNetworks.PredictionModel_Numeric = torchNetworks.PredictionModel_Numeric(
                input_size = X_all_np.shape[1],
                layers = layers,
                dense_activation = dense_activation,
                loss_func = nn.PoissonNLLLoss(
                    log_input=True
                ),
                learning_rate = learning_rate,
                epochs = epochs,
                verbose = verbose,
            )
            
            # Tensorflow cannot handle UInt32 so promote to Int64
            y = y.cast( pl.Int64 ) if isinstance( y, pl.Series ) else y.cast( { col: pl.Int64 for col in y.columns } )
        #
        else:
            raise ValueError(
                "Unrecognized outcomeDescriptor.outcome_type={}".format(outcomeDescriptor.outcome_type)
            )
        #/switch outcomeDescriptor.outcome_type
        return maldImportancesFromModel(
            model = predictionModel,
            X = X,
            Xk = Xk,
            y = y,
            outcome_type = outcome_type,
            local_grad_method = local_grad_method,
            fit = True,
            bandwidth = bandwidth,
            exponent = exponent,
            drop_first = drop_first,
            verbose = verbose,
        )
    #/if outcomeDescriptor.outcome_type == 'categorical'/else
    
    # outcomeDescriptor.outcome_type == 'categorical'
    # Integer class codes (long) for CrossEntropyLoss
    _y_series: pl.Series = y if isinstance( y, pl.Series ) else y.to_series()
    y_np_codes: np.ndarray = _y_series.to_physical().to_numpy().astype( np.int64 )  # (n,)

    predictionModel.fit(
        X = X_all_np,
        y = y_np_codes,
    )

    # -- Get the mald importance

    y_hat: np.ndarray = predictionModel.predict(
        X_all_np
    )
    # Inverse covariance of contrasted logits for Mahalanobis distance
    logit_contrasts: np.ndarray = y_hat[:,1:] - y_hat[:,0:1]  # (n, k-1)
    _cov = np.cov( logit_contrasts, rowvar = False )
    if _cov.ndim == 0:
        # k=2: cov is a scalar; wrap as (1, 1)
        inv_cov: np.ndarray = np.array( [[ 1.0 / float(_cov) ]] )
    else:
        inv_cov: np.ndarray = np.linalg.inv( _cov )
    #
    if local_grad_method == 'auto_diff':
        # Jacobian of logits: (n, k, p)
        auto_diff_tensor: Tensor = torchNetworks.get_logit_jacobian(
            model = predictionModel.model,
            x = torch.tensor( X_all_np ).float().to(
                predictionModel.device
            )
        )
        # Permute to (n, p, k), compute contrasts → (n, p, k-1)
        auto_diff_np: np.ndarray = auto_diff_tensor.permute( 0, 2, 1 ).detach().cpu().numpy()
        auto_diff_np = auto_diff_np[:,:,1:] - auto_diff_np[:,:,0:1]

        # Mahalanobis distance over outcome contrasts: (n, p)
        auto_diff_matrix: np.ndarray = np.einsum(
            'npk,kl,npl->np',
            auto_diff_np,
            inv_cov,
            auto_diff_np,
        )

        y_hat = None
    #
    elif local_grad_method == 'bandwidth':
        auto_diff_matrix = None

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
                    model = predictionModel,
                    bandwidth = bandwidth,
                    inv_cov = inv_cov,
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
                model = predictionModel,
                drop_first = drop_first,
                inv_cov = inv_cov,
            )
            localGrad_matrix[:,j] = column_grad
        #/if isinstance( oheDict[j], int )/else
        j += 1
    #/for j in range( p_out )
    
    importances: np.ndarray = np.mean(
        np.abs( localGrad_matrix )**exponent,
        axis = 0
    )
    
    # If you get all zero importances, perhaps something went wrong
    if np.allclose( importances, 0 ):
        warnings.warn(
            "Got zeros importances",
            UserWarning,
        )
    #/if np.allclose( importances, 0 )
    
    return importances
#/def torchMaldImportance

def rangerMaldImportances(
    X: pl.DataFrame,
    Xk: pl.DataFrame,
    y: pl.Series | pl.DataFrame,
    outcome_type: Literal['continuous','count','categorical',] | None = None,
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
    
    # TODO: update
    return rbridge.rangerMaldImportances(
        X = X,
        Xk = Xk,
        y = y,
        outcome_type = outcome_type,
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
    outcome_type: Literal['continuous','count','categorical',] | None = None,
    verbose: int = 0,
    **kwargs,
    ) -> np.ndarray:
    from . import rbridge
    
    # TODO: update
    return rbridge.rangerGiniImportances(
        X = X,
        Xk = Xk,
        y = y,
        outcome_type = outcome_type,
        verbose = verbose,
        **kwargs,
    )
#/def rangerGiniImportances

def lassoImportances(
    X: pl.DataFrame,
    Xk: pl.DataFrame,
    y: pl.Series | pl.DataFrame,
    outcome_type: Literal['continuous','count','categorical',] | None = None,
    fit_intercept: bool = True,
    exponent: float = 1.0,
    verbose: int = 0,
    **kwargs,
    ) -> np.ndarray:
    
    
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
            X = utilities.get_ohe_np(
                X = X_all_df,
                drop_first = True,
            ),
            y = y,
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
            X = utilities.get_ohe_np( X = X_all_df, drop_first = True ),
            y = y,
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
            X = utilities.get_ohe_np( X = X_all_df, drop_first = True ),
            y = y.to_numpy().ravel(),
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
                    utilities.get_ohe_np( X = X_all_df, drop_first = True )
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
            np.abs( lasso_coefficients[ oheDict[col] ] ) if isinstance( oheDict[col], int )\
                else max(
                    np.max( lasso_coefficients[ list( oheDict[col] ) ] ), 0,
                ) - min(
                    np.min( lasso_coefficients[ list( oheDict[col] ) ] ), 0,
                )\
                    for col in X_all_df.columns
                #/
        ),
        dtype = float,
    )**exponent
        
    return importances
#/def lassoImportances

def ridgeImportances(
    X: pl.DataFrame,
    Xk: pl.DataFrame,
    y: pl.Series | pl.DataFrame,
    outcome_type: Literal['continuous','count','categorical',] | None = None,
    fit_intercept: bool = True,
    exponent: float = 1.0,
    verbose: int = 0,
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
    """
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
        ridgeModel.fit( X = X_ohe, y = y )
        ridge_coefficients = ridgeModel.coef_.reshape( -1 )

    elif outcomeDescriptor.outcome_type == 'count':
        from sklearn.linear_model import PoissonRegressor
        from sklearn.model_selection import GridSearchCV
        alphas = kwargs.get( 'alphas', np.logspace( -4, 2, 10 ) )

        if verbose > 0:
            print("Fitting PoissonRegressor (L2) via GridSearchCV:")
            print("  max_iter={}".format( max_iter ))
            print("  n_splits={}".format( n_splits ))
            print("  alphas={}".format( alphas ))

        poissonModel = GridSearchCV(
            PoissonRegressor(
                fit_intercept = fit_intercept,
                max_iter = max_iter,
            ),
            param_grid = { 'alpha': alphas },
            cv = n_splits,
            scoring = 'neg_mean_poisson_deviance',
        )
        poissonModel.fit( X = X_ohe, y = y )
        ridge_coefficients = poissonModel.best_estimator_.coef_

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
            np.abs( ridge_coefficients[ oheDict[col] ] ) if isinstance( oheDict[col], int )\
                else max(
                    np.max( ridge_coefficients[ list( oheDict[col] ) ] ), 0,
                ) - min(
                    np.min( ridge_coefficients[ list( oheDict[col] ) ] ), 0,
                )\
                    for col in X_all_df.columns
            #/
        ),
        dtype = float,
    )**exponent

    return importances
#/def ridgeImportances

def elasticImportances(
    X: pl.DataFrame,
    Xk: pl.DataFrame,
    y: pl.Series | pl.DataFrame,
    outcome_type: Literal['continuous','count','categorical',] | None = None,
    l1_ratio: float = 0.5,
    fit_intercept: bool = True,
    exponent: float = 1.0,
    verbose: int = 0,
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
    """
    if l1_ratio == 1.0:
        return lassoImportances(
            X = X, Xk = Xk, y = y,
            outcome_type = outcome_type,
            fit_intercept = fit_intercept,
            exponent = exponent,
            verbose = verbose,
            **kwargs,
        )
    if l1_ratio == 0.0:
        return ridgeImportances(
            X = X, Xk = Xk, y = y,
            outcome_type = outcome_type,
            fit_intercept = fit_intercept,
            exponent = exponent,
            verbose = verbose,
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
        elasticModel.fit( X = X_ohe, y = y )
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
        glmModel.fit( X = X_ohe, y = y )
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
            np.abs( elastic_coefficients[ oheDict[col] ] ) if isinstance( oheDict[col], int )\
                else max(
                    np.max( elastic_coefficients[ list( oheDict[col] ) ] ), 0,
                ) - min(
                    np.min( elastic_coefficients[ list( oheDict[col] ) ] ), 0,
                )\
                    for col in X_all_df.columns
            #/
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

def wFromModel(
    model: PredictionModel,
    X: np.ndarray | pl.DataFrame,
    Xk: np.ndarray | pl.DataFrame,
    y: np.ndarray | pl.Series | pl.DataFrame,
    outcome_type: Literal['continuous','count','categorical',] | None = None,
    W_method: Literal['difference','signed_max'] = 'difference',
    local_grad_method: Literal['auto_diff','bandwidth'] = 'auto_diff',
    fit: bool = True,
    bandwidth: float | None = None,
    exponent: float = 2.0,
    drop_first: bool = True,
    verbose: int = 0
    ) -> np.ndarray:
    """
        :param model: Predictor which can use autodifferentiation
        :param X: Explanatory data
        :param Xk: Knockoff explanatory data
        :param y: Outcome data. It may be wider than 1, for a joint outcome.
        :param outcome_type: Type of `y` data; if None, gets inferred from the datatype of y:
        
            - int: Count
            - category: Categorical
            - float: continuous
            
            If you have a categorical outcome, do not one-hot encode it; a multi variate integer outcome will default to count data. Providing one hot encoded data while specifying 'categorical' will assume a multi variate, two category per outcome.
            
            Count data will use the change in log expected value, while categorical data will use the change in log odds ratio, with the first category as base line.
        :param W_method: How to calculate W statistics from importance measures, given the two most common methods.
        :param local_grad_method: Method of MALD. Defaults to `'auto_diff'` for exact autodifferentiation. `'bandwidth'` uses the bandwidth approximation when auto differentiation is not available.
        :param fit: Whether to fit `model`. Set to `False` if you have already trained it to your satisfaction on the combined explanatory and knockoff data together.
        :param bandwidth: Width if `local_grad_method = 'bandwidth'`
        :param exponent: Power to take of each MALD value. 1.0 and 2.0 both work reasonably well.
        :param drop_first: How to handle one-hot-encoding categorical variables. If `True` the number of associated columns is the number of categories minus 1.
        :param verbose: How much to print out, for mostly for debugging.
        
        :returns: W statistics, half the length of `importances`, the same length as the original number of variables
        
        A one step method of getting W statistics given a `PredictionModel`, wrapping ``importancesFromModel()`` and ``wFromImportances()``
    """
    importances: np.ndarray = importancesFromModel(
        model = model,
        X = X,
        Xk = Xk,
        y = y,
        outcome_type = outcome_type,
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
