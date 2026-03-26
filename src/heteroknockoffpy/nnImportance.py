"""
    One step interface to use ``maldImportance.importance`` without a model in hand by using ``maldImportance.superBasicNetworks.SimpleNN``.
"""

import numpy as np
import polars as pl

from typing import Literal

def importances(
    X: np.ndarray | pl.DataFrame,
    Xk: np.ndarray | pl.DataFrame,
    y: np.ndarray | pl.Series | pl.DataFrame,
    local_grad_method: Literal['auto_diff','bandwidth'] = 'auto_diff',
    bandwidth: float | None = None,
    exponent: float = 2.0,
    drop_first: bool = True,
    # SuperBasicNetworks Parameters
    save_root: str = '',
    save_name: str = '',
    epochs: int = 500,
    dense_activation: str = 'relu', # 'relu', 'sigmoid',...
    # Hyper parameters
    first_layer_width: float | int = 0.25, # int = absolute size, float = proportion of input
    layers: int = 2,
    layer_shrink_factor: float = 0.25,
    learning_rate: float = 0.01,
    verbose: int = 0
    ) -> np.ndarray:
    """
        :param np.ndarray|pl.DataFrame X: Explanatory data
        :param np.ndarray|pl.DataFrame Xk: Knockoff explanatory data
        :param np.ndarray|pl.Series|pl.DataFrame y: Outcome data
        :param Literal['auto_diff','bandwidth'] local_grad_method: Method of MALD. Defaults to `'auto_diff'` for exact autodifferentiation. `'bandwidth'` uses the bandwidth approximation when auto differentiation is not available.
        :param float|None bandwidth: Width if `local_grad_method = 'bandwidth'`
        :param float exponent: Power to take of each MALD value. 1.0 and 2.0 both work reasonably well.
        :param bool drop_first: How to handle one-hot-encoding categorical variables. If `True` the number of associated columns is the number of categories minus 1.
        :param str save_root: Dir to make saves
        :param str save_name: Name to use for saving checkpoints
        :param int epochs: Epochs for `.fit()`
        :param str dense_activation: Activation of internal layers, most likely 'relu,' 'sigmoid', 'leaky_relu', etc
        :param float|int first_layer_width: Size of first layer after input. If an integer, it uses that value. If a float, it's a multiplier of the input size.
        :param int layers: Number of internal layers
        :param float layer_shrink_factor: With multiple internal layers, the width of each is this multiplied by the previous layer's width
        :param float learning_rate: Learning rate for `.fit()`; common values are 0.01, 0.005, 0.001, 0.0005, 0.0001
        :param int verbose: How much to print out, for mostly for debugging.
        :returns: Array of importances, with length equal to twice the width of `X`
        :rtype: np.ndarray
        
        Creates a SuperBasicNetwork and uses it for MALD Importance. See :doc:`importance` and :doc:`superBasicNetworks`
    """
    from . import superBasicNetworks
    from . import importance
    
    network = superBasicNetworks.SimpleNN(
        save_root = save_root,
        save_name = save_name,
        epochs = epochs,
        dense_activation = dense_activation,
        first_layer_width = first_layer_width,
        layers = layers,
        layer_shrink_factor = layer_shrink_factor,
        learning_rate = learning_rate,
        verbose = verbose
    )
    
    return importance.importancesFromModel(
        model = network,
        X = X,
        Xk = Xk,
        y = y,
        fit = True,
        exponent = exponent,
        drop_first = drop_first,
        verbose = verbose
    )
#/def importances

def wStats(
    X: np.ndarray | pl.DataFrame,
    Xk: np.ndarray | pl.DataFrame,
    y: np.ndarray | pl.Series | pl.DataFrame,
    local_grad_method: Literal['auto_diff','bandwidth'] = 'auto_diff',
    W_method: Literal['difference','signed_max'] = 'difference',
    bandwidth: float | None = None,
    exponent: float = 2.0,
    drop_first: bool = True,
    # SuperBasicNetworks Parameters
    save_root: str = '',
    save_name: str = '',
    epochs: int = 500,
    dense_activation: str = 'relu', # 'relu', 'sigmoid',...
    # Hyper parameters
    first_layer_width: float | int = 0.25, # int = absolute size, float = proportion of input
    layers: int = 2,
    layer_shrink_factor: float = 0.25,
    learning_rate: float = 0.01,
    verbose: int = 0
    ) -> np.ndarray:
    """
        :param np.ndarray|pl.DataFrame X: Explanatory data
        :param np.ndarray|pl.DataFrame Xk: Knockoff explanatory data
        :param np.ndarray|pl.Series|pl.DataFrame y: Outcome data
        :param Literal['auto_diff','bandwidth'] local_grad_method: Method of MALD. Defaults to `'auto_diff'` for exact autodifferentiation. `'bandwidth'` uses the bandwidth approximation when auto differentiation is not available.
        :param float|None bandwidth: Width if `local_grad_method = 'bandwidth'`
        :param float exponent: Power to take of each MALD value. 1.0 and 2.0 both work reasonably well.
        :param bool drop_first: How to handle one-hot-encoding categorical variables. If `True` the number of associated columns is the number of categories minus 1.
        :param str save_root: Dir to make saves
        :param str save_name: Name to use for saving checkpoints
        :param int epochs: Epochs for `.fit()`
        :param str dense_activation: Activation of internal layers, most likely 'relu,' 'sigmoid', 'leaky_relu', etc
        :param float|int first_layer_width: Size of first layer after input. If an integer, it uses that value. If a float, it's a multiplier of the input size.
        :param int layers: Number of internal layers
        :param float layer_shrink_factor: With multiple internal layers, the width of each is this multiplied by the previous layer's width
        :param float learning_rate: Learning rate for `.fit()`; common values are 0.01, 0.005, 0.001, 0.0005, 0.0001
        :param int verbose: How much to print out, for mostly for debugging.
        :returns: W statistics, half the length of `importances`, the same length as the original number of variables
        :rtype: np.ndarray
        
        Creates a SuperBasicNetwork and uses it for MALD Importance, giving W stats. Equivalent to running `importances` followed by ``maldImportance.importance.wFromImportances()``. See :doc:`importance` and :doc:`superBasicNetworks`
    """
    from . import importance
    _importances: np.ndarray = importances(
        X = X,
        Xk = Xk,
        y = y,
        exponent = exponent,
        drop_first = drop_first,
        # SuperBasicNetworks Parameters
        save_root = save_root,
        save_name = save_name,
        epochs = epochs,
        dense_activation = dense_activation,
        # Hyper parameters
        first_layer_width = first_layer_width, # int = absolute size, float = proportion of input
        layers = layers,
        layer_shrink_factor = layer_shrink_factor,
        learning_rate = learning_rate,
        verbose = verbose
    )
    
    return importance.wFromImportances(
        importances = _importances,
        W_method = W_method,
        verbose = verbose
    )
#/def wStats
