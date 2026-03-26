"""
    Interface for creating simple Keras neural networks for outcome variables, and calculating the Mean Absolute Local Derivatives (MALD)
    
    If you have your own preferred networks or other predictors, you can skip using this module.
"""
from . import utilities

import tensorflow as tf
import numpy as np
import polars as pl

from typing import Literal, Self, Sequence

class SimpleNN():
    """
        :param str save_root: Dir to make saves
        :param str save_name: Name to use for saving checkpoints
        :param int epochs: Epochs for `.fit()`
        :param str dense_activation: Activation of internal layers, most likely 'relu,' 'sigmoid', 'leaky_relu', etc
        :param Sequence[int] layers: Sizes of dense layers
        :param float learning_rate: Learning rate for `.fit()`; common values are 0.01, 0.005, 0.001, 0.0005, 0.0001
        :param int verbose: How much to print out, for mostly for debugging.
        
        See `https://keras.io/api/models/ <https://keras.io/api/models/>`_
        
        Easy interface for dense, simply connected Neural Networks, a good starting point for estimation of single continuous outcomes of i.i.d. explanatory data. The key is that this class conforms to the `PredictionModel` Protocol, used by other modules, by including:
            
            * `.fit()`
            * `.predict()`
            * `.call()`
    """
    def __init__(
        self: Self,
        save_root: str,
        save_name: str,
        epochs: int,
        dense_activation: str, # 'relu', 'sigmoid',...
        # Hyper parameters
        layers: int,
        learning_rate: float,
        verbose: int = 0
        ) -> None:
        self.save_root = save_root
        self.save_name = save_name
        self.epochs = epochs
        self.dense_activation = dense_activation
        
        # Hyperparameters
        self.hyperparameters: dict[{
            "layers": Sequence[ int ],
            "learning_rate": float
        }] = {
            "layers": layers,
            "learning_rate": learning_rate
        }
        
        self.verbose = verbose
        
        self.network = None
        super().__init__()
    #
    
    def getNetwork(
        self: Self,
        input_length: int,
        hyperparameters: dict[ str, any ]
        ) -> tf.keras.Model:
        """
            :param int input_length: Dimension of explanatory input data
            :param dict[ str, any ] hyperparamters: Network architecture and training hyperparamters.
            
                
            
            Builds a network with architecture from hyperparameters (see `get_simpleNN()`)
            
                * `layers Sequence[int]`: Width of each layer
                * `learning_rate float`
        """
        if self.verbose > 1:
            print("# Building SimpleNN network with hyper parameters:")
            for _key, _val in hyperparameters.items():
                print("#   {}: {}".format(_key,_val))
            #
        #
        
        input = tf.keras.layers.Input(
            shape = (input_length,),
            name = 'keras_tensor'
        )
        layers: Sequence[ int ] = hyperparameters['layers']
        
        # Keep track of "true" layer size so that we
        #   don't have compounding rounding issues
        
        dense_next = tf.keras.layers.Dense(
            round( layers[0] ),
            activation = self.dense_activation
        )( input )
        
        if len( layers ) > 1:
            for layer_width in layers[1:]:
                dense_next = tf.keras.layers.Dense(
                    layer_width,
                    activation = self.dense_activation
                )( dense_next )
            #/for layer_width in layers[1:]
        #/if len( layers ) > 1
        
        # Output
        output = tf.keras.layers.Dense(
            1,
            activation = 'linear'
        )( dense_next )
        
        model: tf.keras.Model = tf.keras.Model(
            input,
            output
        )
        
        model.compile(
            optimizer = tf.keras.optimizers.Adam(
                learning_rate = hyperparameters[
                    'learning_rate'
                ]
            ),
            loss = tf.keras.losses.MeanSquaredError(),
            run_eagerly = True,
        )
        return model
    #/def getNetwork
    
    def _fit_graph(self: Self, X: np.ndarray, y: np.ndarray) -> None:
        """TF1 graph-mode training via tf.compat.v1.Session."""
        learning_rate: float = self.hyperparameters['learning_rate']
        layer_sizes: Sequence[int] = self.hyperparameters['layers']

        graph = tf.compat.v1.Graph()
        with graph.as_default():
            x_ph = tf.compat.v1.placeholder(tf.float32, [None, X.shape[1]], name='X')
            y_ph = tf.compat.v1.placeholder(tf.float32, [None], name='y')

            h = x_ph
            for width in layer_sizes:
                h = tf.compat.v1.layers.dense(h, int(round(width)), activation=self.dense_activation)
            pred = tf.compat.v1.layers.dense(h, 1, activation=None)
            pred = tf.squeeze(pred, axis=1)

            loss_op = tf.reduce_mean(tf.square(pred - y_ph))
            train_op = tf.compat.v1.train.AdamOptimizer(
                learning_rate=learning_rate
            ).minimize(loss_op)
            init_op = tf.compat.v1.global_variables_initializer()

        self._x_ph = x_ph
        self._pred_op = pred
        self._session = tf.compat.v1.Session(graph=graph)
        self._session.run(init_op)

        for epoch in range(self.epochs):
            _, loss_val = self._session.run(
                [train_op, loss_op],
                feed_dict={x_ph: X, y_ph: y}
            )
            if self.verbose > 0:
                print("{}/{} - loss = {}".format(epoch, self.epochs, loss_val))
    #

    def fit(
        self: Self,
        X: np.ndarray,
        y: np.ndarray
        ) -> None:
        """
            :param np.ndarray X: Explanatory data, likely including the knockoffs
            :param np.ndarray y: Outcome data

            Initializes the `.network` if necessary and calls `.fit` on it
        """
        if tf.executing_eagerly():
            if self.network is None:
                self.network = self.getNetwork(
                    input_length=X.shape[1],
                    hyperparameters=self.hyperparameters,
                )
            for epoch in range(self.epochs):
                loss = self.network.train_on_batch(X, y)
                if self.verbose > 0:
                    print("{}/{} - loss = {}".format(epoch, self.epochs, loss))
        else:
            self._fit_graph(X, y)

        return
    #

    def predict( self: Self, X: np.ndarray ) -> np.ndarray:
        """
            :param np.ndarray X: Explanatory data, likely including the knockoffs
            :returns: Predictions from the trained network
            :rtype: np.ndarray

            Calls `self.network.predict( X )`
        """
        if tf.executing_eagerly():
            return self.network.predict(X).reshape((X.shape[0],))
        else:
            return self._session.run(
                self._pred_op,
                feed_dict={self._x_ph: X}
            )
    #
    
    def call( self: Self, X: np.ndarray | tf.Tensor ) -> np.ndarray | tf.Tensor:
        """
            :param np.ndarray|tf.Tensor X: Explanatory data, likely including the knockoffs
            :returns: Prediction result from `self.network.call(X)`
            :rtype: np.ndarray|tf.Tensor
        """
        return self.network.call( X )
    #
    
    def auto_diff(
        self: Self,
        X: np.ndarray,
        ) -> np.ndarray:
        _X = tf.constant( X )
    
        #y_pred = model.predict( _X )
        
        tape: tf.GradientTape
        
        with tf.GradientTape() as tape:
            tape.watch( _X )
            y_hat: tf.Tensor = self.call( _X )
        #
        
        return tape.gradient( y_hat, _X ).numpy()
    #/def auto_diff
#/class SimpleNN

def fit_SimpleNN(
    X: np.ndarray | pl.DataFrame,
    Xk: np.ndarray | pl.DataFrame,
    y: np.ndarray | pl.Series | pl.DataFrame,
    drop_first: bool = True,
    save_root: str = '',
    save_name: str = '',
    epochs: int = 500,
    dense_activation: str = 'relu', # 'relu', 'sigmoid',...
    # Hyper parameters
    #first_layer_width: float | int = 0.25, # int = absolute size, float = proportion of input
    layers: Sequence[int] | None = None,
    #layer_shrink_factor: float = 0.25,
    learning_rate: float = 0.01,
    verbose: int = 0
    ) -> SimpleNN:
    """
        :param np.ndarray|pl.DataFrame X: Explanatory data
        :param np.ndarray|pl.DataFrame Xk: Knockoff explanatory data
        :param np.ndarray|pl.Series|pl.DataFrame y: Outcome data
        :param bool drop_first: How to handle one-hot-encoding categorical variables. If `True` the number of associated columns is the number of categories minus 1.
        :param str save_root: Dir to make saves
        :param str save_name: Name to use for saving checkpoints
        :param int epochs: Epochs for `.fit()`
        :param str dense_activation: Activation of internal layers, most likely 'relu,' 'sigmoid', 'leaky_relu', etc
        :param Sequence[int]|None layers: Internal layers. If not provided, defaults to 2 layers, 1/4 and 1/16 of input size
        :param float learning_rate: Learning rate for `.fit()`; common values are 0.01, 0.005, 0.001, 0.0005, 0.0001
        :param int verbose: How much to print out, for mostly for debugging.
        
        Initializing and fits a ``SimpleNN`` with defaults.
    """
    assert X.shape == Xk.shape
    
    if layers is None:
        layers = tuple(
            round((0.25**(l-1))*2*X.shape[1])\
                for l in range(2)
        )
    #/if layers is None
    
    model: SimpleNN = SimpleNN(
        save_root = save_root,
        save_name = save_name,
        epochs = epochs,
        dense_activation = dense_activation,
        layers = layers,
        learning_rate = learning_rate,
        verbose = verbose
    )
    
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
    y_np = np.reshape( y_np, ( X_all_np.shape[0], ) )
    
    model.fit( X_all_np, y_np )
    
    return model
#/def fit_SimpleNN
