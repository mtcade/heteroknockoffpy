#
#//  torchNetworks.py
#//  heteroknockoffpy
#//
#//  Created by Evan Mason on 3/18/26.
#//

import torch
import torch.nn as nn
import torch.optim as optim

import polars as pl
import numpy as np
import math

from tqdm import tqdm
from typing import Sequence, Self

# -- KnockoffGAN networks

def _knockoff_xavier_init(layer: nn.Linear) -> None:
    """Match the original TF xavier_init: std = 1/sqrt(in_dim/2)."""
    std = 1.0 / math.sqrt(layer.in_features / 2.0)
    nn.init.normal_(layer.weight, mean=0.0, std=std)
    nn.init.zeros_(layer.bias)


def _build_sequential(
    input_size: int,
    layers: Sequence[int],
    activation: type[nn.Module],
    output_size: int,
    output_activation: type[nn.Module] | None = None,
) -> nn.Sequential:
    parts: list[nn.Module] = []
    prev = input_size
    for width in layers:
        parts += [nn.Linear(prev, width), activation()]
        prev = width
    parts.append(nn.Linear(prev, output_size))
    if output_activation is not None:
        parts.append(output_activation())
    return nn.Sequential(*parts)


class KnockoffGenerator(nn.Module):
    """
    Generator G(X, Z) → Xk.

    Input:  concat(X, Z)  shape (batch, x_dim + z_dim)
    Output: Xk            shape (batch, x_dim)
    Hidden: tanh → linear (no output activation)
    """
    def __init__(
        self: Self,
        shape: tuple[int, int],
        layers: Sequence[int] | None = None,
        z_dim: int | None = None,
    ) -> None:
        super().__init__()
        x_dim = shape[1]
        z = z_dim if z_dim is not None else x_dim

        self.z_dim = z
        self.net = _build_sequential(
            input_size = x_dim + z,
            layers = layers if layers is not None else (x_dim,),
            activation = nn.Tanh,
            output_size = x_dim,
        )
    #/def __init__

    def forward(
        self: Self,
        X: torch.Tensor,
        Z: torch.Tensor,
    ) -> torch.Tensor:
        return self.net(torch.cat([X, Z], dim=1))
    #/def forward
#/class KnockoffGenerator


class KnockoffDiscriminator(nn.Module):
    """
    Discriminator D(SwapA, SwapB, Hint) → probabilities per feature.

    Input:  concat(SwapA, SwapB, Hint)  shape (batch, 3*x_dim)
    Output: swap probabilities           shape (batch, x_dim)
    Hidden: tanh → sigmoid
    """
    def __init__(
        self: Self,
        shape: tuple[int, int],
        layers: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        x_dim = shape[1]

        self.net = _build_sequential(
            input_size = x_dim * 3,
            layers = layers if layers is not None else (x_dim,),
            activation = nn.Tanh,
            output_size = x_dim,
            output_activation = nn.Sigmoid,
        )
    #/def __init__

    def forward(
        self: Self,
        swap_a: torch.Tensor,
        swap_b: torch.Tensor,
        hint: torch.Tensor,
    ) -> torch.Tensor:
        return self.net(torch.cat([swap_a, swap_b, hint], dim=1))
    #/def forward
#/class KnockoffDiscriminator


class KnockoffWGANDiscriminator(nn.Module):
    """
    Wasserstein critic WD(Xk) → scalar score.

    Input:  Xk       shape (batch, x_dim)
    Output: score    shape (batch, 1)
    Hidden: ReLU → linear (unbounded, for Wasserstein loss)
    """
    def __init__(
        self: Self,
        shape: tuple[int, int],
        layers: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        x_dim = shape[1]

        self.net = _build_sequential(
            input_size = x_dim,
            layers = layers if layers is not None else (x_dim,),
            activation = nn.ReLU,
            output_size = 1,
        )
    #/def __init__

    def forward(
        self: Self,
        Xk: torch.Tensor,
    ) -> torch.Tensor:
        return self.net(Xk)
    #/def forward
#/class KnockoffWGANDiscriminator


class KnockoffMINE(nn.Module):
    """
    Mutual Information Neural Estimator between X and Xk.

    All weights are element-wise 1D vectors (shape x_dim) — no cross-feature
    projection. Matches the original TF implementation exactly:
        h  = tanh(WA * X + WB * Xk + b)   # shape (2, x_dim), run in parallel
        out = W3 * h.sum(0) + b3

    Output: shape (batch, x_dim) — one value per feature per sample.
    """
    def __init__(
        self: Self,
        shape: tuple[int, int],
    ) -> None:
        super().__init__()
        x_dim = shape[1]
        bound = 1.0 / math.sqrt(x_dim)

        self.WA = nn.Parameter(torch.empty(2, x_dim))
        self.WB = nn.Parameter(torch.empty(2, x_dim))
        self.b  = nn.Parameter(torch.zeros(2, x_dim))
        self.W3 = nn.Parameter(torch.empty(x_dim))
        self.b3 = nn.Parameter(torch.zeros(x_dim))

        nn.init.uniform_(self.WA, -bound, bound)
        nn.init.uniform_(self.WB, -bound, bound)
        nn.init.uniform_(self.W3, -bound, bound)
    #/def __init__

    def forward(
        self: Self,
        X: torch.Tensor,
        Xk: torch.Tensor,
    ) -> torch.Tensor:
        # X, Xk: (batch, x_dim) → unsqueeze to (batch, 1, x_dim) for broadcast
        h = torch.tanh(
            self.WA * X.unsqueeze(1) +
            self.WB * Xk.unsqueeze(1) +
            self.b
        )  # (batch, 2, x_dim)
        return self.W3 * h.sum(dim=1) + self.b3
    #/def forward
#/class KnockoffMINE

# -- Torch Importance

class TorchSimpleDense_Numeric( nn.Module ):
    def __init__(
        self: Self,
        input_size: int,
        layers: Sequence[ int ],
        internalModule: nn.Module, # nn.ReLU, ...
        ) -> None:
        super().__init__()
        
        # Input to first internal layer
        sequential_list: list[ nn.Module ] = [
            nn.Linear( input_size, layers[0] ),
            internalModule(),
        ]
         
        if len( layers ) > 1:
            for i in range( len( layers) - 1 ):
                sequential_list.extend(
                    [
                        nn.Linear(
                            layers[i], layers[i+1],
                        ),
                        internalModule()
                    ]
                )
            #
        #
        
        # Final to output
        sequential_list.append(
            nn.Linear(
                layers[-1], 1,
            )
        )
        
        self.network = nn.Sequential(
            *sequential_list
        )
        
        return
    #/def __init__
    
    def forward(
        self: Self,
        X: torch.tensor,
        ) -> torch.tensor:
        
        return self.network( X )
    #/def forward
#/class TorchSimpleDense_Numeric

_nnModule_dict: dict[ str, nn.Module ] = {
    'leaky_relu': nn.LeakyReLU,
    'relu': nn.ReLU,
    'sigmoid': nn.Sigmoid,
}

class PredictionModel_Numeric():
    def __init__(
        self: Self,
        input_size: int,
        layers: Sequence[ int ],
        dense_activation: str | nn.Module = nn.ReLU,
        learning_rate: float = 0.01,
        epochs: int = 500,
        verbose: int = 0
        ) -> None:
        
        internalModule: nn.Module
        if isinstance( dense_activation, str ):
            internalModule = _nnModule_dict[ dense_activation ]
        #
        else:
            internalModule = dense_
        #
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.model = TorchSimpleDense_Numeric(
            input_size = input_size,
            layers = layers,
            internalModule = internalModule,
        ).to( self.device )
        
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.verbose = verbose
        
        return
    #/def __init__
    
    def fit(
        self: Self,
        X: np.ndarray,
        y: np.ndarray,
        **kwargs,
        ) -> None:
        loss_func: nn.MSELoss = nn.MSELoss()
        
        optimizer: optim.Adam = optim.Adam(
            self.model.parameters(),
            lr = self.learning_rate,
        )
        
        X_tensor: torch.Tensor = torch.tensor( X ).float().to( self.device )
        y_tensor: torch.Tensor = torch.tensor( y ).float().to( self.device )
        
        for epoch in tqdm( range( self.epochs ) ):
            y_pred = self.model( X_tensor )
            
            loss = loss_func( y_pred.squeeze(1), y_tensor )
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        #/for epoch in range( self.epochs )
        
        return
    #/def fit
    
    def predict(
        self: Self,
        X: np.ndarray,
        ) -> np.ndarray:
        return self.model(
            torch.tensor( X ).float().to( self.device )
        ).detach().numpy().reshape( (X.shape[0],) )
    #/def predict
    
    def auto_diff(
        self: Self,
        X: np.ndarray,
        ) -> np.ndarray:
        X_torch: torch.tensor = torch.tensor( X ).float().to( self.device )
        X_torch.requires_grad = True
        
        y_pred: torch.Tensor = self.model( X_torch )
        
        grad: tuple[ torch.Tensor,... ] = torch.autograd.grad(
            outputs = y_pred,
            inputs = X_torch,
            grad_outputs = torch.ones_like( y_pred ),
        )
        
        return torch.cat(
            grad, dim=0
        ).numpy()
    #/def auto_diff
#class PredictionModel_Numeric():

