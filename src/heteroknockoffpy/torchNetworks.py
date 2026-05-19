#
#//  torchNetworks.py
#//  heteroknockoffpy
#//
#//  Created by Evan Mason on 3/18/26.
#//

import torch
import torch.nn as nn
import torch.optim as optim
from torch.func import vmap, jacrev

import polars as pl
import numpy as np
import math

from tqdm import tqdm
from typing import Callable, Sequence, Self, Type

# -- KnockoffGAN networks

def _knockoff_xavier_init(layer: nn.Linear) -> None:
    """Match the original TF xavier_init: std = 1/sqrt(in_dim/2)."""
    std = 1.0 / math.sqrt(layer.in_features / 2.0)
    nn.init.normal_(layer.weight, mean=0.0, std=std)
    nn.init.zeros_(layer.bias)
#

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
#

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

def get_logit_jacobian(
    model: nn.Module,
    x: torch.Tensor,
    ) -> torch.Tensor:
    """
    Computes the (n, k, p) Jacobian for a batched input.

        :param model: A PyTorch nn.Module (classifier).
        :param x: Input tensor of shape (n, p).
        
        :returns: A tensor of shape (n, k, p) where k is the number of logits.
    """
    # 1. Put model in eval mode to disable dropout/batchnorm updates
    model.eval()

    # 2. Define a pure function for a single sample (removes batch dim)
    # torch.func requires functional calls, so we use torch.func.functional_call
    # or a simple wrapper if the model doesn't use complex state.
    def model_single(x_single):
        # We add a dummy batch dim [1, p] because most layers expect it,
        # then squeeze it back out to return [k]
        return model(x_single.unsqueeze(0)).squeeze(0)

    # 3. Use jacrev for the derivative and vmap to parallelize over the batch
    # jacrev(model_single) computes (k, p)
    # vmap(...) pushes the batch dimension 'n' to the front
    with torch.no_grad():
        jacobian_batch = vmap(jacrev(model_single))(x)
    #
    return jacobian_batch
#/def get_logit_jacobian

class TorchSimpleDense_Numeric( nn.Module ):
    def __init__(
        self: Self,
        input_size: int,
        layers: Sequence[ int ],
        internalModule: nn.Module, # nn.ReLU, ...
        output_dimension: int = 1
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
                layers[-1], output_dimension,
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
        dense_activation: str | Type[ nn.Module ] = nn.ReLU,
        loss_func: nn.Module = nn.MSELoss(),
        output_dimension: int = 1,
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
        
        self.device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
        
        self.model = TorchSimpleDense_Numeric(
            input_size = input_size,
            layers = layers,
            internalModule = internalModule,
            output_dimension = output_dimension,
        ).to( self.device )
        
        self.loss_func = loss_func
        self.output_dimension = output_dimension
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
        
        optimizer: optim.Adam = optim.Adam(
            self.model.parameters(),
            lr = self.learning_rate,
        )
        
        loss_func: nn.Module = self.loss_func
        
        X_tensor: torch.Tensor = torch.tensor( X ).float().to( self.device )
        _y = np.asarray( y )
        if np.issubdtype( _y.dtype, np.integer ):
            y_tensor: torch.Tensor = torch.tensor( _y ).long().to( self.device )
        else:
            y_tensor: torch.Tensor = torch.tensor( _y ).float().to( self.device )
        #

        for epoch in tqdm( range( self.epochs ) ):
            y_pred = self.model( X_tensor )

            loss = loss_func( y_pred.squeeze(1), y_tensor )
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        #/for epoch in range( self.epochs )
        
        return
    #/def fit
    
    def predict_t(
        self: Self,
        X_t: torch.Tensor,
        ) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            result = self.model( X_t )
        self.model.train()
        return result
    #/def predict_t

    def predict(
        self: Self,
        X: np.ndarray,
        ) -> np.ndarray:
        return self.predict_t(
            torch.tensor( X, dtype=torch.float32 ).to( self.device )
        ).cpu().numpy()
    #/def predict

    def auto_diff_t(
        self: Self,
        X_t: torch.Tensor,
        ) -> torch.Tensor:
        self.model.eval()
        X_t = X_t.detach().requires_grad_( True )
        y_pred: torch.Tensor = self.model( X_t )
        grad = torch.autograd.grad(
            outputs = y_pred,
            inputs = X_t,
            grad_outputs = torch.ones_like( y_pred ),
        )
        self.model.train()
        return grad[0].detach()
    #/def auto_diff_t

    def auto_diff(
        self: Self,
        X: np.ndarray,
        ) -> np.ndarray:
        return self.auto_diff_t(
            torch.tensor( X, dtype=torch.float32 ).to( self.device )
        ).cpu().numpy()
    #/def auto_diff
#class PredictionModel_Numeric():

class PredictionModel_Categorical():
    def __init__(
        self: Self,
        input_size: int,
        layers: Sequence[ int ],
        dense_activation: str | nn.Module = nn.ReLU,
        learning_rate: float = 0.01,
        epochs: int = 500,
        verbose: int = 0
        ) -> None:
        ...
    #/def __init__
#/class PredictionModel_Categorical

# -- Knockoffs

def sample_Z(
    m: int,
    n: int,
    x_name: str,
    device: str = 'cpu',
    z_scale: float = 1.0,
) -> torch.Tensor:
    std = z_scale * (1.0 / 3000.0) ** 0.5
    if x_name in ('Normal', 'AR_Normal'):
        return torch.normal(
            mean = 0.0,
            std = std,
            size = (m, n),
            device = device,
        )
    elif x_name in ('Uniform', 'AR_Uniform'):
        return torch.empty(
            m,
            n,
            device = device,
        ).uniform_(-3 * std, 3 * std)
    else:
        raise ValueError(f"Unknown x_name: {x_name!r}")
#/def sample_Z


class TorchGAN(nn.Module):
    def __init__(
        self: Self,
        shape: tuple[int, int],
        x_name: str,
        lamda: float = 1,
        mu: float = 1,
        lam: float = 10,
        lr: float = 1e-4,
        mb_size: int = 128,
        niter: int = 2000,
        combined_inner: bool = False,
    ) -> None:
        super().__init__()

        self.device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

        self.x_dim = shape[1]
        self.z_dim = shape[1]

        self.x_name = x_name
        self.lamda = lamda
        self.mu = mu
        self.lam = lam
        self.lr = lr
        self.mb_size = mb_size
        self.niter = niter
        self.combined_inner = combined_inner

        self.generator = KnockoffGenerator(shape).to(self.device)
        self.discriminator = KnockoffDiscriminator(shape).to(self.device)
        self.wgan_discriminator = KnockoffWGANDiscriminator(shape).to(self.device)
        self.mine = KnockoffMINE(shape).to(self.device)

        self.opt_G = optim.Adam(
            self.generator.parameters(),
            lr = lr,
            betas = (0.5, 0.999),
        )

        if combined_inner:
            self.opt_inner = optim.Adam(
                list(self.discriminator.parameters()) +
                list(self.wgan_discriminator.parameters()) +
                list(self.mine.parameters()),
                lr = lr,
                betas = (0.5, 0.999),
            )
        else:
            self.opt_D = optim.Adam(
                self.discriminator.parameters(),
                lr = lr,
                betas = (0.5, 0.999),
            )
            self.opt_WD = optim.Adam(
                self.wgan_discriminator.parameters(),
                lr = lr,
                betas = (0.5, 0.999),
            )
            self.opt_M = optim.Adam(
                self.mine.parameters(),
                lr = lr,
                betas = (0.5, 0.999),
            )
        #/if combined_inner
    #/def __init__

    def _wgan_gradient_penalty(
        self: Self,
        X_mb: torch.Tensor,
        G_sample: torch.Tensor,
    ) -> torch.Tensor:
        eps = torch.rand(
            X_mb.size(0),
            1,
            device = self.device,
        )
        X_inter = (eps * X_mb + (1.0 - eps) * G_sample).requires_grad_(True)

        score = self.wgan_discriminator(X_inter)

        grad = torch.autograd.grad(
            outputs = score,
            inputs = X_inter,
            grad_outputs = torch.ones_like(score),
            create_graph = True,
        )[0]

        grad_norm = (grad ** 2 + 1e-8).sum(dim = 1).sqrt()
        return self.lam * ((grad_norm - 1) ** 2).mean()
    #/def _wgan_gradient_penalty

    def _wgan_loss(
        self: Self,
        X_mb: torch.Tensor,
        G_sample: torch.Tensor,
    ) -> torch.Tensor:
        WD_real = self.wgan_discriminator(X_mb)
        WD_fake = self.wgan_discriminator(G_sample)
        return WD_fake.mean() - WD_real.mean() + self._wgan_gradient_penalty(X_mb, G_sample)
    #/def _wgan_loss

    def _discriminator_loss(
        self: Self,
        X_mb: torch.Tensor,
        G_sample: torch.Tensor,
        S: torch.Tensor,
        H: torch.Tensor,
    ) -> torch.Tensor:
        swap_a = S * X_mb + (1 - S) * G_sample
        swap_b = (1 - S) * X_mb + S * G_sample
        D_out = self.discriminator(swap_a, swap_b, H * S)
        return -(
            S * (1 - H) * (D_out + 1e-8).log() +
            (1 - S) * (1 - H) * (1 - D_out + 1e-8).log()
        ).mean()
    #/def _discriminator_loss

    def _mine_loss(
        self: Self,
        X_mb: torch.Tensor,
        X_perm: torch.Tensor,
        G_sample: torch.Tensor,
    ) -> torch.Tensor:
        M_out = self.mine(X_mb, G_sample)
        Exp_M_out = self.mine(X_perm, G_sample).exp()
        return (M_out.mean(dim = 0) - Exp_M_out.mean(dim = 0).log()).sum()
    #/def _mine_loss

    def _generator_loss(
        self: Self,
        X_mb: torch.Tensor,
        X_perm: torch.Tensor,
        G_sample: torch.Tensor,
        S: torch.Tensor,
        H: torch.Tensor,
    ) -> torch.Tensor:
        D_loss = self._discriminator_loss(X_mb, G_sample, S, H)
        WD_fake = self.wgan_discriminator(G_sample)
        M_loss = self._mine_loss(X_mb, X_perm, G_sample)
        return -D_loss + self.mu * -WD_fake.mean() + self.lamda * M_loss
    #/def _generator_loss

    def fit(
        self: Self,
        x_train: np.ndarray,
    ) -> None:
        n = x_train.shape[0]
        self.train()

        for _ in tqdm(range(self.niter)):

            # -- train WD, D, MINE
            for _ in range(5):
                idx = np.random.permutation(n)[:self.mb_size]
                X_mb = torch.tensor(
                    x_train[idx],
                    dtype = torch.float32,
                    device = self.device,
                )
                X_perm = X_mb[torch.randperm(X_mb.size(0))]
                batch = X_mb.size(0)

                Z = sample_Z(
                    batch,
                    self.z_dim,
                    self.x_name,
                    self.device,
                )
                S = torch.bernoulli(torch.full(
                    (batch, self.x_dim),
                    0.5,
                    device = self.device,
                ))
                H = torch.bernoulli(torch.full(
                    (batch, self.x_dim),
                    0.9,
                    device = self.device,
                ))

                G_sample = self.generator(X_mb, Z).detach()

                if self.combined_inner:
                    self.opt_inner.zero_grad()
                    (
                        self._wgan_loss(X_mb, G_sample) +
                        self._discriminator_loss(X_mb, G_sample, S, H) -
                        self._mine_loss(X_mb, X_perm, G_sample)
                    ).backward()
                    self.opt_inner.step()
                else:
                    self.opt_WD.zero_grad()
                    self._wgan_loss(X_mb, G_sample).backward()
                    self.opt_WD.step()

                    self.opt_D.zero_grad()
                    self._discriminator_loss(X_mb, G_sample, S, H).backward()
                    self.opt_D.step()

                    self.opt_M.zero_grad()
                    (-self._mine_loss(X_mb, X_perm, G_sample)).backward()
                    self.opt_M.step()
                #/if combined_inner
            #/for inner

            # -- train G
            idx = np.random.permutation(n)[:self.mb_size]
            X_mb = torch.tensor(
                x_train[idx],
                dtype = torch.float32,
                device = self.device,
            )
            X_perm = X_mb[torch.randperm(X_mb.size(0))]
            batch = X_mb.size(0)

            Z = sample_Z(
                batch,
                self.z_dim,
                self.x_name,
                self.device,
            )
            S = torch.bernoulli(torch.full(
                (batch, self.x_dim),
                0.5,
                device = self.device,
            ))
            H = torch.zeros(
                batch,
                self.x_dim,
                device = self.device,
            )

            G_sample = self.generator(X_mb, Z)

            self.opt_G.zero_grad()
            self._generator_loss(X_mb, X_perm, G_sample, S, H).backward()
            self.opt_G.step()
        #/for niter
    #/def fit

    def forward(
        self: Self,
        X: np.ndarray,
        ) -> np.ndarray:
        """
            Standard torch forward call.


        """
        # Allow possibility of BatchNorm, Dropout, or other training effects
        self.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(
                X,
                dtype = torch.float32,
                device = self.device,
            )
            Z = sample_Z(
                X_tensor.size(0),
                self.z_dim,
                self.x_name,
                self.device,
            )
            Xk = self.generator(X_tensor, Z)

        # Back to training mode
        self.train()
        return Xk.cpu().numpy()
    #/def forward
#/class TorchGAN

# -- PRISM

class PRISMNetwork(nn.Module):
    """
    MLP with a separate input_layer so group-lasso proximal steps and group norms
    can be applied directly to the first-layer weights.

    forward: rest(activation(input_layer(x)))
    """
    def __init__(
        self: Self,
        input_size: int,
        layers: Sequence[int],
        activation_class: type[nn.Module],
        output_size: int,
        output_activation: type[nn.Module] | None = None,
    ) -> None:
        super().__init__()
        self.input_layer = nn.Linear(input_size, layers[0])
        self.activation = activation_class()
        self.rest = _build_sequential(
            input_size = layers[0],
            layers = layers[1:],
            activation = activation_class,
            output_size = output_size,
            output_activation = output_activation,
        )
    #/def __init__

    def forward(
        self: Self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.rest(self.activation(self.input_layer(x)))
    #/def forward
#/class PRISMNetwork


class PRISMPredictionModel:
    """
    Wraps PRISMNetwork for PRISM-W and PRISM-G importance computation.

    Trains over a lambda_path (regularization path), recording an importance
    snapshot at the end of each lambda stage. The final importance is the
    caller's average over snapshots.

    Implements fit / predict / auto_diff to satisfy the PredictionModel protocol.
    """
    def __init__(
        self: Self,
        input_size: int,
        layers: Sequence[int],
        dense_activation: str | type[nn.Module] = 'relu',
        loss_func: nn.Module = nn.MSELoss(),
        output_dimension: int = 1,
        learning_rate: float = 0.01,
        epochs_per_batch: int = 10,
        model_type: str = 'mlp',
        verbose: int = 0,
    ) -> None:
        activation_class: type[nn.Module]
        if isinstance(dense_activation, str):
            activation_class = _nnModule_dict[dense_activation]
        else:
            activation_class = dense_activation
        #

        self.device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

        self.model = PRISMNetwork(
            input_size = input_size,
            layers = list(layers),
            activation_class = activation_class,
            output_size = output_dimension,
        ).to(self.device)

        self.loss_func = loss_func
        self.learning_rate = learning_rate
        self.epochs_per_batch = epochs_per_batch
        self.verbose = verbose
    #/def __init__

    def fit(
        self: Self,
        X: np.ndarray,
        y: np.ndarray,
        groups: list[list[int]],
        lambda_path: Sequence[float],
        a_min: float | None = None,
        snapshot_fn: Callable[['PRISMPredictionModel', torch.Tensor], np.ndarray] | None = None,
    ) -> list[np.ndarray]:
        """
        Train over the lambda_path; record one importance snapshot per lambda stage.

        :param groups: One list of OHE column indices per original feature (length 2*p).
        :param lambda_path: Sequence of lambda values (e.g. large→small).
        :param a_min: When set, enables adaptive scaling: eff_lambda_j = lambda / max(init_norm_j, a_min).
        :param snapshot_fn: If None, snapshots use get_group_importances (PRISM-W).
                            Otherwise called as snapshot_fn(self) (for PRISM-G).
        :returns: List of importance arrays, one per lambda in lambda_path.
        """
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        _y = np.asarray(y)
        if np.issubdtype(_y.dtype, np.integer):
            y_tensor = torch.tensor(_y).long().to(self.device)
        else:
            y_tensor = torch.tensor(_y, dtype=torch.float32).to(self.device)
        #

        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.model.train()

        # Compute group norms before training for adaptive scaling
        init_norms: list[float] | None = None
        if a_min is not None:
            with torch.no_grad():
                w = self.model.input_layer.weight.data
                init_norms = [
                    torch.norm(w[:, group_j]).item() if group_j else 0.0
                    for group_j in groups
                ]
            #
        #

        snapshots: list[np.ndarray] = []

        # Precompute group structure — single-column (numeric) vs multi-column (categorical).
        _single_pairs = [(j, g[0]) for j, g in enumerate(groups) if len(g) == 1]
        _cat_groups   = [(j, g)    for j, g in enumerate(groups) if len(g) > 1]

        _num_cols: torch.Tensor | None = None
        if _single_pairs:
            _num_cols = torch.tensor([c for _, c in _single_pairs], device=self.device)
        #

        # Adaptive proximal denominators — only needed when a_min is set.
        _num_eff_denom: torch.Tensor | None = None
        _cat_eff_denoms: list[float] | None = None
        if a_min is not None:
            if _single_pairs:
                _num_eff_denom = torch.tensor(
                    [1.0 / max(init_norms[j], a_min) for j, _ in _single_pairs],
                    device=self.device, dtype=torch.float32,
                )
            #
            if _cat_groups:
                _cat_eff_denoms = [1.0 / max(init_norms[j], a_min) for j, _ in _cat_groups]
            #
        #

        total_epochs = len(lambda_path) * self.epochs_per_batch

        with tqdm(total=total_epochs) as pbar:
            for lambda_b in lambda_path:
                lb = float(lambda_b)
                pbar.set_postfix({'lambda': f'{lb:.3g}'})
                for _ in range(self.epochs_per_batch):
                    y_pred = self.model(X_tensor)

                    if a_min is None:
                        # Differentiable group-lasso penalty — no proximal step needed.
                        # Numeric groups: one vectorized op; categorical: small loop.
                        w_inp = self.model.input_layer.weight
                        penalty = w_inp.new_zeros(())
                        if _num_cols is not None:
                            penalty = penalty + (w_inp[:, _num_cols] ** 2).sum(dim=0).clamp(min=1e-12).sqrt().sum()
                        for _, group_j in _cat_groups:
                            penalty = penalty + (w_inp[:, group_j] ** 2).sum().clamp(min=1e-12).sqrt()
                        loss = self.loss_func(y_pred.squeeze(1), y_tensor) + lb * penalty
                    else:
                        loss = self.loss_func(y_pred.squeeze(1), y_tensor)
                    #

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    if a_min is not None:
                        # Adaptive proximal step: vectorized for numeric, loop for categorical.
                        with torch.no_grad():
                            w = self.model.input_layer.weight.data
                            if _num_cols is not None:
                                w_cols = w[:, _num_cols]
                                norms  = w_cols.norm(dim=0)
                                eff    = lb * _num_eff_denom
                                scales = (1.0 - eff / norms.clamp(min=1e-12)).clamp(min=0.0)
                                w[:, _num_cols] = w_cols * scales
                            #
                            for k, (_, group_j) in enumerate(_cat_groups):
                                v    = w[:, group_j]
                                norm = v.norm()
                                eff_lambda = lb * _cat_eff_denoms[k]
                                if norm > eff_lambda:
                                    w[:, group_j] = v * (1.0 - eff_lambda / norm)
                                else:
                                    w[:, group_j] = 0.0
                            #/for categorical groups
                        #/with no_grad
                    #
                    pbar.update(1)
                #/for epochs_per_batch

                if snapshot_fn is not None:
                    snapshots.append(snapshot_fn(self, X_tensor))
                else:
                    snapshots.append(self.get_group_importances(groups))
                #
            #/for lambda_b
        #/with tqdm

        return snapshots
    #/def fit

    def get_group_importances(
        self: Self,
        groups: list[list[int]],
    ) -> np.ndarray:
        """Returns ||input_layer.weight[:, group_j]||_F for each group."""
        with torch.no_grad():
            w = self.model.input_layer.weight.detach().cpu()
            return np.array([
                torch.norm(w[:, group_j]).item() if group_j else 0.0
                for group_j in groups
            ])
        #
    #/def get_group_importances

    def predict_t(
        self: Self,
        X_t: torch.Tensor,
    ) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            result = self.model( X_t )
        #
        self.model.train()
        return result
    #/def predict_t

    def predict(
        self: Self,
        X: np.ndarray,
    ) -> np.ndarray:
        return self.predict_t(
            torch.tensor(X, dtype=torch.float32).to(self.device)
        ).cpu().numpy()
    #/def predict

    def auto_diff_t(
        self: Self,
        X_t: torch.Tensor,
    ) -> torch.Tensor:
        self.model.eval()
        X_t = X_t.detach().requires_grad_( True )
        y_pred = self.model( X_t )
        grad = torch.autograd.grad(
            outputs = y_pred,
            inputs = X_t,
            grad_outputs = torch.ones_like( y_pred ),
        )
        self.model.train()
        return grad[0].detach()
    #/def auto_diff_t

    def auto_diff(
        self: Self,
        X: np.ndarray,
    ) -> np.ndarray:
        return self.auto_diff_t(
            torch.tensor(X, dtype=torch.float32).to(self.device)
        ).cpu().numpy()
    #/def auto_diff
#/class PRISMPredictionModel
