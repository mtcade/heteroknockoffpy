#
#//  torchKnockoffs.py
#//  heteroknockoffpy
#//
#//  Created by Evan Mason on 6/1/26.
#//

import torch
import torch.nn as nn
import torch.optim as optim
import math
from tqdm import tqdm
from typing import Sequence, Self, Type

from .torchUtil import _nnModule_dict, _build_sequential

# -- KnockoffGAN networks

def _knockoff_xavier_init(layer: nn.Linear) -> None:
    """Match the original TF xavier_init: std = 1/sqrt(in_dim/2)."""
    std = 1.0 / math.sqrt(layer.in_features / 2.0)
    nn.init.normal_(layer.weight, mean=0.0, std=std)
    nn.init.zeros_(layer.bias)
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
        x_t = torch.tensor(x_train, dtype=torch.float32, device=self.device)
        n   = x_t.shape[0]
        self.train()

        for _ in tqdm(range(self.niter)):

            # -- train WD, D, MINE
            for _ in range(5):
                idx  = torch.randperm(n, device=self.device)[:self.mb_size]
                X_mb = x_t[idx]
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
            idx  = torch.randperm(n, device=self.device)[:self.mb_size]
            X_mb = x_t[idx]
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

        self.train()
        return Xk.cpu().numpy()
    #/def forward
#/class TorchGAN
