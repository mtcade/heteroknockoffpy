#
#//  torchGAN.py
#//  heteroknockoffpy
#//
#//  Created by Evan Mason on 3/19/26.
#//
#//  Torch version of KnockoffGAN, from
#//  https://github.com/firmai/tsgan/blob/master/alg/knockoffgan/KnockoffGAN.py

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from typing import Self

from heteroknockoffpy.torchNetworks import (
    KnockoffGenerator,
    KnockoffDiscriminator,
    KnockoffWGANDiscriminator,
    KnockoffMINE,
)

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

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

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


def KnockoffGAN(
    x_train: np.ndarray,
    **kwargs,
    ) -> np.ndarray:
    raise Exception("UC")
#/def KnockoffGAN
