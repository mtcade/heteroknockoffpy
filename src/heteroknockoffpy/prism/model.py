#
#//  model.py
#//  heteroknockoffpy
#//
#//  Created by Evan Mason on 6/2/26.
#//
#// Adapted from https://github.com/feizhe/novel_knockoffs/tree/master/PRISM
"""
Models for PRISM-G.

AugmentedMLP
    Standard MLP on 2p-dimensional input [X, X_tilde].
    Group regularisation on first-layer column norms.

PairwiseFilterMLP
    Adds a DeepPINK-style pairwise filter layer before the MLP.
    The filter forces explicit competition between X_j and X_tilde_j:

        f_j = (z_j / (|z_j| + |z_{j+p}| + eps)) * x_j
            + (z_{j+p} / (|z_j| + |z_{j+p}| + eps)) * x_tilde_j

    so each f_j is a convex combination of x_j and x_tilde_j.
    Under this constraint the model MUST choose between them.
    Antisymmetry follows from architectural exchange symmetry.

AdditiveMLP
    Feature-wise additive architecture: one small 2-input sub-network
    per feature pair (x_j, x_tilde_j), outputs summed:

        f_j = MLP_j(x_j, x_tilde_j)    [Linear(2,h) -> ReLU -> Linear(h,1)]
        y_hat = sum_j f_j

    Group regularisation targets the two first-layer input columns of each
    sub-network separately, creating direct x_j vs x_tilde_j competition:

        R = lambda * sum_j [ (||W_j[:,0]||^2 + eps)^a      <- x_j channel
                           + (||W_j[:,1]||^2 + eps)^a ]    <- x_tilde_j channel

    Advantages over joint MLP:
      - O(p*h) parameters vs O(2p*H): far fewer, harder to memorise
      - Group regularisation is exact, not a proxy via first-layer columns
      - Model family contains the true additive signal y = sum f(x_j)
      - Competition between x_j and x_tilde_j is local and interpretable

Group regularisation (all models):
    R(g; lambda, a) = lambda * sum_j (||w_j||_2^2 + eps)^a
    a = 1:   ridge-like (no sparsity)
    a = 0.5: Group Lasso (convex, shrinks groups to zero)
    a → 0:   approximates L0 (aggressive competitive exclusion)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AugmentedMLP(nn.Module):
    """
    MLP with 2p-dimensional augmented input and scalar output.

    Parameters
    ----------
    input_dim   : int   -- should be 2*p
    hidden_dims : tuple -- widths of hidden layers, e.g. (64, 32)
    task        : str   -- 'regression' (MSE) or 'classification' (BCE + sigmoid)
    """

    def __init__(self, input_dim: int, hidden_dims=(64, 32), task: str = "regression"):
        super().__init__()
        self.task = task

        dims = [input_dim] + list(hidden_dims) + [1]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x).squeeze(-1)
        if self.task == "classification":
            out = torch.sigmoid(out)
        return out

    def group_regularization(
        self,
        lambda_val: float,
        a: float,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """
        R(g; lambda, a) = lambda * sum_j  (||w_j||_2^2 + eps)^a

        w_j = W[:, j] is the j-th column of the first-layer weight matrix.
        """
        first_layer: nn.Linear = self.net[0]
        W = first_layer.weight                          # (hidden_dim, 2p)
        col_sq_norms = (W ** 2).sum(dim=0)              # (2p,)
        penalty = (col_sq_norms + eps).pow(a).sum()
        return lambda_val * penalty


class PairwiseFilterMLP(nn.Module):
    """
    MLP with pairwise filter layer (DeepPINK / DiffKnock style).

    Input: z = [x, x_tilde] in R^{2p}.
    The filter layer produces p filtered features:

        f_j = alpha_j * x_j + (1 - alpha_j) * x_tilde_j

    where alpha_j = |v_j| / (|v_j| + |v_{j+p}| + eps),
    v_j and v_{j+p} are learnable scalars.

    This forces explicit competition: each f_j is a blend of x_j
    and x_tilde_j weighted by the model's learned preference.

    Antisymmetry (exchange symmetry):
        Swapping columns j and j+p in z exchanges (v_j, v_{j+p}),
        which exchanges alpha_j -> 1-alpha_j, reflecting the signal
        from x_j to x_tilde_j.  The knockoff statistic
            W_j = phi_j(col j) - phi_j(col j+p)
        therefore flips sign.

    Parameters
    ----------
    p           : int   -- number of original features
    hidden_dims : tuple -- MLP widths after the filter layer
    task        : str
    """

    def __init__(self, p: int, hidden_dims=(32,), task: str = "regression"):
        super().__init__()
        self.p    = p
        self.task = task

        # Learnable scalar weights for pairwise filter, shape (2p,)
        self.v = nn.Parameter(torch.randn(2 * p) * 0.1)

        # MLP after filter (input is p-dimensional filtered features)
        dims = [p] + list(hidden_dims) + [1]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.mlp = nn.Sequential(*layers)

    def _filter(self, z: torch.Tensor) -> torch.Tensor:
        """Apply pairwise filter to 2p-dim input, return p-dim filtered output."""
        p   = self.p
        x   = z[:, :p]          # (n, p)
        xt  = z[:, p:]          # (n, p)
        eps = 1e-8
        v_x  = self.v[:p]       # (p,)
        v_xt = self.v[p:]       # (p,)
        denom = v_x.abs() + v_xt.abs() + eps
        alpha = v_x.abs() / denom   # (p,)  -- weight on x_j
        return alpha * x + (1.0 - alpha) * xt   # (n, p)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        f   = self._filter(z)
        out = self.mlp(f).squeeze(-1)
        if self.task == "classification":
            out = torch.sigmoid(out)
        return out

    def group_regularization(
        self,
        lambda_val: float,
        a: float,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """
        Regularise the filter weights v_j and v_{j+p} as scalar groups.
        R = lambda * sum_j (v_j^2 + eps)^a + (v_{j+p}^2 + eps)^a
        """
        v_sq = self.v ** 2                              # (2p,)
        penalty = (v_sq + eps).pow(a).sum()
        return lambda_val * penalty


class AdditiveMLP(nn.Module):
    """
    Feature-wise additive MLP for PRISM.

    For each j = 0..p-1, a small sub-network:
        f_j = ReLU( [x_j, x_tilde_j] @ W_j^T + b_j ) @ v_j + c_j
    Output: y_hat = sum_j f_j  (scalar per sample)

    hidden_dim  : width of each sub-network's hidden layer
                  (pass hidden_dims=(h,) from the PRISM classes)
    """

    def __init__(self, p: int, hidden_dim: int = 8, task: str = "regression"):
        super().__init__()
        self.p          = p
        self.hidden_dim = hidden_dim
        self.task       = task

        h = hidden_dim
        # Batched first-layer weights: W1[j] in R^{h x 2}, b1[j] in R^h
        self.W1 = nn.Parameter(torch.randn(p, h, 2) * 0.1)   # (p, h, 2)
        self.b1 = nn.Parameter(torch.zeros(p, h))              # (p, h)
        # Batched output-layer weights: W2[j] in R^{1 x h}, b2[j] in R^1
        self.W2 = nn.Parameter(torch.randn(p, 1, h) * 0.1)   # (p, 1, h)
        self.b2 = nn.Parameter(torch.zeros(p, 1))              # (p, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z : (n, 2p) — [X, X_tilde] concatenated
        returns : (n,) scalar predictions
        """
        p  = self.p
        x  = z[:, :p]   # (n, p)  original features
        xt = z[:, p:]   # (n, p)  knockoff features

        # inp[i, j, :] = [x_j, x_tilde_j] for sample i
        inp = torch.stack([x, xt], dim=-1)              # (n, p, 2)

        # Hidden layer: contract over input dim i (size 2); W1 is (p, h, 2)
        h = torch.relu(torch.einsum('npi,phi->nph', inp, self.W1) + self.b1)

        # Output layer: contract over h; W2 is (p, 1, h) -> (n, p, 1)
        out = torch.einsum('nph,poh->npo', h, self.W2) + self.b2

        # Sum over features -> (n,)
        out = out.squeeze(-1).sum(dim=1)

        if self.task == "classification":
            out = torch.sigmoid(out)
        return out

    def group_regularization(
        self,
        lambda_val: float,
        a: float,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """
        Separate groups for the x_j and x_tilde_j input channels of each
        sub-network's first layer:

            R = lambda * sum_j [ (||W1_j[:,0]||^2 + eps)^a     (x_j channel)
                               + (||W1_j[:,1]||^2 + eps)^a ]   (x_tilde_j channel)

        W1: (p, h, 2) -> columns W1[:,:,0] and W1[:,:,1], each (p, h)
        """
        col0_sq = (self.W1[:, :, 0] ** 2).sum(dim=1)   # (p,) norms^2 for x_j
        col1_sq = (self.W1[:, :, 1] ** 2).sum(dim=1)   # (p,) norms^2 for x_tilde_j
        penalty = ((col0_sq + eps).pow(a) + (col1_sq + eps).pow(a)).sum()
        return lambda_val * penalty
