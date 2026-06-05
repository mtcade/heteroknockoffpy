#
#//  prism_g.py
#//  heteroknockoffpy
#//
#//  Created by Evan Mason on 6/2/26.
#//
#// Adapted from https://github.com/feizhe/novel_knockoffs/tree/master/PRISM

"""
PRISM-G: Persistent Regularization-Integrated Sensitivity Measure
         (Gradient variant)

Architecture
------------
The model is 2p-dimensional: g_hat : R^{2p} -> R, input = [X, X_tilde].
There is NO data augmentation (swapping orderings). The antisymmetry is
architectural, following GRIP2 Proposition 1.

Antisymmetry proof (architecture-based, no augmentation)
---------------------------------------------------------
Let Z = [X, X_tilde], Z' = swap_j(Z) (columns j and j+p exchanged).
By the exchange symmetry of the regularized objective in (j, j+p):

    g'  = argmin_g  L(g; Z', y) + R(g)   =>   g'(z) = g(swap_j(z))

where g = argmin_g L(g; Z, y) + R(g). Then:

    phi_j(g'; Z') = (1/n) sum_i |g(Z_i + sigma*e_{j+p}) - g(Z_i - sigma*e_{j+p})| / 2sigma
                  = phi_{j+p}(g; Z)

Therefore:

    W_j(Z') = T_j(Z') - T_{j+p}(Z') = T_{j+p}(Z) - T_j(Z) = -W_j(Z)   QED

Why T_j > T_{j+p} for signals
------------------------------
For a signal feature j (X_j -> y), the sparsity-geometry regularisation forces
competition: the model must choose between first-layer weights w_j (for X_j)
and w_{j+p} (for X_tilde_j).  Since X_j is truly predictive while X_tilde_j
is conditionally independent of y given X, the optimizer allocates
||w_j||_2 > ||w_{j+p}||_2.  For a ReLU network, phi_j ≈ c * ||w_j||_2, so
T_j > T_{j+p} = T_tilde_j.  Hence W_j > 0 for signal features.

For null features j: X_j and X_tilde_j carry equal predictive information,
so ||w_j||_2 ≈ ||w_{j+p}||_2 and W_j ≈ 0.

Note on data augmentation
-------------------------
Symmetric training (both orderings) would force phi_j = phi_{j+p} at every
BSS snapshot, giving W_j = 0 always.  This is why data augmentation is
WRONG for the 2p model — it prevents the oracle asymmetry from emerging.
"""

from .model import AugmentedMLP, PairwiseFilterMLP, AdditiveMLP

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from collections.abc import Sequence
from typing import Self

# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────

def _cycle(loader: DataLoader):
    """Yield batches from a DataLoader indefinitely."""
    while True:
        yield from loader
    #
#/def _cycle


# ──────────────────────────────────────────────────────────────────────────────
# Type-specific gradient operator  (continuous, central difference)
# ──────────────────────────────────────────────────────────────────────────────

def _phi_j(
    model: AugmentedMLP,
    Z: torch.Tensor,
    j: int,
    sigma_j: float,
    ) -> float:
    """
    Central-difference output sensitivity for column j of Z.

    Parameters
    ----------
    model   : AugmentedMLP (eval mode, 2p-dim input)
    Z       : FloatTensor (n, 2p)  [X, X_tilde] concatenated
    j       : column index in Z (0..2p-1)
    sigma_j : bandwidth = std of Z[:, j]
    """
    Z_plus  = Z.clone(); Z_plus[:, j]  = Z[:, j] + sigma_j
    Z_minus = Z.clone(); Z_minus[:, j] = Z[:, j] - sigma_j
    with torch.no_grad():
        diff = model(Z_plus) - model(Z_minus)
    return (diff.abs() / (2.0 * sigma_j)).mean().item()
#/def _phi_j

# ──────────────────────────────────────────────────────────────────────────────
# PRISM-G estimator
# ──────────────────────────────────────────────────────────────────────────────

class PRISM_G:
    """
    PRISM-G knockoff importance statistic.

    Fits a 2p-dimensional MLP on [X, X_tilde] with Block Stochastic Sampling
    (BSS) over a 2D (lambda, a) regularisation surface, then computes:

        T_j       = (1/B) sum_b  phi_j    ( g^(b), Z )   for original col j
        T_tilde_j = (1/B) sum_b  phi_{j+p}( g^(b), Z )   for knockoff  col j+p
        W_j       = T_j - T_tilde_j

    The antisymmetry W_j(swap_j(Z)) = -W_j(Z) follows from the exchange
    symmetry of the regularised objective in (j, j+p), without any data
    augmentation.

    Parameters
    ----------
    hidden_dims           : tuple  -- MLP hidden layer widths
    model_type            : str    -- 'mlp' (AugmentedMLP, 2p input) or
                                      'pairwise' (PairwiseFilterMLP, enforces
                                      explicit per-feature competition)
    M                     : int    -- minibatch steps per block
    n_warmup              : int    -- maximum warm-up steps (hard cap)
    warmup_patience       : int    -- stop warmup early if eval loss does not
                                      improve by warmup_tol for this many
                                      consecutive checks; 0 = fixed warmup
    warmup_check_interval : int    -- steps between eval-loss checks during warmup
    warmup_tol            : float  -- minimum absolute improvement to reset patience
    warmup_val_frac       : float  -- fraction of data held out as a validation set
                                      during warmup; patience is applied to VAL loss
                                      so that early stopping is robust to memorisation
                                      of the training set (set 0 to use train loss)
    warmup_weight_decay   : float  -- L2 weight-decay applied only during warmup
    reset_optimizer       : bool   -- if True, reset Adam moments at the start of
                                      each BSS block so each block's M steps follow
                                      a clean gradient direction
    lambda_path  : Sequence[float] | None -- explicit regularisation path; number of
                                            blocks = len(lambda_path).  Default:
                                            np.logspace(log10(0.1), log10(1e-3), 30)
    a_path       : Sequence[float] | None -- per-block input-layer penalty values;
                                            None → use lambda_path values (mirrors
                                            torchNetworks convention)
    lr           : float  -- Adam learning rate
    batch_size   : int
    task         : 'regression' or 'classification'
    device       : str
    verbose      : bool
    """

    def __init__(
        self,
        hidden_dims: tuple = (32,),
        model_type: str = "pairwise",
        M: int = 50,
        n_warmup: int = 5000,
        warmup_patience: int = 20,
        warmup_check_interval: int = 50,
        warmup_tol: float = 1e-4,
        warmup_val_frac: float = 0.2,
        warmup_weight_decay: float = 1e-4,
        reset_optimizer: bool = True,
        lambda_path: Sequence[float] | None = None,
        a_path: Sequence[float] | None = None,
        lr: float = 3e-3,
        batch_size: int = 256,
        task: str = "regression",
        device: str = "cpu",
        verbose: bool = False,
        ) -> None:
        
        self.hidden_dims = hidden_dims
        self.model_type = model_type
        self.M = M
        self.n_warmup = n_warmup
        self.warmup_patience = warmup_patience
        self.warmup_check_interval = warmup_check_interval
        self.warmup_tol = warmup_tol
        self.warmup_val_frac = warmup_val_frac
        self.warmup_weight_decay = warmup_weight_decay
        self.reset_optimizer = reset_optimizer
        self.lambda_path = lambda_path
        self.a_path = a_path
        self.lr = lr
        self.batch_size = batch_size
        self.task = task
        self.device = device
        self.verbose = verbose
    #/def __init__
    
    # ── public ────────────────────────────────────────────────────────────

    def fit(
        self,
        X: np.ndarray,
        X_tilde: np.ndarray,
        y: np.ndarray,
        random_state=None,
        ) -> Self:
        """
        Run BSS and compute knockoff statistics W_j = T_j - T_tilde_j.

        Parameters
        ----------
        X       : ndarray (n, p)
        X_tilde : ndarray (n, p)
        y       : ndarray (n,)

        Sets
        ----
        self.W_        : ndarray (p,)
        self.T_orig_   : ndarray (p,)
        self.T_knock_  : ndarray (p,)
        """
        if random_state is not None:
            torch.manual_seed(random_state)

        _lp = (list(self.lambda_path) if self.lambda_path is not None
               else list(np.logspace(np.log10(0.1), np.log10(1e-3), 30)))
        _ap = list(self.a_path) if self.a_path is not None else _lp
        B   = len(_lp)

        n, p = X.shape

        # ── Augmented input: Z = [X, X_tilde], NO swap augmentation ──────
        # The antisymmetry is architectural; symmetric training would zero W_j.
        Z_np = np.concatenate([X, X_tilde], axis=1)           # (n, 2p)

        # ── Standardise each column to unit variance, zero mean ──────────
        # Rationale: central-difference bandwidth sigma_j := sd(Z_j) couples
        # the evaluation step to feature scale.  On heterogeneous scales
        # (see results/new_p50_k10/sim_new_3_2_scale.txt) this pushed
        # g(X_j ± sd) past the activation saturation range of the MLP,
        # zeroing phi_j on large-variance signals.  After standardisation
        # every column has sd=1, so the central difference evaluates at
        # +/- 1 across all features, well inside the trained activation
        # range.  Column-wise statistics preserve the joint (X, X_tilde)
        # exchange symmetry: each X_j and X_tilde_j column is scaled
        # independently by its own (approximately identical) sample sd.
        self._mu_ = Z_np.mean(axis=0)                           # (2p,)
        self._sd_ = np.maximum(Z_np.std(axis=0), 1e-8)          # (2p,)
        Z_np = (Z_np - self._mu_) / self._sd_

        Z_tr = torch.tensor(Z_np, dtype=torch.float32, device=self.device)
        y_tr = torch.tensor(y,    dtype=torch.float32, device=self.device)

        Z_ev = Z_tr  # evaluation tensor = same as training tensor

        # Bandwidths for phi_j (now ~1 everywhere after standardisation)
        sigma_all = np.maximum(Z_np.std(axis=0), 1e-8)         # (2p,)

        # ── Model and optimisers ──────────────────────────────────────────
        if self.model_type == "pairwise":
            model = PairwiseFilterMLP(p, self.hidden_dims, task=self.task).to(self.device)
        elif self.model_type == "additive":
            model = AdditiveMLP(p, self.hidden_dims[0], task=self.task).to(self.device)
        else:
            model = AugmentedMLP(2 * p, self.hidden_dims, task=self.task).to(self.device)
        #/switch self.model_type
        
        # Warmup uses weight-decay to prevent memorisation; BSS blocks use no decay
        # so the group regularisation penalty drives all feature selection pressure.
        warmup_opt = Adam(model.parameters(), lr=self.lr,
                          weight_decay=self.warmup_weight_decay)
        optimiser  = Adam(model.parameters(), lr=self.lr)
        loss_fn    = nn.MSELoss() if self.task == "regression" else nn.BCELoss()

        loader = DataLoader(
            TensorDataset(Z_tr, y_tr),
            batch_size=self.batch_size,
            shuffle=True,
        )

        # ── Warm-up: train to near-convergence without regularisation ─────
        # A validation split (warmup_val_frac) is held out so that the patience
        # criterion tracks val loss rather than train loss, preventing the loop
        # from stopping at memorisation (train loss → 0) instead of the true
        # noise floor.  BSS afterwards uses the full loader on all n samples.
        warmup_steps   = 0
        patience_count = 0
        if self.n_warmup > 0:
            # ── build train / val split for warmup ────────────────────────
            if self.warmup_val_frac > 0 and self.warmup_patience > 0:
                gen     = torch.Generator().manual_seed(
                              random_state if random_state is not None else 0)
                perm    = torch.randperm(n, generator=gen)
                n_val   = max(1, int(n * self.warmup_val_frac))
                val_idx = perm[:n_val]
                trn_idx = perm[n_val:]
                Z_trn_w = Z_tr[trn_idx];  y_trn_w = y_tr[trn_idx]
                Z_val_w = Z_tr[val_idx];  y_val_w = y_tr[val_idx]
                warm_loader = DataLoader(
                    TensorDataset(Z_trn_w, y_trn_w),
                    batch_size=self.batch_size, shuffle=True,
                )
                def _check_loss(m): return self._eval_loss(m, Z_val_w, y_val_w, loss_fn)
            else:
                warm_loader = loader
                def _check_loss(m): return self._eval_loss(m, Z_ev, y_tr, loss_fn)

            best_loss = float("inf")
            model.train()
            for Zb, yb in _cycle(warm_loader):
                warmup_opt.zero_grad()
                loss_fn(model(Zb), yb).backward()
                warmup_opt.step()
                warmup_steps += 1

                if (
                    self.warmup_patience > 0
                    and warmup_steps % self.warmup_check_interval == 0
                ):
                    wl = _check_loss(model)
                    if wl < best_loss - self.warmup_tol:
                        best_loss      = wl
                        patience_count = 0
                    else:
                        patience_count += 1
                    #/if wl < best_loss - self.warmup_tol
                    
                    if patience_count >= self.warmup_patience:
                        break
                    #
                #/if ...

                if warmup_steps >= self.n_warmup:
                    break
                #
            #/for Zb, yb in _cycle(warm_loader)

            if self.verbose:
                tr_loss = self._eval_loss(model, Z_ev, y_tr, loss_fn)
                vl_loss = _check_loss(model)
                status  = ("converged" if patience_count >= self.warmup_patience
                           else "max steps")
                print(f"  warm-up: {warmup_steps} steps [{status}]"
                      f"  train={tr_loss:.4f}  val={vl_loss:.4f}")
            #/if self.verbose
        #/if self.n_warmup > 0

        # ── BSS loop ──────────────────────────────────────────────────────
        T      = np.zeros(p)   # col j:   phi_j (original)
        T_tild = np.zeros(p)   # col j+p: phi_{j+p} (knockoff)

        for b, (lambda_b, a_b) in enumerate(zip(_lp, _ap)):

            # Fresh Adam moments each block: each block's M steps follow the
            # gradient of its own (lambda_b, a_b) objective without momentum
            # contamination from the previous block's regularisation regime.
            if self.reset_optimizer:
                optimiser = Adam(model.parameters(), lr=self.lr)

            model.train()
            for steps, (Zb, yb) in enumerate(_cycle(loader)):
                if steps >= self.M:
                    break
                optimiser.zero_grad()
                pred = model(Zb)
                loss = loss_fn(pred, yb) + model.group_regularization(lambda_b, a_b)
                loss.backward()
                optimiser.step()

            model.eval()
            for j in range(p):
                T[j]      += _phi_j(model, Z_ev, j,     sigma_all[j])     / B
                T_tild[j] += _phi_j(model, Z_ev, j + p, sigma_all[j + p]) / B

            if self.verbose:
                el = self._eval_loss(model, Z_ev, y_tr, loss_fn)
                print(
                    f"  block {b + 1:3d}/{B}"
                    f"  lambda={lambda_b:.4f}  a={a_b:.3f}"
                    f"  eval_loss={el:.4f}"
                )
            #/if self.verbose
        #/for b, (lambda_b, a_b) in enumerate(zip(_lp, _ap))

        self.T_orig_  = T
        self.T_knock_ = T_tild
        self.W_       = T - T_tild
        self.p_       = p
        return self
    #/def fit
    
    # ── helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _eval_loss(
        model: AugmentedMLP,
        Z: torch.Tensor,
        y: torch.Tensor,
        loss_fn: nn.Module,
        ) -> float:
        model.eval()
        with torch.no_grad():
            return loss_fn(model(Z), y).item()
        #
    #/def _eval_loss
#/class PRISM_G:
