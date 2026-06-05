#
#//  prism_w.py
#//  heteroknockoffpy
#//
#//  Created by Evan Mason on 6/2/26.
#//
#// Adapted from https://github.com/feizhe/novel_knockoffs/tree/master/PRISM

"""
PRISM-W: Persistent Regularization-Integrated Sensitivity Measure
         (Weight-norm variant)

Architecture
------------
Uses the same 2p-dimensional model as PRISM-G but replaces the
gradient-based sensitivity phi_j with first-layer column norms:

    AugmentedMLP:       T_j = (1/B) sum_b  ||W^(1)[:, j]||_2
    PairwiseFilterMLP:  T_j = (1/B) sum_b  |v_j|

where W^(1) is the first linear layer weight matrix (shape: H x 2p)
and v is the pairwise filter's scalar weight vector (shape: 2p).

Antisymmetry
------------
The same exchange-symmetry argument from PRISM-G applies:
swapping columns j and j+p in the input Z swaps ||w_j|| and ||w_{j+p}||
(resp. |v_j| and |v_{j+p}|), so W_j -> -W_j.

Why T_j > T_{j+p} for signals
------------------------------
Under group regularisation the optimizer must allocate column-norm
budget competitively.  For a true signal j, the gradient signal
drives ||w_j|| > ||w_{j+p}||, so W_j > 0.  For nulls the two norms
are exchangeable and W_j ≈ 0.

Note
----
PRISM-W is cheaper than PRISM-G (no finite-difference evaluations)
at the cost of a looser connection to the model's actual sensitivity.
"""

from .model import AugmentedMLP, PairwiseFilterMLP, AdditiveMLP

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from collections.abc import Sequence
from typing import Self

def _cycle(loader: DataLoader):
    while True:
        yield from loader
    #/while True
#/def _cycle


def _col_norms_augmented(model: AugmentedMLP, p: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (T_j, T_tilde_j) for AugmentedMLP: first-layer column L2 norms."""
    W = model.net[0].weight.detach().cpu().numpy()   # (H, 2p)
    norms = np.linalg.norm(W, axis=0)               # (2p,)
    return norms[:p], norms[p:]
#/def _col_norms_augmented

def _col_norms_additive(model: AdditiveMLP, p: int) -> tuple[np.ndarray, np.ndarray]:
    """
    T_j  = ||W1_j[:,0]||_2  (first-layer weight column for x_j in sub-network j)
    T~_j = ||W1_j[:,1]||_2  (first-layer weight column for x_tilde_j)
    W1 : (p, h, 2) -> columns W1[:,:,0] and W1[:,:,1], each (p, h)
    """
    W1 = model.W1.detach().cpu().numpy()            # (p, h, 2)
    T  = np.linalg.norm(W1[:, :, 0], axis=1)       # (p,)
    Tt = np.linalg.norm(W1[:, :, 1], axis=1)       # (p,)
    return T, Tt
#/def _col_norms_additive


def _col_norms_pairwise(model: PairwiseFilterMLP, p: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (T_j, T_tilde_j) for PairwiseFilterMLP: absolute filter weights |v|."""
    v = model.v.detach().cpu().numpy()               # (2p,)
    v_abs = np.abs(v)
    return v_abs[:p], v_abs[p:]
#/def _col_norms_pairwise

class PRISM_W:
    """
    PRISM-W knockoff importance statistic (weight-norm variant).

    Fits a 2p-dimensional MLP on [X, X_tilde] with Block Stochastic
    Sampling (BSS) over a 2D (lambda, a) regularisation surface, then:

        T_j       = (1/B) sum_b  ||w_j^(b)||_2    (original  col j)
        T_tilde_j = (1/B) sum_b  ||w_{j+p}^(b)||_2 (knockoff col j+p)
        W_j       = T_j - T_tilde_j

    For PairwiseFilterMLP the L2 norm is replaced by |v_j|.

    Parameters
    ----------
    hidden_dims           : tuple  -- MLP hidden layer widths
    model_type            : str    -- 'mlp' (AugmentedMLP) or 'pairwise' (PairwiseFilterMLP)
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
    #/init
    
    def fit(
        self,
        X: np.ndarray,
        X_tilde: np.ndarray,
        y: np.ndarray,
        random_state=None,
        ) -> Self:
        """
        Run BSS and compute W_j = T_j - T_tilde_j via weight norms.

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

        Z_np = np.concatenate([X, X_tilde], axis=1)           # (n, 2p)

        # ── Standardise each column (parity with PRISM-G) ────────────────
        # Puts all first-layer input columns on the same scale so that
        # ||w_j|| is directly comparable across features.  On
        # homogeneous-scale designs this is a no-op; on heterogeneous
        # scales it keeps PRISM-W from under-weighting large-variance
        # features the way PRISM-G was under-weighting them before this
        # fix.  Column-wise stats preserve (X, X_tilde) exchange symmetry.
        self._mu_ = Z_np.mean(axis=0)                           # (2p,)
        self._sd_ = np.maximum(Z_np.std(axis=0), 1e-8)          # (2p,)
        Z_np = (Z_np - self._mu_) / self._sd_

        Z_tr = torch.tensor(Z_np, dtype=torch.float32, device=self.device)
        y_tr = torch.tensor(y,    dtype=torch.float32, device=self.device)

        # ── Model and optimisers ──────────────────────────────────────────
        if self.model_type == "pairwise":
            model = PairwiseFilterMLP(p, self.hidden_dims, task=self.task).to(self.device)
            _get_norms = lambda m: _col_norms_pairwise(m, p)
        elif self.model_type == "additive":
            model = AdditiveMLP(p, self.hidden_dims[0], task=self.task).to(self.device)
            _get_norms = lambda m: _col_norms_additive(m, p)
        else:
            model = AugmentedMLP(2 * p, self.hidden_dims, task=self.task).to(self.device)
            _get_norms = lambda m: _col_norms_augmented(m, p)
        #/switch self.model_type
        
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
                def _check_loss(m):
                    return self._eval_loss(m, Z_val_w, y_val_w, loss_fn)
                #/def _check_loss(m)
            #
            else:
                warm_loader = loader
                def _check_loss(m):
                    return self._eval_loss(m, Z_tr, y_tr, loss_fn)
                #/def _check_loss(m)
            #/if self.warmup_val_frac > 0 and self.warmup_patience > 0

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
                    #/if patience_count >= self.warmup_patience
                #
                if warmup_steps >= self.n_warmup:
                    break
                #
            #/for Zb, yb in _cycle(warm_loader)
            if self.verbose:
                tr_loss = self._eval_loss(model, Z_tr, y_tr, loss_fn)
                vl_loss = _check_loss(model)
                status  = ("converged" if patience_count >= self.warmup_patience
                           else "max steps")
                print(f"  warm-up: {warmup_steps} steps [{status}]"
                      f"  train={tr_loss:.4f}  val={vl_loss:.4f}")
            #/if self.verbose
        #/if self.n_warmup > 0
        
        # ── BSS loop ──────────────────────────────────────────────────────
        T      = np.zeros(p)
        T_tild = np.zeros(p)

        for b, (lambda_b, a_b) in enumerate(zip(_lp, _ap)):

            # Fresh Adam moments each block so each block's M steps follow the
            # gradient of its own (lambda_b, a_b) objective without momentum
            # contamination from the previous block's regularisation regime.
            if self.reset_optimizer:
                optimiser = Adam(model.parameters(), lr=self.lr)
            #/if self.reset_optimizer
            
            model.train()
            for steps, (Zb, yb) in enumerate(_cycle(loader)):
                if steps >= self.M:
                    break
                #
                optimiser.zero_grad()
                pred = model(Zb)
                loss = loss_fn(pred, yb) + model.group_regularization(lambda_b, a_b)
                loss.backward()
                optimiser.step()
            #/for steps, (Zb, yb) in enumerate(_cycle(loader))

            model.eval()
            t_j, tt_j = _get_norms(model)
            T      += t_j  / B
            T_tild += tt_j / B

            if self.verbose:
                el = self._eval_loss(model, Z_tr, y_tr, loss_fn)
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
    
    @staticmethod
    def _eval_loss(model, Z, y, loss_fn):
        model.eval()
        with torch.no_grad():
            return loss_fn(model(Z), y).item()
        #
    #/def _eval_loss
#/class PRISM_W
