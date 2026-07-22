#
#//  torchImportances.py
#//  heteroknockoffpy
#//
#//  Created by Evan Mason on 6/1/26.
#//

import torch
import torch.nn as nn
import torch.optim as optim
from torch.func import vmap, jacrev
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
import sys
from abc import ABC, abstractmethod
from tqdm import tqdm
from typing import Callable, Iterable, Literal, Sequence, Self, Type

from .torchUtil import _nnModule_dict, _build_sequential


def _prism_cycle(loader: DataLoader):
    while True:
        yield from loader
#/def _prism_cycle


class _SqueezeLast(nn.Module):
    """
    Squeezes a module's trailing size-1 output dim, mirroring the
    `out.squeeze(-1) if self.output_size == 1 else out` convention every
    `_PRISMNetworkBase.forward()` uses -- `build_prefit_module()` composes
    existing layers directly (bypassing that `forward()`), so this needs to be
    appended explicitly wherever build_prefit_module's raw output would
    otherwise stay (n, 1) while the loss target is (n,), silently broadcasting
    into an (n, n) pairwise comparison instead of an elementwise one.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.squeeze(-1)
    #/def forward
#/class _SqueezeLast


def _weighted_loss(
    loss_func: nn.Module,
    pred: torch.Tensor,
    y: torch.Tensor,
    weight: torch.Tensor | None,
) -> torch.Tensor:
    """
    Compute loss_func(pred, y), weighted by a per-sample weight if given.

    loss_func is an arbitrary already-constructed nn loss module (MSELoss,
    CrossEntropyLoss, PoissonNLLLoss, ...) with its own default reduction
    (almost always 'mean'). To weight it without needing every caller to
    know/rebuild the specific loss class with reduction='none', this
    temporarily flips loss_func.reduction to 'none' (standard torch loss
    modules all expose this attribute), takes the per-sample loss, reduces
    any non-batch dims, then computes a weighted average -- equivalent to
    the unweighted loss_func(pred, y) when weight is None or all-ones.
    """
    if weight is None:
        return loss_func( pred, y )
    if not hasattr( loss_func, 'reduction' ):
        raise ValueError(
            f"_weighted_loss: loss_func of type {type(loss_func).__name__} has no "
            "'reduction' attribute -- can't compute a per-sample loss to weight."
        )
    orig_reduction = loss_func.reduction
    loss_func.reduction = 'none'
    try:
        per_sample = loss_func( pred, y )
    finally:
        loss_func.reduction = orig_reduction
    if per_sample.dim() > 1:
        per_sample = per_sample.mean( dim=tuple( range( 1, per_sample.dim() ) ) )
    return ( per_sample * weight ).sum() / weight.sum()
#/def _weighted_loss


# -- Network architectures

class _PRISMNetworkBase(nn.Module, ABC):
    """
    Shared interface for all PRISM network architectures.

    Subclasses implement forward / _precompute_group_reg / group_regularization /
    get_group_importances on their own discrimination-layer tensors (they differ
    per architecture, so no shared concrete body is provided for these).
    """
    @abstractmethod
    def forward(self, z: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def _precompute_group_reg(self, groups: list[list[int]], device: str) -> None: ...

    @abstractmethod
    def group_regularization(
        self,
        lambda_val: float,
        a: float,
        groups: list[list[int]] | None = None,
        eps: float = 1e-8,
    ) -> torch.Tensor: ...

    @abstractmethod
    def get_group_importances(self, groups: list[list[int]]) -> np.ndarray: ...

    @abstractmethod
    def build_prefit_module(self) -> nn.Module:
        """
        Build a single-sided (p-wide, not 2p-wide) module for `vertical_prefit`
        training: no swap/discrimination structure, just "some row's features" ->
        prediction. Everything downstream of the swap layer (later net layers, mlp,
        b1/W2/b2/activation, the tied `combine` module for NU variants) is shared by
        reference, so training this module's parameters trains the real model's
        shared parameters in place.
        """
        ...

    @abstractmethod
    def transfer_from_prefit(self, prefit_module: nn.Module, noise_std: float = 0.0) -> None:
        """
        After `vertical_prefit` training, install the pretrained single-sided
        discrimination weights into the real (paired) swap/discrimination parameter
        -- duplicated onto both the X and Xk halves for MLP-family/Additive, or a
        symmetric `v = 0.5` for Pairwise-family (which had no discrimination
        parameter active during pre-fit at all). An exact symmetric start is a
        degenerate saddle point (both halves compute an identical function, so
        there's no immediate gradient signal to prefer one over the other); if
        `noise_std > 0`, independent N(0, noise_std) antisymmetric noise is added
        to break the tie -- +noise on the X half, -noise on the Xk half (or
        equivalently on `v`'s two halves) -- while keeping the two halves' *expected*
        values equal.
        """
        ...

    def no_decay_parameters(self) -> list[nn.Parameter]:
        """Parameters that should never receive weight_decay (e.g. warmup). Empty by default."""
        return []
    #/def no_decay_parameters
#/class _PRISMNetworkBase


class _PRISMNetworkMLP(_PRISMNetworkBase):
    """
    Flat MLP on 2p-dimensional augmented input [X, Xk].
    Group regularisation: differentiable block-Frobenius penalty over each group.
    Supports OHE groups (multi-column) and multi-class output.
    """
    def __init__(
        self,
        input_size: int,
        layers: Sequence[int],
        activation_class: Type[nn.Module],
        output_size: int = 1,
    ) -> None:
        super().__init__()
        self.output_size = output_size
        dims = [input_size] + list(layers) + [output_size]
        parts: list[nn.Module] = []
        for i in range(len(dims) - 1):
            parts.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                parts.append(activation_class())
        self.net = nn.Sequential(*parts)
    #/def __init__

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        return out.squeeze(-1) if self.output_size == 1 else out
    #/def forward

    def _precompute_group_reg(self, groups: list[list[int]], device: str) -> None:
        active = [g for g in groups if g]
        input_size = self.net[0].weight.shape[1]
        col_to_group = torch.zeros(input_size, dtype=torch.long, device=device)
        for gidx, g in enumerate(active):
            for c in g:
                col_to_group[c] = gidx
        self._col_to_group = col_to_group
        self._n_groups = len(active)
    #/def _precompute_group_reg

    def group_regularization(
        self,
        lambda_val: float,
        a: float,
        groups: list[list[int]] | None = None,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        w      = self.net[0].weight                                      # (H, 2p)
        col_sq = w.pow(2).sum(dim=0)                                     # (2p,)
        grp_sq = col_sq.new_zeros(self._n_groups).scatter_add(0, self._col_to_group, col_sq)
        return lambda_val * (grp_sq + eps).pow(a).sum()
    #/def group_regularization

    def get_group_importances(self, groups: list[list[int]]) -> np.ndarray:
        with torch.no_grad():
            w = self.net[0].weight.detach().cpu()
            return np.array([
                torch.norm(w[:, g]).item() if g else 0.0
                for g in groups
            ])
    #/def get_group_importances

    def build_prefit_module(self) -> nn.Module:
        hidden, input_size = self.net[0].weight.shape
        p_ohe = input_size // 2
        prefit_first = nn.Linear(p_ohe, hidden)
        modules = [prefit_first, *list(self.net.children())[1:]]
        if self.output_size == 1:
            modules.append(_SqueezeLast())
        #
        return nn.Sequential(*modules)
    #/def build_prefit_module

    def transfer_from_prefit(self, prefit_module: nn.Module, noise_std: float = 0.0) -> None:
        prefit_first = prefit_module[0]
        p_ohe = prefit_first.weight.shape[1]
        with torch.no_grad():
            noise = torch.randn_like(prefit_first.weight) * noise_std
            self.net[0].weight[:, :p_ohe].copy_(prefit_first.weight + noise)
            self.net[0].weight[:, p_ohe:].copy_(prefit_first.weight - noise)
            self.net[0].bias.copy_(prefit_first.bias)
    #/def transfer_from_prefit
#/class _PRISMNetworkMLP


class _PRISMNetworkPairwise(_PRISMNetworkBase):
    """
    DeepPINK-style pairwise filter for p feature positions, followed by an MLP.
    Input: z of shape (n, 2p). Each OHE column at position j competes with its
    knockoff at position j+p via learnable scalar filter weights v.

    Group regularisation uses block-Frobenius on v[g] per group, so OHE columns
    of the same original variable are regularised jointly.
    """
    def __init__(
        self,
        p: int,
        layers: Sequence[int],
        activation_class: Type[nn.Module],
        output_size: int = 1,
    ) -> None:
        super().__init__()
        self.p = p
        self.output_size = output_size
        self.v = nn.Parameter(torch.randn(2 * p) * 0.1)
        dims = [p] + list(layers) + [output_size]
        parts: list[nn.Module] = []
        for i in range(len(dims) - 1):
            parts.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                parts.append(activation_class())
        self.mlp = nn.Sequential(*parts)
    #/def __init__

    def _filter(self, z: torch.Tensor) -> torch.Tensor:
        p = self.p
        x, xt = z[:, :p], z[:, p:]
        v_x, v_xt = self.v[:p], self.v[p:]
        alpha = v_x.abs() / (v_x.abs() + v_xt.abs() + 1e-8)
        return alpha * x + (1.0 - alpha) * xt
    #/def _filter

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        out = self.mlp(self._filter(z))
        return out.squeeze(-1) if self.output_size == 1 else out
    #/def forward

    def _precompute_group_reg(self, groups: list[list[int]], device: str) -> None:
        active = [g for g in groups if g]
        input_size = len(self.v)
        col_to_group = torch.zeros(input_size, dtype=torch.long, device=device)
        for gidx, g in enumerate(active):
            for c in g:
                col_to_group[c] = gidx
        self._col_to_group = col_to_group
        self._n_groups = len(active)
    #/def _precompute_group_reg

    def group_regularization(
        self,
        lambda_val: float,
        a: float,
        groups: list[list[int]] | None = None,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        v_sq   = self.v.pow(2)                                           # (2p,)
        grp_sq = v_sq.new_zeros(self._n_groups).scatter_add(0, self._col_to_group, v_sq)
        return lambda_val * (grp_sq + eps).pow(a).sum()
    #/def group_regularization

    def get_group_importances(self, groups: list[list[int]]) -> np.ndarray:
        with torch.no_grad():
            v = self.v.detach().cpu()
            return np.array([v[g].norm().item() if g else 0.0 for g in groups])
    #/def get_group_importances

    def build_prefit_module(self) -> nn.Module:
        # self.mlp is already sized for single-sided (post-filter, p-wide) input --
        # pre-fit trains it directly, in place, with no swap/filter step at all.
        # Wrapping (rather than mutating) self.mlp keeps it shared by reference
        # for the real model's forward(), which applies its own squeeze separately.
        if self.output_size == 1:
            return nn.Sequential(self.mlp, _SqueezeLast())
        #
        return self.mlp
    #/def build_prefit_module

    def transfer_from_prefit(self, prefit_module: nn.Module, noise_std: float = 0.0) -> None:
        # prefit_module IS self.mlp (already trained in place); only self.v never
        # participated in pre-fit, so it gets a fresh (anti)symmetric start.
        with torch.no_grad():
            noise = torch.randn(self.p, device=self.v.device) * noise_std
            self.v[:self.p].copy_(0.5 + noise)
            self.v[self.p:].copy_(0.5 - noise)
    #/def transfer_from_prefit
#/class _PRISMNetworkPairwise


class _PRISMNetworkAdditive(_PRISMNetworkBase):
    """
    Feature-wise additive MLP: one 2-input sub-network per input position, outputs summed.
    Input: z of shape (n, 2p) where the first p columns are X and the last p are Xk.
    Sub-network j handles (z[:, j], z[:, j+p]).

    For OHE variables spanning multiple positions, the block-Frobenius regularisation
    jointly penalises all sub-networks belonging to the same original variable, and
    get_group_importances returns the Frobenius norm of the block.

    Only layers[0] is used as the sub-network hidden dim.
    """
    def __init__(
        self,
        p: int,
        layers: Sequence[int],
        activation_class: Type[nn.Module],
        output_size: int = 1,
    ) -> None:
        super().__init__()
        self.p = p
        self.output_size = output_size
        h = layers[0]
        self.activation = activation_class()
        self.W1 = nn.Parameter(torch.randn(p, h, 2) * 0.1)        # (p, h, 2)
        self.b1 = nn.Parameter(torch.zeros(p, h))
        self.W2 = nn.Parameter(torch.randn(p, output_size, h) * 0.1)  # (p, k, h)
        self.b2 = nn.Parameter(torch.zeros(p, output_size))
    #/def __init__

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        p = self.p
        x, xt = z[:, :p], z[:, p:]
        inp = torch.stack([x, xt], dim=-1)                                    # (n, p, 2)
        h   = self.activation(
            torch.einsum('npi,phi->nph', inp, self.W1) + self.b1
        )                                                                      # (n, p, hidden)
        out = torch.einsum('nph,poh->npo', h, self.W2) + self.b2             # (n, p, output_size)
        out = out.sum(dim=1)                                                   # (n, output_size)
        return out.squeeze(-1) if self.output_size == 1 else out
    #/def forward

    def _precompute_group_reg(self, groups: list[list[int]], device: str) -> None:
        p = self.p
        x_groups  = [(i, g) for i, g in enumerate(groups) if g and min(g) < p]
        xk_groups = [(i, g) for i, g in enumerate(groups) if g and min(g) >= p]

        x_col_to_group  = torch.zeros(p, dtype=torch.long, device=device)
        xk_col_to_group = torch.zeros(p, dtype=torch.long, device=device)
        for new_idx, (_, g) in enumerate(x_groups):
            for c in g:
                x_col_to_group[c] = new_idx
        for new_idx, (_, g) in enumerate(xk_groups):
            for c in g:
                xk_col_to_group[c - p] = new_idx

        self._x_col_to_group  = x_col_to_group
        self._xk_col_to_group = xk_col_to_group
        self._n_x_groups      = len(x_groups)
        self._n_xk_groups     = len(xk_groups)
    #/def _precompute_group_reg

    def group_regularization(
        self,
        lambda_val: float,
        a: float,
        groups: list[list[int]] | None = None,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        col_sq_x  = self.W1[:, :, 0].pow(2).sum(dim=1)                  # (p,)
        col_sq_xk = self.W1[:, :, 1].pow(2).sum(dim=1)                  # (p,)
        grp_x  = col_sq_x.new_zeros(self._n_x_groups).scatter_add(0, self._x_col_to_group,  col_sq_x)
        grp_xk = col_sq_xk.new_zeros(self._n_xk_groups).scatter_add(0, self._xk_col_to_group, col_sq_xk)
        return lambda_val * ((grp_x + eps).pow(a).sum() + (grp_xk + eps).pow(a).sum())
    #/def group_regularization

    def get_group_importances(self, groups: list[list[int]]) -> np.ndarray:
        """
        Frobenius norm of the W1 block for each group.
        X groups → W1[g, :, 0]; Xk groups → W1[g-p, :, 1].
        """
        p = self.p
        with torch.no_grad():
            W1 = self.W1.detach().cpu()
            result = []
            for g in groups:
                if not g:
                    result.append(0.0)
                elif min(g) >= p:
                    g_adj = [c - p for c in g]
                    result.append(W1[g_adj, :, 1].norm().item())
                else:
                    result.append(W1[g, :, 0].norm().item())
            return np.array(result)
    #/def get_group_importances

    def build_prefit_module(self) -> nn.Module:
        p, h = self.W1.shape[0], self.W1.shape[1]
        prefit_W1 = nn.Parameter(torch.randn(p, h) * 0.1)
        activation, b1, W2, b2, output_size = self.activation, self.b1, self.W2, self.b2, self.output_size

        class _AdditivePrefit(nn.Module):
            def __init__(self_) -> None:
                super().__init__()
                self_.W1 = prefit_W1
                self_.activation = activation
                self_.b1 = b1
                self_.W2 = W2
                self_.b2 = b2

            def forward(self_, x_single: torch.Tensor) -> torch.Tensor:
                h_ = self_.activation(
                    torch.einsum('np,ph->nph', x_single, self_.W1) + self_.b1
                )
                out_ = torch.einsum('nph,poh->npo', h_, self_.W2) + self_.b2
                out_ = out_.sum(dim=1)
                return out_.squeeze(-1) if output_size == 1 else out_
            #/def forward
        #/class _AdditivePrefit

        return _AdditivePrefit()
    #/def build_prefit_module

    def transfer_from_prefit(self, prefit_module: nn.Module, noise_std: float = 0.0) -> None:
        with torch.no_grad():
            noise = torch.randn_like(prefit_module.W1) * noise_std
            self.W1[:, :, 0].copy_(prefit_module.W1 + noise)
            self.W1[:, :, 1].copy_(prefit_module.W1 - noise)
    #/def transfer_from_prefit
#/class _PRISMNetworkAdditive


# -- PRISM prediction model

class PRISMPredictionModel:
    """
    Wraps one of three PRISM network architectures for PRISM-W and PRISM-G importance computation.

    model_type
    ----------
    'mlp'         : flat MLP on full OHE input; full OHE group support
    'pairwise'    : DeepPINK pairwise filter + MLP; OHE columns treated as independent positions
    'additive'    : feature-wise additive sub-networks; OHE columns treated as independent positions

    For 'pairwise' and 'additive', p = input_size // 2 (one sub-network per OHE column pair).
    Group regularisation uses block-Frobenius norms so multi-column OHE groups are
    regularised jointly, matching 'mlp' behaviour.

    Warmup
    ------
    If n_warmup > 0, trains for up to n_warmup steps before the lambda_path loop using
    Adam with warmup_weight_decay. If warmup_patience > 0 and warmup_val_frac > 0, a
    hold-out val set is used for patience-based early stopping.

    vertical_prefit
    ---------------
    If True, REPLACES the warmup phase above (same n_warmup step budget) with a
    different procedure: X and Xk rows are vertically stacked into one (2n, p_ohe)
    pool and randomly shuffled, so the network only ever sees "some row's features,"
    never a paired [X,Xk] comparison. A single-sided (p-wide) version of the model
    (`build_prefit_module`) is trained unregularized on this pool for n_warmup steps,
    then `transfer_from_prefit` installs the pretrained weights into the real
    swap/discrimination parameter (duplicated onto both halves for mlp/additive, or
    v=0.5 for pairwise) before the normal lambda-path training proceeds. An exact
    symmetric transfer is a degenerate saddle point (both
    halves compute an identical function), so `prefit_noise_std` (default 0.01) adds
    independent N(0, prefit_noise_std) antisymmetric noise -- +noise on the X half,
    -noise on the Xk half -- to break the tie while keeping both halves' expected
    values equal. Set to 0.0 for an exact symmetric transfer.

    Implements fit / predict / predict_t / auto_diff / auto_diff_t / jacobian_t.
    """

    def __init__(
        self: Self,
        input_size: int,
        layers: Sequence[int],
        dense_activation: str | Type[nn.Module] = 'relu',
        loss_func: nn.Module = nn.MSELoss(),
        output_dimension: int = 1,
        learning_rate: float = 0.01,
        epochs: int = 500,
        model_type: Literal['mlp','pairwise','additive',] = 'pairwise',
        n_warmup: int = 0,
        warmup_patience: int = 20,
        warmup_check_interval: int = 50,
        warmup_tol: float = 1e-4,
        warmup_val_frac: float = 0.2,
        warmup_weight_decay: float = 1e-4,
        verbose: int = 0,
        vertical_prefit: bool = False,
        prefit_noise_std: float = 0.01,
        reset_optimizer: bool = True,
    ) -> None:
        activation_class: Type[nn.Module]
        if isinstance(dense_activation, str):
            activation_class = _nnModule_dict[dense_activation]
        else:
            activation_class = dense_activation
        #

        self.device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
        self.loss_func     = loss_func
        self.learning_rate = learning_rate
        self.epochs        = epochs
        self.model_type    = model_type
        self.n_warmup               = n_warmup
        self.warmup_patience        = warmup_patience
        self.warmup_check_interval  = warmup_check_interval
        self.warmup_tol             = warmup_tol
        self.warmup_val_frac        = warmup_val_frac
        self.warmup_weight_decay    = warmup_weight_decay
        self.verbose = verbose
        self.vertical_prefit = vertical_prefit
        self.prefit_noise_std = prefit_noise_std
        self.reset_optimizer = reset_optimizer

        if model_type == 'pairwise':
            if input_size % 2 != 0:
                raise ValueError(f"input_size must be even for model_type='pairwise'; got {input_size}")
            self.model = _PRISMNetworkPairwise(
                p               = input_size // 2,
                layers          = list(layers),
                activation_class= activation_class,
                output_size     = output_dimension,
            ).to(self.device)
        elif model_type == 'additive':
            if input_size % 2 != 0:
                raise ValueError(f"input_size must be even for model_type='additive'; got {input_size}")
            self.model = _PRISMNetworkAdditive(
                p               = input_size // 2,
                layers          = list(layers),
                activation_class= activation_class,
                output_size     = output_dimension,
            ).to(self.device)
        else:  # 'mlp'
            self.model = _PRISMNetworkMLP(
                input_size      = input_size,
                layers          = list(layers),
                activation_class= activation_class,
                output_size     = output_dimension,
            ).to(self.device)
        #/switch model_type
    #/def __init__

    def fit(
        self: Self,
        X: np.ndarray,
        y: np.ndarray,
        groups: list[list[int]],
        lambda_path: Sequence[float] | None = None,
        a_path: Iterable[float] | None = None,
        batch_size: int | None = None,
        snapshot_fn: Callable[['PRISMPredictionModel', torch.Tensor], np.ndarray] | None = None,
        weight: np.ndarray | None = None,
    ) -> list[np.ndarray]:
        """
        Train over the lambda_path; record one importance snapshot per lambda stage.

        :param groups: One list of OHE column indices per original feature (length 2*p).
        :param lambda_path: Sequence of lambda values. If None, trains once without regularisation.
        :param a_path: Per-stage penalty values. If None, uses lambda_path values.
        :param snapshot_fn: If None, snapshots use get_group_importances (PRISM-W).
                            Otherwise called as snapshot_fn(self, X_tensor) (for PRISM-G).
        :param weight: Optional length-n sample weight (see `_weighted_loss`),
            applied to the main lambda_path/no-lambda-path training loop
            (the one that actually produces the returned snapshots) and to
            the warmup phase's patience-check loss. The warmup/vertical_prefit
            phases' own gradient steps stay unweighted -- they're only a
            preliminary initialization before the real (weighted) fit, not
            what the reported importances come from. None (default) trains
            entirely unweighted, unchanged from before this parameter existed.
        :returns: List of importance arrays, one per lambda in lambda_path.
        """
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        _y = np.asarray(y)
        if np.issubdtype(_y.dtype, np.integer):
            y_tensor = torch.tensor(_y).long().to(self.device)
        else:
            y_tensor = torch.tensor(_y, dtype=torch.float32).to(self.device)
        #
        weight_tensor: torch.Tensor | None = (
            torch.tensor(np.asarray(weight, dtype=np.float32)).to(self.device)
            if weight is not None else None
        )

        self.model._precompute_group_reg(groups, self.device)

        n   = X_tensor.shape[0]
        _bs = batch_size if batch_size is not None and batch_size < n else n
        _use_minibatch = _bs < n

        # ── Warmup ────────────────────────────────────────────────────────────
        if self.n_warmup > 0 and self.vertical_prefit:
            self._vertical_prefit(X_tensor, y_tensor, batch_size)
        elif self.n_warmup > 0:
            no_decay_ids    = {id(p) for p in self.model.no_decay_parameters()}
            decay_params    = [p for p in self.model.parameters() if id(p) not in no_decay_ids]
            no_decay_params = [p for p in self.model.parameters() if id(p) in no_decay_ids]
            warmup_opt = optim.Adam(
                [
                    {'params': decay_params,    'weight_decay': self.warmup_weight_decay},
                    {'params': no_decay_params, 'weight_decay': 0.0},
                ],
                lr = self.learning_rate,
            )

            if self.warmup_val_frac > 0 and self.warmup_patience > 0:
                n_val   = max(1, int(n * self.warmup_val_frac))
                perm    = torch.randperm(n, device=self.device)
                val_idx = perm[:n_val]
                trn_idx = perm[n_val:]
                warm_loader = DataLoader(
                    TensorDataset(X_tensor[trn_idx], y_tensor[trn_idx]),
                    batch_size=_bs, shuffle=True,
                )
                def _check_loss(m):
                    return self._eval_loss(m, X_tensor[val_idx], y_tensor[val_idx], self.loss_func, weight_tensor[val_idx] if weight_tensor is not None else None)
                #/def _check_loss
            else:
                warm_loader = DataLoader(
                    TensorDataset(X_tensor, y_tensor),
                    batch_size=_bs, shuffle=True,
                )
                def _check_loss(m):
                    return self._eval_loss(m, X_tensor, y_tensor, self.loss_func, weight_tensor)
                #/def _check_loss
            #/if self.warmup_val_frac > 0 and self.warmup_patience > 0

            best_loss    = float('inf')
            patience_cnt = 0
            warmup_steps = 0

            self.model.train()
            for Zb, yb in _prism_cycle(warm_loader):
                warmup_opt.zero_grad()
                _weighted_loss(self.loss_func, self.model(Zb), yb, None).backward()
                warmup_opt.step()
                warmup_steps += 1

                if (
                    self.warmup_patience > 0
                    and warmup_steps % self.warmup_check_interval == 0
                ):
                    wl = _check_loss(self.model)
                    if wl < best_loss - self.warmup_tol:
                        best_loss    = wl
                        patience_cnt = 0
                    else:
                        patience_cnt += 1
                    #
                    if patience_cnt >= self.warmup_patience:
                        break
                    #
                #

                if warmup_steps >= self.n_warmup:
                    break
                #
            #/for Zb, yb in _prism_cycle(warm_loader)

            if self.verbose:
                tr_loss = self._eval_loss(self.model, X_tensor, y_tensor, self.loss_func, weight_tensor)
                vl_loss = _check_loss(self.model)
                status  = 'converged' if patience_cnt >= self.warmup_patience else 'max steps'
                print(f"  warm-up: {warmup_steps} steps [{status}]"
                      f"  train={tr_loss:.4f}  val={vl_loss:.4f}")
            #/if self.verbose
        #/if self.n_warmup > 0

        # ── No lambda_path: single unregularised pass ──────────────────────────
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.model.train()

        if lambda_path is None:
            for _ in range(self.epochs):
                if _use_minibatch:
                    perm = torch.randperm(n, device=self.device)
                    for start in range(0, n, _bs):
                        idx  = perm[start : start + _bs]
                        w_idx = weight_tensor[idx] if weight_tensor is not None else None
                        loss = _weighted_loss(self.loss_func, self.model(X_tensor[idx]), y_tensor[idx], w_idx)
                        optimizer.zero_grad(); loss.backward(); optimizer.step()
                else:
                    loss = _weighted_loss(self.loss_func, self.model(X_tensor), y_tensor, weight_tensor)
                    optimizer.zero_grad(); loss.backward(); optimizer.step()
            #/for

            snapshot = snapshot_fn(self, X_tensor) if snapshot_fn is not None else self.get_group_importances(groups)
            return [snapshot]
        #

        # ── Lambda path: distributed epoch BSS loop ────────────────────────────
        n_stages = len(lambda_path)
        _base, _rem = divmod(self.epochs, n_stages)
        stage_epochs = [_base + 1] * _rem + [_base] * (n_stages - _rem)

        _a_path: list[float] | None = list(a_path) if a_path is not None else None

        snapshots: list[np.ndarray] = []

        if not self.reset_optimizer:
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        #

        with tqdm(total=self.epochs, file=sys.stderr, disable=False) as pbar:
            for stage_idx, lambda_b in enumerate(lambda_path):
                if self.reset_optimizer:
                    optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
                #
                lb  = float(lambda_b)
                a_b = _a_path[stage_idx] if _a_path is not None else lb
                pbar.set_postfix({'lambda': f'{lb:.3g}'})

                for _ in range(stage_epochs[stage_idx]):
                    if _use_minibatch:
                        perm = torch.randperm(n, device=self.device)
                        for start in range(0, n, _bs):
                            idx  = perm[start : start + _bs]
                            w_idx = weight_tensor[idx] if weight_tensor is not None else None
                            pred = self.model(X_tensor[idx])
                            loss = _weighted_loss(self.loss_func, pred, y_tensor[idx], w_idx) + self.model.group_regularization(lb, a_b, groups)
                            optimizer.zero_grad(); loss.backward(); optimizer.step()
                    else:
                        pred = self.model(X_tensor)
                        loss = _weighted_loss(self.loss_func, pred, y_tensor, weight_tensor) + self.model.group_regularization(lb, a_b, groups)
                        optimizer.zero_grad(); loss.backward(); optimizer.step()
                    #
                    pbar.update(1)
                #/for epoch

                if snapshot_fn is not None:
                    snapshots.append(snapshot_fn(self, X_tensor))
                else:
                    snapshots.append(self.get_group_importances(groups))
                #
            #/for lambda_b
        #/with tqdm

        return snapshots
    #/def fit

    def _vertical_prefit(
        self: Self,
        X_tensor: torch.Tensor,
        y_tensor: torch.Tensor,
        batch_size: int | None = None,
    ) -> None:
        """
        Pretrain a single-sided (p-wide) version of the model on X and Xk rows
        vertically stacked and randomly shuffled together (no paired comparison),
        unregularized, for n_warmup steps -- then transfer the result into the real
        swap/discrimination parameter. See `vertical_prefit` in the class docstring.
        """
        p_ohe = X_tensor.shape[1] // 2
        X_stacked = torch.cat([X_tensor[:, :p_ohe], X_tensor[:, p_ohe:]], dim=0)
        y_stacked = torch.cat([y_tensor, y_tensor], dim=0)
        perm = torch.randperm(X_stacked.shape[0], device=self.device)
        X_stacked, y_stacked = X_stacked[perm], y_stacked[perm]

        n   = X_stacked.shape[0]
        _bs = batch_size if batch_size is not None and batch_size < n else n
        prefit_module = self.model.build_prefit_module().to(self.device)
        prefit_opt = optim.Adam(prefit_module.parameters(), lr=self.learning_rate)

        prefit_loader = DataLoader(
            TensorDataset(X_stacked, y_stacked),
            batch_size=_bs, shuffle=True,
        )

        prefit_module.train()
        steps = 0
        for Xb, yb in _prism_cycle(prefit_loader):
            prefit_opt.zero_grad()
            self.loss_func(prefit_module(Xb), yb).backward()
            prefit_opt.step()
            steps += 1
            if steps >= self.n_warmup:
                break
            #
        #/for Xb, yb in _prism_cycle(prefit_loader)

        self.model.transfer_from_prefit(prefit_module, noise_std=self.prefit_noise_std)

        if self.verbose:
            print(f"  vertical_prefit: {steps} steps")
        #/if self.verbose
    #/def _vertical_prefit

    def get_group_importances(
        self: Self,
        groups: list[list[int]],
    ) -> np.ndarray:
        return self.model.get_group_importances(groups)
    #/def get_group_importances

    def predict_t(
        self: Self,
        X_t: torch.Tensor,
    ) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            result = self.model(X_t)
        self.model.train()
        # Callers (importance.py) expect (n, output_dim) — unsqueeze scalar output.
        return result.unsqueeze(-1) if result.dim() == 1 else result
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
        X_t = X_t.detach().requires_grad_(True)
        y_pred = self.model(X_t)
        grad = torch.autograd.grad(
            outputs      = y_pred,
            inputs       = X_t,
            grad_outputs = torch.ones_like(y_pred),
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

    def jacobian_t(
        self: Self,
        X_t: torch.Tensor,
    ) -> torch.Tensor:
        """Per-sample Jacobian (n, k, p) via vmap+jacrev. Used for categorical outcomes."""
        def _forward_single(x_single: torch.Tensor) -> torch.Tensor:
            return self.model(x_single.unsqueeze(0)).squeeze(0)
        self.model.eval()
        with torch.no_grad():
            jac = vmap(jacrev(_forward_single))(X_t)
        self.model.train()
        return jac
    #/def jacobian_t

    @staticmethod
    def _eval_loss(
        model: nn.Module,
        X_t: torch.Tensor,
        y_t: torch.Tensor,
        loss_fn: nn.Module,
        weight_t: torch.Tensor | None = None,
    ) -> float:
        model.eval()
        with torch.no_grad():
            return _weighted_loss(loss_fn, model(X_t), y_t, weight_t).item()
        #
    #/def _eval_loss
#/class PRISMPredictionModel
