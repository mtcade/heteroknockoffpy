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


def _range_importance(vals: torch.Tensor) -> float:
    """
    Zero-anchored spread statistic for one group's per-column norms `vals`
    (length >= 1, all originally from a nonnegative-by-construction source
    like an L2 column norm, but the clamps below make this correct even if a
    signed per-column value is used in the future):

    - a singleton group (`len(vals) == 1`) returns its own value unchanged.
    - a multi-column (categorical) group returns
      `max_g.clamp(min=0) - min_g.clamp(max=0)`: equals `max_g` when every
      value in the group is >= 0 (the common case today, since T_j is a
      nonnegative norm), `-min_g` (i.e. `|min_g|`) when every value is <= 0,
      and the ordinary `max_g - min_g` range for a group whose values span
      both signs.
    """
    if vals.numel() == 1:
        return vals.item()
    #
    max_g = vals.max().clamp(min=0)
    min_g = vals.min().clamp(max=0)
    return (max_g - min_g).item()
#/def _range_importance


def _group_importance(col_norms: torch.Tensor, categorical_collapse_method: str) -> float:
    """
    Collapse one group's per-column L2 norms `col_norms` (length >= 1, e.g.
    one entry per OHE dummy column of a categorical variable) into a single
    importance value. A singleton group (numeric variable) returns its own
    value unchanged regardless of `categorical_collapse_method` -- the choice
    only matters for multi-column (categorical) groups:

    - 'l2_norm' (default): Frobenius norm of the group's column-norm vector,
      i.e. `sqrt(sum(col_norms**2))` -- identical to the L2/Frobenius norm of
      the whole underlying weight block. Aggregates signal across every
      column in the group, so it's stronger when a categorical variable's
      true effect is spread across several/all of its categories, but gets
      diluted when the effect is concentrated in just one category.
    - 'range': `_range_importance` -- a zero-anchored max-min spread that
      isolates the single most extreme column instead of aggregating.
      Stronger when only one category actually deviates, weaker when the
      effect is spread across categories (the other categories' real signal
      is simply discarded).
    """
    if col_norms.numel() == 1:
        return col_norms.item()
    #
    if categorical_collapse_method == 'range':
        return _range_importance(col_norms)
    elif categorical_collapse_method == 'l2_norm':
        return col_norms.norm().item()
    #
    raise ValueError(
        f"Unrecognized categorical_collapse_method={categorical_collapse_method!r}; "
        "expected 'l2_norm' or 'range'."
    )
#/def _group_importance


def _bss_step_batches(n: int, _bs: int, use_minibatch: bool, device: str):
    """
    Infinite generator yielding one gradient step's row-index tensor (or None
    for full-batch) at a time, crossing epoch boundaries transparently --
    the BSS block loop pulls an exact step count per block from this same
    generator instance, so a block can end mid-epoch instead of being
    quantized to whole passes over the data.

    - use_minibatch False: yields None forever (caller uses the full tensors,
      no indexing).
    - use_minibatch True: yields shuffled index tensors of size _bs (the
      last batch of each internal pass may be smaller, matching
      `range(0, n, _bs)`'s tail-batch behavior), drawing a fresh
      torch.randperm(n) each time the current permutation is exhausted.
    """
    if not use_minibatch:
        while True:
            yield None
        #
    #
    while True:
        perm = torch.randperm(n, device=device)
        for start in range(0, n, _bs):
            yield perm[start : start + _bs]
        #
    #
#/def _bss_step_batches


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
    def get_group_importances(
        self, groups: list[list[int]], categorical_collapse_method: str = 'l2_norm',
    ) -> np.ndarray: ...

    @abstractmethod
    def group_parameters(self) -> list[nn.Parameter]:
        """
        Parameter(s) penalized by group_regularization (the first-layer group
        weights). Excluded from the BSS-phase deep-layer weight decay so the
        two penalties never double-count the same weights -- GRIP2 Remark 1
        keeps the deep-layer ridge term on theta_deep = theta \\ {W} only.
        """
        ...

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
        # Column count per group -- divides grp_sq below so a K-column
        # categorical group isn't penalized K^(a/2)x harder than a numeric
        # singleton at equal per-column weight magnitude.
        self._group_sizes = torch.tensor(
            [len(g) for g in active], dtype=torch.float32, device=device
        )
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
        # Mean (not sum) of squared column norms per group, so groups of
        # different cardinality are penalized comparably at equal per-column
        # weight magnitude -- see _group_sizes above.
        grp_sq_mean = grp_sq / self._group_sizes
        # GRIP2 Eq. 2 penalizes ||w_j||_2^a (the norm, not the squared norm);
        # grp_sq_mean is squared, so raise it to the a/2 power.
        return lambda_val * (grp_sq_mean + eps).pow(a / 2).sum()
    #/def group_regularization

    def get_group_importances(
        self, groups: list[list[int]], categorical_collapse_method: str = 'l2_norm',
    ) -> np.ndarray:
        with torch.no_grad():
            w = self.net[0].weight.detach().cpu()             # (H, 2p)
            col_norms = torch.norm(w, dim=0)                   # (2p,) T_j per OHE column
            return np.array([
                _group_importance(col_norms[g], categorical_collapse_method) if g else 0.0
                for g in groups
            ])
    #/def get_group_importances

    def group_parameters(self) -> list[nn.Parameter]:
        return [self.net[0].weight]
    #/def group_parameters

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
        # Column count per group -- divides grp_sq below so a K-column
        # categorical group isn't penalized K^(a/2)x harder than a numeric
        # singleton at equal per-column weight magnitude.
        self._group_sizes = torch.tensor(
            [len(g) for g in active], dtype=torch.float32, device=device
        )
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
        # Mean (not sum) of squared column norms per group, so groups of
        # different cardinality are penalized comparably at equal per-column
        # weight magnitude -- see _group_sizes above.
        grp_sq_mean = grp_sq / self._group_sizes
        # GRIP2 Eq. 2 penalizes ||v_j||_2^a (the norm, not the squared norm);
        # grp_sq_mean is squared, so raise it to the a/2 power.
        return lambda_val * (grp_sq_mean + eps).pow(a / 2).sum()
    #/def group_regularization

    def get_group_importances(
        self, groups: list[list[int]], categorical_collapse_method: str = 'l2_norm',
    ) -> np.ndarray:
        with torch.no_grad():
            col_norms = self.v.detach().cpu().abs()            # (2p,) T_j per OHE column
            return np.array([
                _group_importance(col_norms[g], categorical_collapse_method) if g else 0.0
                for g in groups
            ])
    #/def get_group_importances

    def group_parameters(self) -> list[nn.Parameter]:
        return [self.v]
    #/def group_parameters

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
        # Column count per group -- divides grp_x/grp_xk below so a K-column
        # categorical group isn't penalized K^(a/2)x harder than a numeric
        # singleton at equal per-column weight magnitude.
        self._x_group_sizes  = torch.tensor(
            [len(g) for _, g in x_groups], dtype=torch.float32, device=device
        )
        self._xk_group_sizes = torch.tensor(
            [len(g) for _, g in xk_groups], dtype=torch.float32, device=device
        )
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
        # Mean (not sum) of squared column norms per group, so groups of
        # different cardinality are penalized comparably at equal per-column
        # weight magnitude -- see _x_group_sizes/_xk_group_sizes above.
        grp_x_mean  = grp_x  / self._x_group_sizes
        grp_xk_mean = grp_xk / self._xk_group_sizes
        # GRIP2 Eq. 2 penalizes ||w_j||_2^a (the norm, not the squared norm);
        # grp_x_mean/grp_xk_mean are squared, so raise them to the a/2 power.
        return lambda_val * ((grp_x_mean + eps).pow(a / 2).sum() + (grp_xk_mean + eps).pow(a / 2).sum())
    #/def group_regularization

    def get_group_importances(
        self, groups: list[list[int]], categorical_collapse_method: str = 'l2_norm',
    ) -> np.ndarray:
        """
        T_j = ||W1[j,:,0]|| (X side) or ||W1[j,:,1]|| (Xk side) per column;
        group importance is _group_importance over the group's T_j values
        (its own T_j for a numeric singleton). X groups draw from column
        norms of W1[:,:,0]; Xk groups → W1[:,:,1], offset by -p.
        """
        p = self.p
        with torch.no_grad():
            W1 = self.W1.detach().cpu()
            col_norms_x  = torch.norm(W1[:, :, 0], dim=1)      # (p,) T_j per X-side column
            col_norms_xk = torch.norm(W1[:, :, 1], dim=1)      # (p,) T_j per Xk-side column
            result = []
            for g in groups:
                if not g:
                    result.append(0.0)
                elif min(g) >= p:
                    result.append(_group_importance(col_norms_xk[[c - p for c in g]], categorical_collapse_method))
                else:
                    result.append(_group_importance(col_norms_x[g], categorical_collapse_method))
            return np.array(result)
    #/def get_group_importances

    def group_parameters(self) -> list[nn.Parameter]:
        return [self.W1]
    #/def group_parameters

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
    Wraps one of three PRISM network architectures for GRIP2 and Torch PRISM importance computation.

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
    Adam with warmup_weight_decay applied to all parameters. If warmup_patience > 0
    and warmup_val_frac > 0, a hold-out val set is used for patience-based early
    stopping.

    Deep-layer decay during BSS
    ----------------------------
    During the lambda_path loop, warmup_weight_decay is also applied (as
    Adam weight_decay) to every parameter except the group-regularized
    first-layer weights (self.model.group_parameters()) -- i.e. the same
    coefficient plays the role of GRIP2 Eq. 2's persistent
    gamma/2 * ||theta_deep||^2 ridge term. This penalty stays active in every
    BSS stage, not just warmup, so the first-layer group weights can't be
    gamed by shrinking them while inflating deeper layers via the network's
    scaling symmetry (GRIP2 Remark 1).

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

    rng
    ---
    If given, seeds torch's global RNG (`torch.manual_seed`) once at construction,
    before any model parameters are created. Every source of torch randomness in
    this class and `_vertical_prefit` (parameter init, `DataLoader(shuffle=True)`,
    `torch.randperm`) draws from that global generator with no explicit
    `generator=` kwarg, so this one seed call makes an entire `fit()` run fully
    reproducible. `None` (default) leaves torch's ambient RNG state untouched.

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
        model_type: Literal['mlp','pairwise','additive',] = 'mlp',
        n_warmup: int = 5000,
        warmup_patience: int = 20,
        warmup_check_interval: int = 50,
        warmup_tol: float = 1e-4,
        warmup_val_frac: float = 0.2,
        warmup_weight_decay: float = 1e-4,
        verbose: int = 0,
        vertical_prefit: bool = False,
        prefit_noise_std: float = 0.01,
        reset_optimizer: bool = True,
        rng: np.random.Generator | None = None,
    ) -> None:
        activation_class: Type[nn.Module]
        if isinstance(dense_activation, str):
            activation_class = _nnModule_dict[dense_activation]
        else:
            activation_class = dense_activation
        #

        # Seed torch's global RNG once, before any model parameters are
        # constructed below -- every torch.randn/randperm/DataLoader(shuffle=True)
        # call in this module and _vertical_prefit draws from that global
        # generator with no explicit `generator=` kwarg, so this single call is
        # both necessary and sufficient for full run-to-run reproducibility.
        if rng is not None:
            torch.manual_seed(int(rng.integers(0, 2**63)))
        #

        # Stored (not just consumed above) so fit()'s post-warmup calibration
        # branch can draw the calibrated lambda_path/a_path from the same
        # generator, instead of needing its own separate rng parameter.
        self.rng = rng

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
        calibrate: bool = False,
        n_blocks: int = 30,
        a_min: float = 0.3,
        a_max: float = 1.0,
        calibrate_rmin: float = 0.01,
        calibrate_rmax: float = 1.0,
        batch_size: int | None = None,
        total_steps: int | None = None,
        snapshot_fn: Callable[['PRISMPredictionModel', torch.Tensor], np.ndarray] | None = None,
        weight: np.ndarray | None = None,
        categorical_collapse_method: str = 'l2_norm',
    ) -> list[np.ndarray]:
        """
        Train over the lambda_path; record one importance snapshot per lambda stage.

        :param groups: One list of OHE column indices per original feature (length 2*p).
        :param lambda_path: Sequence of lambda values. If None and calibrate=False, trains
            once without regularisation. If None and calibrate=True, resolved by the
            calibration pilot pass below instead.
        :param a_path: Per-stage penalty values. If None, uses lambda_path values.
        :param calibrate: If True, runs GRIP2 Eq. 5's gradient-ratio calibration
            immediately after warmup (on the near-converged post-warmup model, since
            the ratio is meaningless at random init) to derive lambda_path/a_path,
            instead of requiring them to be passed in. Callers (the public
            prism*Importances functions) are responsible for having already validated
            that lambda_path/a_path are both None when calibrate=True -- this method
            assumes that precondition and does not re-check it.
        :param n_blocks: Number of BSS blocks to sample when calibrate=True (ignored
            otherwise -- non-calibrate path resolution happens entirely in the caller's
            `_resolve_lambda_a_path`, which produces an already-sized lambda_path/a_path).
        :param a_min: Lower bound for a_path's Uniform(a_min, a_max) draw when
            calibrate=True (ignored otherwise, same reason as n_blocks).
        :param a_max: Upper bound for a_path's Uniform(a_min, a_max) draw when
            calibrate=True (ignored otherwise).
        :param calibrate_rmin: Target lower bound for the gradient-ratio
            ||grad_W R|| / ||grad_W L_pred|| that calibration solves for (GRIP2 Eq. 5's
            r_min). Only consulted when calibrate=True.
        :param calibrate_rmax: Target upper bound for that same gradient ratio (r_max).
            Only consulted when calibrate=True.
        :param batch_size: Minibatch size; None (default) or >= n trains full-batch.
        :param total_steps: Total gradient-step budget for the lambda_path loop (and
            the unregularised single pass when lambda_path is None), overriding
            `self.epochs`. If None (default), the budget is derived as
            `self.epochs * ceil(n / effective_batch_size)` -- i.e. `epochs` full-batch
            -equivalent passes, converted to raw steps. Pass `total_steps` explicitly
            to pin the exact step count independent of `n`/`batch_size` (mirrors
            HuggingFace Trainer's `max_steps` overriding `num_train_epochs`, or
            PyTorch Lightning's `max_steps` overriding `max_epochs`).
            When `lambda_path` is given, this budget is split as evenly as possible
            (in raw-step units, not whole epochs) across `len(lambda_path)` BSS
            blocks, so changing the number of blocks redistributes the same total
            budget rather than changing it. A single minibatch-index generator
            cycles continuously across block boundaries, so a block can end mid-
            epoch instead of being quantized to whole passes over the data.
        :param snapshot_fn: If None, snapshots use get_group_importances (GRIP2).
                            Otherwise called as snapshot_fn(self, X_tensor) (for Torch PRISM).
        :param categorical_collapse_method: Forwarded to get_group_importances when
            snapshot_fn is None -- ignored otherwise, since a custom snapshot_fn
            (e.g. Torch PRISM/PRISM-GRIP2's) is responsible for calling
            get_group_importances itself if it wants one. See
            get_group_importances/_group_importance for what 'l2_norm' (default)
            vs 'range' mean.
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
        # val_idx: set below only in the plain-warmup branch when a genuine
        # held-out split exists (warmup_val_frac>0 and warmup_patience>0) --
        # stays None for vertical_prefit, n_warmup==0, or no-val-split cases,
        # so the calibration call site below can fall back to the full
        # training tensors in those cases.
        val_idx: torch.Tensor | None = None
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
            best_state: dict[str, torch.Tensor] | None = None
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
                        # Snapshot (not reference) the best-so-far weights -- restored
                        # below once the loop ends, so calibration/BSS start from the
                        # genuinely best-converged warmup state rather than whatever
                        # arbitrary later (possibly overfit) state the loop stops at.
                        best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
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

            if best_state is not None:
                self.model.load_state_dict(best_state)
            #

            if self.verbose:
                tr_loss = self._eval_loss(self.model, X_tensor, y_tensor, self.loss_func, weight_tensor)
                vl_loss = _check_loss(self.model)
                status  = 'converged' if patience_cnt >= self.warmup_patience else 'max steps'
                print(f"  warm-up: {warmup_steps} steps [{status}]"
                      f"  train={tr_loss:.4f}  val={vl_loss:.4f}")
            #/if self.verbose
        #/if self.n_warmup > 0

        # ── Calibration (GRIP2 sec 2.4): derive lambda_path/a_path from the
        #    post-warmup model's gradient ratio, in place of requiring them to be
        #    passed in. Must run here, not earlier -- the ratio needs a real
        #    (near-converged, unregularised) model + real data, neither of which
        #    existed before warmup finished. By construction (validated in the
        #    caller's _resolve_lambda_a_path) lambda_path/a_path are still None
        #    at this point whenever calibrate=True.
        if calibrate:
            _rng = self.rng if self.rng is not None else np.random.default_rng()

            # a's range isn't calibrated (GRIP2 fixes a's upper bound at 1 and
            # only ever varies a_min) -- draw a_path immediately, then average
            # the gradient ratio over it (not a single fixed a_ref) so the
            # calibrated lambda range reflects the actual a values BSS will use.
            a_path = list(_rng.uniform(a_min, a_max, size=n_blocks))

            # Measure grad_L/grad_R on the held-out warmup validation split
            # (val_idx), not the training set, whenever one exists -- the
            # training set is exactly what the model was just fit on, so its
            # gradient there can be contaminated by memorized noise rather
            # than genuine predictive signal (worse the faster a loss overfits,
            # e.g. count/categorical outcomes vs. continuous MSE). Falls back
            # to the full tensors when no val split exists (warmup_val_frac==0
            # or warmup_patience==0), matching prior behavior.
            if val_idx is not None:
                _calib_X, _calib_y = X_tensor[val_idx], y_tensor[val_idx]
                _calib_weight = weight_tensor[val_idx] if weight_tensor is not None else None
            else:
                _calib_X, _calib_y, _calib_weight = X_tensor, y_tensor, weight_tensor
            #

            lambda_min_calib, lambda_max_calib = self._calibrate_lambda_range(
                _calib_X, _calib_y, groups, _calib_weight,
                a_path=a_path, r_min=calibrate_rmin, r_max=calibrate_rmax,
            )

            log_low, log_high = np.log10(lambda_min_calib), np.log10(lambda_max_calib)
            lambda_path = list(10.0 ** _rng.uniform(log_low, log_high, size=n_blocks))

            if self.verbose:
                print(f"  calibrate: lambda in [{lambda_min_calib:.3g}, "
                      f"{lambda_max_calib:.3g}]  (target ratio "
                      f"[{calibrate_rmin:.3g}, {calibrate_rmax:.3g}])")
            #/if self.verbose
        #/if calibrate

        # ── Total step budget: epochs (full-batch-equivalent passes), converted
        #    to raw steps, unless total_steps pins the exact count directly ──────
        _steps_per_epoch = -(-n // _bs)                       # ceil(n / _bs); 1 when full-batch
        resolved_total_steps = (
            total_steps if total_steps is not None
            else self.epochs * _steps_per_epoch
        )

        step_iter = _bss_step_batches(n, _bs, _use_minibatch, self.device)

        # ── No lambda_path: single unregularised pass ──────────────────────────
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.model.train()

        if lambda_path is None:
            for _ in range(resolved_total_steps):
                idx = next(step_iter)
                if idx is None:
                    Xb, yb, wb = X_tensor, y_tensor, weight_tensor
                else:
                    Xb, yb = X_tensor[idx], y_tensor[idx]
                    wb = weight_tensor[idx] if weight_tensor is not None else None
                #
                loss = _weighted_loss(self.loss_func, self.model(Xb), yb, wb)
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            #/for

            snapshot = (
                snapshot_fn(self, X_tensor) if snapshot_fn is not None
                else self.get_group_importances(groups, categorical_collapse_method)
            )
            return [snapshot]
        #

        # ── Lambda path: distributed step BSS loop ──────────────────────────────
        # GRIP2 Eq. 2 keeps a persistent gamma/2 * ||theta_deep||^2 ridge penalty
        # active on every non-first-layer parameter throughout BSS, not just
        # warmup (Remark 1): without it the model can game the group penalty by
        # shrinking the first-layer group weights while inflating the deeper
        # layers to compensate, via the network's scaling symmetry, which would
        # make the recorded ||w_j|| snapshots an unfaithful activity proxy. We
        # reuse warmup_weight_decay as gamma and exclude the group-regularized
        # parameters (already penalized by group_regularization) from this decay.
        group_param_ids  = {id(p) for p in self.model.group_parameters()}
        deep_params      = [p for p in self.model.parameters() if id(p) not in group_param_ids]
        group_params     = [p for p in self.model.parameters() if id(p) in group_param_ids]

        def _make_bss_optimizer() -> optim.Adam:
            return optim.Adam(
                [
                    {'params': deep_params,  'weight_decay': self.warmup_weight_decay},
                    {'params': group_params, 'weight_decay': 0.0},
                ],
                lr = self.learning_rate,
            )
        #/def _make_bss_optimizer

        # Same total budget, split evenly in raw-step units (not whole epochs)
        # across blocks -- a strict generalization of the old epoch-quantized
        # split: reduces to it exactly when _steps_per_epoch == 1 (full-batch).
        # Changing n_stages (block count) redistributes resolved_total_steps,
        # it never changes it.
        n_stages = len(lambda_path)
        _base, _rem = divmod(resolved_total_steps, n_stages)
        stage_steps = [_base + 1] * _rem + [_base] * (n_stages - _rem)

        _a_path: list[float] | None = list(a_path) if a_path is not None else None

        snapshots: list[np.ndarray] = []

        if not self.reset_optimizer:
            optimizer = _make_bss_optimizer()
        #

        with tqdm(total=resolved_total_steps, file=sys.stderr, disable=False) as pbar:
            for stage_idx, lambda_b in enumerate(lambda_path):
                if self.reset_optimizer:
                    optimizer = _make_bss_optimizer()
                #
                lb  = float(lambda_b)
                a_b = _a_path[stage_idx] if _a_path is not None else lb
                pbar.set_postfix({'lambda': f'{lb:.3g}'})

                for _ in range(stage_steps[stage_idx]):
                    idx = next(step_iter)
                    if idx is None:
                        Xb, yb, wb = X_tensor, y_tensor, weight_tensor
                    else:
                        Xb, yb = X_tensor[idx], y_tensor[idx]
                        wb = weight_tensor[idx] if weight_tensor is not None else None
                    #
                    pred = self.model(Xb)
                    loss = _weighted_loss(self.loss_func, pred, yb, wb) + self.model.group_regularization(lb, a_b, groups)
                    optimizer.zero_grad(); loss.backward(); optimizer.step()
                    pbar.update(1)
                #/for step

                if snapshot_fn is not None:
                    snapshots.append(snapshot_fn(self, X_tensor))
                else:
                    snapshots.append(self.get_group_importances(groups, categorical_collapse_method))
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
        categorical_collapse_method: str = 'l2_norm',
    ) -> np.ndarray:
        return self.model.get_group_importances(groups, categorical_collapse_method)
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

    def _calibrate_lambda_range(
        self: Self,
        X_tensor: torch.Tensor,
        y_tensor: torch.Tensor,
        groups: list[list[int]],
        weight_tensor: torch.Tensor | None,
        a_path: Sequence[float],
        r_min: float,
        r_max: float,
    ) -> tuple[float, float]:
        """
        GRIP2 Eq. 5's gradient-ratio calibration: solve for (lambda_min, lambda_max)
        such that ||grad_W R(theta; lambda, a)||_F / ||grad_W L_pred(theta)||_F
        spans [r_min, r_max] as lambda ranges over the result, averaged over a_path.

        group_regularization(lambda_val, a, groups) is exactly
        `lambda_val * (...)` -- linear in lambda_val -- so grad_W R at any lambda
        is grad_W R at lambda=1 scaled by that lambda. One gradient computation
        per a_path entry (at lambda=1) is therefore enough to solve for both
        endpoints directly, without iterating over candidate lambda values.

        grad_W is taken w.r.t. self.model.group_parameters() (the exact first-layer
        weight tensor(s) group_regularization differentiates through) for BOTH
        R and L_pred, so the two Frobenius norms live in the same parameter space
        and their ratio is directly comparable, matching the paper's grad_W R /
        grad_W L_pred.
        """
        self.model.eval()

        params = list(self.model.group_parameters())

        pred = self.model(X_tensor)
        loss = _weighted_loss(self.loss_func, pred, y_tensor, weight_tensor)
        grad_L = torch.autograd.grad(loss, params)
        grad_L_norm = torch.sqrt(sum(g.pow(2).sum() for g in grad_L))

        ratios: list[float] = []
        for a_b in a_path:
            R = self.model.group_regularization(1.0, float(a_b), groups)
            grad_R = torch.autograd.grad(R, params)
            grad_R_norm = torch.sqrt(sum(g.pow(2).sum() for g in grad_R))
            ratios.append((grad_R_norm / (grad_L_norm + 1e-12)).item())
        #/for a_b

        self.model.train()

        ratio_avg = sum(ratios) / len(ratios)
        return r_min / ratio_avg, r_max / ratio_avg
    #/def _calibrate_lambda_range
#/class PRISMPredictionModel
