#
#//  tests/heteroknockofftorch/test_torchImportances.py
#//  heteroknockoffpy
#//
#//  Verifies heteroknockofftorch.torchImportances is importable and exposes
#//  its public API (PRISMPredictionModel.fit for each model type), plus
#//  structural tests for the vertical_prefit training method: build a prefit
#//  module, do one forward/backward step, transfer into the real
#//  swap/discrimination parameter, and check the transfer landed correctly --
#//  no full lambda-path training needed.
#//
import numpy as np
import pytest
import torch
import torch.nn as nn

from heteroknockoffpy.heteroknockofftorch.torchImportances import (
    PRISMPredictionModel,
    _PRISMNetworkMLP,
    _PRISMNetworkPairwise,
    _PRISMNetworkAdditive,
)


def test_torch_importances_importable():
    # Construct and do a minimal fit for each model type.
    rng = np.random.default_rng(0)
    n, p = 60, 4
    X_all = np.concatenate(
        [rng.standard_normal((n, p)), rng.standard_normal((n, p))], axis=1
    )
    y = rng.standard_normal(n)
    groups = [[j] for j in range(2 * p)]

    for mt in ("mlp", "pairwise", "additive"):
        m = PRISMPredictionModel(input_size=2 * p, layers=[8], model_type=mt, epochs=2)
        snaps = m.fit(X_all, y, groups, lambda_path=np.logspace(-1, -2, 2))
        assert np.array(snaps).shape == (2, 2 * p), f"{mt}: unexpected shape"


P_OHE = 6          # input_size = 2*P_OHE
X_GROUPS = [[0], [1], [2, 3, 4], [5]]
N_VARS = len(X_GROUPS)


def _make_mlp():
    return _PRISMNetworkMLP(input_size=2 * P_OHE, layers=[8], activation_class=nn.ReLU)


def _make_pairwise():
    return _PRISMNetworkPairwise(p=P_OHE, layers=[8], activation_class=nn.ReLU)


def _make_additive():
    return _PRISMNetworkAdditive(p=P_OHE, layers=[8], activation_class=nn.ReLU)


def _train_one_step(prefit_module, input_width):
    opt = torch.optim.Adam(prefit_module.parameters(), lr=0.1)
    x = torch.randn(16, input_width)
    y = torch.randn(16)
    for _ in range(5):
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(prefit_module(x), y)
        loss.backward()
        opt.step()
    return prefit_module


# ---------------------------------------------------------------------------
# antisymmetric noise: breaks the exact-duplicate tie while keeping the
# expected value of both halves equal to the pretrained weight
# ---------------------------------------------------------------------------

def test_mlp_transfer_antisymmetric_noise():
    net = _make_mlp()
    prefit = net.build_prefit_module()
    _train_one_step(prefit, P_OHE)

    net.transfer_from_prefit(prefit, noise_std=0.05)
    w = net.net[0].weight.detach()
    assert not torch.allclose(w[:, :P_OHE], w[:, P_OHE:])
    # average of the two halves recovers the (noiseless) pretrained weight
    assert torch.allclose((w[:, :P_OHE] + w[:, P_OHE:]) / 2, prefit[0].weight.detach())


def test_pairwise_transfer_antisymmetric_noise():
    net = _make_pairwise()
    prefit = net.build_prefit_module()
    _train_one_step(prefit, P_OHE)

    net.transfer_from_prefit(prefit, noise_std=0.05)
    v = net.v.detach()
    assert not torch.allclose(v[:P_OHE], v[P_OHE:])
    assert torch.allclose((v[:P_OHE] + v[P_OHE:]) / 2, torch.full((P_OHE,), 0.5))


# ---------------------------------------------------------------------------
# mlp: duplicate pretrained first-layer weights onto both halves
# ---------------------------------------------------------------------------

def test_mlp_prefit_transfer_duplicates_both_halves():
    net = _make_mlp()
    prefit = net.build_prefit_module()
    assert prefit[0].weight.shape == (8, P_OHE)
    _train_one_step(prefit, P_OHE)

    net.transfer_from_prefit(prefit)
    w = net.net[0].weight.detach()
    assert torch.allclose(w[:, :P_OHE], w[:, P_OHE:])
    assert torch.allclose(w[:, :P_OHE], prefit[0].weight.detach())


# ---------------------------------------------------------------------------
# pairwise: prefit IS self.mlp; v gets fresh 0.5 fill
# ---------------------------------------------------------------------------

def test_pairwise_prefit_reuses_mlp_and_v_becomes_half():
    net = _make_pairwise()
    v_before = net.v.detach().clone()

    prefit = net.build_prefit_module()
    assert prefit is net.mlp
    _train_one_step(prefit, P_OHE)

    net.transfer_from_prefit(prefit)
    assert torch.allclose(net.v.detach(), torch.full_like(net.v, 0.5))
    assert not torch.allclose(net.v.detach(), v_before)


# ---------------------------------------------------------------------------
# additive: duplicate pretrained W1 onto both slices
# ---------------------------------------------------------------------------

def test_additive_prefit_transfer_duplicates_both_slices():
    net = _make_additive()
    prefit = net.build_prefit_module()
    assert prefit.W1.shape == (P_OHE, 8)
    _train_one_step(prefit, P_OHE)

    net.transfer_from_prefit(prefit)
    W1 = net.W1.detach()
    assert torch.allclose(W1[:, :, 0], W1[:, :, 1])
    assert torch.allclose(W1[:, :, 0], prefit.W1.detach())


# ---------------------------------------------------------------------------
# Downstream params shared by reference actually train in place
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("make_net", [_make_mlp])
def test_mlp_family_downstream_layers_train_in_place(make_net):
    net = make_net()
    last_linear_before = [p.clone() for p in net.net[-1].parameters()]

    prefit = net.build_prefit_module()
    _train_one_step(prefit, P_OHE)

    last_linear_after = list(net.net[-1].parameters())
    assert any(
        not torch.allclose(before, after)
        for before, after in zip(last_linear_before, last_linear_after)
    )


@pytest.mark.parametrize("make_net", [_make_pairwise])
def test_pairwise_family_mlp_trains_in_place(make_net):
    net = make_net()
    mlp_before = [p.clone() for p in net.mlp.parameters()]

    prefit = net.build_prefit_module()
    _train_one_step(prefit, P_OHE)

    mlp_after = list(net.mlp.parameters())
    assert any(
        not torch.allclose(before, after)
        for before, after in zip(mlp_before, mlp_after)
    )


def test_additive_downstream_params_train_in_place():
    net = _make_additive()
    b1_before, W2_before, b2_before = net.b1.clone(), net.W2.clone(), net.b2.clone()

    prefit = net.build_prefit_module()
    _train_one_step(prefit, P_OHE)

    assert (
        not torch.allclose(net.b1, b1_before)
        or not torch.allclose(net.W2, W2_before)
        or not torch.allclose(net.b2, b2_before)
    )


# ---------------------------------------------------------------------------
# End-to-end: PRISMPredictionModel.fit with vertical_prefit=True runs without error
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model_type", ["mlp", "pairwise", "additive"])
def test_vertical_prefit_end_to_end(model_type):
    rng = np.random.default_rng(0)
    n = 60
    X_all = np.concatenate(
        [rng.standard_normal((n, P_OHE)), rng.standard_normal((n, P_OHE))], axis=1
    )
    y = rng.standard_normal(n)
    groups = X_GROUPS + [[c + P_OHE for c in g] for g in X_GROUPS]

    m = PRISMPredictionModel(
        input_size=2 * P_OHE, layers=[8], model_type=model_type,
        n_warmup=20, vertical_prefit=True,
    )
    snaps = m.fit(X_all, y, groups, lambda_path=np.logspace(-1, -2, 2))
    arr = np.array(snaps)
    assert arr.shape == (2, 2 * N_VARS)
    assert np.all(arr >= 0)
