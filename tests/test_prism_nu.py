#
#//  test_prism_nu.py
#//  heteroknockoffpy
#//
#//  Structural unit tests for the NU (numeric-unified) PRISM network architectures.
#//  These construct networks directly and check architectural properties (weight
#//  tying, regularization exclusion, positional get_group_importances) via a single
#//  forward/backward pass -- no optimizer loop, epochs, or lambda-path training.
#//
import numpy as np
import pytest
import torch
import torch.nn as nn

from heteroknockoffpy.torchImportances import (
    PRISMPredictionModel,
    _PRISMNetworkMLP_NU,
    _PRISMNetworkPairwise_NU,
)


NU_CLASSES = (_PRISMNetworkMLP_NU, _PRISMNetworkPairwise_NU)
NU_MODEL_TYPES = ("mlp_nu", "pairwise_nu")

# Fixture structure: 2 numeric vars, 1 categorical (K=3), 1 binary categorical
# (K=1 dummy after drop_first) -- covers the drop_first edge case explicitly.
X_GROUPS = [[0], [1], [2, 3, 4], [5]]
VAR_IS_CATEGORICAL = [False, False, True, True]
N_VARS = len(X_GROUPS)
P_OHE = 6


def _make_net(cls):
    return cls(
        layers=[8],
        activation_class=nn.ReLU,
        x_groups=X_GROUPS,
        var_is_categorical=VAR_IS_CATEGORICAL,
        output_size=1,
    )


def _full_groups():
    """The (X-side, Xk-side) groups list of length 2*N_VARS that callers pass in."""
    return X_GROUPS + [[c + P_OHE for c in g] for g in X_GROUPS]


# ---------------------------------------------------------------------------
# A. Construction / validation (via PRISMPredictionModel)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model_type", NU_MODEL_TYPES)
def test_nu_requires_groups_and_ohedict(model_type):
    with pytest.raises(ValueError):
        PRISMPredictionModel(input_size=2 * P_OHE, layers=[8], model_type=model_type)


def _make_ohedict():
    """Matches the real _prism_setup shape: one entry per variable per side
    (X columns first, then Xk~-suffixed columns), consistent with `groups`."""
    return {
        "a": 0, "b": 1, "c": (2, 3, 4), "d": (5,),
        "a~": 6, "b~": 7, "c~": (8, 9, 10), "d~": (11,),
    }


@pytest.mark.parametrize("model_type", NU_MODEL_TYPES)
def test_nu_requires_even_input_size(model_type):
    groups = _full_groups()
    oheDict = _make_ohedict()
    with pytest.raises(ValueError):
        PRISMPredictionModel(
            input_size=2 * P_OHE + 1, layers=[8], model_type=model_type,
            groups=groups, oheDict=oheDict,
        )


@pytest.mark.parametrize("model_type", NU_MODEL_TYPES)
def test_nu_binary_categorical_gets_combiner_not_passthrough(model_type):
    """A K=1-dummy categorical (oheDict value is a length-1 tuple) must still get a
    real combining module, not be treated as numeric passthrough, even though its
    column count alone is indistinguishable from a numeric variable."""
    groups = _full_groups()
    oheDict = _make_ohedict()
    m = PRISMPredictionModel(
        input_size=2 * P_OHE, layers=[8], model_type=model_type,
        groups=groups, oheDict=oheDict,
    )
    assert "3" in m.model.combine  # var index 3 == "d", the binary categorical
    assert "0" not in m.model.combine and "1" not in m.model.combine  # numeric vars
    assert isinstance(m.model.combine["3"], nn.Linear)
    assert m.model.combine["3"].weight.shape == (1, 1)
    assert m.model.combine["3"].bias is None


# ---------------------------------------------------------------------------
# B. Direct network unit tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls", NU_CLASSES)
def test_weight_tying_identity(cls):
    """Same combiner weight applied to X-side and Xk-side blocks: if the raw one-hot
    inputs for a categorical variable are identical on both sides, the combined
    scalars must be bit-for-bit identical."""
    net = _make_net(cls)
    with torch.no_grad():
        net.combine["2"].weight.copy_(torch.tensor([[1.5, -2.0, 0.5]]))

    onehot = torch.tensor([0.0, 1.0, 0.0])  # category 1 of the K=3 variable
    z = torch.zeros(1, 2 * P_OHE)
    z[0, 2:5] = onehot          # X-side block for var 2
    z[0, P_OHE + 2 : P_OHE + 5] = onehot  # Xk-side block, identical

    combined = net._combine(z)
    assert torch.allclose(combined[0, 2], combined[0, N_VARS + 2])


@pytest.mark.parametrize("cls", NU_CLASSES)
def test_manual_combining_correctness(cls):
    """No bias, no activation: output must equal exactly w . onehot."""
    net = _make_net(cls)
    w = torch.tensor([[2.0, -3.0, 0.25]])
    with torch.no_grad():
        net.combine["2"].weight.copy_(w)

    for cat in range(3):
        onehot = torch.zeros(1, 3)
        onehot[0, cat] = 1.0
        out = net.combine["2"](onehot)
        assert torch.allclose(out, w[:, cat : cat + 1])


@pytest.mark.parametrize("cls", NU_CLASSES)
def test_numeric_passthrough(cls):
    """Numeric variable columns must pass through _combine_side unchanged."""
    net = _make_net(cls)
    side = torch.zeros(1, P_OHE)
    side[0, 0] = 3.14   # var 0, numeric
    side[0, 1] = -1.23  # var 1, numeric
    combined = net._combine_side(side)
    assert torch.allclose(combined[0, 0], torch.tensor(3.14))
    assert torch.allclose(combined[0, 1], torch.tensor(-1.23))


@pytest.mark.parametrize("cls", NU_CLASSES)
def test_swap_antisymmetry(cls):
    """Swapping a variable's X-side and Xk-side one-hot blocks in the raw input must
    swap the corresponding combined scalars (X-half <-> Xk-half) -- the property the
    tied-weight design exists to guarantee for valid knockoff swap statistics."""
    net = _make_net(cls)
    with torch.no_grad():
        net.combine["2"].weight.copy_(torch.tensor([[1.0, 2.0, 3.0]]))

    rng = np.random.default_rng(0)
    z = torch.zeros(1, 2 * P_OHE)
    x_block = torch.tensor(rng.standard_normal(P_OHE), dtype=torch.float32)
    xk_block = torch.tensor(rng.standard_normal(P_OHE), dtype=torch.float32)
    z[0, :P_OHE] = x_block
    z[0, P_OHE:] = xk_block

    z_swapped = torch.zeros_like(z)
    z_swapped[0, :P_OHE] = xk_block
    z_swapped[0, P_OHE:] = x_block

    combined = net._combine(z)
    combined_swapped = net._combine(z_swapped)

    # Every variable's X-half/Xk-half pair should swap.
    assert torch.allclose(combined[0, :N_VARS], combined_swapped[0, N_VARS:])
    assert torch.allclose(combined[0, N_VARS:], combined_swapped[0, :N_VARS])


@pytest.mark.parametrize("cls", NU_CLASSES)
def test_forward_shape_and_discrimination_width(cls):
    net = _make_net(cls)
    z = torch.randn(5, 2 * P_OHE)
    out = net(z)
    assert out.shape == (5,)

    if cls is _PRISMNetworkMLP_NU:
        assert net.net[0].weight.shape[1] == 2 * N_VARS
    else:
        assert len(net.v) == 2 * N_VARS


# ---------------------------------------------------------------------------
# C. Regularization / optimizer plumbing (single backward pass, no training loop)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls", NU_CLASSES)
def test_group_regularization_excludes_combining_layer(cls):
    net = _make_net(cls)
    groups = _full_groups()
    net._precompute_group_reg(groups, device="cpu")

    loss = net.group_regularization(lambda_val=1.0, a=1.0, groups=groups)
    loss.backward()

    for p in net.combine.parameters():
        assert p.grad is None

    if cls is _PRISMNetworkMLP_NU:
        disc_grad = net.net[0].weight.grad
    else:
        disc_grad = net.v.grad
    assert disc_grad is not None
    assert torch.any(disc_grad != 0)


@pytest.mark.parametrize("cls", NU_CLASSES)
def test_no_decay_parameters_identity(cls):
    net = _make_net(cls)
    no_decay = net.no_decay_parameters()
    combine_params = list(net.combine.parameters())
    assert {id(p) for p in no_decay} == {id(p) for p in combine_params}
    assert len(no_decay) == sum(VAR_IS_CATEGORICAL)  # 2 categorical vars


@pytest.mark.parametrize("cls", NU_CLASSES)
def test_warmup_param_group_partition(cls):
    """Replicates the split PRISMPredictionModel.fit performs for the warmup
    optimizer: every model parameter must land in exactly one of decay/no_decay."""
    net = _make_net(cls)
    no_decay_ids = {id(p) for p in net.no_decay_parameters()}
    all_params = list(net.parameters())
    decay_params = [p for p in all_params if id(p) not in no_decay_ids]
    no_decay_params = [p for p in all_params if id(p) in no_decay_ids]

    assert len(decay_params) + len(no_decay_params) == len(all_params)
    assert {id(p) for p in decay_params} & {id(p) for p in no_decay_params} == set()
    assert {id(p) for p in decay_params} | {id(p) for p in no_decay_params} == {
        id(p) for p in all_params
    }


# ---------------------------------------------------------------------------
# D. get_group_importances correctness (no training)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls", NU_CLASSES)
def test_get_group_importances_ignores_group_content(cls):
    """Only len(groups)/position is used -- the actual OHE-index content of `groups`
    must be irrelevant, since every variable is already exactly one column here."""
    net = _make_net(cls)
    groups = _full_groups()
    net._precompute_group_reg(groups, device="cpu")

    if cls is _PRISMNetworkMLP_NU:
        with torch.no_grad():
            net.net[0].weight.copy_(torch.randn_like(net.net[0].weight))
    else:
        with torch.no_grad():
            net.v.copy_(torch.randn_like(net.v))

    real_result = net.get_group_importances(groups)

    # Deliberately scrambled/nonsensical index content, same length.
    adversarial_groups = [[999] for _ in range(2 * N_VARS)]
    adversarial_result = net.get_group_importances(adversarial_groups)

    assert np.allclose(real_result, adversarial_result)


@pytest.mark.parametrize("cls", NU_CLASSES)
def test_get_group_importances_length_assertion(cls):
    net = _make_net(cls)
    groups = _full_groups()
    net._precompute_group_reg(groups, device="cpu")
    with pytest.raises(AssertionError):
        net.get_group_importances(groups[:-1])


@pytest.mark.parametrize("cls", NU_CLASSES)
def test_precompute_group_reg_length_assertion(cls):
    net = _make_net(cls)
    with pytest.raises(AssertionError):
        net._precompute_group_reg(_full_groups()[:-1], device="cpu")
