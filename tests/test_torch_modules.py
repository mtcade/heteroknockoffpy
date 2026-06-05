#
#//  test_torch_modules.py
#//  heteroknockoffpy
#//
#//  Verifies the torch module split: torchNetworks is gone, torchKnockoffs /
#//  torchImportances / torchUtil are importable and expose their public API.
#//
import pytest


def test_torch_networks_raises_import_error():
    """torchNetworks must raise ImportError directing users to the new modules."""
    import importlib, sys
    # Clear any cached (failed) entry so the module re-executes.
    sys.modules.pop("heteroknockoffpy.torchNetworks", None)
    with pytest.raises(ImportError, match="torchNetworks has been split"):
        importlib.import_module("heteroknockoffpy.torchNetworks")


def test_torch_util_importable():
    from heteroknockoffpy.torchUtil import _nnModule_dict, _build_sequential
    import torch.nn as nn
    assert "relu" in _nnModule_dict
    assert _nnModule_dict["relu"] is nn.ReLU
    # Verify _build_sequential produces a Sequential with correct input→output sizes.
    net = _build_sequential(input_size=4, layers=[8], activation=nn.ReLU, output_size=2)
    assert isinstance(net, nn.Sequential)
    import torch
    out = net(torch.zeros(3, 4))
    assert out.shape == (3, 2)


def test_torch_knockoffs_importable():
    from heteroknockoffpy.torchKnockoffs import (
        TorchGAN,
        KnockoffGenerator,
        KnockoffDiscriminator,
        KnockoffWGANDiscriminator,
        KnockoffMINE,
        sample_Z,
    )
    import torch
    # Smoke-test sample_Z and a forward pass through KnockoffGenerator.
    shape = (10, 4)
    gen = KnockoffGenerator(shape)
    Z = sample_Z(m=3, n=4, x_name="Normal")
    X = torch.zeros(3, 4)
    out = gen(X, Z)
    assert out.shape == (3, 4)


def test_torch_importances_importable():
    from heteroknockoffpy.torchImportances import PRISMPredictionModel
    import numpy as np
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
