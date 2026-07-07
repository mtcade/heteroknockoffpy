#
#//  tests/heteroknockofftorch/test_torchKnockoffs.py
#//  heteroknockoffpy
#//
#//  Verifies heteroknockofftorch.torchKnockoffs is importable and exposes
#//  its public API.
#//

def test_torch_knockoffs_importable():
    from heteroknockoffpy.heteroknockofftorch.torchKnockoffs import (
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
