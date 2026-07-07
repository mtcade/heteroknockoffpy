#
#//  tests/heteroknockofftorch/test_torchUtil.py
#//  heteroknockoffpy
#//
#//  Verifies heteroknockofftorch.torchUtil is importable and its shared
#//  building blocks work.
#//

def test_torch_util_importable():
    from heteroknockoffpy.heteroknockofftorch.torchUtil import _nnModule_dict, _build_sequential
    import torch.nn as nn
    assert "relu" in _nnModule_dict
    assert _nnModule_dict["relu"] is nn.ReLU
    # Verify _build_sequential produces a Sequential with correct input→output sizes.
    net = _build_sequential(input_size=4, layers=[8], activation=nn.ReLU, output_size=2)
    assert isinstance(net, nn.Sequential)
    import torch
    out = net(torch.zeros(3, 4))
    assert out.shape == (3, 2)
