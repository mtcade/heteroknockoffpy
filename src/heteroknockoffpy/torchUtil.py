#
#//  torchUtil.py
#//  heteroknockoffpy
#//
#//  Created by Evan Mason on 6/2/26.
#//

import torch.nn as nn
import math
from typing import Sequence, Type


_nnModule_dict: dict[str, type[nn.Module]] = {
    'leaky_relu': nn.LeakyReLU,
    'relu':       nn.ReLU,
    'sigmoid':    nn.Sigmoid,
}


def _build_sequential(
    input_size: int,
    layers: Sequence[int],
    activation: Type[nn.Module],
    output_size: int,
    output_activation: Type[nn.Module] | None = None,
) -> nn.Sequential:
    parts: list[nn.Module] = []
    prev = input_size
    for width in layers:
        parts += [nn.Linear(prev, width), activation()]
        prev = width
    parts.append(nn.Linear(prev, output_size))
    if output_activation is not None:
        parts.append(output_activation())
    return nn.Sequential(*parts)
#/def _build_sequential
