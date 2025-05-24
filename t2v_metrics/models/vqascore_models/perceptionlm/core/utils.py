import math
from dataclasses import dataclass
from functools import partial
from typing import Callable, Optional

import torch


@dataclass
class InitArgs:
    use_gaussian: bool = True  # gaussian vs uniform
    coeff_std: Optional[float] = None  # std coeff multiplier
    no_init: bool = False


def get_init_fn(
    args: InitArgs, input_dim: int, init_depth: Optional[int]
) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Init functions.
    """
    if args.no_init:
        return lambda x: x

    # standard deviation
    std = 1 / math.sqrt(input_dim)
    std = std if args.coeff_std is None else (args.coeff_std * std)

    # rescale with depth
    if init_depth is not None:
        std = std / math.sqrt(2 * init_depth)

    # gaussian vs uniform
    if args.use_gaussian:
        return partial(
            torch.nn.init.trunc_normal_, mean=0.0, std=std, a=-3 * std, b=3 * std
        )
    else:
        bound = math.sqrt(3) * std  # ensure the standard deviation is `std`
        return partial(torch.nn.init.uniform_, a=-bound, b=bound)
