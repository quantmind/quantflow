from typing import Callable

import numpy as np

from .poisson import Param, PoissonProcess

JumpSampler = Callable[[int], np.array]


class CompoundPoissonProcess(PoissonProcess):
    pass


class ExponentialPoissonProcess(CompoundPoissonProcess):
    r"""
    1D Poisson process.

    It's a process where the inter-arrival time is exponentially distributed
    with rate :math:`\lambda`

    .. attribute:: rate

        The arrival rate of events. Must be positive.
    """

    def __init__(self, rate: float, decay: float):
        super().__init__(rate)
        self.decay = Param(
            "decay", decay, bounds=(0, None), description="Jump size decay rate"
        )

    def jumps(self, n: int) -> np.array:
        """Sample jump sizes from an exponential distribution with rate
        parameter :class:b
        """
        exp_rate = 1.0 / self.decay.value
        return np.random.exponential(scale=exp_rate, size=n)
