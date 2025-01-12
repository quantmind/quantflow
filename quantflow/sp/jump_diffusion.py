from __future__ import annotations

from typing import Generic

import numpy as np
from pydantic import Field

from ..ta.paths import Paths
from ..utils.types import FloatArrayLike, Vector
from .base import StochasticProcess1D
from .poisson import CompoundPoissonProcess, D
from .weiner import WeinerProcess


class JumpDiffusion(StochasticProcess1D, Generic[D]):
    r"""A generic jump-diffusion model

    .. math::
        dx_t = \sigma d w_t + d N_t

    where :math:`w_t` is a Weiner process with standard deviation :math:`\sigma`
    and :math:`N_t` is a :class:`.CompoundPoissonProcess`
    with intensity :math:`\lambda` and generic jump distribution `D`
    """

    diffusion: WeinerProcess = Field(
        default_factory=WeinerProcess, description="diffusion"
    )
    """The diffusion process is a standard :class:`.WeinerProcess`"""
    jumps: CompoundPoissonProcess[D] = Field(description="jump process")
    """The jump process is a generic :class:`.CompoundPoissonProcess`"""

    def characteristic_exponent(self, t: FloatArrayLike, u: Vector) -> Vector:
        return self.diffusion.characteristic_exponent(
            t, u
        ) + self.jumps.characteristic_exponent(t, u)

    def sample(self, n: int, time_horizon: float = 1, time_steps: int = 100) -> Paths:
        dw1 = Paths.normal_draws(n, time_horizon, time_steps)
        return self.sample_from_draws(dw1)

    def sample_from_draws(self, path_w: Paths, *args: Paths) -> Paths:
        if args:
            path_j = args[0]
        else:
            path_j = self.jumps.sample(path_w.samples, path_w.t, path_w.time_steps)
        path_w = self.diffusion.sample_from_draws(path_w)
        return Paths(t=path_w.t, data=path_w.data + path_j.data)

    def analytical_mean(self, t: FloatArrayLike) -> FloatArrayLike:
        return self.diffusion.analytical_mean(t) + self.jumps.analytical_mean(t)

    def analytical_variance(self, t: FloatArrayLike) -> FloatArrayLike:
        return self.diffusion.analytical_variance(t) + self.jumps.analytical_variance(t)

    @classmethod
    def create(
        cls,
        jump_distribution: type[D],
        vol: float = 0.5,
        jump_intensity: float = 100,
        jump_fraction: float = 0.5,
        jump_asymmetry: float = 0.0,
    ) -> JumpDiffusion[D]:
        """Create a jump-diffusion model with a given jump distribution, volatility
        and jump fraction.

        :param jump_distribution: The distribution of jump sizes (currently only
            :class:`.Normal` and :class:`.DoubleExponential` are supported)
        :param vol: total annualized standard deviation
        :param jump_intensity: The average number of jumps per year
        :param jump_fraction: The fraction of variance due to jumps (between 0 and 1)
        :param jump_asymmetry: The asymmetry of the jump distribution (0 for symmetric,
            only used by distributions with asymmetry)

        If the jump distribution is set to the :class:`.Normal` distribution, the
        model reduces to a Merton jump-diffusion model.
        """
        variance = vol * vol
        if jump_fraction >= 1:
            raise ValueError("jump_fraction must be less than 1")
        elif jump_fraction <= 0:
            raise ValueError("jump_fraction must be greater than 0")
        else:
            jump_variance = variance * jump_fraction
            jump_distribution_variance = jump_variance / jump_intensity
            jumps = jump_distribution.from_variance_and_asymmetry(
                jump_distribution_variance, jump_asymmetry
            )
            return cls(
                diffusion=WeinerProcess(sigma=np.sqrt(variance * (1 - jump_fraction))),
                jumps=CompoundPoissonProcess(intensity=jump_intensity, jumps=jumps),
            )
