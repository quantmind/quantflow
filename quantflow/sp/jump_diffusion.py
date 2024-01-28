from __future__ import annotations

from typing import Generic

import numpy as np
from pydantic import Field

from ..utils.distributions import Normal
from ..utils.paths import Paths
from ..utils.types import FloatArrayLike, Vector
from .base import StochasticProcess1D
from .poisson import CompoundPoissonProcess, D
from .weiner import WeinerProcess


class JumpDiffusion(StochasticProcess1D, Generic[D]):
    r"""A jump-diffusion model

    math::
        dx_t = \sigma d w_t + d N_t

        where N_t is a compound poisson process with intensity \lambda
        and jump distribution D
    """

    diffusion: WeinerProcess = Field(
        default_factory=WeinerProcess, description="diffusion"
    )
    jumps: CompoundPoissonProcess[D] = Field(description="jump process")

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


class Merton(JumpDiffusion[Normal]):
    """Merton jump-diffusion model"""

    @classmethod
    def create(
        cls,
        vol: float = 0.5,
        diffusion_percentage: float = 0.5,
        jump_intensity: float = 100,
        jump_skew: float = 0.0,
    ) -> Merton:
        variance = vol * vol
        jump_std = 1.0
        jump_mean = 0.0
        if diffusion_percentage > 1:
            raise ValueError("diffusion_percentage must be less than 1")
        elif diffusion_percentage < 0:
            raise ValueError("diffusion_percentage must be greater than 0")
        elif diffusion_percentage == 1:
            jump_intensity = 0
        else:
            jump_std = np.sqrt(variance * (1 - diffusion_percentage) / jump_intensity)
            jump_mean = jump_skew / jump_intensity
        return cls(
            diffusion=WeinerProcess(sigma=np.sqrt(variance * diffusion_percentage)),
            jumps=CompoundPoissonProcess[Normal](
                intensity=jump_intensity, jumps=Normal(mu=jump_mean, sigma=jump_std)
            ),
        )
