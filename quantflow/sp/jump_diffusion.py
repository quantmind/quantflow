from __future__ import annotations

from typing import Generic

import numpy as np
from pydantic import Field
from typing_extensions import Annotated, Doc

from ..ta.paths import Paths
from ..utils.types import FloatArrayLike, Vector
from .base import StochasticProcess1D
from .poisson import CompoundPoissonProcess, D
from .weiner import WeinerProcess


class JumpDiffusion(StochasticProcess1D, Generic[D]):
    r"""A generic jump-diffusion model

    \begin{equation}
        dx_t = \sigma d w_t + d N_t
    \end{equation}

    where $w_t$ is a [WeinerProcess][quantflow.sp.weiner.WeinerProcess] process
    with standard deviation $\sigma$ and $N_t$ is a
    [CompoundPoissonProcess][quantflow.sp.poisson.CompoundPoissonProcess]
    with intensity $\lambda$ and generic jump distribution $D$
    """

    diffusion: WeinerProcess = Field(
        default_factory=WeinerProcess, description="diffusion process"
    )
    jumps: CompoundPoissonProcess[D] = Field(description="The jump process")

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
        jump_distribution: Annotated[
            type[D],
            Doc(
                "The distribution of jump sizes. Currently "
                "[Normal][quantflow.utils.distributions.Normal] and "
                "[DoubleExponential][quantflow.utils.distributions.DoubleExponential] "
                "are supported. If the jump distribution is set to the Normal "
                "distribution, the model reduces to a Merton jump-diffusion."
            ),
        ],
        vol: Annotated[float, Doc("total standard deviation per unit time")] = 0.5,
        jump_intensity: Annotated[
            float, Doc("The expected number of jumps per unit time")
        ] = 100,
        jump_fraction: Annotated[
            float, Doc("The fraction of variance due to jumps (between 0 and 1)")
        ] = 0.5,
        jump_asymmetry: Annotated[
            float,
            Doc(
                "The asymmetry of the jump distribution "
                "(0 for symmetric, only used by distributions with asymmetry)"
            ),
        ] = 0.0,
    ) -> JumpDiffusion[D]:
        """Create a jump-diffusion model with a given jump distribution, volatility
        and jump fraction.
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
