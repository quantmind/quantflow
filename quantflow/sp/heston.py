from __future__ import annotations

import numpy as np
from pydantic import Field

from ..utils.distributions import DoubleExponential, Exponential
from ..utils.paths import Paths
from ..utils.types import FloatArrayLike, Vector
from .base import StochasticProcess1D
from .cir import CIR
from .poisson import CompoundPoissonProcess


class Heston(StochasticProcess1D):
    r"""The Heston stochastic volatility model

    THe classical square-root stochastic volatility model of Heston (1993)
    can be regarded as a standard Brownian motion $x_t$ time changed by a CIR
    activity rate process.

    .. math::

        d x_t = d w^1_t \\
        d v_t = (a - \kappa v_t) dt + \nu \sqrt{v_t} dw^2_t
        \rho dt = \E[dw^1 dw^2]
    """

    variance_process: CIR = Field(default_factory=CIR, description="Variance process")
    rho: float = Field(default=0, ge=-1, le=1, description="Correlation")

    @classmethod
    def create(
        cls,
        vol: float = 0.5,
        kappa: float = 1,
        sigma: float = 0.8,
        rho: float = 0,
        theta: float | None = None,
    ) -> Heston:
        rate = vol * vol
        if theta is None:
            theta = rate
        return cls(
            variance_process=CIR(rate=rate, kappa=kappa, sigma=sigma, theta=theta),
            rho=rho,
        )

    def characteristic_exponent(self, t: FloatArrayLike, u: Vector) -> Vector:
        eta = self.variance_process.sigma
        eta2 = eta * eta
        theta_kappa = self.variance_process.theta * self.variance_process.kappa
        # adjusted drift
        kappa = self.variance_process.kappa - 1j * u * eta * self.rho
        u2 = u * u
        gamma = np.sqrt(kappa * kappa + u2 * eta2)
        egt = np.exp(-gamma * t)
        c = (gamma - 0.5 * (gamma - kappa) * (1 - egt)) / gamma
        b = u2 * (1 - egt) / ((gamma + kappa) + (gamma - kappa) * egt)
        a = theta_kappa * (2 * np.log(c) + (gamma - kappa) * t) / eta2
        return a + b * self.variance_process.rate

    def sample(self, n: int, time_horizon: float = 1, time_steps: int = 100) -> Paths:
        dw1 = Paths.normal_draws(n, time_horizon, time_steps)
        dw2 = Paths.normal_draws(n, time_horizon, time_steps)
        return self.sample_from_draws(dw1, dw2)

    def sample_from_draws(self, path1: Paths, *args: Paths) -> Paths:
        if args:
            path2 = args[0]
        else:
            path2 = Paths.normal_draws(path1.samples, path1.t, path1.time_steps)
        dz = path1.data
        dw = self.rho * dz + np.sqrt(1 - self.rho * self.rho) * path2.data
        v = self.variance_process.sample_from_draws(path1)
        dx = dw * np.sqrt(v.data * path1.dt)
        paths = np.zeros(dx.shape)
        paths[1:] = np.cumsum(dx[:-1], axis=0)
        return Paths(t=path1.t, data=paths)


class HestonJ(Heston):
    jumps_up: CompoundPoissonProcess[Exponential] = Field(
        description="positive jumps process"
    )
    jumps_down: CompoundPoissonProcess[Exponential] = Field(
        description="negative jumps process"
    )

    @classmethod
    def create(
        cls,
        vol: float = 0.5,
        kappa: float = 1,
        sigma: float = 0.8,
        rho: float = 0,
        theta: float | None = None,
        jump_intensity: float = 100,  # number of jumps per year
        jump_fraction: float = 0.1,  # percentage of variance due to jumps
        jump_asymmetry: float = 1,
    ) -> HestonJ:
        if jump_fraction <= 0 or jump_fraction >= 1:
            raise ValueError("jump_percentage must be between 0 and 1")
        total_variance = vol * vol
        jump_variance = total_variance * jump_fraction
        diffusion_variance = total_variance - jump_variance
        jump_distribution_variance = 0.5 * jump_variance / jump_intensity
        de = DoubleExponential(
            decay=1.0 / np.sqrt(jump_distribution_variance), k=jump_asymmetry
        )
        return cls(
            variance_process=CIR(
                rate=diffusion_variance,
                kappa=kappa,
                sigma=sigma,
                theta=theta if theta is not None else diffusion_variance,
            ),
            rho=rho,
            jumps_up=CompoundPoissonProcess[Exponential](
                intensity=jump_intensity,
                jumps=Exponential(decay=1 / de.scale_up),
            ),
            jumps_down=CompoundPoissonProcess[Exponential](
                intensity=jump_intensity,
                jumps=Exponential(decay=1 / de.scale_down),
            ),
        )

    def characteristic_exponent(self, t: FloatArrayLike, u: Vector) -> Vector:
        return (
            super().characteristic_exponent(t, u)
            + self.jumps_up.characteristic_exponent(t, u)
            - self.jumps_down.characteristic_exponent(t, u)
        )
