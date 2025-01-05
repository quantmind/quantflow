from __future__ import annotations

from typing import Generic, Self

import numpy as np
from pydantic import Field

from quantflow.ta.paths import Paths
from quantflow.utils.distributions import DoubleExponential
from quantflow.utils.types import FloatArrayLike, Vector

from .base import StochasticProcess1D
from .cir import CIR
from .poisson import CompoundPoissonProcess, D


class Heston(StochasticProcess1D):
    r"""The Heston stochastic volatility model

    The classical square-root stochastic volatility model of Heston (1993)
    can be regarded as a standard Brownian motion :math:`x_t` time changed by a CIR
    activity rate process.

    .. math::
        \begin{align}
            d x_t &= d w^1_t \\
            d v_t &= \kappa (\theta - v_t) dt + \nu \sqrt{v_t} dw^2_t \\
            \rho dt &= {\tt E}[dw^1 dw^2]
        \end{align}
    """

    variance_process: CIR = Field(default_factory=CIR, description="Variance process")
    """The variance process is a Cox-Ingersoll-Ross (:class:`.CIR`) process"""
    rho: float = Field(default=0, ge=-1, le=1, description="Correlation")
    """Correlation between the Brownian motions - provides the leverage effect"""

    @classmethod
    def create(
        cls,
        *,
        rate: float = 1.0,
        vol: float = 0.5,
        kappa: float = 1,
        sigma: float = 0.8,
        rho: float = 0,
        theta: float | None = None,
    ) -> Self:
        r"""Create an Heston model.

        To understand the parameters lets introduce the following notation:

        .. math::
            \begin{align}
                {\tt var} &= {\tt vol}^2 \\
                v_0 &= {\tt rate}\cdot{\tt var}
            \end{align}

        :param rate: define the initial value of the variance process
        :param vol: The standard deviation of the price process, normalized by the
            square root of time, as time tends to infinity
            (the long term standard deviation)
        :param kappa: The mean reversion speed for the variance process
        :param sigma: The volatility of the variance process
        :param rho: The correlation between the Brownian motions of the
            variance and price processes
        :param theta: The long-term mean of the variance process, if `None`, it
            defaults to the variance given by :math:`{\tt var}`
        """
        variance = vol * vol
        return cls(
            variance_process=CIR(
                rate=rate * variance,
                kappa=kappa,
                sigma=sigma,
                theta=theta if theta is not None else variance,
            ),
            rho=rho,
        )

    def characteristic_exponent(self, t: FloatArrayLike, u: Vector) -> Vector:
        """The characteristic exponent of the Heston model has a closed form"""
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


class HestonJ(Heston, Generic[D]):
    r"""The Heston stochastic volatility model with jumps

    The Heston model with jumps is an extension of the classical square-root
    stochastic volatility model of Heston (1993) with the addition of jump
    processes. The jumps are modeled as compound Poisson processes

    .. math::
        d x_t &= d w^1_t + d N_t\\
        d v_t &= \kappa (\theta - v_t) dt + \nu \sqrt{v_t} dw^2_t \\
        \rho dt &= {\tt E}[dw^1 dw^2]
    """

    jumps: CompoundPoissonProcess[D] = Field(description="jumps process")
    """Jump process driven by a compound Poisson process"""

    @classmethod
    def exponential(
        cls,
        *,
        rate: float = 1.0,
        vol: float = 0.5,
        kappa: float = 1,
        sigma: float = 0.8,
        rho: float = 0,
        theta: float | None = None,
        jump_intensity: float = 100,  # number of jumps per year
        jump_fraction: float = 0.1,  # percentage of variance due to jumps
        jump_asymmetry: float = 1,
    ) -> HestonJ[DoubleExponential]:
        r"""Create an Heston model with :class:`.DoubleExponential` jumps.

        To understand the parameters lets introduce the following notation:

        .. math::
            \begin{align}
                {\tt var} &= {\tt vol}^2 \\
                {\tt var}_j &= {\tt var} \cdot {\tt jump\_fraction} \\
                {\tt var}_d &= {\tt var} - {\tt var}_j \\
                v_0 &= {\tt rate}\cdot{\tt var}_d
            \end{align}

        :param rate: define the initial value of the variance process
        :param vol: The standard deviation of the price process, normalized by the
            square root of time, as time tends to infinity
            (the long term standard deviation)
        :param kappa: The mean reversion speed for the variance process
        :param sigma: The volatility of the variance process
        :param rho: The correlation between the Brownian motions of the
            variance and price processes
        :param theta: The long-term mean of the variance process, if `None`, it
            defaults to the diffusion variance given by :math:`{\tt var}_d`
        :param jump_intensity: The number of jumps per year
        :param jump_fraction: The percentage of variance due to jumps
        :param jump_asymmetry: The asymmetry of the jump distribution (1 for symmetric)
        :param jump_distribution: The distribution of the jumps, either 'normal'
            or 'double_exponential' (default)
        """
        if jump_fraction <= 0 or jump_fraction >= 1:
            raise ValueError("jump_percentage must be between 0 and 1")
        total_variance = vol * vol
        jump_variance = total_variance * jump_fraction
        diffusion_variance = total_variance - jump_variance
        jump_distribution_variance = jump_variance / jump_intensity
        jumps = DoubleExponential.from_moments(
            variance=jump_distribution_variance, kappa=jump_asymmetry
        )
        return HestonJ[DoubleExponential](
            variance_process=CIR(
                rate=rate * diffusion_variance,
                kappa=kappa,
                sigma=sigma,
                theta=theta if theta is not None else diffusion_variance,
            ),
            rho=rho,
            jumps=CompoundPoissonProcess(
                intensity=jump_intensity,
                jumps=jumps,
            ),
        )

    def characteristic_exponent(self, t: FloatArrayLike, u: Vector) -> Vector:
        """The characteristic exponent is given by the sum of the exponent of the
        classic Heston model and the exponent of the jumps"""
        return super().characteristic_exponent(
            t, u
        ) + self.jumps.characteristic_exponent(t, u)
