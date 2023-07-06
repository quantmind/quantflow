from __future__ import annotations

from functools import lru_cache

import numpy as np
from pydantic import Field
from scipy.stats import norm

from ..utils.types import Vector
from .base import StochasticProcess1D, StochasticProcess1DMarginal


class Weiner(StochasticProcess1D):
    r"""The Heston stochastic volatility model

    THe classical square-root stochastic volatility model of Heston (1993)
    can be regarded as a standard Brownian motion $x_t$ time changed by a CIR
    activity rate process.

    .. math::

        d x_t = d w^1_t \\
        d v_t = (a - \kappa v_t) dt + \nu \sqrt{v_t} dw^2_t
        \rho dt = \E[dw^1 dw^2]
    """
    sigma: float = Field(default=1, ge=0, description="volatility")

    def marginal(self, t: float, N: int = 128) -> StochasticProcess1DMarginal:
        return WeinerMarginal(self, t, N)

    def characteristic(self, t: float, u: Vector) -> Vector:
        su = self.sigma * u
        return np.exp(-0.5 * su * su * t)


class WeinerMarginal(StochasticProcess1DMarginal[Weiner]):
    def variance(self) -> float:
        s = self.process.sigma
        return s * s * self.t

    @lru_cache
    def create_pdf(self) -> Vector:
        u = -2 * np.pi * np.fft.rfftfreq(self.N)
        psi = self.characteristic(u)
        return np.fft.irfft(psi)

    def pdf(self, n: Vector) -> Vector:
        return norm.pdf(n, scale=self.std())

    def cdf(self, n: Vector) -> Vector:
        """
        Compute the cumulative distribution function of the process.

        :param t: time horizon
        :param n: Location in the stochastic process domain space. If a numpy array,
            the output should have the same shape as the input.
        """
        return n
