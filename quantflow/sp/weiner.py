from functools import lru_cache

import numpy as np
from scipy.stats import norm

from ..utils.param import Param, Parameters
from ..utils.types import Vector
from .base import StochasticProcess1D, StochasticProcess1DMarginal


class WeinerMarginal(StochasticProcess1DMarginal):
    def variance(self) -> float:
        s = self.process.sigma.value
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

    def __init__(self, sigma: float = 1) -> None:
        super().__init__()
        self.sigma = Param("sigma", sigma, bounds=(0, None), description="volatility")

    @property
    def parameters(self) -> Parameters:
        return Parameters(self.sigma)

    def marginal(self, t: float, N: int = 128) -> StochasticProcess1DMarginal:
        return WeinerMarginal(self, t, N)

    def characteristic(self, t: float, u: Vector) -> Vector:
        su = self.sigma.value * u
        return np.exp(-0.5 * su * su * t)

    def cdf(self):
        pass
