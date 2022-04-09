import numpy as np

from ..utils.param import Param, Parameters
from ..utils.types import Vector
from .base import StochasticProcess1D, StochasticProcess1DMarginal
from .cir import CIR


class HestonMarginal(StochasticProcess1DMarginal):
    def cdf(self, n: Vector) -> Vector:
        """
        Compute the cumulative distribution function of the process.

        :param t: time horizon
        :param n: Location in the stochastic process domain space. If a numpy array,
            the output should have the same shape as the input.
        """
        return n


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

    def __init__(self, variance_process: CIR, rho: float) -> None:
        super().__init__()
        self.variance_process = variance_process
        self.rho = Param("rho", rho, bounds=(-1, 1), description="Correlation")

    @classmethod
    def create(
        cls, vol: float = 0.5, kappa: float = 1, sigma: float = 0.1, rho: float = 0
    ) -> "Heston":
        """Create an Heston model from a long-term volatility,
        mean reversion speed and vol of volatility"""
        return cls(CIR(vol * vol, kappa, sigma), rho)

    @property
    def parameters(self) -> Parameters:
        params = self.variance_process.parameters
        params.append(self.rho)
        return params

    def marginal(self, t: float, N: int = 128) -> StochasticProcess1DMarginal:
        return HestonMarginal(self, t, N)

    def characteristic(self, t: float, u: Vector) -> Vector:
        rho = self.rho.value
        eta = self.variance_process.sigma.value
        eta2 = eta * eta
        theta_kappa = (
            self.variance_process.theta.value * self.variance_process.kappa.value
        )
        # adjusted drift
        kappa = self.variance_process.kappa.value - 1j * u * eta * rho
        u2 = u * u
        gamma = np.sqrt(kappa * kappa + u2 * eta2)
        egt = np.exp(-gamma * t)
        c = (gamma - 0.5 * (gamma - kappa) * (1 - egt)) / gamma
        b = u2 * (1 - egt) / ((gamma + kappa) + (gamma - kappa) * egt)
        a = theta_kappa * (2 * np.log(c) + (gamma - kappa) * t) / eta2
        return np.exp(-a - b * self.variance_process.rate.value)

    def pdf(self, t: float, n: Vector) -> Vector:
        raise NotImplementedError

    def cdf(self, t: float, n: Vector) -> Vector:
        raise NotImplementedError
