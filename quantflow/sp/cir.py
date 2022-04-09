import numpy as np
from numpy.random import normal

from ..utils.param import Param, Parameters
from ..utils.types import Vector
from .base import Im, IntensityProcess, StochasticProcess1DMarginal


class CIR(IntensityProcess):
    r"""The Cox–Ingersoll–Ross (CIR) model is a mean-reverting square-root diffusion
    process.

    The process :math:`x_t` that satisfies the following stochastic
    differential equation with Wiener process :math:`w_t`:

    .. math::
        dx_t = \kappa (\theta - x_t) dt + \sigma \sqrt{x_t}dw_t

    This process is guaranteed to be positive if

    .. math::
        2 \kappa \theta >= \sigma^2

    :param rate: The initial value of the process :math:`x_0`
    :param kappa: Mean reversion speed :math:`\kappa`
    :param sigma: Volatility parameter :math:`\sigma`
    :param theta: Long term mean rate :math:`\theta`
    """

    def __init__(
        self, rate: float, kappa: float, sigma: float, theta: float = 0
    ) -> None:
        super().__init__(rate, kappa)
        self.sigma = Param("sigma", sigma, bounds=(0, None), description="Volatility")
        self.theta = Param(
            "theta", theta or self.rate.value, bounds=(0, None), description="Mean rate"
        )

    @property
    def parameters(self) -> Parameters:
        return Parameters(self.rate, self.kappa, self.sigma, self.theta)

    @property
    def is_positive(self) -> bool:
        return (
            self.kappa.value * self.theta.value
            >= 0.5 * self.sigma.value * self.sigma.value
        )

    def marginal(self, t: float, N: int) -> StochasticProcess1DMarginal:
        return None

    def cdf(self, t: float, n: Vector) -> Vector:
        pass

    def mean(self, t: float) -> float:
        ekt = np.exp(-self.kappa.value * t)
        return self.rate.value * ekt + self.theta.value * (1 - ekt)

    def std(self, t: float) -> float:
        kappa = self.kappa.value
        ekt = np.exp(-kappa * t)
        return np.sqrt(
            self.sigma.value
            * self.sigma.value
            * (1 - ekt)
            * (self.rate.value * ekt + 0.5 * self.theta.value * (1 - ekt))
            / kappa
        )

    def sample(self, n: int, t: float = 1, steps: int = 0) -> np.array:
        size, dt = self.sample_dt(t, steps)
        kappa = self.kappa.value
        theta = self.theta.value
        sdt = self.sigma.value * np.sqrt(dt)
        paths = np.zeros((size + 1, n))
        paths[0, :] = self.rate.value
        for p in range(n):
            w = normal(scale=sdt, size=size)
            for i in range(size):
                x = paths[i, p]
                dx = kappa * (theta - x) * dt + np.sqrt(x) * w[i]
                paths[i + 1, p] = x + dx
        return paths

    def characteristic(self, t: float, u: Vector) -> Vector:
        iu = Im * u
        sigma = self.sigma.value
        kappa = self.kappa.value
        kt = kappa * t
        ekt = np.exp(kt)
        sigma2 = sigma * sigma
        s2u = iu * sigma2
        c = s2u + (2 * kappa - s2u) * ekt
        b = 2 * kappa * iu / c
        a = 2 * kappa * self.theta.value * (kt + np.log(2 * kappa / c)) / sigma2
        return np.exp(a + b * self.rate.value)

    def cumulative_characteristic(self, t: float, u: Vector) -> Vector:
        iu = Im * u
        sigma = self.sigma.value
        kappa = self.kappa.value
        sigma2 = sigma * sigma
        gamma = np.sqrt(kappa * kappa - 2 * iu * sigma2)
        egt = np.exp(gamma * t)
        c = (gamma + kappa) * (1 - egt) - 2 * gamma
        d = 2 * gamma * np.exp(0.5 * (gamma + kappa) * t)
        a = 2 * self.theta.value * kappa * np.log(-d / c) / sigma2
        b = 2 * iu * (1 - egt) / c
        return np.exp(a + b * self.rate.value)
