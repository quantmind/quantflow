import enum

import numpy as np
from numpy.random import normal
from pydantic import Field
from scipy import special

from quantflow.utils.types import Vector

from .base import Im, IntensityProcess


class SamplingAlgorithm(str, enum.Enum):
    euler = "euler"
    implicit = "implicit"


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
    sigma: float = Field(default=1.0, gt=0, description="Volatility")
    theta: float = Field(default=1.0, gt=0, description="Mean rate")
    sample_algo: SamplingAlgorithm = Field(
        default=SamplingAlgorithm.implicit, description="Sampling algorithm", repr=False
    )

    @property
    def is_positive(self) -> bool:
        return self.kappa * self.theta >= 0.5 * self.sigma * self.sigma

    def pdf(self, t: float, x: Vector) -> Vector:
        k = self.kappa
        s2 = self.sigma * self.sigma
        ekt = np.exp(-k * t)
        c = 2 * k / (1 - ekt) / s2
        q = 2 * k * self.theta / s2 - 1
        u = c * ekt * self.rate
        v = c * x
        return (
            c
            * np.exp(-v - u)
            * np.power(v / u, 0.5 * q)
            * special.iv(q, 2 * np.sqrt(u * v))
        )

    def mean(self, t: float) -> float:
        ekt = np.exp(-self.kappa * t)
        return self.rate * ekt + self.theta * (1 - ekt)

    def std(self, t: float) -> float:
        kappa = self.kappa
        ekt = np.exp(-kappa * t)
        return np.sqrt(
            self.sigma
            * self.sigma
            * (1 - ekt)
            * (self.rate * ekt + 0.5 * self.theta * (1 - ekt))
            / kappa
        )

    def sample(self, n: int, t: float = 1, steps: int = 0) -> np.ndarray:
        if self.sample_algo == SamplingAlgorithm.euler:
            return self.sample_euler(n, t, steps)
        else:
            return self.sample_implicit(n, t, steps)

    def sample_euler(self, n: int, t: float = 1, steps: int = 0) -> np.ndarray:
        size, dt = self.sample_dt(t, steps)
        kappa = self.kappa
        theta = self.theta
        sdt = self.sigma * np.sqrt(dt)
        paths = np.zeros((size + 1, n))
        paths[0, :] = self.rate
        for p in range(n):
            w = normal(scale=sdt, size=size)
            for i in range(size):
                x = paths[i, p]
                dx = kappa * (theta - x) * dt + np.sqrt(x) * w[i]
                paths[i + 1, p] = x + dx
        return paths

    def sample_implicit(self, n: int, t: float = 1, steps: int = 0) -> np.ndarray:
        """Use an implicit scheme to preserve positivity of the process."""
        size, dt = self.sample_dt(t, steps)
        kappa = self.kappa
        theta = self.theta
        sigma = self.sigma
        kdt2 = 2 * (kappa * dt + 1)
        kts = (kappa * theta - 0.5 * sigma * sigma) * dt
        sdt = self.sigma * np.sqrt(dt)
        paths = np.zeros((size + 1, n))
        paths[0, :] = self.rate
        for p in range(n):
            w = normal(scale=sdt, size=size)
            for i in range(size):
                x = paths[i, p]
                sw = w[i]
                xs = (sw + np.sqrt(sw * sw + 2 * (x + kts) * kdt2)) / kdt2
                paths[i + 1, p] = xs * xs
        return paths

    def characteristic(self, t: float, u: Vector) -> Vector:
        iu = Im * u
        sigma = self.sigma
        kappa = self.kappa
        kt = kappa * t
        ekt = np.exp(kt)
        sigma2 = sigma * sigma
        s2u = iu * sigma2
        c = s2u + (2 * kappa - s2u) * ekt
        b = 2 * kappa * iu / c
        a = 2 * kappa * self.theta * (kt + np.log(2 * kappa / c)) / sigma2
        return np.exp(a + b * self.rate)

    def cumulative_characteristic(self, t: float, u: Vector) -> Vector:
        iu = Im * u
        sigma = self.sigma
        kappa = self.kappa
        sigma2 = sigma * sigma
        gamma = np.sqrt(kappa * kappa - 2 * iu * sigma2)
        egt = np.exp(gamma * t)
        c = (gamma + kappa) * (1 - egt) - 2 * gamma
        d = 2 * gamma * np.exp(0.5 * (gamma + kappa) * t)
        a = 2 * self.theta * kappa * np.log(-d / c) / sigma2
        b = 2 * iu * (1 - egt) / c
        return np.exp(a + b * self.rate)
