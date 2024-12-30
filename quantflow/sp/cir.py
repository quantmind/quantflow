import enum

import numpy as np
from pydantic import Field
from scipy import special
from scipy.optimize import Bounds

from quantflow.utils.types import FloatArrayLike, Vector

from ..ta.paths import Paths
from .base import Im, IntensityProcess


class SamplingAlgorithm(str, enum.Enum):
    euler = "euler"
    milstein = "milstein"
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
        default=SamplingAlgorithm.implicit, description="Sampling algorithm"
    )

    @property
    def is_positive(self) -> bool:
        return self.kappa * self.theta >= 0.5 * self.sigma2

    @property
    def sigma2(self) -> float:
        return self.sigma * self.sigma

    def sample(
        self, paths: int, time_horizon: float = 1, time_steps: int = 100
    ) -> Paths:
        draws = Paths.normal_draws(paths, time_horizon, time_steps)
        return self.sample_from_draws(draws)

    def sample_from_draws(self, paths: Paths, *args: Paths) -> Paths:
        match self.sample_algo:
            case SamplingAlgorithm.euler:
                return self.sample_euler(paths)
            case SamplingAlgorithm.milstein:
                return self.sample_euler(paths, 0.25)
            case SamplingAlgorithm.implicit:
                return self.sample_implicit(paths)

    def sample_euler(self, draws: Paths, ic: float = 0.0) -> Paths:
        kappa = self.kappa
        theta = self.theta
        dt = draws.dt
        sdt = self.sigma * np.sqrt(dt)
        sdt2 = sdt * sdt
        paths = np.zeros(draws.data.shape)
        paths[0, :] = self.rate
        for t in range(draws.time_steps):
            w = sdt * draws.data[t, :]
            x = paths[t, :]
            xplus = np.clip(x, 0, None)
            dx = kappa * (theta - xplus) * dt + np.sqrt(xplus) * w + ic * (w * w - sdt2)
            paths[t + 1, :] = x + dx
        return Paths(t=draws.t, data=paths)

    def sample_implicit(self, draws: Paths) -> Paths:
        """Use an implicit scheme to preserve positivity of the process."""
        kappa = self.kappa
        theta = self.theta
        dt = draws.dt
        kdt2 = 2 * (1 + kappa * dt)
        kts = (kappa * theta - 0.5 * self.sigma2) * dt
        sdt = self.sigma * np.sqrt(dt)
        paths = np.zeros(draws.data.shape)
        paths[0, :] = self.rate
        for t in range(draws.time_steps):
            w = sdt * draws.data[t, :]
            x = paths[t, :]
            w2p = np.clip(w * w + 2 * (x + kts) * kdt2, 0, None)
            xs = (w + np.sqrt(w2p)) / kdt2
            paths[t + 1, :] = xs * xs
        return Paths(t=draws.t, data=paths)

    def characteristic_exponent(self, t: Vector, u: Vector) -> Vector:
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
        return -a - b * self.rate

    def integrated_log_laplace(self, t: Vector, u: Vector) -> Vector:
        """Integrated log Laplace transform of the process

        This is the log of the Laplace transform of the process integrated
        over time.
        """
        kappa = self.kappa
        sigma2 = self.sigma2
        gamma = np.sqrt(kappa * kappa + 2 * u * sigma2)
        kts = 2 * kappa * self.theta / sigma2
        gamma_kappa = gamma + kappa
        egt = np.exp(gamma * t)
        d = 2 * gamma + gamma_kappa * (egt - 1)
        a = 2 * gamma * np.exp(0.5 * gamma_kappa * t) / d
        b = 2 * u * (egt - 1) / d
        return kts * np.log(a) - b * self.rate

    def domain_range(self) -> Bounds:
        return Bounds(0, np.inf)

    def analytical_mean(self, t: FloatArrayLike) -> FloatArrayLike:
        """Analytical mean of the process at time `t`

        This has a closed form solution.
        """
        ekt = self.ekt(t)
        return self.rate * ekt + self.theta * (1 - ekt)

    def analytical_variance(self, t: FloatArrayLike) -> FloatArrayLike:
        kappa = self.kappa
        ekt = self.ekt(t)
        return (
            self.sigma2
            * (1 - ekt)
            * (self.rate * ekt + 0.5 * self.theta * (1 - ekt))
            / kappa
        )

    def analytical_pdf(self, t: FloatArrayLike, x: FloatArrayLike) -> FloatArrayLike:
        k = self.kappa
        s2 = self.sigma2
        ekt = self.ekt(t)
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
