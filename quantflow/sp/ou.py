from __future__ import annotations

from abc import abstractmethod
from typing import Generic, Self

import numpy as np
from pydantic import Field
from scipy.optimize import Bounds
from scipy.stats import gamma, norm

from ..ta.paths import Paths
from ..utils.distributions import Exponential
from ..utils.types import Float, FloatArrayLike, Vector
from .base import Im, IntensityProcess
from .poisson import CompoundPoissonProcess, D
from .weiner import WeinerProcess


class Vasicek(IntensityProcess):
    r"""Gaussian OU process, also know as the
    [Vasiceck model](https://en.wikipedia.org/wiki/Vasicek_model)

    Historically, the Vasicek model was used to model the short rate, but it can be
    used to model any process that reverts to a mean level at a rate proportional to
    the difference between the current level and the mean level.

    \begin{equation}
        dx_t = \kappa (\theta - x_t) dt + \sigma dw_t
    \end{equation}

    where $\kappa$ is the mean reversion speed, $\theta$ is the mean level, and
    $\sigma$ is the volatility of the process. The solution to the SDE is given by

    \begin{equation}
        x_t = x_0 e^{-\kappa t} + \theta (1 - e^{-\kappa t}) +
            \sigma \int_0^t e^{-\kappa (t-s)} dw_s
    \end{equation}
    """

    bdlp: WeinerProcess = Field(
        default_factory=WeinerProcess,
        description="Background driving Weiner process",
    )
    theta: float = Field(default=1.0, gt=0, description=r"Mean rate $\theta$")

    def characteristic_exponent(self, t: FloatArrayLike, u: Vector) -> Vector:
        r"""The characteristic exponent of the Vasicek process is given by

        \begin{equation}
            \phi_{x_t}(u) = - iu \mathbb{E}[x_t] + \frac{1}{2} u^2 \text{Var}[x_t]
        \end{equation}
        """
        mu = self.analytical_mean(t)
        var = self.analytical_variance(t)
        return u * (-1j * mu + 0.5 * var * u)

    def integrated_log_laplace(self, t: FloatArrayLike, u: Vector) -> Vector:
        raise NotImplementedError

    def sample(self, n: int, time_horizon: float = 1, time_steps: int = 100) -> Paths:
        paths = Paths.normal_draws(n, time_horizon, time_steps)
        return self.sample_from_draws(paths)

    def sample_from_draws(self, draws: Paths, *args: Paths) -> Paths:
        kappa = self.kappa
        theta = self.theta
        dt = draws.dt
        sdt = self.bdlp.sigma * np.sqrt(dt)
        paths = np.zeros(draws.data.shape)
        paths[0, :] = self.rate
        for t in range(draws.time_steps):
            x = paths[t, :]
            dx = kappa * (theta - x) * dt + sdt * draws.data[t, :]
            paths[t + 1, :] = x + dx
        return Paths(t=draws.t, data=paths)

    def domain_range(self) -> Bounds:
        return Bounds(-np.inf, np.inf)

    def analytical_mean(self, t: FloatArrayLike) -> FloatArrayLike:
        r"""The analytical mean of the Vasicek process is given by

        \begin{equation}
            \begin{aligned}
            \mathbb{E}[x_t] &= x_0 e^{-\kappa t} + \theta (1 - e^{-\kappa t}) \\
            &= \theta \qquad \text{as } t \to \infty
            \end{aligned}
        \end{equation}
        """
        ekt = self.ekt(t)
        return self.rate * ekt + self.theta * (1 - ekt)

    def analytical_variance(self, t: FloatArrayLike) -> FloatArrayLike:
        r"""The analytical variance of the Vasicek process is given by

        \begin{equation}
            \begin{aligned}
            \text{Var}[x_t] &= \frac{\sigma^2}{2\kappa} (1 - e^{-2\kappa t}) \\
            &= \frac{\sigma^2}{2\kappa} \qquad \text{as } t \to \infty
            \end{aligned}
        \end{equation}
        """
        ekt = self.ekt(2 * t)
        return 0.5 * self.bdlp.sigma2 * (1 - ekt) / self.kappa

    def analytical_pdf(self, t: FloatArrayLike, x: FloatArrayLike) -> FloatArrayLike:
        return norm.pdf(x, loc=self.analytical_mean(t), scale=self.analytical_std(t))

    def analytical_cdf(self, t: FloatArrayLike, x: FloatArrayLike) -> FloatArrayLike:
        return norm.cdf(x, loc=self.analytical_mean(t), scale=self.analytical_std(t))


class NGOU(IntensityProcess, Generic[D]):
    r"""The general definition of a non-Gaussian OU process is defined as
    the solution to the SDE

    \begin{equation}
        dx_t = -\kappa x_t dt + dZ_{\kappa t}
    \end{equation}

    where $Z_{\kappa t}$ is a pure jump Lévy process, also known as the background
    driving Lévy process (BDLP).
    The process is mean-reverting with mean reversion speed $\kappa>0$.

    The unusual timing $dZ_{\kappa t}$ is deliberately chosen so that it will turn
    out that whatever the value of $\kappa$, the marginal distribution of $x_t$
    will be unchanged.

    The solution to the SDE is given by

    \begin{equation}
        x_t = x_0 e^{-\kappa t} + \int_0^t e^{-\kappa (t-s)} dZ_{\kappa s}
    \end{equation}
    """

    bdlp: CompoundPoissonProcess[D] = Field(
        description=(
            "Background driving Lévy process is a "
            "[CompoundPoissonProcess][quantflow.sp.poisson.CompoundPoissonProcess] "
            "with jump distribution D"
        ),
    )

    @property
    def intensity(self) -> float:
        """The intensity of the NGOU process is the intensity of the background
        driving Lévy process"""
        return self.bdlp.intensity

    @abstractmethod
    def characteristic_exponent(self, t: FloatArrayLike, u: Vector) -> Vector:
        r"""
        \begin{equation}
            \phi_{x_t,u} = iu x_0 e^{-\kappa t} +
                \int_{ue^{-\kappa t}}^{u} \frac{\psi_Z(v)}{v} , dv
        \end{equation}
        """

    def sample_from_draws(self, draws: Paths, *args: Paths) -> Paths:
        raise NotImplementedError


class GammaOU(NGOU[Exponential]):
    """Gamma OU process is a non-Gaussian OU process where the background
    driving Lévy process is a compound Poisson process with exponential jumps"""

    @property
    def beta(self) -> float:
        return self.bdlp.jumps.decay

    @classmethod
    def create(cls, rate: float = 1, decay: float = 1, kappa: float = 1) -> Self:
        """Convenience constructor for GammaOU process"""
        return cls(
            rate=rate,
            kappa=kappa,
            bdlp=CompoundPoissonProcess[Exponential](
                intensity=rate * decay, jumps=Exponential(decay=decay)
            ),
        )

    def characteristic_exponent(self, t: FloatArrayLike, u: Vector) -> Vector:
        r"""
        \begin{equation}
            \phi_{x_t,u} =
                - iu x_0 e^{-\kappa t}
                + \lambda \log\!\left(\frac{\beta - iu}{\beta - iu e^{-\kappa t}}\right)
        \end{equation}
        """
        b = self.beta
        iu = Im * u
        c1 = iu * np.exp(-self.kappa * t)
        c0 = self.intensity * np.log((b - c1) / (b - iu))
        return -c0 - c1 * self.rate

    def integrated_log_laplace(self, t: FloatArrayLike, u: Vector) -> Vector:
        kappa = self.kappa
        b = self.beta
        iu = Im * u
        iuk = iu / kappa
        ekt = np.exp(-kappa * t)
        c1 = iuk * (1 - ekt)
        c0 = self.intensity * (
            b * np.log(b / (iuk + (b - iuk) / ekt)) / (iuk - b) - kappa * t
        )
        return np.exp(c0 + c1 * self.rate)

    def sample(self, n: int, time_horizon: float = 1, time_steps: int = 100) -> Paths:
        dt = time_horizon / time_steps
        jump_process = self.bdlp
        paths = np.zeros((time_steps + 1, n))
        paths[0, :] = self.rate
        for p in range(n):
            arrivals = jump_process.arrivals(self.kappa * time_horizon)
            jumps = jump_process.sample_jumps(len(arrivals))
            pp = paths[:, p]
            i = 1
            for arrival, jump in zip(arrivals, jumps):
                arrival /= self.kappa
                while i * dt < arrival:
                    i = self._advance(i, pp, dt)
                if i <= time_steps:
                    i = self._advance(i, pp, dt, arrival, jump)
            while i <= time_steps:
                i = self._advance(i, pp, dt)
        return Paths(t=time_horizon, data=paths)

    def _advance(
        self,
        i: int,
        pp: np.ndarray,
        dt: Float,
        arrival: Float = 0,
        jump: Float = 0,
    ) -> int:
        x = pp[i - 1]
        kappa = self.kappa
        t0 = i * dt
        t1 = t0 + dt
        a = arrival or t1
        pp[i] = x - kappa * x * (a - t0) - kappa * (x + jump) * (t1 - a) + jump
        return i + 1

    def cumulative_characteristic2(self, t: FloatArrayLike, u: Vector) -> Vector:
        """Formula from a paper"""
        kappa = self.kappa
        b = self.beta
        iu = Im * u
        iuk = iu / kappa
        ekt = np.exp(-kappa * t)
        c1 = iuk * (1 - ekt)
        c0 = self.intensity * (b * np.log(b / (b - c1)) - iu * t) / (iuk - b)
        return np.exp(c0 + c1 * self.rate)

    def analytical_mean(self, t: FloatArrayLike) -> FloatArrayLike:
        return self.intensity / self.beta

    def analytical_variance(self, t: FloatArrayLike) -> FloatArrayLike:
        return self.intensity / self.beta / self.beta

    def analytical_pdf(self, t: FloatArrayLike, x: FloatArrayLike) -> FloatArrayLike:
        return gamma.pdf(x, self.intensity, scale=1 / self.beta)
