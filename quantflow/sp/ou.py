from __future__ import annotations

from typing import Generic

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
    r"""Gaussian OU process, also know as the `Vasiceck model`_.

    Historically, the Vasicek model was used to model the short rate, but it can be
    used to model any process that reverts to a mean level at a rate proportional to
    the difference between the current level and the mean level.

    .. math::
        dx_t = \kappa (\theta - x_t) dt + \sigma dw_t

    It derives from :class:`.IntensityProcess`, although, it is not strictly
    an intensity process since it is not positive.

    .. _`Vasiceck model`: https://en.wikipedia.org/wiki/Vasicek_model
    """

    bdlp: WeinerProcess = Field(
        default_factory=WeinerProcess,
        description="Background driving Weiner process",
    )
    """Background driving Weiner process"""
    theta: float = Field(default=1.0, gt=0, description="Mean rate")
    r"""Mean rate :math:`\theta`"""

    def characteristic_exponent(self, t: FloatArrayLike, u: Vector) -> Vector:
        mu = self.analytical_mean(t)
        var = self.analytical_variance(t)
        return u * (-complex(0, 1) * mu + 0.5 * var * u)

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
        ekt = self.ekt(t)
        return self.rate * ekt + self.theta * (1 - ekt)

    def analytical_variance(self, t: FloatArrayLike) -> FloatArrayLike:
        ekt = self.ekt(2 * t)
        return 0.5 * self.bdlp.sigma2 * (1 - ekt) / self.kappa

    def analytical_pdf(self, t: FloatArrayLike, x: FloatArrayLike) -> FloatArrayLike:
        return norm.pdf(x, loc=self.analytical_mean(t), scale=self.analytical_std(t))

    def analytical_cdf(self, t: FloatArrayLike, x: FloatArrayLike) -> FloatArrayLike:
        return norm.cdf(x, loc=self.analytical_mean(t), scale=self.analytical_std(t))


class NGOU(IntensityProcess, Generic[D]):
    bdlp: CompoundPoissonProcess[D] = Field(
        description="Background driving Levy process",
    )

    def sample_from_draws(self, draws: Paths, *args: Paths) -> Paths:
        raise NotImplementedError


class GammaOU(NGOU[Exponential]):
    @property
    def intensity(self) -> float:
        return self.bdlp.intensity

    @property
    def beta(self) -> float:
        # TODO: find a better way for this
        return self.bdlp.jumps.decay

    @classmethod
    def create(cls, rate: float = 1, decay: float = 1, kappa: float = 1) -> GammaOU:
        return cls(
            rate=rate,
            kappa=kappa,
            bdlp=CompoundPoissonProcess[Exponential](
                intensity=rate * decay, jumps=Exponential(decay=decay)
            ),
        )

    def characteristic_exponent(self, t: FloatArrayLike, u: Vector) -> Vector:
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
