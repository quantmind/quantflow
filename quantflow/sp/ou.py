from __future__ import annotations

import numpy as np
from pydantic import Field
from scipy.stats import gamma

from ..utils.paths import Paths
from ..utils.types import Vector
from .base import Im, IntensityProcess, StochasticProcess1DMarginal
from .poisson import ExponentialPoissonProcess


class NGOU(IntensityProcess):
    r"""Non-Gaussian Ornstein-Uhlenbeck process

    The process :math:`x_t` that satisfies the following stochastic
    differential equation:

    .. math::
        dx_t =-\kappa x_t dt + d j_t
    """

    def sample_from_draws(self, draws: Paths, *args: Paths) -> Paths:
        raise NotImplementedError


class GammaOU(NGOU):
    bdlp: ExponentialPoissonProcess = Field(
        default_factory=ExponentialPoissonProcess,
        description="Background driving Levy process",
    )

    @property
    def alpha(self) -> float:
        return self.bdlp.intensity

    @property
    def beta(self) -> float:
        return self.bdlp.decay

    @classmethod
    def create(cls, rate: float = 1, decay: float = 1, kappa: float = 1) -> GammaOU:
        return cls(
            rate=rate,
            kappa=kappa,
            bdlp=ExponentialPoissonProcess(intensity=rate * decay, decay=decay),
        )

    def marginal(self, t: float, N: int = 128) -> StochasticProcess1DMarginal:
        return GammaOUMarginal(self, t, N)

    def characteristic(self, t: float, u: Vector) -> Vector:
        kappa = self.kappa
        a = self.alpha
        b = self.beta
        iu = Im * u
        kt = kappa * t
        ekt = np.exp(-kt)
        c1 = iu * ekt
        c0 = a * (np.log((b / ekt - iu) / (b - iu)) - kt)
        return np.exp(c0 + c1 * self.rate)

    def cumulative_characteristic(self, t: float, u: Vector) -> Vector:
        kappa = self.kappa
        b = self.beta
        iu = Im * u
        iuk = iu / kappa
        ekt = np.exp(-kappa * t)
        c1 = iuk * (1 - ekt)
        c0 = self.alpha * (
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
            jumps = jump_process.jumps(len(arrivals))
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
        self, i: int, pp: np.ndarray, dt: float, arrival: float = 0, jump: float = 0
    ) -> int:
        x = pp[i - 1]
        kappa = self.kappa
        t0 = i * dt
        t1 = t0 + dt
        a = arrival or t1
        pp[i] = x - kappa * x * (a - t0) - kappa * (x + jump) * (t1 - a) + jump
        return i + 1

    def cumulative_characteristic2(self, t: float, u: Vector) -> Vector:
        """Formula from a paper"""
        kappa = self.kappa
        b = self.beta
        iu = Im * u
        iuk = iu / kappa
        ekt = np.exp(-kappa * t)
        c1 = iuk * (1 - ekt)
        c0 = self.alpha * (b * np.log(b / (b - c1)) - iu * t) / (iuk - b)
        return np.exp(c0 + c1 * self.rate)


class GammaOUMarginal(StochasticProcess1DMarginal[GammaOU]):
    def mean(self) -> float:
        return self.process.alpha / self.process.beta

    def variance(self) -> float:
        return self.process.alpha / self.process.beta / self.process.beta

    def pdf(self, x: Vector) -> Vector:
        return gamma.pdf(x, self.process.alpha, scale=1 / self.process.beta)
