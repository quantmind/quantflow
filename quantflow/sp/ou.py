from __future__ import annotations

import numpy as np
from pydantic import Field

from ..utils.types import Vector
from .base import Im, IntensityProcess
from .poisson import ExponentialPoissonProcess


class OU(IntensityProcess):
    r"""Non-Gaussian Ornstein-Uhlenbeck process

    The process :math:`x_t` that satisfies the following stochastic
    differential equation:

    .. math::
        dx_t =-\kappa x_t dt + d j_t
    """
    a: float = Field(default=1, ge=0, description="Jump intensity")
    decay: float = Field(default=0, gt=0, description="Jump size exponential decay")

    @classmethod
    def create(
        cls, rate: float = 1, kappa: float = 1, a: float = 1, decay: float | None = None
    ) -> OU:
        return cls(rate=rate, kappa=kappa, a=a, decay=decay or a / rate)

    @property
    def jump_process(self) -> ExponentialPoissonProcess:
        return ExponentialPoissonProcess(rate=self.kappa * self.a, decay=self.decay)

    def cdf(self, t: float, n: Vector) -> Vector:
        raise NotImplementedError

    def characteristic(self, t: float, u: Vector) -> Vector:
        kappa = self.kappa
        b = self.decay
        iu = Im * u
        kt = kappa * t
        ekt = np.exp(-kt)
        c1 = iu * ekt
        c0 = self.a * (np.log((b / ekt - iu) / (b - iu)) - kt)
        return np.exp(c0 + c1 * self.rate)

    def cumulative_characteristic(self, t: float, u: Vector) -> Vector:
        kappa = self.kappa
        b = self.decay
        iu = Im * u
        iuk = iu / kappa
        ekt = np.exp(-kappa * t)
        c1 = iuk * (1 - ekt)
        c0 = self.a * (b * np.log(b / (iuk + (b - iuk) / ekt)) / (iuk - b) - kappa * t)
        return np.exp(c0 + c1 * self.rate)

    def sample(self, n: int, t: float = 1, steps: int = 0) -> np.ndarray:
        size, dt = self.sample_dt(t, steps)
        jump_process = self.jump_process
        paths = np.zeros((size + 1, n))
        paths[0, :] = self.rate
        for p in range(n):
            arrivals = jump_process.arrivals(t)
            jumps = jump_process.jumps(len(arrivals))
            pp = paths[:, p]
            i = 1
            for arrival, jump in zip(arrivals, jumps):
                while i * dt < arrival:
                    i = self._advance(i, pp, dt)
                if i <= size:
                    i = self._advance(i, pp, dt, arrival, jump)
            while i <= size:
                i = self._advance(i, pp, dt)
        return paths

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


class OU2(OU):
    def cumulative_characteristic(self, t: float, u: Vector) -> Vector:
        """Formula from a paper"""
        kappa = self.kappa
        b = self.decay
        iu = Im * u
        iuk = iu / kappa
        ekt = np.exp(-kappa * t)
        c1 = iuk * (1 - ekt)
        c0 = self.a * (b * np.log(b / (b - c1)) - iu * t) / (iuk - b)
        return np.exp(c0 + c1 * self.rate)
