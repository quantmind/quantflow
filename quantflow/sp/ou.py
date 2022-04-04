import numpy as np

from ..utils.param import Param, Parameters
from ..utils.types import Vector
from .base import Im, IntensityProcess
from .compoundp import ExponentialPoissonProcess


class OU(IntensityProcess):
    r"""Non-Gaussian Ornstein-Uhlenbeck process

    The process :math:`x_t` that satisfies the following stochastic
    differential equation:

    .. math::
        dx_t =-\kappa x_t dt + d j_t
    """

    def __init__(self, rate: float, kappa: float, a: float = 1, b: float = 0) -> None:
        super().__init__(rate, kappa)
        self.a = Param("a", a, bounds=(0, None), description="Jump intensity")
        self.b = Param(
            "b",
            b or a / self.rate.value,
            bounds=(0, None),
            description="Jump size exponential decay",
        )

    @property
    def parameters(self) -> Parameters:
        return Parameters(self.rate, self.kappa, self.a, self.b)

    @property
    def jump_process(self) -> ExponentialPoissonProcess:
        return ExponentialPoissonProcess(self.kappa.value * self.a.value, self.b.value)

    def cdf(self, t: float, n: Vector) -> Vector:
        raise NotImplementedError

    def characteristic(self, t: float, u: Vector) -> Vector:
        kappa = self.kappa.value
        b = self.b.value
        iu = Im * u
        kt = kappa * t
        ekt = np.exp(-kt)
        c1 = iu * ekt
        c0 = self.a.value * (np.log((b / ekt - iu) / (b - iu)) - kt)
        return np.exp(c0 + c1 * self.rate.value)

    def cumulative_characteristic(self, t: float, u: Vector) -> Vector:
        kappa = self.kappa.value
        b = self.b.value
        iu = Im * u
        iuk = iu / kappa
        ekt = np.exp(-kappa * t)
        c1 = iuk * (1 - ekt)
        c0 = self.a.value * (
            b * np.log(b / (iuk + (b - iuk) / ekt)) / (iuk - b) - kappa * t
        )
        return np.exp(c0 + c1 * self.rate.value)

    def sample(self, n: int, t: float = 1, steps: int = 0) -> np.array:
        size, dt = self.sample_dt(t, steps)
        jump_process = self.jump_process
        paths = np.zeros((size + 1, n))
        paths[0, :] = self.rate.value
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
        self, i: int, pp: np.array, dt: float, arrival: float = 0, jump: float = 0
    ) -> int:
        x = pp[i - 1]
        kappa = self.kappa.value
        t0 = i * dt
        t1 = t0 + dt
        a = arrival or t1
        pp[i] = x - kappa * x * (a - t0) - kappa * (x + jump) * (t1 - a) + jump
        return i + 1


class OU2(OU):
    def cumulative_characteristic(self, t: float, u: Vector) -> Vector:
        """Formula from a paper"""
        kappa = self.kappa.value
        b = self.b.value
        iu = Im * u
        iuk = iu / kappa
        ekt = np.exp(-kappa * t)
        c1 = iuk * (1 - ekt)
        c0 = self.a.value * (b * np.log(b / (b - c1)) - iu * t) / (iuk - b)
        return np.exp(c0 + c1 * self.rate.value)
