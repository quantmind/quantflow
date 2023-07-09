from abc import abstractmethod

import numpy as np
from pydantic import Field
from scipy.optimize import Bounds
from scipy.stats import poisson

from ..utils.functions import factorial
from ..utils.paths import Paths
from ..utils.types import Vector
from .base import Im, StochasticProcess1D, StochasticProcess1DMarginal


class PoissonBase(StochasticProcess1D):
    @abstractmethod
    def jumps(self, n: int) -> np.ndarray:
        """Generate a list of jump sizes"""

    @abstractmethod
    def characteristic_exponent(self, u: Vector) -> Vector:
        """Characteristic exponent of the process"""

    @abstractmethod
    def arrivals(self, time_horizon: float = 1) -> list[float]:
        """Generate a list of jump arrivals times up to time t"""

    def characteristic(self, t: float, u: Vector) -> Vector:
        return np.exp(-t * self.characteristic_exponent(u))

    def sample(self, n: int, time_horizon: float = 1, time_steps: int = 100) -> Paths:
        dt = time_horizon / time_steps
        paths = np.zeros((time_steps + 1, n))
        for p in range(n):
            if arrivals := self.arrivals(time_horizon):
                jumps = self.jumps(len(arrivals))
                i = 1
                y = 0.0
                for j, arrival in enumerate(arrivals):
                    while i * dt < arrival:
                        paths[i, p] = y
                        i += 1
                    y += jumps[j]
                paths[i:, p] = y
        return Paths(t=time_horizon, data=paths)

    def sample_from_draws(self, draws: Paths, *args: Paths) -> Paths:
        raise NotImplementedError

    def domain_range(self) -> Bounds:
        return Bounds(0, np.inf)


def poisson_arrivals(intensity: float, time_horizon: float = 1) -> list[float]:
    """Generate a list of jump arrivals times up to time t"""
    exp_rate = 1.0 / intensity
    arrivals = []
    tt = 0.0
    while tt < time_horizon:
        dt = np.random.exponential(scale=exp_rate)
        tt += dt
        if tt <= time_horizon:
            arrivals.append(tt)
    return arrivals


class PoissonProcess(PoissonBase):
    intensity: float = Field(default=1.0, ge=0, description="intensity rate")

    def marginal(self, t: float, N: int = 128) -> StochasticProcess1DMarginal:
        return PoissonMarginal(self, t, N)

    def characteristic_exponent(self, u: Vector) -> Vector:
        return -self.intensity * (np.exp(Im * u) - 1)

    def arrivals(self, time_horizon: float = 1) -> list[float]:
        return poisson_arrivals(self.intensity, time_horizon)

    def jumps(self, n: int) -> np.ndarray:
        """For a poisson process this is just a list of 1s"""
        return np.ones((n,))

    def max_frequency(self, t: float) -> float:
        """Maximum frequency of the process"""
        return 2 * np.pi


class ExponentialPoissonProcess(PoissonBase):
    r"""
    1D Poisson process.

    It's a process where the inter-arrival time is exponentially distributed
    with rate :math:`\lambda`

    .. attribute:: rate

        The arrival rate of events. Must be positive.
    """
    intensity: float = Field(default=1.0, ge=0, description="intensity rate")
    decay: float = Field(default=1.0, ge=0, description="Jump size decay rate")

    def marginal(self, t: float, N: int = 128) -> StochasticProcess1DMarginal:
        return ExponentialPoissonMarginal(self, t, N)

    def characteristic_exponent(self, u: Vector) -> Vector:
        iu = Im * u
        return -self.intensity * iu / (self.decay - iu)

    def arrivals(self, time_horizon: float = 1) -> list[float]:
        return poisson_arrivals(self.intensity, time_horizon)

    def jumps(self, n: int) -> np.ndarray:
        """Sample jump sizes from an exponential distribution with rate
        parameter :class:b
        """
        exp_rate = 1.0 / self.decay
        return np.random.exponential(scale=exp_rate, size=n)


class PoissonMarginal(StochasticProcess1DMarginal[PoissonProcess]):
    def mean(self) -> float:
        """Expected value at a time horizon"""
        return self.process.intensity * self.t

    def variance(self) -> float:
        """Expected variance at a time horizon"""
        return self.process.intensity * self.t

    def cdf(self, n: Vector) -> Vector:
        r"""
        CDF of the number of events at time ``t``.

        It's given by

        .. math::
            :label: poisson_cdf

            F_{X}\left(n\right)=\frac{\Gamma\left(\left\lfloor n+1\right\rfloor
            ,\lambda\right)}{\left\lfloor n\right\rfloor !}

        where :math:`\Gamma` is the upper incomplete gamma function.
        """
        return poisson.cdf(n, self.t * self.process.intensity)

    def pdf(self, n: Vector = 0) -> Vector:
        r"""
        Probability density function of the number of events at time ``t``.

        It's given by

        \begin{equation}
           f_{X}\left(n\right)=\frac{\lambda^{n}e^{-\lambda}}{n!}
        \end{equation}
        """
        return poisson.pmf(n, self.t * self.process.intensity)

    def cdf_jacobian(self, n: Vector) -> np.ndarray:
        r"""
        Jacobian of the CDF

        It's given by

        .. math::

            \frac{\partial F}{\partial\lambda}=-\frac{\lambda^{\left\lfloor
            n\right\rfloor }e^{-\lambda}}{\left\lfloor n\right\rfloor !}
        """
        k = np.floor(n).astype(int)
        rate = self.process.intensity
        return np.array([-(rate**k) * np.exp(-rate)]) / factorial(k)


class ExponentialPoissonMarginal(
    StochasticProcess1DMarginal[ExponentialPoissonProcess]
):
    def mean(self) -> float:
        """Expected value at a time horizon"""
        return self.process.intensity * self.t / self.process.decay

    def variance(self) -> float:
        """Expected variance at a time horizon"""
        return 2 * self.mean() / self.process.decay
