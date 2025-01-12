from __future__ import annotations

from abc import abstractmethod
from typing import Any, Generic, TypeVar

import numpy as np
from pydantic import Field
from scipy.integrate import simpson
from scipy.optimize import Bounds
from scipy.stats import poisson

from quantflow.ta.paths import Paths
from quantflow.utils.distributions import Distribution1D
from quantflow.utils.functions import factorial
from quantflow.utils.transforms import TransformResult
from quantflow.utils.types import FloatArray, FloatArrayLike, Vector

from .base import Im, StochasticProcess1D, StochasticProcess1DMarginal

D = TypeVar("D", bound=Distribution1D)


class PoissonBase(StochasticProcess1D):
    @abstractmethod
    def sample_jumps(self, n: int) -> np.ndarray:
        """Generate a list of jump sizes"""

    @abstractmethod
    def arrivals(self, time_horizon: float = 1) -> list[float]:
        """Generate a list of jump arrivals times up to time t"""

    def sample(self, n: int, time_horizon: float = 1, time_steps: int = 100) -> Paths:
        dt = time_horizon / time_steps
        paths = np.zeros((time_steps + 1, n))
        for p in range(n):
            if arrivals := self.arrivals(time_horizon):
                jumps = self.sample_jumps(len(arrivals))
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
    r"""Intensity rate :math:`\lambda` of the Poisson process"""

    def marginal(self, t: FloatArrayLike) -> StochasticProcess1DMarginal:
        return MarginalDiscrete1D(process=self, t=t)

    def characteristic_exponent(self, t: Vector, u: Vector) -> Vector:
        return t * self.intensity * (1 - np.exp(Im * u))

    def arrivals(self, time_horizon: float = 1) -> list[float]:
        return poisson_arrivals(self.intensity, time_horizon)

    def sample_jumps(self, n: int) -> np.ndarray:
        """For a poisson process this is just a list of 1s"""
        return np.ones((n,))

    def frequency_range(self, std: float, max_frequency: float | None = None) -> Bounds:
        """Frequency range of the process"""
        return Bounds(0, np.pi)

    def support(self, mean: float, std: float, points: int) -> FloatArray:
        """Support of the process at time `t`"""
        return np.linspace(0, points, points + 1)

    def analytical_mean(self, t: FloatArrayLike) -> FloatArrayLike:
        """Expected value at a time horizon"""
        return self.intensity * t

    def analytical_variance(self, t: FloatArrayLike) -> FloatArrayLike:
        """Expected variance at a time horizon"""
        return self.intensity * t

    def analytical_cdf(self, t: FloatArrayLike, n: FloatArrayLike) -> FloatArrayLike:
        r"""
        CDF of the number of events at time ``t``.

        It's given by

        .. math::
            :label: poisson_cdf

            F\left(n\right)=\frac{\Gamma\left(\left\lfloor n+1\right\rfloor
            ,\lambda\right)}{\left\lfloor n\right\rfloor !}

        where :math:`\Gamma` is the upper incomplete gamma function.
        """
        return poisson.cdf(n, t * self.intensity)

    def analytical_pdf(self, t: FloatArrayLike, n: FloatArrayLike) -> FloatArrayLike:
        r"""
        Probability density function of the number of events at time ``t``.

        It's given by

        .. math::
            :label: poisson_pdf

            f\left(n\right)=\frac{\lambda^{n}e^{-\lambda}}{n!}
        """
        return poisson.pmf(n, t * self.intensity)

    def cdf_jacobian(self, t: FloatArrayLike, n: Vector) -> np.ndarray:
        r"""
        Jacobian of the CDF

        It's given by

        .. math::
            :label: poisson_cdf_jacobian

            \frac{\partial F}{\partial\lambda}=-\frac{\lambda^{\left\lfloor
            n\right\rfloor }e^{-\lambda}}{\left\lfloor n\right\rfloor !}
        """
        k = np.floor(n).astype(int)
        rate = self.intensity
        return np.array([-(rate**k) * np.exp(-rate)]) / factorial(k)


class CompoundPoissonProcess(PoissonBase, Generic[D]):
    """A generic Compound Poisson process."""

    intensity: float = Field(default=1.0, gt=0, description="jump intensity rate")
    r"""Intensity rate :math:`\lambda` of the Poisson process

    It determines the number of jumps in the same way as the :class:`.PoissonProcess`
    """
    jumps: D = Field(description="Jump size distribution")
    """Jump size distribution"""

    def characteristic_exponent(self, t: FloatArrayLike, u: Vector) -> Vector:
        r"""The characteristic exponent of the Compound Poisson process,
        given by

        .. math::
            \phi_{x_t,u} = t\lambda \left(1 - \Phi_{j,u}\right)

        where :math:`\Phi_{j,u}` is the characteristic function
        of the jump distribution
        """
        return t * self.intensity * (1 - self.jumps.characteristic(u))

    def arrivals(self, time_horizon: float = 1) -> list[float]:
        """Same as Poisson process"""
        return poisson_arrivals(self.intensity, time_horizon)

    def sample_jumps(self, n: int) -> FloatArray:
        """Sample jump sizes from an exponential distribution with rate
        parameter :class:b
        """
        return self.jumps.sample(n)

    def analytical_mean(self, t: FloatArrayLike) -> FloatArrayLike:
        """Expected value at a time horizon"""
        return self.intensity * t * self.jumps.mean()

    def analytical_variance(self, t: FloatArrayLike) -> FloatArrayLike:
        """Expected variance at a time horizon"""
        return self.intensity * t * (self.jumps.variance() + self.jumps.mean() ** 2)

    @classmethod
    def create(
        cls,
        jump_distribution: type[D],
        *,
        vol: float = 0.5,
        jump_intensity: float = 100,
        jump_asymmetry: float = 0.0,
    ) -> CompoundPoissonProcess[D]:
        """Create a Compound Poisson process with a given jump distribution, volatility,
        jump intensity a nd jump asymmetry .

        :param jump_distribution: The distribution of jump size (currently only
            :class:`.Normal` and :class:`.DoubleExponential` are supported)
        :param vol: Annualized standard deviation
        :param jump_intensity: The average number of jumps per year
        :param jump_asymmetry: The asymmetry of the jump distribution (0 for symmetric,
            only used by distributions with asymmetry)
        """
        variance = vol * vol
        jump_distribution_variance = variance / jump_intensity
        jumps = jump_distribution.from_variance_and_asymmetry(
            jump_distribution_variance, jump_asymmetry
        )
        return cls(intensity=jump_intensity, jumps=jumps)


class MarginalDiscrete1D(StochasticProcess1DMarginal):
    def pdf_from_characteristic(
        self,
        n: int | None = None,
        **kwargs: Any,
    ) -> TransformResult:
        cdf = self.cdf_from_characteristic(n, **kwargs)
        return cdf._replace(y=np.diff(cdf.y, prepend=0))

    def cdf_from_characteristic(
        self,
        n: int | None = None,
        *,
        frequency_n: int | None = None,
        simpson_rule: bool = True,
        **kwargs: Any,
    ) -> TransformResult:
        n = n or 10
        transform = self.get_transform(frequency_n, self.support, **kwargs)
        frequency = transform.frequency_domain
        c = self.characteristic(frequency)
        a = 1 / np.pi
        result = []
        x = np.arange(n or 10)
        for m in x:
            d = np.sin(0.5 * frequency)
            d[0] = 1.0
            f = (
                np.sin(0.5 * (m + 1) * frequency)
                * (c * np.exp(-0.5 * Im * m * frequency)).real
                / d
            )
            f[0] = c[0].real  # type: ignore[index]
            if simpson_rule:
                result.append(a * simpson(f, x=frequency))
            else:
                result.append(a * np.trapezoid(f, frequency))
        pdf = np.maximum(np.diff(result, prepend=0), 0)
        return TransformResult(x=x, y=np.cumsum(pdf))  # type: ignore[arg-type]
