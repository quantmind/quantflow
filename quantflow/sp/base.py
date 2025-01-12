from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from scipy.optimize import Bounds

from quantflow.ta.paths import Paths
from quantflow.utils.marginal import Marginal1D, default_bounds
from quantflow.utils.numbers import sigfig
from quantflow.utils.transforms import lower_bound, upper_bound
from quantflow.utils.types import FloatArray, FloatArrayLike, Vector

Im = complex(0, 1)


class StochasticProcess(BaseModel, ABC, extra="forbid"):
    """
    Base class for stochastic processes in continuous time
    """

    @abstractmethod
    def sample_from_draws(self, draws: Paths, *args: Paths) -> Paths:
        """Sample :class:`.Paths` from the process given a set of draws"""

    @abstractmethod
    def sample(self, n: int, time_horizon: float = 1, time_steps: int = 100) -> Paths:
        """Generate random :class:`.Paths` from the process.

        :param n: number of paths
        :param time_horizon: time horizon
        :param time_steps: number of time steps to arrive at horizon
        """

    @abstractmethod
    def characteristic_exponent(self, t: FloatArrayLike, u: Vector) -> Vector:
        """Characteristic exponent at time `t` for a given input parameter"""

    def characteristic(self, t: FloatArrayLike, u: Vector) -> Vector:
        r"""Characteristic function at time `t` for a given input parameter

        The characteristic function represents the Fourier transform of the
        probability density function

        .. math::
            \phi = {\mathbb E} \left[e^{i u x_t}\right]

        :param t: time horizon
        :param u: characteristic function input parameter
        """
        return np.exp(-self.characteristic_exponent(t, u))

    def convexity_correction(self, t: FloatArrayLike) -> Vector:
        """Convexity correction for the process"""
        return -self.characteristic_exponent(t, complex(0, -1)).real

    def analytical_std(self, t: FloatArrayLike) -> FloatArrayLike:
        return np.sqrt(self.analytical_variance(t))

    def analytical_mean(self, t: FloatArrayLike) -> FloatArrayLike:
        """Analytical mean of the process at time `t`

        Implement if available
        """
        raise NotImplementedError

    def analytical_variance(self, t: FloatArrayLike) -> FloatArrayLike:
        """Analytical variance of the process at time `t`

        Implement if available
        """
        raise NotImplementedError

    def analytical_pdf(self, t: FloatArrayLike, x: FloatArrayLike) -> FloatArrayLike:
        """Analytical pdf of the process at time `t`

        Implement if available
        """
        raise NotImplementedError

    def analytical_cdf(self, t: FloatArrayLike, x: FloatArrayLike) -> FloatArrayLike:
        """Analytical cdf of the process at time `t`

        Implement if available
        """
        raise NotImplementedError


class StochasticProcess1D(StochasticProcess):
    """
    Base class for 1D stochastic process in continuous time
    """

    def marginal(self, t: FloatArrayLike) -> StochasticProcess1DMarginal:
        return StochasticProcess1DMarginal(process=self, t=t)

    def domain_range(self) -> Bounds:
        return default_bounds()

    def frequency_range(self, std: float, max_frequency: float | None = None) -> Bounds:
        """Maximum frequency when calculating characteristic functions"""
        if max_frequency is None:
            max_frequency = np.sqrt(40 / std / std)
        return Bounds(0, max_frequency)

    def support(self, mean: float, std: float, points: int) -> FloatArray:
        """Support of the process at time `t`"""
        bounds = self.domain_range()
        start = float(sigfig(lower_bound(bounds.lb, mean - std)))
        end = float(sigfig(upper_bound(bounds.ub, mean + std)))
        return np.linspace(start, end, points + 1)


P = TypeVar("P", bound=StochasticProcess1D)


class StochasticProcess1DMarginal(Marginal1D, Generic[P]):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    process: P
    t: FloatArrayLike

    def std_norm(self) -> Vector:
        """Standard deviation at a time horizon normalized by the time"""
        return np.sqrt(self.variance() / self.t)

    def characteristic(self, u: Vector) -> Vector:
        return self.process.characteristic(self.t, u)

    def domain_range(self) -> Bounds:
        return self.process.domain_range()

    def frequency_range(self, max_frequency: float | None = None) -> Bounds:
        std = float(np.min(self.std()))
        return self.process.frequency_range(std, max_frequency=max_frequency)

    def pdf(self, x: FloatArrayLike) -> FloatArrayLike:
        return self.process.analytical_pdf(self.t, x)

    def cdf(self, x: FloatArrayLike) -> FloatArrayLike:
        return self.process.analytical_cdf(self.t, x)

    def mean(self) -> FloatArrayLike:
        try:
            return self.process.analytical_mean(self.t)
        except NotImplementedError:
            return self.mean_from_characteristic()

    def variance(self) -> FloatArrayLike:
        try:
            return self.process.analytical_variance(self.t)
        except NotImplementedError:
            return self.variance_from_characteristic()

    def support(self, points: int = 100, *, std_mult: float = 4) -> FloatArray:
        return self.process.support(
            float(self.mean()), std_mult * float(self.std()), points
        )

    def option_alpha(self) -> float:
        """Option alpha parameter for integrability of call option transform"""
        return max(8 * np.max(np.exp(-2 * self.t)), 0.5)


class IntensityProcess(StochasticProcess1D):
    """Base class for mean reverting 1D processes which can be used
    as stochastic intensity
    """

    rate: float = Field(default=1.0, gt=0, description="Instantaneous initial rate")
    r"""Instantaneous initial rate :math:`r_0`"""
    kappa: float = Field(default=1.0, gt=0, description="Mean reversion speed")
    r"""Mean reversion speed :math:`\kappa`"""

    @abstractmethod
    def integrated_log_laplace(self, t: FloatArrayLike, u: Vector) -> Vector:
        r"""The log-Laplace transform of the cumulative process:

        .. math::
            e^{\phi_{t, u}} = {\mathbb E} \left[e^{i u \int_0^t x_s ds}\right]

        :param t: time horizon
        :param u: frequency
        """

    def domain_range(self) -> Bounds:
        return Bounds(0, np.inf)

    def ekt(self, t: FloatArrayLike) -> FloatArrayLike:
        return np.exp(-self.kappa * t)
