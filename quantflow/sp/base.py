from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from scipy.optimize import Bounds

from quantflow.utils.marginal import Marginal1D, default_bounds
from quantflow.utils.numbers import sigfig
from quantflow.utils.paths import Paths
from quantflow.utils.transforms import lower_bound, upper_bound
from quantflow.utils.types import FloatArray, Vector

Im = complex(0, 1)


class StochasticProcess(BaseModel, ABC):
    """
    Base class for stochastic processes in continuous time
    """

    @abstractmethod
    def sample_from_draws(self, draws: Paths, *args: Paths) -> Paths:
        """Sample paths from the process given a set of draws"""

    @abstractmethod
    def sample(
        self, paths: int, time_horizon: float = 1, time_steps: int = 100
    ) -> Paths:
        """Generate random paths from the process.

        :param paths: Number of paths
        :param time_horizon: time horizon
        :param time_steps: number of time steps to arrive at horizon
        """

    @abstractmethod
    def characteristic_exponent(self, t: Vector, u: Vector) -> Vector:
        """Characteristic exponent at time `t` for a given input parameter"""

    def characteristic(self, t: Vector, u: Vector) -> Vector:
        r"""Characteristic function at time `t` for a given input parameter

        The characteristic function represents the Fourier transform of the
        probability density function

        .. math::
            \phi = {\mathbb E} \left[e^{i u x_t}\right]

        :param t: time horizon
        :param u: characteristic function input parameter
        """
        return np.exp(-self.characteristic_exponent(t, u))

    def convexity_correction(self, t: Vector) -> Vector:
        """Convexity correction for the process"""
        return -self.characteristic_exponent(t, complex(0, -1)).real


class StochasticProcess1D(StochasticProcess):
    """
    Base class for 1D stochastic process in continuous time
    """

    def marginal(self, t: Vector) -> StochasticProcess1DMarginal:
        return StochasticProcess1DMarginal(process=self, t=t)

    def domain_range(self) -> Bounds:
        return default_bounds()

    def max_frequency(self, std: float) -> float:
        """Maximum frequency when calculating characteristic functions"""
        return np.sqrt(40 / std / std)

    def support(self, mean: float, std: float, points: int) -> FloatArray:
        """Support of the process at time `t`"""
        bounds = self.domain_range()
        start = float(sigfig(lower_bound(bounds.lb, mean - std)))
        end = float(sigfig(upper_bound(bounds.ub, mean + std)))
        return np.linspace(start, end, points)


P = TypeVar("P", bound=StochasticProcess1D)


class StochasticProcess1DMarginal(Marginal1D, Generic[P]):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    process: P
    t: Vector

    def std_norm(self) -> Vector:
        """Standard deviation at a time horizon normalized by the time"""
        return np.sqrt(self.variance() / self.t)

    def characteristic(self, u: Vector) -> Vector:
        return self.process.characteristic(self.t, u)

    def domain_range(self) -> Bounds:
        return self.process.domain_range()

    def max_frequency(self) -> float:
        return self.process.max_frequency(np.min(self.std()))

    def support(self, points: int = 100, *, std_mult: float = 4) -> FloatArray:
        return self.process.support(
            np.max(self.mean()), std_mult * np.max(self.std()), points
        )

    def option_alpha(self) -> float:
        """Option alpha parameter for integrability of call option transform"""
        return max(8 * np.max(np.exp(-2 * self.t)), 0.5)


class IntensityProcess(StochasticProcess1D):
    """Base class for mean reverting 1D processes which can be used
    as stochastic intensity
    """

    rate: float = Field(default=1.0, gt=0, description="Instantaneous initial rate")
    kappa: float = Field(default=1.0, gt=0, description="Mean reversion speed")

    @abstractmethod
    def cumulative_characteristic(self, t: Vector, u: Vector) -> Vector:
        r"""The characteristic function of the cumulative process:

        .. math::
            \phi = {\mathbb E} \left[e^{i u \int_0^t x_s ds}\right]

        :param t: time horizon
        :param u: frequency
        """

    def domain_range(self) -> Bounds:
        return Bounds(0, np.inf)

    def ekt(self, t: Vector) -> Vector:
        return np.exp(-self.kappa * t)
