from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import numpy as np
from pydantic import BaseModel, Field
from scipy.optimize import Bounds

from quantflow.utils.marginal import Marginal1D, default_bounds
from quantflow.utils.paths import Paths
from quantflow.utils.types import Vector

Im = complex(0, 1)


class StochasticProcess(BaseModel, ABC):
    """
    Base class for stochastic processes in continuous time
    """

    @abstractmethod
    def sample_from_draws(self, path: Paths, *args: Paths) -> Paths:
        """Sample a path from the process given a set of draws"""

    @abstractmethod
    def sample(
        self, paths: int, time_horizon: float = 1, time_steps: int = 100
    ) -> Paths:
        """Generate random paths from the process.

        :param paths: Number of paths
        :param time_horizon: time horizon
        :param time_steps: number of time steps to arrive at horizon
        """


class StochasticProcess1D(StochasticProcess):
    """
    Base class for 1D stochastic process in continuous time
    """

    def marginal(self, t: float, N: int = 128) -> StochasticProcess1DMarginal:
        return StochasticProcess1DMarginal(self, t, N)

    def characteristic(self, t: float, u: Vector) -> Vector:
        r"""Characteristic function at time `t` for a given input parameter

        The characteristic function represents the Fourier transform of the
        probability density function

        .. math::
            \phi = {\mathbb E} \left[e^{i u x_t}\right]

        :param t: time horizon
        :param u: characteristic function input parameter
        """
        raise NotImplementedError

    def domain_range(self) -> Bounds:
        return default_bounds()

    def max_frequency(self, t: float) -> float:
        """Maximum frequency of the process"""
        return 20


P = TypeVar("P", bound=StochasticProcess1D)


class StochasticProcess1DMarginal(Marginal1D, Generic[P]):
    def __init__(self, process: P, t: float, N: int) -> None:
        self.process = process
        self.t = t
        self.N = N

    def std_norm(self) -> float:
        """Standard deviation at a time horizon normalized by the time"""
        return np.sqrt(self.variance() / self.t)

    def characteristic(self, u: Vector) -> Vector:
        return self.process.characteristic(self.t, u)

    def domain_range(self) -> Bounds:
        return self.process.domain_range()

    def max_frequency(self) -> float:
        return self.process.max_frequency(self.t)


class IntensityProcess(StochasticProcess1D):
    """Base class for mean reverting 1D processes which can be used
    as stochastic intensity
    """

    rate: float = Field(default=1.0, gt=0, description="Instantaneous initial rate")
    kappa: float = Field(default=1.0, gt=0, description="Mean reversion speed")

    @abstractmethod
    def cumulative_characteristic(self, t: float, u: Vector) -> Vector:
        r"""The characteristic function of the cumulative process:

        .. math::
            \phi = {\mathbb E} \left[e^{i u \int_0^t x_s ds}\right]

        :param t: time horizon
        :param u: frequency
        """

    def domain_range(self) -> Bounds:
        return Bounds(0, np.inf)
