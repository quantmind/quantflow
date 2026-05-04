from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from scipy.optimize import Bounds
from typing_extensions import Annotated, Doc

from quantflow.ta.paths import Paths
from quantflow.utils.marginal import Marginal1D, default_bounds
from quantflow.utils.numbers import sigfig
from quantflow.utils.transforms import bound_from_any
from quantflow.utils.types import FloatArray, FloatArrayLike, Vector

Im = complex(0, 1)


class StochasticProcess(BaseModel, ABC, extra="forbid"):
    """
    Base class for stochastic processes in continuous time
    """

    @abstractmethod
    def sample_from_draws(self, draws: Paths, *args: Paths) -> Paths:
        """Sample [Paths][quantflow.ta.paths.Paths]
        from the process given a set of draws"""

    @abstractmethod
    def sample(
        self,
        n: Annotated[int, Doc("number of paths")],
        time_horizon: Annotated[float, Doc("time horizon")] = 1,
        time_steps: Annotated[
            int, Doc("number of time steps to arrive at horizon")
        ] = 100,
    ) -> Paths:
        """Generate random [Paths][quantflow.ta.paths.Paths] from the process."""

    @abstractmethod
    def characteristic_exponent(self, t: FloatArrayLike, u: Vector) -> Vector:
        """Characteristic exponent at time `t` for a given input parameter"""

    def characteristic(
        self,
        t: Annotated[FloatArrayLike, Doc("Time horizon")],
        u: Annotated[Vector, Doc("Characteristic function input parameter")],
    ) -> Vector:
        r"""Characteristic function at time `t` for a given input parameter `u`

        The characteristic function represents the Fourier transform of the
        probability density function

        \begin{equation}
            \Phi = {\mathbb E} \left[e^{i u x_t}\right] = e^{-\phi(t, u)}
        \end{equation}

        where $\phi$ is the characteristic exponent, which can be more easily
        computed for many processes.
        """
        return np.exp(-self.characteristic_exponent(t, u))

    def convexity_correction(self, t: FloatArrayLike) -> Vector:
        """Convexity correction for the process"""
        return -self.characteristic_exponent(t, complex(0, -1)).real

    def analytical_std(self, t: FloatArrayLike) -> FloatArrayLike:
        """Analytical standard deviation of the process at time `t`

        This has a closed form solution if the process has an analytical variance
        """
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
        """Marginal distribution of the process at time `t`"""
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
        start = float(sigfig(bound_from_any(bounds.lb, mean - std)))
        end = float(sigfig(bound_from_any(bounds.ub, mean + std)))
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

    def call_option_carr_madan_alpha(self) -> float:
        """Option alpha parameter for integrability of call option transform
        in the Carr-Madan formula.

        The choice of alpha is crucial for the numerical stability of the Carr-Madan
        formula. A common choice is to set alpha to a value that ensures the integrand
        decays sufficiently fast at high frequencies.
        """
        return max(8 * np.max(np.exp(-2 * self.t)), 0.5)


class IntensityProcess(StochasticProcess1D):
    """Base class for mean reverting 1D processes which can be used
    as stochastic intensity
    """

    rate: float = Field(
        default=1.0, gt=0, description=r"Instantaneous initial rate $x_0$"
    )
    kappa: float = Field(
        default=1.0, gt=0, description=r"Mean reversion speed $\kappa$"
    )

    @abstractmethod
    def integrated_log_laplace(
        self,
        t: Annotated[FloatArrayLike, Doc("time horizon")],
        u: Annotated[Vector, Doc("frequency")],
    ) -> Vector:
        r"""The log-Laplace transform of the cumulative process:

        \begin{equation}
            \phi_{t, u} = \log {\mathbb E} \left[e^{-u \int_0^t x_s ds}\right]
        \end{equation}
        """

    def domain_range(self) -> Bounds:
        return Bounds(0, np.inf)

    def ekt(self, t: FloatArrayLike) -> FloatArrayLike:
        return np.exp(-self.kappa * t)
