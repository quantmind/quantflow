from __future__ import annotations

import numpy as np
from pydantic import Field
from scipy.stats import norm

from ..utils.paths import Paths
from ..utils.types import Vector
from .base import StochasticProcess1D, StochasticProcess1DMarginal


class WeinerProcess(StochasticProcess1D):
    sigma: float = Field(default=1, ge=0, description="volatility")

    @property
    def sigma2(self) -> float:
        return self.sigma * self.sigma

    def marginal(self, t: Vector) -> StochasticProcess1DMarginal:
        return WeinerMarginal(process=self, t=t)

    def characteristic_exponent(self, t: Vector, u: Vector) -> Vector:
        su = self.sigma * u
        return 0.5 * su * su * t

    def sample(self, n: int, time_horizon: float = 1, time_steps: int = 100) -> Paths:
        paths = Paths.normal_draws(n, time_horizon, time_steps)
        return self.sample_from_draws(paths)

    def sample_from_draws(self, draws: Paths, *args: Paths) -> Paths:
        self.sigma * np.sqrt(draws.dt)
        paths = np.zeros(draws.data.shape)
        paths[1:] = np.cumsum(draws.data[:-1], axis=0)
        return Paths(t=draws.t, data=paths)


class WeinerMarginal(StochasticProcess1DMarginal[WeinerProcess]):
    def mean(self) -> Vector:
        return 0 * self.t

    def variance(self) -> Vector:
        s = self.process.sigma
        return s * s * self.t

    def pdf(self, n: Vector) -> Vector:
        return norm.pdf(n, scale=self.std())

    def cdf(self, n: Vector) -> Vector:
        return norm.cdf(n, scale=self.std())
