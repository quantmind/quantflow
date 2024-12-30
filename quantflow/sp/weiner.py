from __future__ import annotations

import numpy as np
from pydantic import Field
from scipy.stats import norm

from ..ta.paths import Paths
from ..utils.types import FloatArrayLike, Vector
from .base import StochasticProcess1D


class WeinerProcess(StochasticProcess1D):
    sigma: float = Field(default=1, ge=0, description="volatility")

    @property
    def sigma2(self) -> float:
        return self.sigma * self.sigma

    def characteristic_exponent(self, t: Vector, u: Vector) -> Vector:
        su = self.sigma * u
        return 0.5 * su * su * t

    def sample(self, n: int, time_horizon: float = 1, time_steps: int = 100) -> Paths:
        paths = Paths.normal_draws(n, time_horizon, time_steps)
        return self.sample_from_draws(paths)

    def sample_from_draws(self, draws: Paths, *args: Paths) -> Paths:
        sdt = self.sigma * np.sqrt(draws.dt)
        paths = np.zeros(draws.data.shape)
        paths[1:] = np.cumsum(draws.data[:-1], axis=0)
        return Paths(t=draws.t, data=sdt * paths)

    def analytical_mean(self, t: FloatArrayLike) -> FloatArrayLike:
        return 0 * t

    def analytical_variance(self, t: FloatArrayLike) -> FloatArrayLike:
        return t * self.sigma2

    def analytical_pdf(self, t: FloatArrayLike, x: FloatArrayLike) -> FloatArrayLike:
        return norm.pdf(x, scale=self.analytical_std(t))

    def analytical_cdf(self, t: FloatArrayLike, x: FloatArrayLike) -> FloatArrayLike:
        return norm.cdf(x, scale=self.analytical_std(t))
