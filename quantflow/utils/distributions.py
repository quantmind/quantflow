from abc import abstractmethod

import numpy as np
from pydantic import Field

from .marginal import Marginal1D
from .types import FloatArray, Vector


class Distribution1D(Marginal1D):
    @abstractmethod
    def sample(self, n: int) -> np.ndarray:
        """Sample from the distribution"""


class Exponential(Distribution1D):
    decay: float = Field(default=1, gt=0, description="exponential decay rate")

    @property
    def scale(self) -> float:
        return 1 / self.decay

    def characteristic(self, u: Vector) -> Vector:
        return self.decay / (self.decay - complex(0, 1) * u)

    def mean(self) -> float:
        return self.scale

    def variance(self) -> float:
        return self.scale**2

    def sample(self, n: int) -> np.ndarray:
        return np.random.exponential(scale=self.scale, size=n)

    def support(self, points: int = 100, *, std_mult: float = 4) -> FloatArray:
        return np.linspace(0, std_mult * np.max(self.std()), points)


class Normal(Distribution1D):
    mu: float = Field(default=0, description="mean")
    sigma: float = Field(default=1, gt=0, description="standard deviation")

    def characteristic(self, u: Vector) -> Vector:
        return np.exp(complex(0, 1) * u * self.mu - self.sigma**2 * u**2 / 2)

    def mean(self) -> float:
        return self.mu

    def variance(self) -> float:
        return self.sigma**2

    def sample(self, n: int) -> np.ndarray:
        return np.random.normal(loc=self.mu, scale=self.sigma, size=n)

    def support(self, points: int = 100, *, std_mult: float = 4) -> FloatArray:
        return np.linspace(
            self.mu - std_mult * self.sigma,
            self.mu + std_mult * self.sigma,
            points,
        )
