from abc import abstractmethod

import numpy as np
from pydantic import Field

from .marginal import Marginal1D
from .types import FloatArray, Vector


class Distribution1D(Marginal1D):
    """Base class for 1D distributions to be used as
    Jump distributions in Compound Poisson processes
    """

    @abstractmethod
    def sample(self, n: int) -> np.ndarray:
        """Sample from the distribution"""


class Exponential(Distribution1D):
    decay: float = Field(default=1, gt=0, description="exponential decay rate")

    @property
    def scale(self) -> float:
        return 1 / self.decay

    @property
    def scale2(self) -> float:
        return self.scale**2

    def characteristic(self, u: Vector) -> Vector:
        return self.decay / (self.decay - complex(0, 1) * u)

    def mean(self) -> float:
        return self.scale

    def variance(self) -> float:
        return self.scale2

    def sample(self, n: int) -> np.ndarray:
        return np.random.exponential(scale=self.scale, size=n)

    def support(self, points: int = 100, *, std_mult: float = 4) -> FloatArray:
        return np.linspace(0, std_mult * np.max(self.std()), points)


class Normal(Distribution1D):
    mu: float = Field(default=0, description="mean")
    sigma: float = Field(default=1, gt=0, description="standard deviation")

    @property
    def sigma2(self) -> float:
        return self.sigma**2

    def characteristic(self, u: Vector) -> Vector:
        return np.exp(complex(0, 1) * u * self.mu - self.sigma2 * u**2 / 2)

    def mean(self) -> float:
        return self.mu

    def variance(self) -> float:
        return self.sigma2

    def sample(self, n: int) -> np.ndarray:
        return np.random.normal(loc=self.mu, scale=self.sigma, size=n)

    def support(self, points: int = 100, *, std_mult: float = 4) -> FloatArray:
        return np.linspace(
            self.mu - std_mult * self.sigma,
            self.mu + std_mult * self.sigma,
            points,
        )


class DoubleExponential(Exponential):
    """Double exponential distribution

    AKA Asymmetric Laplace distribution
    """

    k: float = Field(default=1, gt=0, description="asymmetric parameter")

    @property
    def k2(self) -> float:
        return self.k**2

    @property
    def k2m(self) -> float:
        k2 = self.k2
        return k2 / (k2 + 1)

    @property
    def scale_up(self) -> float:
        return np.sqrt(self.scale2_up)

    @property
    def scale_down(self) -> float:
        return np.sqrt(self.scale2_down)

    @property
    def scale2_up(self) -> float:
        return self.scale2 * self.k2m

    @property
    def scale2_down(self) -> float:
        return self.scale2 - self.scale2_up

    def characteristic(self, u: Vector) -> Vector:
        return np.exp(complex(0, 1) * u * self.mean()) / (1 - self.scale2 * u * u)

    def mean(self) -> float:
        return self.scale_up - self.scale_down

    def variance(self) -> float:
        return self.scale2

    def sample(self, n: int) -> np.ndarray:
        return np.random.exponential(scale=self.scale_up, size=n)

    def support(self, points: int = 100, *, std_mult: float = 4) -> FloatArray:
        return np.linspace(
            self.mean() - std_mult * self.std(),
            self.mean() + std_mult * self.std(),
            points,
        )
