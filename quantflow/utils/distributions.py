from abc import abstractmethod

import numpy as np
from pydantic import Field

from .marginal import Marginal1D
from .types import Vector


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
