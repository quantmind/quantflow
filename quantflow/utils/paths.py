from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from scipy.integrate import cumtrapz

from . import plot


class Paths(BaseModel):
    """Paths of a stochastic process"""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    t: float = Field(description="time horizon")
    data: np.ndarray = Field(description="paths")

    @property
    def dt(self) -> float:
        return self.t / self.steps

    @property
    def samples(self) -> int:
        return self.data.shape[1]

    @property
    def steps(self) -> int:
        return self.data.shape[0] - 1

    @property
    def time(self) -> np.ndarray:
        return np.linspace(0.0, self.t, num=self.steps + 1)

    @property
    def df(self) -> pd.DataFrame:
        return pd.DataFrame(self.data)

    @property
    def xs(self) -> list[np.ndarray]:
        """Time as list of list (for visualization tools)"""
        return self.samples * [self.time]

    @property
    def ys(self) -> list[list[float]]:
        """Paths as list of list (for visualization tools)"""
        return self.data.transpose().tolist()

    def integrate(self) -> Paths:
        """Integrate paths"""
        return self.__class__(
            t=self.t, data=cumtrapz(self.data, dx=self.dt, axis=0, initial=0)
        )

    def plot(self) -> Any:
        return plot.plot_lines(self.data)
