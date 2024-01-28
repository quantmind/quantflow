from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd
from numpy.random import normal
from pydantic import BaseModel, Field
from scipy.integrate import cumulative_trapezoid

from . import plot
from .bins import pdf as bins_pdf
from .types import FloatArray


class Paths(BaseModel, arbitrary_types_allowed=True):
    """Paths of a stochastic process"""

    t: float = Field(description="time horizon")
    data: FloatArray = Field(description="paths")

    @property
    def dt(self) -> float:
        return self.t / self.time_steps

    @property
    def samples(self) -> int:
        """Number of samples"""
        return self.data.shape[1]

    @property
    def time_steps(self) -> int:
        """Number of time steps"""
        return self.data.shape[0] - 1

    @property
    def time(self) -> FloatArray:
        """Time as numpy array"""
        return np.linspace(0.0, self.t, num=self.time_steps + 1)

    @property
    def df(self) -> pd.DataFrame:
        """Paths as pandas DataFrame"""
        return pd.DataFrame(self.data, index=self.time)

    @property
    def xs(self) -> list[np.ndarray]:
        """Time as list of list (for visualization tools)"""
        return self.samples * [self.time]

    @property
    def ys(self) -> list[list[float]]:
        """Paths as list of list (for visualization tools)"""
        return self.data.transpose().tolist()

    def mean(self) -> FloatArray:
        """Mean of paths"""
        return np.mean(self.data, axis=1)

    def std(self) -> FloatArray:
        """Standard deviation of paths"""
        return np.std(self.data, axis=1)

    def var(self) -> FloatArray:
        """Variance of paths"""
        return np.var(self.data, axis=1)

    def integrate(self) -> Paths:
        """Integrate paths"""
        return self.__class__(
            t=self.t,
            data=cumulative_trapezoid(self.data, dx=self.dt, axis=0, initial=0),
        )

    def cross_section(self, t: float | None = None) -> FloatArray:
        """Cross section of paths at time t"""
        index = self.time_steps
        if t is not None:
            index = cast(int, t // self.dt)
        return self.data[index, :]

    def pdf(
        self,
        t: float | None = None,
        num_bins: int | None = None,
        delta: float | None = None,
        symmetric: float | None = None,
    ) -> pd.DataFrame:
        """Probability density function of paths

        Calculate a DataFrame with the probability density function of the paths
        at a given cross section of time. By default it take the last section.

        :param t: time at which to calculate the pdf
        :param num_bins: number of bins
        :param delta: optional size of bins (cannot be set with num_bins)
        :param symmetric: optional center of bins
        """
        return bins_pdf(
            self.cross_section(t), num_bins=num_bins, delta=delta, symmetric=symmetric
        )

    def plot(self, **kwargs: Any) -> Any:
        """Plot paths

        It requires plotly installed
        """
        return plot.plot_lines(self.df, **kwargs)

    @classmethod
    def normal_draws(
        cls,
        paths: int,
        time_horizon: float = 1,
        time_steps: int = 1000,
        antithetic_variates: bool = True,
    ) -> Paths:
        """Generate normal draws

        :param paths: number of paths
        :param time_horizon: time horizon
        :param time_steps: number of time steps to arrive at horizon
        :param antithetic_variates: whether to use antithetic variates
        """
        time_horizon / time_steps
        odd = 0
        if antithetic_variates:
            odd = paths % 2
            paths = paths // 2
        data = normal(size=(time_steps + 1, paths))
        if antithetic_variates:
            data = np.concatenate((data, -data), axis=1)
            if odd:
                extra_data = normal(size=(time_steps + 1, odd))
                data = np.concatenate((data, extra_data), axis=1)
        return cls(t=time_horizon, data=data)
