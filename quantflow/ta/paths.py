from __future__ import annotations

from datetime import datetime
from typing import Any, Self, cast

import numpy as np
import pandas as pd
from numpy.random import normal
from pydantic import BaseModel, Field
from scipy.integrate import cumulative_trapezoid
from typing_extensions import Annotated, Doc

from quantflow.utils import plot
from quantflow.utils.bins import pdf as bins_pdf
from quantflow.utils.dates import utcnow
from quantflow.utils.types import FloatArray


class Paths(BaseModel, arbitrary_types_allowed=True):
    """Paths of a stochastic process

    This is the output from a simulation of a stochastic process.
    """

    t: float = Field(description="Time horizon - the unit of time is not specified")
    data: FloatArray = Field(description="Paths of the stochastic process")

    @property
    def dt(self) -> float:
        """Time step given by the time horizon divided by time steps"""
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
        return self.data.transpose().tolist()  # type: ignore

    def path(self, i: int) -> FloatArray:
        """Path i"""
        return self.data[:, i]

    def dates(
        self, *, start: datetime | None = None, unit: str = "d"
    ) -> pd.DatetimeIndex:
        """Dates of paths as a pandas DatetimeIndex"""
        start = start or utcnow()
        end = start + pd.to_timedelta(self.t, unit=unit)
        return pd.date_range(start=start, end=end, periods=self.time_steps + 1)

    def mean(self) -> FloatArray:
        """Paths cross-section mean"""
        return np.mean(self.data, axis=1)

    def std(self) -> FloatArray:
        """Paths cross-section standard deviation"""
        return np.std(self.data, axis=1)

    def var(self) -> FloatArray:
        """Paths cross-section variance"""
        return np.var(self.data, axis=1)

    def paths_mean(self, *, scaled: bool = False) -> FloatArray:
        """mean for each path

        If scaled is True, the mean is scaled by the time step
        """
        scale = self.dt if scaled else 1.0
        return np.mean(self.data, axis=0) / scale

    def paths_std(self, *, scaled: bool = False) -> FloatArray:
        """standard deviation for each path

        If scaled is True, the standard deviation is scaled by the square
        root of the time step
        """
        scale = np.sqrt(self.dt) if scaled else 1.0
        return np.std(np.diff(self.data, axis=0), axis=0) / scale

    def paths_var(self, *, scaled: bool = False) -> FloatArray:
        """variance for each path

        If scaled is True, the variance is scaled by the time step
        """
        scale = self.dt if scaled else 1.0
        return np.var(np.diff(self.data, axis=0), axis=0) / scale

    def as_datetime_df(
        self, *, start: datetime | None = None, unit: str = "d"
    ) -> pd.DataFrame:
        """Paths as pandas DataFrame with datetime index"""
        return pd.DataFrame(self.data, index=self.dates(start=start, unit=unit))

    def integrate(self) -> Paths:
        """Integrate paths"""
        return self.__class__(
            t=self.t,
            data=cumulative_trapezoid(self.data, dx=self.dt, axis=0, initial=0),
        )

    def hurst_exponent(
        self,
        steps: Annotated[
            int | None,
            Doc(
                "number of lags to consider, if not provided it uses half "
                "of the time steps capped at 100"
            ),
        ] = None,
    ) -> float:
        """Estimate the Hurst exponent from all paths"""
        ts = self.time_steps // 2
        n = min(steps or ts, 100)
        lags = []
        tau = []
        for lag in range(2, n):
            variances = np.var(self.data[lag:, :] - self.data[:-lag, :], axis=0)
            tau.extend(variances)
            lags.extend([lag] * self.samples)
        return float(np.polyfit(np.log(lags), np.log(tau), 1)[0]) / 2.0

    def cross_section(
        self, t: Annotated[float | None, Doc("time of cross section")] = None
    ) -> FloatArray:
        """Cross section of paths at time t"""
        index = self.time_steps
        if t is not None:
            index = cast(int, t // self.dt)
        return self.data[index, :]

    def pdf(
        self,
        t: Annotated[float | None, Doc("time at which to calculate the pdf")] = None,
        num_bins: Annotated[int | None, Doc("number of bins to use")] = None,
        delta: Annotated[
            float | None, Doc("optional size of bins (cannot be set with num_bins)")
        ] = None,
        symmetric: Annotated[
            float | None, Doc("An optional value where to center bins")
        ] = None,
    ) -> pd.DataFrame:
        """Estimate the Probability density function from paths at a given
        time horizon.

        This method calculates a DataFrame with the probability density function
        of the paths at a given cross section of time.
        By default it take the last section.
        """
        return bins_pdf(
            self.cross_section(t),
            num_bins=num_bins,
            delta=delta,
            symmetric=symmetric,
        )

    def plot(self, **kwargs: Any) -> Any:
        """Plot paths as lines

        It requires plotly installed
        """
        return plot.plot_lines(self.df, **kwargs)

    @classmethod
    def normal_draws(
        cls,
        paths: Annotated[int, Doc("Number of paths to simulate")],
        time_horizon: Annotated[float, Doc("Time horizon")] = 1,
        time_steps: Annotated[
            int, Doc("Number of time steps to arrive at horizon")
        ] = 1000,
        antithetic_variates: Annotated[
            bool,
            Doc(
                "Whether to use [antithetic variates]"
                "(https://en.wikipedia.org/wiki/Antithetic_variates)"
                " to reduce variance by generating pairs of paths that are mirror"
                " images of each other"
            ),
        ] = True,
    ) -> Self:
        """Create paths from normal draws"""
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
