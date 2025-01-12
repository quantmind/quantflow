from __future__ import annotations

from typing import Any, Generic, NamedTuple, TypeVar, cast

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from quantflow.sp.base import StochasticProcess1D
from quantflow.utils import plot

from ..utils.types import FloatArray
from .bs import black_call, implied_black_volatility

M = TypeVar("M", bound=StochasticProcess1D)


TTM_FACTOR = 10000


def get_intrinsic_value(moneyness: FloatArray) -> FloatArray:
    return 1.0 - np.exp(np.clip(moneyness, None, 0))


class MaturityPricer(NamedTuple):
    """Result of option pricing for a given Time to Maturity"""

    ttm: float
    """Time to Maturity"""
    std: float
    """Standard deviation model of log returns"""
    moneyness: FloatArray
    """Moneyness as log Strike/Forward"""
    call: FloatArray
    """Call prices for the given :attr`.moneyness`"""
    name: str = ""
    """Name of the model"""

    @property
    def moneyness_ttm(self) -> FloatArray:
        """Time adjusted moneyness array"""
        return self.moneyness / np.sqrt(self.ttm)

    @property
    def time_value(self) -> FloatArray:
        """Time value of the option"""
        return self.call - self.intrinsic_value

    @property
    def intrinsic_value(self) -> FloatArray:
        """Intrinsic value of the option"""
        return get_intrinsic_value(self.moneyness)

    @property
    def implied_vols(self) -> FloatArray:
        """Calculate the implied volatility"""
        return implied_black_volatility(
            self.moneyness,
            self.call,
            ttm=self.ttm,
            initial_sigma=0.5 * np.ones_like(self.moneyness),
            call_put=1.0,
        ).root

    @property
    def df(self) -> pd.DataFrame:
        """Get a dataframe with the results"""
        return pd.DataFrame(
            {
                "moneyness": self.moneyness,
                "moneyness_ttm": self.moneyness_ttm,
                "call": self.call,
                "implied_vol": self.implied_vols,
                "time_value": self.time_value,
            }
        )

    def call_price(self, moneyness: float) -> float:
        """Price a single call option"""
        return float(np.interp(moneyness, self.moneyness, self.call))

    def interp(self, moneyness: FloatArray) -> MaturityPricer:
        """get interpolated prices"""
        return self._replace(
            moneyness=moneyness,
            call=np.interp(moneyness, self.moneyness, self.call),
        )

    def max_moneyness_ttm(
        self, max_moneyness_ttm: float = 1.0, support: int = 51
    ) -> MaturityPricer:
        """Calculate the implied volatility"""
        moneyness_ttm = np.linspace(-max_moneyness_ttm, max_moneyness_ttm, support)
        moneyness = np.asarray(moneyness_ttm) * np.sqrt(self.ttm)
        return self.interp(moneyness)

    def black(self) -> MaturityPricer:
        """Calculate the Maturity Result for the Black model with same std"""
        return self._replace(
            call=black_call(self.moneyness, self.std / np.sqrt(self.ttm), ttm=self.ttm),
            name="Black",
        )

    def plot(self, series: str = "implied_vol", **kwargs: Any) -> Any:
        """Plot the results"""
        return plot.plot_vol_cross(self.df, series=series, **kwargs)


class OptionPricer(BaseModel, Generic[M], arbitrary_types_allowed=True):
    """Pricer for options"""

    model: M
    """The stochastic process used for pricing"""
    ttm: dict[int, MaturityPricer] = Field(
        default_factory=dict, repr=False, exclude=True
    )
    """Cache for :class:`.MaturityPricer` for different time to maturity"""
    n: int = 128
    """NUmber of discretization points for the marginal distribution"""
    max_moneyness_ttm: float = 1.5
    """Max time-adjusted moneyness to calculate prices"""

    def reset(self) -> None:
        """Clear the :attr:`.ttm` cache"""
        self.ttm.clear()

    def maturity(self, ttm: float, **kwargs: Any) -> MaturityPricer:
        """Get a :class:`.MaturityPricer` from cache or create
        a new one and return it"""
        ttm_int = int(TTM_FACTOR * ttm)
        if ttm_int not in self.ttm:
            ttmr = ttm_int / TTM_FACTOR
            marginal = self.model.marginal(ttmr)
            transform = marginal.call_option(
                self.n, max_moneyness=self.max_moneyness_ttm * np.sqrt(ttmr), **kwargs
            )
            self.ttm[ttm_int] = MaturityPricer(
                ttm=ttmr,
                std=float(np.max(marginal.std())),
                moneyness=transform.x,
                call=transform.y,
                name=type(self.model).__name__,
            )
        return self.ttm[ttm_int]

    def call_price(self, ttm: float, moneyness: float) -> float:
        """Price a single call option

        :param ttm: Time to maturity
        :param moneyness: Moneyness as log(Strike/Forward)

        This method will use the cache to get the maturity pricer if possible
        """
        return self.maturity(ttm).call_price(moneyness)

    def plot3d(
        self,
        max_moneyness_ttm: float = 1.0,
        support: int = 51,
        ttm: FloatArray | None = None,
        **kwargs: Any,
    ) -> Any:
        """Plot the implied vols surface

        It requires plotly to be installed
        """
        if ttm is None:
            ttm = np.arange(0.05, 1.0, 0.05)
        moneyness_ttm = np.linspace(-max_moneyness_ttm, max_moneyness_ttm, support)
        implied = np.zeros((len(ttm), len(moneyness_ttm)))
        for i, t in enumerate(ttm):
            maturity = self.maturity(cast(float, t))
            implied[i, :] = maturity.interp(moneyness_ttm * np.sqrt(t)).implied_vols
        properties: dict = dict(
            xaxis_title="moneyness_ttm",
            yaxis_title="TTM",
            colorscale="viridis",
            scene=dict(
                xaxis=dict(title="moneyness_ttm"),
                yaxis=dict(title="TTM"),
                zaxis=dict(title="implied_vol"),
            ),
            scene_camera=dict(eye=dict(x=1.2, y=-1.8, z=0.3)),
            contours=dict(
                x=dict(show=True, color="white"), y=dict(show=True, color="white")
            ),
        )
        properties.update(kwargs)
        return plot.plot3d(
            x=moneyness_ttm,
            y=ttm,
            z=implied,
            **properties,
        )
