from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Generic, NamedTuple, TypeVar

import numpy as np
import pandas as pd

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
    """Call prices"""
    name: str = ""
    """Name of the model"""

    @property
    def moneyness_ttm(self) -> FloatArray:
        return self.moneyness / np.sqrt(self.ttm)

    @property
    def time_value(self) -> FloatArray:
        return self.call - self.intrinsic_value

    @property
    def intrinsic_value(self) -> FloatArray:
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
        """Call price a single option"""
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

    def plot(self, series: str = "implied_vol") -> Any:
        """Plot the results"""
        return plot.plot_vol_cross(self.df, series=series)


@dataclass
class OptionPricer(Generic[M]):
    """Pricer for options"""

    model: M
    """The stochastic process"""
    ttm: dict[int, MaturityPricer] = field(default_factory=dict, repr=False)
    """Cache for the maturity pricer"""
    n: int = 128
    max_moneyness_ttm: float = 1.5

    def reset(self) -> None:
        """Clear the cache"""
        self.ttm.clear()

    def maturity(self, ttm: float, **kwargs: Any) -> MaturityPricer:
        """Get the maturity cache or create a new one and return it"""
        ttm_int = int(TTM_FACTOR * ttm)
        if ttm_int not in self.ttm:
            ttmr = ttm_int / TTM_FACTOR
            marginal = self.model.marginal(ttmr)
            transform = marginal.call_option(
                self.n, max_moneyness=self.max_moneyness_ttm * np.sqrt(ttmr), **kwargs
            )
            self.ttm[ttm_int] = MaturityPricer(
                ttm=ttmr,
                std=np.max(marginal.std()),
                moneyness=transform.x,
                call=transform.y,
                name=type(self.model).__name__,
            )
        return self.ttm[ttm_int]

    def call_price(self, ttm: float, moneyness: float) -> float:
        """Price a single call option"""
        return self.maturity(ttm).call_price(moneyness)
