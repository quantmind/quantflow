from dataclasses import dataclass, field
from typing import Generic, TypeVar

import numpy as np

from quantflow.sp.base import StochasticProcess1D
from quantflow.utils.transforms import TransformResult

M = TypeVar("M", bound=StochasticProcess1D)


TTM_FACTOR = 10000


@dataclass
class MaturityPricer:
    prices: TransformResult

    def price(self, moneyness: float) -> float:
        """Price a single option"""
        return float(np.interp(moneyness, self.prices.x, self.prices.y))


@dataclass
class OptionPricer(Generic[M]):
    """Pricer for options"""

    model: M
    ttm: dict[int, MaturityPricer] = field(default_factory=dict)

    def reset(self) -> None:
        """Clear the cache"""
        self.ttm.clear()

    def price(self, ttm: float, moneyness: float) -> float:
        """Price a single option"""
        ttm_int = int(TTM_FACTOR * ttm)
        if ttm_int not in self.ttm:
            self.ttm[ttm_int] = MaturityPricer(
                prices=self.model.marginal(ttm_int / TTM_FACTOR).call_option()
            )
        return self.ttm[ttm_int].price(moneyness)
