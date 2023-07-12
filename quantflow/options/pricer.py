from dataclasses import dataclass, field
from typing import Generic, TypeVar

import numpy as np
from scipy.optimize import RootResults

from quantflow.sp.base import StochasticProcess1D
from quantflow.utils.transforms import TransformResult

from ..utils.types import FloatArray, FloatArrayLike
from .bs import implied_black_volatility

M = TypeVar("M", bound=StochasticProcess1D)


TTM_FACTOR = 10000


@dataclass
class MaturityPricer:
    ttm: float
    transform: TransformResult

    def price(self, moneyness: float) -> float:
        """Price a single option"""
        return float(np.interp(moneyness, self.transform.x, self.transform.y))

    def prices(self, moneyness: FloatArrayLike) -> FloatArray:
        """get interpolated prices at monenyness"""
        return np.interp(moneyness, self.transform.x, self.transform.y)

    def implied_vols(self, moneyness: FloatArrayLike | None = None) -> RootResults:
        """Calculate the implied volatility"""
        if moneyness is None:
            moneyness = self.transform.x
            prices = self.transform.y
        else:
            moneyness = np.asarray(moneyness)
            prices = self.prices(moneyness)
        return implied_black_volatility(
            moneyness,
            prices,
            ttm=self.ttm,
            initial_sigma=0.5 * np.ones_like(moneyness),
            call_put=1.0,
        )


@dataclass
class OptionPricer(Generic[M]):
    """Pricer for options"""

    model: M
    """The stochastic process"""
    ttm: dict[int, MaturityPricer] = field(default_factory=dict)
    """Cache for the maturity pricer"""
    n: int = 128

    def reset(self) -> None:
        """Clear the cache"""
        self.ttm.clear()

    def maturity(self, ttm: float) -> MaturityPricer:
        """Get the maturity cache or create a new one and return it"""
        ttm_int = int(TTM_FACTOR * ttm)
        if ttm_int not in self.ttm:
            self.ttm[ttm_int] = MaturityPricer(
                ttm=ttm_int / TTM_FACTOR,
                transform=self.model.marginal(ttm_int / TTM_FACTOR).call_option(self.n),
            )
        return self.ttm[ttm_int]

    def price(self, ttm: float, moneyness: float) -> float:
        """Price a single option"""
        return self.maturity(ttm).price(moneyness)

    def implied_vols(
        self, ttm: float, moneyness: FloatArrayLike | None = None
    ) -> RootResults:
        """Calculate the implied volatility"""
        return self.maturity(ttm).implied_vols(moneyness=moneyness)
