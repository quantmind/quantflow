from __future__ import annotations

from abc import abstractmethod
from decimal import Decimal
from typing import Any, Generic, Self, TypeVar, cast

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, computed_field
from typing_extensions import Annotated, Doc

from quantflow.sp.base import StochasticProcess1D
from quantflow.utils import plot
from quantflow.utils.numbers import DecimalNumber, to_decimal

from ..utils.types import FloatArray
from .bs import BlackSensitivities, black_call, implied_black_volatility
from .inputs import OptionType

M = TypeVar("M", bound=StochasticProcess1D)


TTM_FACTOR = 10000


def get_intrinsic_value(moneyness: FloatArray) -> FloatArray:
    return 1.0 - np.exp(np.clip(moneyness, None, 0))


class ModelOptionPrice(BaseModel, frozen=True):
    """Represents the model price and sensitivities of an option for a given strike,
    forward and time to maturity.

    It provides black price and sensitivies too for comparison and analysis
    of the model price.
    """

    strike: DecimalNumber = Field(description="Strike price of the option")
    option_type: OptionType = Field(description="Type of the option, call or put")
    forward: DecimalNumber = Field(description="Forward price of the underlying")
    moneyness: float = Field(description="Moneyness as log(Strike/Forward)")
    ttm: float = Field(default=0, description="Time to maturity in years")
    price: float = Field(description=("Price in forward space"))
    delta: float = Field(description="Model delta of the option")
    gamma: float = Field(description="Model gamma of the option")

    @computed_field  # type: ignore [prop-decorator]
    @property
    def black(self) -> BlackSensitivities:
        """Calculate the Black price for the option using the price as time value and
        moneyness as log-strike"""
        return BlackSensitivities.calculate(
            k=self.moneyness,
            ttm=self.ttm,
            s=self.option_type.call_put(),
            price=self.price,
        )

    @property
    def intrinsic_value(self) -> float:
        """Calculate the intrinsic value of the option in forward space for
        the given moneyness

        This is the value of the option if it were to expire immediately, and it
        depends only on the moneyness and the type of the option.

        For a call option, the intrinsic value is non-negative when the moneyness
        is negative, i.e. when the strike is below the forward price.
        For a put option, the intrinsic value is non-negative when the moneyness
        is positive, i.e. when the strike is above the forward price.
        """
        if self.option_type == OptionType.call:
            return max(0.0, 1.0 - np.exp(self.moneyness))
        else:
            return max(0.0, np.exp(self.moneyness) - 1.0)

    def as_option_type(
        self,
        option_type: Annotated[
            OptionType,
            Doc("Type of the option, call or put"),
        ],
    ) -> Self:
        """Convert the option price to the given option type"""
        if self.option_type == option_type:
            return self
        else:
            return self.model_copy(
                update=dict(
                    option_type=option_type,
                    price=self.price - self.intrinsic_value,
                    delta=self.delta - 1.0,
                    gamma=self.gamma,
                )
            )


class MaturityPricer(BaseModel, arbitrary_types_allowed=True):
    """Result of option pricing for a given Time to Maturity"""

    ttm: float = Field(description="Time to maturity")
    std: float = Field(description="Standard deviation model of log returns")
    moneyness: FloatArray = Field(description="Moneyness as log Strike/Forward")
    call: FloatArray = Field(
        description="Call prices in forward space for the given moneyness"
    )
    name: str = Field(default="", description="Name of the model")

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

    def price(
        self,
        option_type: Annotated[OptionType, Doc("Type of the option (call or put)")],
        strike: Annotated[float | Decimal, Doc("Strike price of the option")],
        forward: Annotated[float | Decimal, Doc("Forward price of the underlying")],
    ) -> ModelOptionPrice:
        """Price a single option

        This method will use the cache to get the maturity pricer if possible
        """
        strike_ = to_decimal(strike)
        forward_ = to_decimal(forward)
        moneyness = float((strike_ / forward_).ln())
        return ModelOptionPrice(
            option_type=OptionType.call,
            strike=strike_,
            forward=forward_,
            moneyness=moneyness,
            ttm=self.ttm,
            price=self.call_price(moneyness),
            delta=self.call_delta(moneyness),
            gamma=self.call_gamma(moneyness),
        ).as_option_type(option_type)

    @property
    def implied_vols(self) -> FloatArray:
        """Calculate the implied volatility"""
        return implied_black_volatility(
            self.moneyness,
            self.call,
            ttm=self.ttm,
            initial_sigma=0.5 * np.ones_like(self.moneyness),
            call_put=1.0,
        ).values

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

    def call_delta(self, moneyness: float) -> float:
        r"""Delta of a call option as change in price per unit change in forward.

        Since prices are stored in forward space (c = C/F) and m = log(K/F),
        the chain rule gives: dC/dF = c - dc/dm
        """
        dc_dm = np.gradient(self.call, self.moneyness)
        return float(np.interp(moneyness, self.moneyness, self.call - dc_dm))

    def call_gamma(self, moneyness: float) -> float:
        """Gamma of a call option as change in delta per unit change in forward.

        Since prices are stored in forward space (c = C/F) and m = log(K/F),
        the chain rule gives: d2C/dF2 = d2c/dm2 - dc/dm
        """
        dc_dm = np.gradient(self.call, self.moneyness)
        d2c_dm2 = np.gradient(dc_dm, self.moneyness)
        return float(np.interp(moneyness, self.moneyness, d2c_dm2 - dc_dm))

    def interp(self, moneyness: FloatArray) -> Self:
        """get interpolated prices"""
        return self.model_copy(
            update=dict(
                moneyness=moneyness,
                call=np.interp(moneyness, self.moneyness, self.call),
            )
        )

    def max_moneyness_ttm(
        self, max_moneyness_ttm: float = 1.0, support: int = 51
    ) -> Self:
        """Calculate the implied volatility"""
        moneyness_ttm = np.linspace(-max_moneyness_ttm, max_moneyness_ttm, support)
        moneyness = np.asarray(moneyness_ttm) * np.sqrt(self.ttm)
        return self.interp(moneyness)

    def black(self) -> Self:
        """Calculate the Maturity Result for the Black model with same std"""
        return self.model_copy(
            update=dict(
                call=np.asarray(
                    black_call(
                        self.moneyness, self.std / np.sqrt(self.ttm), ttm=self.ttm
                    )
                ),
                name="Black",
            )
        )

    def plot(self, series: str = "implied_vol", **kwargs: Any) -> Any:
        """Plot the results"""
        return plot.plot_vol_cross(self.df, series=series, **kwargs)


class OptionPricerBase(BaseModel, arbitrary_types_allowed=True):
    """Abstract base class for option pricers.

    Provides caching of [MaturityPricer][quantflow.options.pricer.MaturityPricer]
    results and generic pricing/plotting methods. Subclasses implement
    ``_compute_maturity`` to define how call prices are computed for a given
    time to maturity.
    """

    ttm: dict[int, MaturityPricer] = Field(
        default_factory=dict,
        repr=False,
        exclude=True,
        description=(
            "Cache for [MaturityPricer][quantflow.options.pricer.MaturityPricer] "
            "for different time to maturity"
        ),
    )

    @abstractmethod
    def _compute_maturity(self, ttm: float, **kwargs: Any) -> MaturityPricer:
        """Compute a [MaturityPricer][quantflow.options.pricer.MaturityPricer]
        for the given time to maturity.

        Called by [maturity][quantflow.options.pricer.OptionPricerBase.maturity]
        on a cache miss. Subclasses must implement this method.
        """

    def reset(self) -> None:
        """Clear the [ttm][quantflow.options.pricer.OptionPricerBase.ttm] cache"""
        self.ttm.clear()

    def maturity(self, ttm: float, **kwargs: Any) -> MaturityPricer:
        """Get a [MaturityPricer][quantflow.options.pricer.MaturityPricer]
        from cache or compute a new one and return it"""
        ttm_int = int(TTM_FACTOR * ttm)
        if ttm_int not in self.ttm:
            ttmr = ttm_int / TTM_FACTOR
            self.ttm[ttm_int] = self._compute_maturity(ttmr, **kwargs)
        return self.ttm[ttm_int]

    def price(
        self,
        option_type: Annotated[OptionType, Doc("Type of the option (call or put)")],
        ttm: Annotated[float, Doc("Time to maturity")],
        strike: Annotated[float, Doc("Strike price of the option")],
        forward: Annotated[float, Doc("Forward price of the underlying")],
    ) -> ModelOptionPrice:
        """Price a single option

        This method will use the cache to get the maturity pricer if possible
        """
        return self.maturity(ttm).price(option_type, strike, forward)

    def call_price(
        self,
        ttm: Annotated[float, Doc("Time to maturity")],
        moneyness: Annotated[float, Doc("Moneyness as log(Strike/Forward)")],
    ) -> float:
        """Price a single call option

        This method will use the cache to get the maturity pricer if possible
        """
        return self.maturity(ttm).call_price(moneyness)

    def plot3d(
        self,
        max_moneyness_ttm: float = 1.0,
        support: int = 51,
        ttm: FloatArray | None = None,
        dragmode: str = "turntable",
        scene_camera: dict | None = None,
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
            dragmode=dragmode,
            scene=dict(
                xaxis=dict(title="moneyness_ttm"),
                yaxis=dict(title="TTM"),
                zaxis=dict(title="implied_vol"),
            ),
            scene_camera=scene_camera or dict(eye=dict(x=1.2, y=-1.8, z=0.3)),
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


class OptionPricer(OptionPricerBase, Generic[M]):
    """Pricer for options based on a stochastic process model.

    Computes call prices via the inverse Fourier transform of the
    characteristic function of the underlying stochastic process.
    """

    model: M = Field(description="Stochastic process model for the underlying")
    n: int = Field(
        default=128,
        description="Number of discretization points for the marginal distribution",
    )
    max_moneyness_ttm: float = Field(
        default=1.5, description="Max time-adjusted moneyness to calculate prices"
    )

    def _compute_maturity(self, ttm: float, **kwargs: Any) -> MaturityPricer:
        marginal = self.model.marginal(ttm)
        transform = marginal.call_option(
            self.n, max_moneyness=self.max_moneyness_ttm * np.sqrt(ttm), **kwargs
        )
        return MaturityPricer(
            ttm=ttm,
            std=float(np.max(marginal.std())),
            moneyness=transform.x,
            call=transform.y,
            name=type(self.model).__name__,
        )
