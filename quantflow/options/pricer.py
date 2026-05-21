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
from quantflow.utils.marginal import OptionPricingMethod, OptionPricingResult
from quantflow.utils.numbers import DecimalNumber, to_decimal

from ..utils.types import FloatArray, FloatArrayLike
from .bs import BlackSensitivities, implied_black_volatility
from .inputs import OptionType

M = TypeVar("M", bound=StochasticProcess1D)


TTM_FACTOR = 10000


class ModelOptionPrice(BaseModel, frozen=True):
    r"""Model price and sensitivities of an option for a given strike,
    forward and time to maturity.

    The [price][.price] field is always in
    [forward space](../../glossary.md#forward-space), regardless of whether
    the underlying market quotes options in the quote currency (e.g. SPX,
    USD) or in the underlying (inverse options, e.g. BTC).

    \begin{equation}
        c = \frac{C}{F}
    \end{equation}

    Use [price_in_quote][.price_in_quote] to recover the quote-currency
    premium $C = c\,F$.

    Also exposes Black price and sensitivities for comparison.
    """

    strike: DecimalNumber = Field(description="Strike price of the option")
    option_type: OptionType = Field(description="Type of the option, call or put")
    forward: DecimalNumber = Field(description="Forward price of the underlying")
    log_strike: float = Field(description="Log strike over forward, i.e. log(K/F)")
    moneyness: float = Field(description="Moneyness")
    ttm: float = Field(default=0, description="Time to maturity in years")
    price: float = Field(
        description=(
            "Option price in"
            " [forward space](../../glossary.md#forward-space)."
            " Multiply by [forward][.forward]"
            " (or read [price_in_quote][.price_in_quote])"
            " to obtain the quote-currency premium."
        )
    )
    delta: float = Field(description="Model delta of the option")
    gamma: float = Field(description="Model gamma of the option")

    @property
    def price_in_quote(self) -> float:
        """Premium in the quote currency: forward-space price times forward.

        For inverse markets (BTC) the conventional premium is in the
        underlying and equals [price][.price] directly; callers should pick
        the convention that matches their downstream consumer.
        """
        return self.price * float(self.forward)

    @property
    def parity(self) -> float:
        """Put call parity value for the option, i.e. the difference between call
        and put price for the same strike and maturity"""
        return 1.0 - float(np.exp(self.log_strike))

    @computed_field
    @property
    def black(self) -> BlackSensitivities:
        """Calculate the Black price for the option using the price as time value and
        log-strike"""
        return BlackSensitivities.calculate(
            k=self.log_strike,
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
            return max(0.0, self.parity)
        else:
            return max(0.0, -self.parity)

    def as_option_type(
        self,
        option_type: Annotated[
            OptionType,
            Doc("Type of the option, call or put"),
        ],
    ) -> Self:
        """Convert the option price to the given option type via put-call parity."""
        if self.option_type == option_type:
            return self
        if self.option_type == OptionType.call:
            new_price = self.price - self.parity
            new_delta = self.delta - 1.0
        else:
            new_price = self.price + self.parity
            new_delta = self.delta + 1.0
        return self.model_copy(
            update=dict(
                option_type=option_type,
                price=new_price,
                delta=new_delta,
            )
        )


class MaturityPricer(BaseModel, arbitrary_types_allowed=True):
    """Result of option pricing for a given Time to Maturity"""

    ttm: float = Field(description="Time to maturity")
    pricing: OptionPricingResult = Field(
        description="Call option pricing result for the maturity"
    )
    name: str = Field(default="", description="Name of the model")

    def moneyness(self, log_strikes: FloatArrayLike) -> FloatArrayLike:
        """Time-adjusted moneyness for one or many log-strikes"""
        return log_strikes / np.sqrt(self.ttm)

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
        log_strike = float((strike_ / forward_).ln())
        result = self.pricing.call_greeks(log_strike)
        return ModelOptionPrice(
            option_type=OptionType.call,
            strike=strike_,
            forward=forward_,
            log_strike=log_strike,
            moneyness=float(self.moneyness(log_strike)),
            ttm=self.ttm,
            price=result.price,
            delta=result.delta,
            gamma=result.gamma,
        ).as_option_type(option_type)

    def prices(self, log_strikes: FloatArray) -> pd.DataFrame:
        """Price a batch of call options with the same TTM and different log-strikes"""
        call_prices = self.pricing.call_price(log_strikes)
        ivs = implied_black_volatility(
            log_strikes,
            call_prices,
            ttm=self.ttm,
            initial_sigma=0.5 * np.ones_like(log_strikes),
            call_put=1.0,
        ).values
        return pd.DataFrame(
            {
                "log_strike": log_strikes,
                "moneyness": self.moneyness(log_strikes),
                "call": call_prices,
                "implied_vol": ivs,
                "time_value": call_prices - np.maximum(0, 1 - np.exp(log_strikes)),
            }
        )


class OptionPricerBase(BaseModel, arbitrary_types_allowed=True):
    """Abstract base class for option pricers.

    Provides caching of [MaturityPricer][..MaturityPricer]
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
    def _compute_maturity(self, ttm: float) -> MaturityPricer:
        """Compute a [MaturityPricer][quantflow.options.pricer.MaturityPricer]
        for the given time to maturity.

        Called by [maturity][quantflow.options.pricer.OptionPricerBase.maturity]
        on a cache miss. Subclasses must implement this method.
        """

    def reset(self) -> None:
        """Clear the [ttm][quantflow.options.pricer.OptionPricerBase.ttm] cache"""
        self.ttm.clear()

    def maturity(self, ttm: float) -> MaturityPricer:
        """Get a [MaturityPricer][quantflow.options.pricer.MaturityPricer]
        from cache or compute a new one and return it"""
        ttm_int = int(TTM_FACTOR * ttm)
        if ttm_int not in self.ttm:
            ttmr = ttm_int / TTM_FACTOR
            self.ttm[ttm_int] = self._compute_maturity(ttmr)
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

    def call_prices(
        self,
        ttms: Annotated[FloatArray, Doc("Vector of time to maturities")],
        log_strikes: Annotated[FloatArray, Doc("Vector of log-strikes log(K/F)")],
    ) -> FloatArray:
        """Price a batch of call options.

        Options are grouped by their `ttm` so each unique maturity pricer is
        retrieved (and cached) once and the corresponding log-strikes are
        interpolated in a single vectorised `np.interp` call.
        """
        out = np.empty_like(log_strikes, dtype=float)
        for ttm in np.unique(ttms):
            mask = ttms == ttm
            mat = self.maturity(float(ttm))
            out[mask] = mat.pricing.call_price(log_strikes[mask])
        return out

    def plot3d(
        self,
        max_moneyness: float = 1.0,
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
        moneyness = np.linspace(-max_moneyness, max_moneyness, support)
        implied = np.zeros((len(ttm), len(moneyness)))
        for i, t in enumerate(ttm):
            maturity = self.maturity(cast(float, t))
            implied[i, :] = maturity.prices(moneyness * np.sqrt(t))["implied_vol"]
        properties: dict = dict(
            xaxis_title="moneyness",
            yaxis_title="TTM",
            colorscale="viridis",
            dragmode=dragmode,
            scene=dict(
                xaxis=dict(title="moneyness"),
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
            x=moneyness,
            y=ttm,
            z=implied,
            **properties,
        )


class OptionPricer(OptionPricerBase, Generic[M]):
    """Pricer for options based on a stochastic process model.

    Computes call prices via the inverse Fourier transform of the
    call option transform function of the underlying stochastic process.
    """

    model: M = Field(description="Stochastic process model for the underlying")
    n: int = Field(
        default=128,
        description="Number of discretization points for the marginal distribution",
    )
    method: OptionPricingMethod = Field(
        default=OptionPricingMethod.CARR_MADAN,
        description="Method to use for option pricing",
    )
    cos_moneyness_std_precision: float = Field(
        default=12.0,
        description="the accuracy of the COS method in number of std at a given TTM",
    )
    max_moneyness: float = Field(
        default=1.5,
        description=(
            "Maximum time-scaled moneyness to use for the pricing grid. "
            "Only used if method is Lewis or Carr & Madan, otherwise ignored."
        ),
    )

    def _compute_maturity(self, ttm: float) -> MaturityPricer:
        marginal = self.model.marginal(ttm)
        option_pricing_result = marginal.call_option(
            self.n,
            pricing_method=self.method,
            max_moneyness=self.max_moneyness,
            cos_moneyness_std_precision=self.cos_moneyness_std_precision,
        )
        return MaturityPricer(
            ttm=ttm,
            pricing=option_pricing_result,
            name=type(self.model).__name__,
        )
