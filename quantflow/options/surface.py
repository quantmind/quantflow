from __future__ import annotations

import enum
import warnings
from datetime import datetime
from decimal import Decimal
from typing import Any, Generic, Iterator, NamedTuple, Self, TypeVar, cast

import numpy as np
import pandas as pd
from ccy.core.daycounter import DayCounter
from pydantic import BaseModel, Field
from typing_extensions import Annotated, Doc

from quantflow.rates import (
    AnyYieldCurve,
    NoDiscount,
    Rate,
    YieldCurve,
    YieldCurveCalibration,
)
from quantflow.rates.options import OptionsDiscountingCalibration
from quantflow.utils import plot
from quantflow.utils.dates import utcnow
from quantflow.utils.numbers import (
    ONE,
    ZERO,
    DecimalNumber,
    Rounding,
    normalize_decimal,
    round_to_step,
    sigfig,
    to_decimal,
    to_decimal_or_none,
)
from quantflow.utils.price import PriceVolume
from quantflow.utils.types import FloatArray

from .bs import black_price, implied_black_volatility
from .inputs import (
    DefaultVolSecurity,
    ForwardInput,
    OptionInput,
    OptionMetadata,
    OptionType,
    Side,
    SpotInput,
    VolSecurityType,
    VolSurfaceInput,
    VolSurfaceInputs,
    VolSurfaceSecurity,
)
from .moneyness import moneyness_from_log_strike
from .parity import PutCallParities, PutCallParity
from .svi import SVI

INITIAL_VOL = 0.5
default_day_counter = DayCounter.ACTACT


S = TypeVar("S", bound=VolSurfaceSecurity)


class OptionSelection(enum.Enum):
    """Option selection method

    This enum is used to select which one between calls and puts are used
    for calculating implied volatility and other operations
    """

    best = enum.auto()
    """Select the OTM option but blend call and put implied volatilities
    near the money. The blending weight transitions linearly from 50/50
    at moneyness 0 to pure OTM at the moneyness threshold."""
    otm = enum.auto()
    """Select Out of the Money options only, where their
    intrinsic value is zero"""
    call = enum.auto()
    """Select the call options only"""
    put = enum.auto()
    """Select the put options only"""
    all = enum.auto()
    """Select all options regardless of their moneyness"""


class SecurityPrice(PriceVolume, Generic[S]):
    """Represents the bid/ask price of a security,
    which can be a spot price, forward price or option price
    """

    security: S = Field(description="The underlying security of the price")

    def is_valid(self) -> bool:
        """Check if the forward price is valid, which means that the bid and ask
        are positive and the bid is less than or equal to the ask"""
        return self.bid > ZERO and self.ask > ZERO and super().is_valid()


class SpotPrice(SecurityPrice[S]):
    """Represents the spot bid/ask price of an underlying asset"""

    def inputs(self) -> SpotInput:
        return SpotInput(
            bid=self.bid,
            ask=self.ask,
            open_interest=self.open_interest,
            volume=self.volume,
        )

    def _implied_forward(self, maturity: datetime, price: Decimal) -> FwdPrice[S]:
        return FwdPrice(
            security=self.security.forward(),
            maturity=maturity,
            bid=price,
            ask=price,
        )


class FwdPrice(SecurityPrice[S]):
    """Represents the forward bid/ask price of an underlying asset
    at a specific maturity"""

    maturity: datetime = Field(description="Maturity date of the forward price")

    def inputs(self) -> ForwardInput:
        return ForwardInput(
            bid=self.bid,
            ask=self.ask,
            maturity=self.maturity,
            open_interest=self.open_interest,
            volume=self.volume,
        )


class OptionPrice(BaseModel):
    """Represents the price of an option quoted in the market along with
    its metadata and implied volatility information."""

    price: DecimalNumber = Field(
        description="Price of the option as a percentage of the forward price"
    )
    meta: OptionMetadata = Field(description="Metadata of the option price")
    forward: DecimalNumber = Field(
        default=ZERO, description="Forward price of the underlying"
    )
    ttm: float = Field(default=0, description="Time to maturity in years")
    iv: float = Field(default=0, description="Implied volatility of the option")
    side: Side = Field(
        default=Side.bid, description="Side of the market for the option price"
    )
    converged: bool = Field(
        default=False,
        description="Flag indicating if implied vol calculation converged",
    )

    @property
    def strike(self) -> Decimal:
        return self.meta.strike

    @property
    def maturity(self) -> datetime:
        return self.meta.maturity

    @property
    def option_type(self) -> OptionType:
        return self.meta.option_type

    @property
    def log_strike(self) -> float:
        """Log strike of the option, calculated as log(strike/forward)"""
        return float(np.log(float(self.strike / self.forward)))

    @property
    def price_in_forward_space(self) -> Decimal:
        """Price of the option as a percentage of the forward price"""
        if self.meta.inverse:
            return self.price
        else:
            return self.price / self.forward

    @property
    def price_bp(self) -> Decimal:
        """Price of the option in basis points, calculated as price in forward space
        multiplied by 10000"""
        return self.price_in_forward_space * 10000

    @property
    def price_in_quote(self) -> Decimal:
        """Price of the option in quote currency"""
        if self.meta.inverse:
            return self.forward * self.price
        else:
            return self.price

    @property
    def price_intrinsic(self) -> Decimal:
        """Intrinsic price of the option in forward space, which is the price
        if the option had zero time value"""
        if self.option_type.is_call():
            return max(self.forward - self.strike, ZERO) / self.forward
        else:
            return max(self.strike - self.forward, ZERO) / self.forward

    @property
    def price_time(self) -> Decimal:
        """Time value of the option in forward space, which is the price minus
        its intrinsic value"""
        return self.price_in_forward_space - self.price_intrinsic

    @property
    def call_price(self) -> Decimal:
        """call price in forward space

        use put-call parity to calculate the call price if a put
        """
        if self.option_type.is_call():
            return self.price_in_forward_space
        else:
            return self.price_in_forward_space + 1 - self.strike / self.forward

    @property
    def put_price(self) -> Decimal:
        """put price in forward space

        use put-call parity to calculate the put price if a call
        """
        if self.option_type.is_call():
            return self.price_in_forward_space - 1 + self.strike / self.forward
        else:
            return self.price_in_forward_space

    def is_in_the_money(self, forward: Decimal) -> bool:
        return self.meta.is_in_the_money(forward)

    def calculate_price(self) -> Self:
        price = Decimal(
            sigfig(
                black_price(
                    np.asarray(self.log_strike),
                    self.iv,
                    self.ttm,
                    1 if self.option_type.is_call() else -1,
                ).sum(),
                8,
            )
        )
        self.price = price if self.meta.inverse else price * self.forward
        return self

    def info_dict(
        self,
        open_interest: Decimal = ZERO,
        volume: Decimal = ZERO,
    ) -> dict[str, Any]:
        return dict(
            strike=float(self.strike),
            forward=float(self.forward),
            maturity=self.maturity,
            log_strike=self.log_strike,
            moneyness=moneyness_from_log_strike(self.log_strike, self.ttm),
            ttm=self.ttm,
            iv=self.iv,
            price=float(self.price_in_forward_space),
            price_bp=float(self.price_bp),
            price_quote=float(self.price_in_quote),
            type=str(self.option_type),
            side=str(self.side),
            open_interest=float(open_interest),
            volume=float(volume),
        )

    def info(
        self,
        open_interest: Decimal = ZERO,
        volume: Decimal = ZERO,
    ) -> OptionInfo:
        """Return a structured [OptionInfo][quantflow.options.surface.OptionInfo]
        representation of this option price"""
        return OptionInfo(
            strike=self.strike,
            forward=self.forward,
            maturity=self.maturity,
            log_strike=to_decimal(self.log_strike),
            moneyness=to_decimal(
                float(moneyness_from_log_strike(self.log_strike, self.ttm))
            ),
            ttm=to_decimal(self.ttm),
            iv=to_decimal(self.iv),
            price=self.price_in_forward_space,
            price_bp=self.price_bp,
            price_quote=self.price_in_quote,
            option_type=self.option_type,
            side=self.side,
            open_interest=open_interest,
            volume=volume,
        )


class OptionInfo(BaseModel):
    """Structured representation of an option price with all computed fields"""

    strike: DecimalNumber = Field(description="Strike price of the option")
    forward: DecimalNumber = Field(
        description="Forward price of the underlying at maturity"
    )
    maturity: datetime = Field(description="Maturity date of the option")
    log_strike: DecimalNumber = Field(
        description="Log strike, calculated as log(strike/forward)"
    )
    moneyness: DecimalNumber = Field(
        description="Standardised moneyness, log(K/F) / sqrt(T)"
    )
    ttm: DecimalNumber = Field(description="Time to maturity in years")
    iv: DecimalNumber = Field(description="Black implied volatility")
    price: DecimalNumber = Field(
        description="Option price as a fraction of the forward price"
    )
    price_bp: DecimalNumber = Field(description="Option price in basis points")
    price_quote: DecimalNumber = Field(description="Option price in quote currency")
    option_type: OptionType = Field(description="Option type (call or put)")
    side: Side = Field(description="Market side (bid or ask)")
    open_interest: DecimalNumber = Field(description="Open interest")
    volume: DecimalNumber = Field(description="Volume traded")


class OptionArrays(NamedTuple):
    """Represents the option data in array form for efficient calculations
    via vectorized operations"""

    options: list[OptionPrice]
    """List of option prices corresponding to the arrays below"""
    log_strike: FloatArray
    """The log strike of the options, calculated as log(strike/forward)"""
    price: FloatArray
    """The option prices"""
    ttm: FloatArray
    """Time to maturity of the options"""
    iv: FloatArray
    """Implied volatility of the options"""
    call_put: FloatArray
    """Indicator for call (1) or put (-1) options"""


class OptionPrices(BaseModel, Generic[S]):
    """Represents the market for a single option contract (identified by its
    strike, maturity and option type), holding the bid and ask sides as separate
    [OptionPrice][quantflow.options.surface.OptionPrice] objects.
    """

    security: S = Field(description="The underlying security of the price")
    meta: OptionMetadata = Field(description="Metadata for the option prices")
    bid: OptionPrice = Field(description="Bid option price")
    ask: OptionPrice = Field(description="Ask option price")
    open_interest: DecimalNumber = Field(
        default=ZERO, description="Open interest of the spot price"
    )
    volume: DecimalNumber = Field(default=ZERO, description="Total volume traded")

    @property
    def converged(self) -> bool:
        """Check if the implied volatility calculation has converged
        for both bid and ask"""
        return self.bid.converged and self.ask.converged

    @property
    def mid(self) -> Decimal:
        """Calculate the mid option price by averaging the bid and ask prices"""
        return (self.bid.price + self.ask.price) / 2

    @property
    def spread(self) -> Decimal:
        """Calculate the bid-ask spread"""
        return self.ask.price - self.bid.price

    def price(self) -> PriceVolume:
        """Convert the option prices to a PriceVolume object"""
        return PriceVolume(
            bid=self.bid.price,
            ask=self.ask.price,
            volume=self.volume,
            open_interest=self.open_interest,
        )

    def iv_bid_ask_spread(self) -> float:
        """Calculate the bid-ask spread of the implied volatility"""
        return self.ask.iv - self.bid.iv

    def iv_mid(self) -> float:
        """Calculate the mid implied volatility"""
        return (self.bid.iv + self.ask.iv) / 2

    def is_in_the_money(self, forward: Decimal) -> bool:
        """Check if the option is in the money given the forward price"""
        return self.meta.is_in_the_money(forward)

    def disable(self) -> None:
        """Disable the option by setting its implied volatility convergence to False"""
        self.bid.converged = False
        self.ask.converged = False

    def prices(
        self,
        forward: Annotated[Decimal, Doc("Forward price of the underlying asset")],
        ttm: Annotated[float, Doc("Time to maturity in years")],
        *,
        initial_vol: Annotated[
            float, Doc("Initial volatility for the root finding algorithm")
        ] = INITIAL_VOL,
    ) -> Iterator[OptionPrice]:
        """Iterator over bid/ask option prices"""
        for o in (self.bid, self.ask):
            o.forward = forward
            o.ttm = ttm
            if not o.iv:
                o.iv = initial_vol
            yield o

    def inputs(self) -> OptionInput:
        """Convert the option prices to an OptionInput instance"""
        return OptionInput(
            bid=self.bid.price,
            ask=self.ask.price,
            open_interest=self.open_interest,
            volume=self.volume,
            strike=self.meta.strike,
            maturity=self.meta.maturity,
            option_type=self.meta.option_type,
            iv_bid=to_decimal_or_none(
                None if np.isnan(self.bid.iv) else round(self.bid.iv, 7)
            ),
            iv_ask=to_decimal_or_none(
                None if np.isnan(self.ask.iv) else round(self.ask.iv, 7)
            ),
        )


class Strike(BaseModel, Generic[S]):
    """Option prices for a single strike"""

    strike: DecimalNumber = Field(description="Strike price of the options")
    call: OptionPrices[S] | None = Field(
        default=None, description="Call option prices for the strike"
    )
    put: OptionPrices[S] | None = Field(
        default=None, description="Put option prices for the strike"
    )

    def put_call_parity(self) -> PutCallParity | None:
        """Return a [PutCallParity][quantflow.rates.calibrator.PutCallParity] for this
        strike, or None if either the call or the put are not available."""
        if self.call is None or self.put is None:
            return None
        return PutCallParity(
            strike=self.strike,
            call=self.call.price(),
            put=self.put.price(),
            inverse=self.call.meta.inverse,
        )

    def options_iter(
        self,
        forward: Annotated[Decimal, Doc("Forward price of the underlying asset")],
        ttm: Annotated[float, Doc("Time to maturity in years")],
        *,
        select: Annotated[
            OptionSelection, Doc("Option selection method")
        ] = OptionSelection.all,
    ) -> Iterator[OptionPrices[S]]:
        """Iterator over option prices for the strike

        It uses the `select` parameter to determine which options to include in
        the iteration. The forward price is used to determine the moneyness of
        the options when the `best` or `otm` selection method is used, in which
        case only the Out of the Money options are included in the iteration.
        """
        match select:
            case OptionSelection.otm:
                if self.call and not self.call.is_in_the_money(forward):
                    yield self.call
                elif self.put and not self.put.is_in_the_money(forward):
                    yield self.put
            case OptionSelection.best:
                if self.call and not self.call.is_in_the_money(forward):
                    yield self.call
                elif self.put and not self.put.is_in_the_money(forward):
                    yield self.put
            case OptionSelection.call:
                if self.call:
                    yield self.call
            case OptionSelection.put:
                if self.put:
                    yield self.put
            case OptionSelection.all:
                if self.call:
                    yield self.call
                if self.put:
                    yield self.put

    def securities(
        self,
        forward: Annotated[Decimal, Doc("Forward price of the underlying asset")],
        ttm: Annotated[float, Doc("Time to maturity in years")] = 0.0,
        *,
        select: Annotated[
            OptionSelection, Doc("Option selection method")
        ] = OptionSelection.all,
        converged: Annotated[
            bool,
            Doc(
                "Include options with implied volatility converged only if True, "
                "otherwise include all options regardless of convergence"
            ),
        ] = False,
    ) -> Iterator[OptionPrices[S]]:
        """Iterator over option prices for the strike"""
        for option in self.options_iter(forward, ttm, select=select):
            if not converged or option.converged:
                yield option

    def option_prices(
        self,
        forward: Annotated[Decimal, Doc("Forward price of the underlying asset")],
        ttm: Annotated[float, Doc("Time to maturity in years")],
        *,
        select: Annotated[
            OptionSelection, Doc("Option selection method")
        ] = OptionSelection.best,
        initial_vol: Annotated[
            float, Doc("Initial volatility for the root finding algorithm")
        ] = INITIAL_VOL,
        converged: Annotated[
            bool,
            Doc(
                "Include options with implied volatility converged only if True, "
                "otherwise include all options regardless of convergence"
            ),
        ] = False,
    ) -> Iterator[OptionPrice]:
        for option in self.options_iter(forward, ttm, select=select):
            if not converged or option.converged:
                yield from option.prices(
                    forward,
                    ttm,
                    initial_vol=initial_vol,
                )


class VolCrossSection(BaseModel, Generic[S]):
    """Represents a cross section of a volatility surface at a specific maturity."""

    maturity: datetime = Field(description="Maturity date of the cross section")
    forward: FwdPrice[S] = Field(
        description=(
            "Forward price of the underlying asset at the time " "of the cross section"
        )
    )
    strikes: tuple[Strike[S], ...] = Field(
        description="Tuple of sorted strikes and their corresponding option prices"
    )
    day_counter: DayCounter = Field(
        default=default_day_counter,
        description=(
            "Day counter for time to maturity calculations "
            "- by default it uses Act/Act"
        ),
    )

    def ttm(self, ref_date: datetime) -> float:
        """Time to maturity in years"""
        return self.day_counter.dcf(ref_date, self.maturity)

    def info_dict(
        self,
        ref_date: datetime,
        spot: Decimal,
        implied_forward: Decimal,
    ) -> dict:
        """Return a dictionary with information about the cross section"""
        ttm = self.ttm(ref_date)
        return dict(
            maturity=self.maturity,
            ttm=ttm,
            forward=self.forward.mid,
            implied_forward=implied_forward,
            forward_basis=implied_forward - self.forward.mid,
            rate=Rate.from_number(float((implied_forward / spot).ln()) / ttm).rate,
            bid_ask_spread=self.forward.spread,
            basis=implied_forward - spot,
            open_interest=self.forward.open_interest,
            volume=self.forward.volume,
        )

    def option_prices(
        self,
        ref_date: Annotated[
            datetime, Doc("Reference date for time to maturity calculation")
        ],
        forward: Annotated[Decimal, Doc("Forward price of the underlying asset")],
        *,
        select: Annotated[
            OptionSelection, Doc("Option selection method")
        ] = OptionSelection.best,
        initial_vol: Annotated[
            float, Doc("Initial volatility for the root finding algorithm")
        ] = INITIAL_VOL,
        converged: Annotated[
            bool, Doc("Whether the calculation has converged")
        ] = False,
    ) -> Iterator[OptionPrice]:
        """Iterator over option prices in the cross section"""
        for s in self.strikes:
            yield from s.option_prices(
                forward,
                self.ttm(ref_date),
                select=select,
                initial_vol=initial_vol,
                converged=converged,
            )

    def securities(
        self,
        *,
        select: Annotated[
            OptionSelection, Doc("Option selection method")
        ] = OptionSelection.all,
        converged: Annotated[
            bool,
            Doc(
                "Include the forward and options with implied volatility "
                "converged only if `True`, otherwise include all securities regardless "
                "of convergence"
            ),
        ] = False,
    ) -> Iterator[FwdPrice[S] | OptionPrices[S]]:
        """Iterator over all securities in the cross section"""
        yield self.forward
        yield from self.option_securities(select=select, converged=converged)

    def option_securities(
        self,
        *,
        select: Annotated[
            OptionSelection, Doc("Option selection method")
        ] = OptionSelection.all,
        converged: Annotated[
            bool,
            Doc(
                "Include the forward and options with implied volatility "
                "converged only if `True`, otherwise include all securities regardless "
                "of convergence"
            ),
        ] = False,
    ) -> Iterator[OptionPrices[S]]:
        """Iterator over all option securities in the cross section"""
        for strike in self.strikes:
            yield from strike.securities(
                self.forward.mid,
                select=select,
                converged=converged,
            )

    def disable_outliers(
        self,
        *,
        ttm: Annotated[float, Doc("Time to maturity in years, used for SVI fitting")],
        bid_ask_spread_fraction: Annotated[
            float,
            Doc(
                "Maximum allowed bid/ask spread as a fraction of the mid implied "
                "volatility. A value of 0.2 means options with a spread greater than "
                "20% of the mid vol are disabled."
            ),
        ] = 0.2,
        svi_residual_fraction: Annotated[
            float,
            Doc(
                "Maximum allowed SVI residual as a fraction of the mid implied "
                "volatility. A value of 0.2 means options whose mid vol deviates "
                "from the SVI fit by more than 20% of their mid vol are disabled."
            ),
        ] = 0.2,
        repeat: Annotated[
            int, Doc("Number of times to repeat the outlier removal process")
        ] = 2,
    ) -> None:
        """Disable outlier options in the cross section by marking them as not
        converged.

        Two passes are applied:

        First pass: options where the bid/ask spread in implied vol space exceeds
        `bid_ask_spread_fraction` of the mid implied vol are disabled.
        For example, a value of 0.2 disables options where the spread is more
        than 20% of the mid vol. Options with a zero mid vol are also disabled.

        Second pass: an [SVI][quantflow.options.svi.SVI] smile is fitted to the
        surviving options (mid implied vol vs log-strike). Options whose
        residual from the SVI fit exceeds `svi_residual_fraction` of their mid
        implied vol are disabled. This is repeated up to `repeat` times,
        refitting after each removal. The loop stops early if no outliers are
        found or fewer than 5 options remain.
        """
        options = list(self.option_securities(converged=True))
        # first remove options with high bid/offer spread
        for option in options:
            spread = option.iv_bid_ask_spread()
            mid = option.iv_mid()
            if mid > 0:
                if spread / mid > bid_ask_spread_fraction:
                    option.disable()
            else:
                option.disable()
        # remove outliers based on residuals from an SVI smile fit
        forward = float(self.forward.mid)
        for _ in range(repeat):
            options = list(self.option_securities(converged=True))
            if len(options) < 5:
                break
            log_m = np.array([np.log(float(o.meta.strike) / forward) for o in options])
            iv_mid = np.array([o.iv_mid() for o in options])
            try:
                svi = SVI.fit(log_m, iv_mid, ttm)
            except Exception:
                break
            iv_fit = svi.iv(log_m, ttm)
            residuals = np.abs(iv_mid - iv_fit) / iv_mid
            found = False
            for option, residual in zip(options, residuals):
                if residual > svi_residual_fraction:
                    option.disable()
                    found = True
            if not found:
                break


class ForwardPricer(BaseModel, Generic[S]):
    """Base class for forward/discount factor pricers"""

    asset: str = Field(
        default="",
        description="Name of the underlying asset",
    )
    spot: SpotPrice[S] | None = Field(
        default=None,
        description="Spot price of the underlying asset",
    )
    quote_curve: AnyYieldCurve = Field(
        default_factory=NoDiscount,
        description="Discount curve for the quote",
    )
    asset_curve: AnyYieldCurve = Field(
        default_factory=NoDiscount,
        description="Discount curve for the asset",
    )
    tick_size_forwards: DecimalNumber | None = Field(
        default=None,
        description="Tick size for rounding forward and spot prices - optional",
    )
    tick_size_options: DecimalNumber | None = Field(
        default=None, description="Tick size for rounding option prices - optional"
    )
    day_counter: DayCounter = Field(
        default=default_day_counter,
        description=(
            "Day counter for time to maturity calculations, "
            "by default it uses Act/Act"
        ),
    )

    @property
    def ref_date(self) -> datetime:
        """Reference date for the volatility surface, taken as the earliest maturity
        or the provided ref_date if it's earlier"""
        return min(self.quote_curve.ref_date, self.asset_curve.ref_date)

    def spot_price(self) -> Decimal:
        """Get the spot price if it exists"""
        if self.spot is None:
            raise ValueError("No spot price provided")
        return self.spot.mid

    def forward(self, maturity: datetime) -> Decimal:
        """Calculate the implied forward for a given maturity"""
        ttm = self.day_counter.dcf(self.ref_date, maturity)
        df_quote = to_decimal(float(self.quote_curve.discount_factor(ttm)))
        df_asset = to_decimal(float(self.asset_curve.discount_factor(ttm)))
        forward_rate = self.spot_price() * df_asset / df_quote
        return self.clip_forward(forward_rate)

    def clip_forward(
        self,
        forward: Decimal,
        rounding: Rounding = Rounding.ZERO,
    ) -> Decimal:
        """Clip the forward price to the nearest tick size if tick_size_forwards
        is set"""
        if self.tick_size_forwards:
            return round_to_step(forward, self.tick_size_forwards, rounding)
        return forward


class VolSurface(ForwardPricer[S]):
    """Represents a volatility surface, which captures the implied volatility of an
    option for different strikes and maturities.

    Key Concepts:

    * **Implied Volatility:** The market's expectation of future volatility, derived
        from the price of an option using a pricing model (e.g., Black-Scholes).
    * **Strike Price:** The price at which the underlying asset can be bought
        (call option) or sold (put option) at the option's expiry.
    * **Time to Maturity:** The time remaining until the option's expiration date.
    * **Volatility Smile/Skew:** The often-observed phenomenon where implied
        volatility varies across different strike prices for the same maturity.
        Typically, it forms a "smile" or "skew" shape.

    This class provides a structure for storing and manipulating volatility
    surface data. It can be used for various tasks, such as:

    * **Option pricing and risk management:** Using the surface to determine the
        appropriate volatility input for pricing models.
    * **Volatility arbitrage:** Identifying mispricings in options by comparing
        market prices to model prices derived from the surface.
    * **Market analysis:** Understanding market sentiment and expectations of
        future volatility.
    """

    maturities: tuple[VolCrossSection[S], ...] = Field(
        default=(),
        description=(
            "Sorted tuple of "
            "[VolCrossSection][quantflow.options.surface.VolCrossSection], "
            "each containing the forward price and option prices for that maturity"
        ),
    )

    def securities(
        self,
        *,
        select: Annotated[
            OptionSelection, Doc("Option selection method")
        ] = OptionSelection.all,
        index: Annotated[
            int | None, Doc("Index of the cross section to use, if None use all")
        ] = None,
        converged: Annotated[
            bool,
            Doc(
                "Include the spot, forwards and options with implied volatility "
                "converged only if True, otherwise include all securities regardless "
                "of convergence"
            ),
        ] = False,
    ) -> Iterator[SpotPrice[S] | FwdPrice[S] | OptionPrices[S]]:
        """Iterator over securities in the volatility surface"""
        if self.spot is not None:
            yield self.spot
        if index is not None:
            yield from self.maturities[index].securities(
                select=select, converged=converged
            )
        else:
            for maturity in self.maturities:
                yield from maturity.securities(select=select, converged=converged)

    def inputs(
        self,
        *,
        select: Annotated[
            OptionSelection, Doc("Option selection method")
        ] = OptionSelection.all,
        index: Annotated[
            int | None, Doc("Index of the cross section to use, if None use all")
        ] = None,
        converged: Annotated[
            bool,
            Doc(
                "Include spot, forwards and options with implied volatility "
                "converged only if True, otherwise include all securities regardless "
                "of convergence"
            ),
        ] = False,
    ) -> VolSurfaceInputs:
        """Convert the volatility surface to a
        [VolSurfaceInputs][quantflow.options.inputs.VolSurfaceInputs] instance"""
        return VolSurfaceInputs(
            asset=self.asset,
            asset_curve=self.asset_curve,
            quote_curve=self.quote_curve,
            inputs=list(
                s.inputs()
                for s in self.securities(
                    select=select, converged=converged, index=index
                )
            ),
        )

    def term_structure(self) -> pd.DataFrame:
        """Return the term structure of the volatility surface as a DataFrame"""
        spot = self.spot_price()
        return pd.DataFrame(
            cross.info_dict(self.ref_date, spot, self.forward(cross.maturity))
            for cross in self.maturities
        )

    def trim(self, num_maturities: int) -> Self:
        """Create a new volatility surface with the last `num_maturities` maturities"""
        return self.model_copy(
            update=dict(maturities=self.maturities[-num_maturities:])
        )

    def option_prices(
        self,
        *,
        select: Annotated[
            OptionSelection, Doc("Option selection method")
        ] = OptionSelection.best,
        index: Annotated[
            int | None, Doc("Index of the cross section to use, if None use all")
        ] = None,
        initial_vol: Annotated[
            float, Doc("Initial volatility for the root finding algorithm")
        ] = INITIAL_VOL,
        converged: Annotated[
            bool,
            Doc(
                "Include options with implied volatility "
                "converged only if True, otherwise include all options regardless "
                "of convergence"
            ),
        ] = False,
    ) -> Iterator[OptionPrice]:
        """Iterator over selected option prices in the surface"""
        if index is not None:
            cross = self.maturities[index]
            yield from cross.option_prices(
                self.ref_date,
                self.forward(cross.maturity),
                select=select,
                initial_vol=initial_vol,
                converged=converged,
            )
        else:
            for cross in self.maturities:
                yield from cross.option_prices(
                    self.ref_date,
                    self.forward(cross.maturity),
                    select=select,
                    initial_vol=initial_vol,
                    converged=converged,
                )

    def option_list(
        self,
        *,
        select: Annotated[
            OptionSelection, Doc("Option selection method")
        ] = OptionSelection.best,
        index: Annotated[
            int | None, Doc("Index of the cross section to use, if None use all")
        ] = None,
        converged: Annotated[
            bool,
            Doc(
                "Include options with implied volatility "
                "converged only if True, otherwise include all options regardless "
                "of convergence"
            ),
        ] = False,
    ) -> list[OptionPrice]:
        "List of selected option prices in the surface"
        return list(self.option_prices(select=select, index=index, converged=converged))

    def bs(
        self,
        *,
        select: Annotated[
            OptionSelection, Doc("Option selection method")
        ] = OptionSelection.best,
        index: Annotated[
            int | None, Doc("Index of the cross section to use, if None use all")
        ] = None,
        initial_vol: Annotated[
            float, Doc("Initial volatility for the root finding algorithm")
        ] = INITIAL_VOL,
    ) -> list[OptionPrice]:
        """Calculate Black-Scholes implied volatility for options
        in the surface.
        For some option prices, the implied volatility calculation may not converge,
        in this case the implied volatility is not
        calculated correctly and the option is marked as not converged.
        """
        self.reset_convergence()
        d = self.as_array(
            select=select,
            index=index,
            initial_vol=initial_vol,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = implied_black_volatility(
                k=d.log_strike,
                price=d.price,
                ttm=d.ttm,
                initial_sigma=d.iv,
                call_put=d.call_put,
            )
        for option, iv, converged in zip(d.options, result.values, result.converged):
            option.iv = float(iv)
            option.converged = converged and not np.isnan(iv)
        return d.options

    def calc_bs_prices(
        self,
        *,
        select: Annotated[
            OptionSelection, Doc("Option selection method")
        ] = OptionSelection.best,
        index: Annotated[
            int | None, Doc("Index of the cross section to use, if None use all")
        ] = None,
    ) -> FloatArray:
        """calculate Black-Scholes prices for all options in the surface

        It uses options with a converged implied volatility calculation only,
        otherwise the price calculation won't be correct.
        """
        d = self.as_array(select=select, index=index, converged=True)
        return black_price(k=d.log_strike, sigma=d.iv, ttm=d.ttm, s=d.call_put)

    def options_df(
        self,
        *,
        select: Annotated[
            OptionSelection, Doc("Option selection method")
        ] = OptionSelection.best,
        index: Annotated[
            int | None, Doc("Index of the cross section to use, if None use all")
        ] = None,
        initial_vol: Annotated[
            float, Doc("Initial volatility for the root finding algorithm")
        ] = INITIAL_VOL,
        converged: Annotated[
            bool, Doc("Whether the calculation has converged")
        ] = False,
    ) -> pd.DataFrame:
        """Time frame of Black-Scholes call input data"""
        data = self.option_prices(
            select=select,
            index=index,
            initial_vol=initial_vol,
            converged=converged,
        )
        return pd.DataFrame([d.info_dict() for d in data])

    def as_array(
        self,
        *,
        select: Annotated[
            OptionSelection, Doc("Option selection method")
        ] = OptionSelection.best,
        index: Annotated[
            int | None, Doc("Index of the cross section to use, if None use all")
        ] = None,
        initial_vol: Annotated[
            float, Doc("Initial volatility for the root finding algorithm")
        ] = INITIAL_VOL,
        converged: Annotated[
            bool,
            Doc(
                "If True, include only options for which the calculation has converged"
            ),
        ] = False,
    ) -> OptionArrays:
        """Organize option prices in a numpy arrays for Black volatility
        and price calculation

        It returns an [OptionArrays][quantflow.options.surface.OptionArrays] instance,
        which contains the option prices and their corresponding log strikes,
        time to maturity and implied volatility in numpy arrays
        for efficient calculations.
        """
        options = list(
            self.option_prices(
                select=select,
                index=index,
                initial_vol=initial_vol,
                converged=converged,
            )
        )
        log_strike = []
        ttm = []
        price = []
        vol = []
        call_put = []
        for option in options:
            log_strike.append(float(option.log_strike))
            price.append(float(option.price_in_forward_space))
            ttm.append(float(option.ttm))
            vol.append(float(option.iv))
            call_put.append(1 if option.option_type.is_call() else -1)
        return OptionArrays(
            options=options,
            log_strike=np.array(log_strike),
            price=np.array(price),
            ttm=np.array(ttm),
            iv=np.array(vol),
            call_put=np.array(call_put),
        )

    def reset_convergence(self) -> None:
        """Reset the convergence flag for all options in the surface"""
        for option in self.option_prices(select=OptionSelection.all):
            option.converged = False

    def disable_outliers(
        self,
        *,
        bid_ask_spread_fraction: Annotated[
            float,
            Doc(
                "Maximum allowed bid/ask spread as a fraction of the mid implied "
                "volatility. A value of 0.2 means options with a spread greater than "
                "20% of the mid vol are disabled."
            ),
        ] = 0.2,
        svi_residual_fraction: Annotated[
            float,
            Doc(
                "Maximum allowed SVI residual as a fraction of the mid implied "
                "volatility. A value of 0.2 means options whose mid vol deviates "
                "from the SVI fit by more than 20% of their mid vol are disabled."
            ),
        ] = 0.2,
        repeat: Annotated[
            int, Doc("Number of times to repeat the outlier removal process")
        ] = 2,
    ) -> Self:
        """Disable outlier options across all maturities in the surface.

        Calls
        [VolCrossSection.disable_outliers]
        [quantflow.options.surface.VolCrossSection.disable_outliers]
        on each maturity with the same parameters.
        """
        for maturity in self.maturities:
            maturity.disable_outliers(
                ttm=maturity.ttm(self.ref_date),
                bid_ask_spread_fraction=bid_ask_spread_fraction,
                svi_residual_fraction=svi_residual_fraction,
                repeat=repeat,
            )
        return self

    def plot(
        self,
        *,
        index: Annotated[
            int | None, Doc("Index of the cross section to use, if None use all")
        ] = None,
        select: Annotated[
            OptionSelection, Doc("Option selection method")
        ] = OptionSelection.best,
        **kwargs: Any,
    ) -> Any:
        """Plot the volatility surface"""
        df = self.options_df(index=index, select=select, converged=True)
        return plot.plot_vol_surface(df, **kwargs)

    def plot3d(
        self,
        *,
        select: Annotated[
            OptionSelection, Doc("Option selection method")
        ] = OptionSelection.best,
        index: Annotated[
            int | None, Doc("Index of the cross section to use, if None use all")
        ] = None,
        dragmode: Annotated[
            str, Doc("Drag interaction mode for the 3D scene")
        ] = "turntable",
        **kwargs: Any,
    ) -> Any:
        """Plot the volatility surface"""
        df = self.options_df(select=select, index=index, converged=True)
        return plot.plot_vol_surface_3d(df, dragmode=dragmode, **kwargs)


class VolCrossSectionLoader(BaseModel, Generic[S]):
    maturity: datetime = Field(description="Maturity date of the cross section")
    forward: FwdPrice[S] | None = Field(
        default=None,
        description=(
            "Forward price of the underlying asset at the time of the cross section"
        ),
    )
    strikes: dict[Decimal, Strike[S]] = Field(
        default_factory=dict,
        description="Dictionary of strikes and their corresponding option prices",
    )
    day_counter: DayCounter = Field(
        default=default_day_counter,
        description=(
            "Day counter for time to maturity calculations "
            "- by default it uses Act/Act"
        ),
    )

    def add_option(
        self,
        security: Annotated[S, Doc("Security for the option")],
        strike: Annotated[Decimal, Doc("Strike price for the option")],
        option_type: Annotated[OptionType, Doc("Type of the option (call or put)")],
        bid: Annotated[Decimal, Doc("Bid price for the option")],
        ask: Annotated[Decimal, Doc("Ask price for the option")],
        open_interest: Annotated[Decimal, Doc("Open interest for the option")] = ZERO,
        volume: Annotated[Decimal, Doc("Volume for the option")] = ZERO,
        inverse: Annotated[bool, Doc("Whether the option is an inverse option")] = True,
    ) -> None:
        """Add an option to the cross section loader"""
        strike = normalize_decimal(strike)
        if strike not in self.strikes:
            self.strikes[strike] = Strike(strike=strike)
        meta = OptionMetadata(
            strike=strike,
            option_type=option_type,
            maturity=self.maturity,
            inverse=inverse,
        )
        option = OptionPrices(
            security=security,
            meta=meta,
            bid=OptionPrice(price=normalize_decimal(bid), meta=meta, side=Side.bid),
            ask=OptionPrice(price=normalize_decimal(ask), meta=meta, side=Side.ask),
            open_interest=normalize_decimal(open_interest),
            volume=normalize_decimal(volume),
        )
        if option_type.is_call():
            self.strikes[strike].call = option
        else:
            self.strikes[strike].put = option

    def _cross_section(self, forward: FwdPrice[S]) -> VolCrossSection[S] | None:
        strikes = []
        for strike in sorted(self.strikes):
            sk = self.strikes[strike]
            if sk.call is None and sk.put is None:
                continue
            strikes.append(sk)
        return (
            VolCrossSection(
                maturity=self.maturity,
                forward=forward,
                strikes=tuple(strikes),
                day_counter=self.day_counter,
            )
            if strikes
            else None
        )

    def put_call_parities(
        self,
        spot: Annotated[Decimal, Doc("Spot price of the underlying asset")],
        *,
        ref_date: Annotated[
            datetime | None, Doc("Reference date for time to maturity calculation")
        ] = None,
        max_pairs: Annotated[
            int, Doc("Maximum number of put-call pairs to consider")
        ] = 10,
    ) -> PutCallParities:
        """Return a list of the most liquid
        [PutCallParity][quantflow.options.parity.PutCallParities]
        from a cross-section loader.

        Liquidity is determined by the bid-ask spread of the put-call parity price.
        """
        ttm = self.day_counter.dcf(ref_date or utcnow(), self.maturity)
        parities = sorted(
            (
                p
                for sk in self.strikes.values()
                if (p := sk.put_call_parity()) is not None
            ),
            key=lambda p: p.spread,
        )[:max_pairs]
        return PutCallParities.from_parities(parities, spot, ttm)


class GenericVolSurfaceLoader(ForwardPricer[S], arbitrary_types_allowed=True):
    """Helper class to build a volatility surface from a list of securities

    Use this class to add spot, forward and option securities with their prices
    and then call the `surface` method to build a `VolSurface` instance
    from the provided data.
    """

    maturities: dict[datetime, VolCrossSectionLoader[S]] = Field(
        default_factory=dict,
        description=(
            "Dictionary of maturities and their corresponding cross section loaders"
        ),
    )
    exclude_open_interest: DecimalNumber | None = Field(
        default=None,
        description="Exclude options with open interest at or below this value",
    )
    exclude_volume: DecimalNumber | None = Field(
        default=None, description="Exclude options with volume at or below this value"
    )

    def get_or_create_maturity(
        self,
        maturity: Annotated[datetime, Doc("Maturity date for the options")],
    ) -> VolCrossSectionLoader[S]:
        """Get or create a
        [VolCrossSectionLoader][quantflow.options.surface.VolCrossSectionLoader]
        for a given maturity"""
        if maturity not in self.maturities:
            self.maturities[maturity] = VolCrossSectionLoader(
                maturity=maturity,
                day_counter=self.day_counter,
            )
        return self.maturities[maturity]

    def add_spot(
        self,
        security: Annotated[S, Doc("Security for the spot price")],
        bid: Annotated[Decimal, Doc("Bid price for the spot")],
        ask: Annotated[Decimal, Doc("Ask price for the spot")],
        open_interest: Annotated[Decimal, Doc("Open interest for the spot")] = ZERO,
        volume: Annotated[Decimal, Doc("Volume for the spot")] = ZERO,
    ) -> None:
        """Add a spot to the volatility surface loader"""
        if security.vol_surface_type() != VolSecurityType.spot:
            raise ValueError("Security is not a spot")
        self.spot = SpotPrice(
            security=security,
            bid=normalize_decimal(bid),
            ask=normalize_decimal(ask),
            open_interest=normalize_decimal(open_interest),
            volume=normalize_decimal(volume),
        )

    def add_forward(
        self,
        security: Annotated[S, Doc("Security for the forward price")],
        maturity: Annotated[datetime, Doc("Maturity date for the forward price")],
        bid: Annotated[Decimal, Doc("Bid price for the forward")],
        ask: Annotated[Decimal, Doc("Ask price for the forward")],
        open_interest: Annotated[Decimal, Doc("Open interest for the forward")] = ZERO,
        volume: Annotated[Decimal, Doc("Volume for the forward")] = ZERO,
    ) -> None:
        """Add a forward to the volatility surface loader"""
        if security.vol_surface_type() != VolSecurityType.forward:
            raise ValueError("Security is not a forward")
        self.get_or_create_maturity(maturity=maturity).forward = FwdPrice(
            security=security,
            bid=normalize_decimal(bid),
            ask=normalize_decimal(ask),
            maturity=maturity,
            open_interest=normalize_decimal(open_interest),
            volume=normalize_decimal(volume),
        )

    def add_option(
        self,
        security: Annotated[S, Doc("Security for the option")],
        strike: Annotated[Decimal, Doc("Strike price for the option")],
        maturity: Annotated[datetime, Doc("Maturity date for the option")],
        option_type: Annotated[OptionType, Doc("Type of the option (call or put)")],
        bid: Annotated[Decimal, Doc("Bid price for the option")],
        ask: Annotated[Decimal, Doc("Ask price for the option")],
        open_interest: Annotated[Decimal, Doc("Open interest for the option")] = ZERO,
        volume: Annotated[Decimal, Doc("Volume for the option")] = ZERO,
        inverse: Annotated[bool, Doc("Whether the option is an inverse option")] = True,
    ) -> None:
        """Add an option to the volatility surface loader"""
        if security.vol_surface_type() != VolSecurityType.option:
            raise ValueError("Security is not an option")
        if self.exclude_volume is not None and volume <= self.exclude_volume:
            return
        if (
            self.exclude_open_interest is not None
            and open_interest <= self.exclude_open_interest
        ):
            return
        self.get_or_create_maturity(maturity=maturity).add_option(
            security=security,
            strike=strike,
            option_type=option_type,
            bid=bid,
            ask=ask,
            open_interest=open_interest,
            volume=volume,
            inverse=inverse,
        )

    def surface(self) -> VolSurface[S]:
        """Build a volatility surface from the provided data"""
        maturities = []
        spot = self.spot
        if spot is None:
            raise ValueError("No spot price provided")
        for maturity in sorted(self.maturities):
            loader = self.maturities[maturity]
            forward = loader.forward
            if forward is None:
                implied_forward_price = self.forward(maturity)
                forward = spot._implied_forward(maturity, implied_forward_price)
            if section := loader._cross_section(forward):
                maturities.append(section)
        return VolSurface(
            asset=self.asset,
            spot=self.spot,
            maturities=tuple(maturities),
            day_counter=self.day_counter,
            quote_curve=self.quote_curve.model_copy(),
            asset_curve=self.asset_curve.model_copy(),
            tick_size_forwards=self.tick_size_forwards,
            tick_size_options=self.tick_size_options,
        )

    def calibrate_spot(
        self,
        *,
        max_ttm: Annotated[
            float,
            Doc(
                "Maximum time to maturity (in years) for maturities used to imply "
                "the spot price. Default is 1/52 (one week)."
            ),
        ] = 1.0
        / 52,
        max_pairs: Annotated[
            int, Doc("Maximum number of put-call pairs to use per maturity")
        ] = 10,
    ) -> Decimal | None:
        """Calibrate the spot price from short-dated put-call parity.

        For short-dated options where discount factors are approximately 1,
        put-call parity simplifies to C - P = S - K, so S = C - P + K.
        This method computes the median implied spot across all put-call pairs
        with time to maturity at or below max_ttm and updates the spot price.

        Returns the implied spot, or None if no maturities fall within max_ttm.
        """
        spot = self.spot
        if spot is None:
            raise ValueError("No spot price provided")
        ref_date = self.ref_date
        implied_spots: list[float] = []
        for maturity in sorted(self.maturities):
            ttm = self.day_counter.dcf(ref_date, maturity)
            if ttm <= 0:
                continue
            if ttm > max_ttm:
                break
            parities = self.maturities[maturity].put_call_parities(
                ONE, ref_date=ref_date, max_pairs=max_pairs
            )
            for p in parities.parities:
                implied_spots.append(float(p.mid + p.strike))
        if not implied_spots:
            return None
        implied_spot = self.clip_forward(to_decimal(float(np.median(implied_spots))))
        self.spot = SpotPrice(
            security=spot.security,
            bid=implied_spot,
            ask=implied_spot,
        )
        return implied_spot

    def calibrate_curves(
        self,
        *,
        quote_curve: Annotated[
            type[YieldCurve] | YieldCurve | None,
            Doc(
                "YieldCurve type or instance to fit the quote currency discount "
                "curve $D_q$ from option prices. "
                "When None the current quote_curve is unchanged."
            ),
        ] = None,
        asset_curve: Annotated[
            type[YieldCurve] | YieldCurve | None,
            Doc(
                "YieldCurve type or instance to fit the asset discount curve $D_a$ "
                "from option prices. "
                "When None the current asset_curve is unchanged."
            ),
        ] = None,
        max_pairs: Annotated[
            int, Doc("Maximum number of put-call pairs to use per maturity")
        ] = 10,
    ) -> None:
        """Calibrate the quote and/or asset discount curves from option prices.

        Three modes are supported:

        Both curves: pass a curve type or instance for both curves.
        A single OLS regression per maturity identifies $D_q$ and $D_a$ simultaneously.

        Asset only: pass a curve type or instance for `asset_curve`, leave
        `quote_curve` as None.
        The existing `quote_curve` is treated as known and $D_a$ is solved analytically.

        Quote only: pass a curve type or instance for `quote_curve`, leave
        `asset_curve` as None.
        The existing `asset_curve` is treated as known and $D_q$ is solved analytically.
        """
        ttm, cp, strikes = self.collect_put_call_parities(max_pairs=max_pairs)
        asset_curve_input = (
            self._curve_calibrator(asset_curve) if asset_curve else self.asset_curve
        )
        quote_curve_input = (
            self._curve_calibrator(quote_curve) if quote_curve else self.quote_curve
        )
        calibration = OptionsDiscountingCalibration(
            asset_curve=asset_curve_input,
            quote_curve=quote_curve_input,
            ttm=ttm,
            cp=cp,
            strikes=strikes,
        )
        calibrated_asset_curve, calibrated_quote_curve = calibration.calibrate()
        self.asset_curve = cast(AnyYieldCurve, calibrated_asset_curve)
        self.quote_curve = cast(AnyYieldCurve, calibrated_quote_curve)

    def _curve_calibrator(
        self,
        curve_type: type[YieldCurve] | YieldCurve,
    ) -> YieldCurveCalibration:
        curve = (
            curve_type(ref_date=self.ref_date)
            if isinstance(curve_type, type)
            else curve_type
        )
        calibrator = curve.calibrator()
        if calibrator is None:
            raise ValueError(f"{type(curve).__name__} does not support calibration")
        return calibrator

    def collect_put_call_parities(
        self,
        *,
        max_pairs: Annotated[
            int, Doc("Maximum number of put-call pairs to use per maturity")
        ] = 10,
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Collect per-maturity continuously compounded rates from put-call parity."""
        if not self.spot or self.spot.mid == ZERO:
            raise ValueError("No spot price provided")
        spot = self.spot.mid
        ttms: list[FloatArray] = []
        cp: list[FloatArray] = []
        strikes: list[FloatArray] = []
        ref_date = self.ref_date
        for maturity, section in sorted(self.maturities.items()):
            ttm = self.day_counter.dcf(ref_date, maturity)
            if ttm <= 0:
                continue
            parities = section.put_call_parities(
                spot,
                ref_date=ref_date,
                max_pairs=max_pairs,
            )
            regressand = parities.regressand()
            if not regressand.size:
                continue
            ttms.append(np.full(regressand.shape, ttm, dtype=float))
            cp.append(regressand)
            strikes.append(parities.regressor())
        if not cp:
            raise ValueError("No put-call parity pairs available")
        return (
            np.concatenate(ttms),
            np.concatenate(cp),
            np.concatenate(strikes),
        )


class VolSurfaceLoader(GenericVolSurfaceLoader[DefaultVolSecurity]):
    """Helper class to build a volatility surface from a list of securities

    Use this class to add spot, forward and option securities with their prices
    and then call the `surface` method to build a `VolSurface` instance
    from the provided data.
    """

    def add(
        self, input: Annotated[VolSurfaceInput, Doc("Volatility surface input data")]
    ) -> None:
        """Add a volatility security input to the loader"""
        if isinstance(input, SpotInput):
            self.add_spot(
                DefaultVolSecurity.spot(),
                bid=input.bid,
                ask=input.ask,
                open_interest=input.open_interest,
                volume=input.volume,
            )
        elif isinstance(input, ForwardInput):
            self.add_forward(
                DefaultVolSecurity.forward(),
                maturity=input.maturity,
                bid=input.bid,
                ask=input.ask,
                open_interest=input.open_interest,
                volume=input.volume,
            )
        elif isinstance(input, OptionInput):
            self.add_option(
                DefaultVolSecurity.option(),
                strike=input.strike,
                option_type=input.option_type,
                maturity=input.maturity,
                bid=input.bid,
                ask=input.ask,
                open_interest=input.open_interest,
                volume=input.volume,
                inverse=input.inverse,
            )
        else:
            raise ValueError(f"Unknown input type {type(input)}")


def surface_from_inputs(
    inputs: Annotated[VolSurfaceInputs, Doc("Volatility surface input data")],
) -> VolSurface[DefaultVolSecurity]:
    """Helper function to build a volatility surface from a
    [VolSurfaceInputs][quantflow.options.inputs.VolSurfaceInputs] instance
    """
    loader = VolSurfaceLoader(
        asset=inputs.asset,
        quote_curve=inputs.quote_curve,
        asset_curve=inputs.asset_curve,
    )
    for input in inputs.inputs:
        loader.add(input)
    return loader.surface()
