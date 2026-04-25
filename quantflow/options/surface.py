from __future__ import annotations

import enum
import math
import warnings
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Generic, Iterator, NamedTuple, Self, TypeVar

import numpy as np
import pandas as pd
from ccy.core.daycounter import DayCounter
from pydantic import BaseModel, Field
from typing_extensions import Annotated, Doc

from quantflow.rates.interest_rate import Rate
from quantflow.utils import plot
from quantflow.utils.dates import utcnow
from quantflow.utils.numbers import (
    ZERO,
    DecimalNumber,
    Number,
    normalize_decimal,
    sigfig,
    to_decimal,
    to_decimal_or_none,
)
from quantflow.utils.types import FloatArray

from .bs import black_price, implied_black_volatility
from .inputs import (
    DefaultVolSecurity,
    ForwardInput,
    OptionInput,
    OptionType,
    Side,
    SpotInput,
    VolSecurityType,
    VolSurfaceInput,
    VolSurfaceInputs,
    VolSurfaceSecurity,
)

INITIAL_VOL = 0.5
default_day_counter = DayCounter.ACTACT


S = TypeVar("S", bound=VolSurfaceSecurity)


class OptionSelection(enum.Enum):
    """Option selection method

    This enum is used to select which one between calls and puts are used
    for calculating implied volatility and other operations
    """

    best = enum.auto()
    """Select the best bid/ask options.

    These are the options which are Out of the Money, where their
    intrinsic value is zero"""
    call = enum.auto()
    """Select the call options only"""
    put = enum.auto()
    """Select the put options only"""
    all = enum.auto()
    """Select all options regardless of their moneyness"""


class Price(BaseModel, Generic[S]):
    """Represents the bid/ask price of a security,
    which can be a spot price, forward price or option price
    """

    security: S = Field(description="The underlying security of the price")
    bid: DecimalNumber = Field(description="Bid price")
    ask: DecimalNumber = Field(description="Ask price")

    @property
    def mid(self) -> Decimal:
        """Calculate the mid price by averaging the bid and ask prices"""
        return (self.bid + self.ask) / 2

    @property
    def spread(self) -> Decimal:
        """Calculate the bid-ask spread"""
        return self.ask - self.bid

    @property
    def bp_spread(self) -> Decimal:
        """Bid-ask spread in basis points, calculated as spread divided by mid
        price and multiplied by 10000"""
        mid = self.mid
        if mid > ZERO:
            return 10000 * self.spread / mid
        else:
            return Decimal("inf")


class SpotPrice(Price[S]):
    """Represents the spot bid/ask price of an underlying asset"""

    open_interest: DecimalNumber = Field(
        default=ZERO, description="Open interest of the spot price"
    )
    volume: DecimalNumber = Field(default=ZERO, description="Total volume traded")

    def inputs(self) -> SpotInput:
        return SpotInput(
            bid=self.bid,
            ask=self.ask,
            open_interest=self.open_interest,
            volume=self.volume,
        )


class FwdPrice(Price[S]):
    """Represents the forward bid/ask price of an underlying asset
    at a specific maturity"""

    maturity: datetime = Field(description="Maturity date of the forward price")
    open_interest: DecimalNumber = Field(
        default=ZERO, description="Open interest of the forward price"
    )
    volume: DecimalNumber = Field(default=ZERO, description="Total volume traded")

    def inputs(self) -> ForwardInput:
        return ForwardInput(
            bid=self.bid,
            ask=self.ask,
            maturity=self.maturity,
            open_interest=self.open_interest,
            volume=self.volume,
        )

    def is_valid(self) -> bool:
        """Check if the forward price is valid, which means that the bid and ask
        are positive and the bid is less than or equal to the ask"""
        return self.bid > ZERO and self.ask > ZERO and self.bid <= self.ask


class ImpliedFwdPrice(FwdPrice[S]):
    """Represents the implied forward price of an underlying asset at a specific
    maturity, extracted from option prices via put-call parity"""

    strike: DecimalNumber = Field(
        description="Strike price of the options used to extract the forward price"
    )

    def moneyness(self, ttm: float) -> float:
        """Moneyness of the implied forward"""
        return math.log(float(self.strike / self.mid)) / math.sqrt(ttm)

    @classmethod
    def aggregate(
        cls,
        forwards: list[Self],
        ttm: float,
        default: FwdPrice[S] | None = None,
        previous_forward: Decimal | None = None,
    ) -> FwdPrice[S] | None:
        r"""Aggregate implied forward prices extracted from put-call parity into a
        single best-estimate forward price.

        Each implied forward is an independent noisy estimate of the true forward,
        obtained at a different strike. Strikes near the money tend to produce the
        most reliable estimates (tightest bid-ask spreads, smallest put-call parity
        error), so the aggregation weights each estimate by three independent factors
        that all reward quality:

        \begin{equation}
            w_i = w^{\text{moneyness}}_i
                  \cdot w^{\text{spread}}_i
                  \cdot w^{\text{proximity}}_i
        \end{equation}

        **Moneyness weight** rewards strikes close to the current mid price.
        It uses the same Gaussian shape as the standard normal density, where
        $m_i = \log(K_i / F_i) / \sqrt{\tau}$ is the normalised log-moneyness:

        \begin{equation}
            w^{\text{moneyness}}_i = \exp\!\left(-\tfrac{m_i^2}{2}\right)
        \end{equation}

        **Spread weight** penalises wide bid-ask markets, which indicate either
        low liquidity or high uncertainty in the put-call parity estimate. It
        decays exponentially with the bid-ask spread relative to the adaptive
        cutoff $c$:

        \begin{equation}
            w^{\text{spread}}_i
                = \exp\!\left(-\frac{\text{bp\_spread}_i}{c}\right)
        \end{equation}

        **Proximity weight** anchors the result to the previous maturity's forward
        $F_{\text{prev}}$ when provided. It down-weights estimates whose mid is
        far from the known anchor, acting as a soft prior that limits how much the
        result can deviate from a recent reliable observation:

        \begin{equation}
            w^{\text{proximity}}_i
                = \exp\!\left(
                    -\frac{1}{2}
                    \left(\frac{F_i - F_{\text{prev}}}{F_{\text{prev}}}\right)^{\!2}
                  \right)
        \end{equation}

        **Adaptive spread filter**: the cutoff $c$ starts at 10 bp and doubles
        until at least half the valid forwards survive. This ensures that in
        illiquid markets with universally wide spreads, the algorithm still
        produces a result rather than discarding everything.

        **Default blending**: the actual market forward (e.g. from futures) is
        optionally blended into the weighted average using only the spread weight.
        This forward may be unreliable (wide spread, stale price), so it receives
        no moneyness or proximity weight.

        **Default fallback**: if the computed mid lies within one default
        spread-width of the default mid, the default is returned unchanged,
        avoiding unnecessary deviation from the observable market price when
        the implied estimate is consistent with it.
        """
        forwards = [f for f in forwards if f.is_valid()]
        if not forwards:
            return default
        weights = 0.0
        values = 0.0
        spreads = 0.0
        target_len = max(1, len(forwards) // 2)
        cleaned: list[Self] = []
        spread_bp_cutoff = 10
        while True:
            cleaned = [
                forward for forward in forwards if forward.bp_spread < spread_bp_cutoff
            ]
            if len(cleaned) < target_len:
                spread_bp_cutoff *= 2
            else:
                forwards = cleaned
                break
        for forward in forwards:
            m = forward.moneyness(ttm)
            moneyness_weight = math.exp(-(m**2) / 2)
            spread_weight = math.exp(-forward.bp_spread / spread_bp_cutoff)
            if previous_forward is not None:
                d = float((forward.mid - previous_forward) / previous_forward)
                proximity_weight = math.exp(-(d**2) / 2)
            else:
                proximity_weight = 1.0
            w = moneyness_weight * spread_weight * proximity_weight
            weights += w
            values += w * float(forward.mid)
            spreads += w * float(forward.spread)
        if (
            default is not None
            and default.is_valid()
            and default.bp_spread < spread_bp_cutoff
        ):
            w = math.exp(-10000 * float(default.spread) / float(default.mid))
            weights += w
            values += w * float(default.mid)
            spreads += w * float(default.spread)
        mid = to_decimal(values / weights)
        spread = to_decimal(spreads / weights)
        if (
            default is not None
            and default.is_valid()
            and abs(mid - default.mid) / default.spread < 1
        ):
            return default
        return FwdPrice(
            security=forwards[0].security.forward(),
            bid=mid - spread / 2,
            ask=mid + spread / 2,
            maturity=forwards[0].maturity,
        )


class OptionMetadata(BaseModel):
    """Represents the metadata of an option, including its strike, type, maturity,
    and other relevant information."""

    strike: DecimalNumber = Field(description="Strike price of the option")
    option_type: OptionType = Field(description="Type of the option, call or put")
    maturity: datetime = Field(description="Maturity date of the option")
    forward: DecimalNumber = Field(
        default=ZERO, description="Forward price of the underlying"
    )
    ttm: float = Field(default=0, description="Time to maturity in years")
    open_interest: DecimalNumber = Field(
        default=ZERO, description="Open interest of the option"
    )
    volume: DecimalNumber = Field(default=ZERO, description="Total volume traded")
    inverse: bool = Field(
        default=True,
        description=(
            "Whether the option is an inverse option (i.e. quoted in terms of the "
            "underlying) or not (i.e. quoted in terms of the quote currency)"
        ),
    )

    def is_in_the_money(self, forward: Decimal) -> bool:
        """Check if the option is in the money given the forward price"""
        if self.option_type.is_call():
            return self.strike < forward
        else:
            return self.strike > forward


class OptionPrice(BaseModel):
    """Represents the price of an option quoted in the market along with
    its metadata and implied volatility information."""

    price: DecimalNumber = Field(
        description="Price of the option as a percentage of the forward price"
    )
    meta: OptionMetadata = Field(description="Metadata of the option price")
    implied_vol: float = Field(
        default=0, description="Implied volatility of the option"
    )
    side: Side = Field(
        default=Side.bid, description="Side of the market for the option price"
    )
    converged: bool = Field(
        default=False,
        description="Flag indicating if implied vol calculation converged",
    )

    @classmethod
    def create(
        cls,
        strike: Number,
        *,
        price: Number = ZERO,
        implied_vol: float = INITIAL_VOL,
        forward: Number | None = None,
        ref_date: datetime | None = None,
        maturity: datetime | None = None,
        day_counter: DayCounter | None = None,
        option_type: OptionType = OptionType.call,
        open_interest: Number = ZERO,
        volume: Number = ZERO,
        inverse: bool = True,
    ) -> OptionPrice:
        """Create an option price

        mainly used for testing
        """
        ref_date = ref_date or utcnow()
        maturity = maturity or ref_date + timedelta(days=365)
        day_counter = day_counter or default_day_counter
        return cls(
            price=to_decimal(price),
            implied_vol=implied_vol,
            meta=OptionMetadata(
                strike=to_decimal(strike),
                forward=to_decimal(forward or strike),
                option_type=option_type,
                maturity=maturity,
                ttm=day_counter.dcf(ref_date, maturity),
                open_interest=to_decimal(open_interest),
                volume=to_decimal(volume),
                inverse=inverse,
            ),
        )

    @property
    def strike(self) -> Decimal:
        return self.meta.strike

    @property
    def forward(self) -> Decimal:
        """Forward price of the underlying asset at the time of the option price"""
        return self.meta.forward

    @property
    def maturity(self) -> datetime:
        return self.meta.maturity

    @property
    def ttm(self) -> float:
        return self.meta.ttm

    @property
    def option_type(self) -> OptionType:
        return self.meta.option_type

    @property
    def open_interest(self) -> Decimal:
        return self.meta.open_interest

    @property
    def volume(self) -> Decimal:
        return self.meta.volume

    @property
    def moneyness(self) -> float:
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

    def calculate_price(self) -> OptionPrice:
        price = Decimal(
            sigfig(
                black_price(
                    np.asarray(self.moneyness),
                    self.implied_vol,
                    self.ttm,
                    1 if self.option_type.is_call() else -1,
                ).sum(),
                8,
            )
        )
        self.price = price if self.meta.inverse else price * self.forward
        return self

    def info_dict(self) -> dict[str, Any]:
        return dict(
            strike=float(self.strike),
            forward=float(self.forward),
            maturity=self.maturity,
            moneyness=self.moneyness,
            moneyness_ttm=self.moneyness / np.sqrt(self.ttm),
            ttm=self.ttm,
            implied_vol=self.implied_vol,
            price=float(self.price_in_forward_space),
            price_bp=float(self.price_bp),
            price_quote=float(self.price_in_quote),
            type=str(self.option_type),
            side=str(self.side),
            open_interest=float(self.open_interest),
            volume=float(self.volume),
        )


class OptionArrays(NamedTuple):
    """Represents the option data in array form for efficient calculations
    via vectorized operations"""

    options: list[OptionPrice]
    """List of option prices corresponding to the arrays below"""
    moneyness: FloatArray
    """The log strike of the options, calculated as log(strike/forward)"""
    price: FloatArray
    """The option prices"""
    ttm: FloatArray
    """Time to maturity of the options"""
    implied_vol: FloatArray
    """Implied volatility of the options"""
    call_put: FloatArray
    """Indicator for call (1) or put (-1) options"""


class OptionPrices(BaseModel, Generic[S]):
    """Represents the market for a single option contract (identified by its
    strike, maturity and option type), holding the bid and ask sides as separate
    [OptionPrice][quantflow.options.surface.OptionPrice] objects.
    """

    security: S = Field(description="The underlying security of the option prices")
    meta: OptionMetadata = Field(description="Metadata for the option prices")
    bid: OptionPrice = Field(description="Bid option price")
    ask: OptionPrice = Field(description="Ask option price")

    @property
    def converged(self) -> bool:
        """Check if the implied volatility calculation has converged
        for both bid and ask"""
        return self.bid.converged and self.ask.converged

    @property
    def mid(self) -> Decimal:
        """Calculate the mid option price by averaging the bid and ask prices"""
        return (self.bid.price + self.ask.price) / 2

    def iv_bid_ask_spread(self) -> float:
        """Calculate the bid-ask spread of the implied volatility"""
        return self.ask.implied_vol - self.bid.implied_vol

    def iv_mid(self) -> float:
        """Calculate the mid implied volatility"""
        return (self.bid.implied_vol + self.ask.implied_vol) / 2

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
        self.meta.forward = forward
        self.meta.ttm = ttm
        for o in (self.bid, self.ask):
            o.meta.forward = forward
            o.meta.ttm = ttm
            if not o.implied_vol:
                o.implied_vol = initial_vol
            yield o

    def inputs(self) -> OptionInput:
        """Convert the option prices to an OptionInput instance"""
        return OptionInput(
            bid=self.bid.price,
            ask=self.ask.price,
            open_interest=self.meta.open_interest,
            volume=self.meta.volume,
            strike=self.meta.strike,
            maturity=self.meta.maturity,
            option_type=self.meta.option_type,
            iv_bid=to_decimal_or_none(
                None
                if np.isnan(self.bid.implied_vol)
                else round(self.bid.implied_vol, 7)
            ),
            iv_ask=to_decimal_or_none(
                None
                if np.isnan(self.ask.implied_vol)
                else round(self.ask.implied_vol, 7)
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

    def implied_forward(self) -> ImpliedFwdPrice[S] | None:
        r"""Extract the implied forward price from put-call parity.

        Requires both a call and a put at this strike. Uses mid prices.
        For inverse options (prices quoted in the underlying currency)
        put-call parity reads

        \begin{equation}
            F = \frac{K}{1 - c + p}
        \end{equation}

        For non-inverse options (prices quoted in the quote currency)

        \begin{equation}
            F = K + C - P
        \end{equation}

        Returns None when the strike does not have both a call and a put,
        or when the denominator is non-positive (arbitrage condition violated).
        """
        if self.call is None or self.put is None:
            return None
        cp_bid = self.call.bid.price - self.put.ask.price
        cp_ask = self.call.ask.price - self.put.bid.price
        if self.call.meta.inverse:
            d_bid = 1 - cp_bid
            d_ask = 1 - cp_ask
            if d_bid <= ZERO or d_ask <= ZERO:
                return None
            bid = self.strike / d_bid
            ask = self.strike / d_ask
        else:
            bid = self.strike + cp_bid
            ask = self.strike + cp_ask
            if bid <= ZERO or ask <= ZERO:
                return None
        if bid > ask:
            return None
        return ImpliedFwdPrice(
            security=self.call.security.forward(),
            strike=self.strike,
            maturity=self.call.meta.maturity,
            bid=bid,
            ask=ask,
        )

    def options_iter(
        self,
        forward: Annotated[Decimal, Doc("Forward price of the underlying asset")],
        *,
        select: Annotated[
            OptionSelection, Doc("Option selection method")
        ] = OptionSelection.all,
    ) -> Iterator[OptionPrices[S]]:
        """Iterator over option prices for the strike

        It uses the `select` parameter to determine which options to include in
        the iteration. The forward price is used to determine the moneyness of
        the options when the `best` selection method is used, in which
        case only the Out of the Money options are included in the iteration.
        """
        match select:
            case OptionSelection.best:
                if self.call:
                    if self.call.is_in_the_money(forward) and self.put:
                        yield self.put
                    else:
                        yield self.call
                elif self.put:
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
        for option in self.options_iter(forward, select=select):
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
        for option in self.options_iter(forward, select=select):
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

    def forward_rate(self, ref_date: datetime, spot: SpotPrice[S]) -> Rate:
        """Compute the implied continuous rate from spot and forward mid"""
        return Rate.from_spot_and_forward(
            spot.mid,
            self.forward.mid,
            ref_date,
            self.maturity,
            day_counter=self.day_counter,
        )

    def forward_spread_fraction(self) -> Decimal:
        """Bid-ask spread of the forward as a fraction of its mid price"""
        mid = self.forward.mid
        if mid <= ZERO:
            return Decimal("Inf")
        return (self.forward.ask - self.forward.bid) / mid

    def info_dict(self, ref_date: datetime, spot: SpotPrice[S]) -> dict:
        """Return a dictionary with information about the cross section"""
        return dict(
            maturity=self.maturity,
            ttm=self.ttm(ref_date),
            forward=self.forward.mid,
            basis=self.forward.mid - spot.mid,
            rate_percent=self.forward_rate(ref_date, spot).percent,
            fwd_spread_pct=round(100 * self.forward_spread_fraction(), 4),
            open_interest=self.forward.open_interest,
            volume=self.forward.volume,
        )

    def option_prices(
        self,
        ref_date: Annotated[
            datetime, Doc("Reference date for time to maturity calculation")
        ],
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
                self.forward.mid,
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
        bid_ask_spread: Annotated[
            float,
            Doc(
                "Maximum allowed bid/ask spread as a fraction of the mid "
                "implied volatility. A value of 0.3 means that options with a bid/ask"
                "spread greater than 30% of the mid implied volatility will be "
                "considered outliers and have their implied volatility convergence "
                "set to False",
            ),
        ] = 0.3,
        quantile: Annotated[float, Doc("Quantile for determining outliers")] = 0.99,
        repeat: Annotated[
            int, Doc("Number of times to repeat the outlier removal process")
        ] = 2,
    ) -> None:
        """Disable outlier options in the cross section by marking them as not
        converged.

        Two passes are applied:


        First pass: options where the bid/ask spread in implied vol space exceeds
        `bid_ask_spread` as a fraction of the mid implied vol are disabled.
        For example, a value of 0.3 disables options where the spread is more
        than 30% of the mid vol. Options with a zero mid vol are also disabled.

        Second pass: a degree-2 polynomial is fitted to the smile (mid implied vol
        vs log-moneyness). Options whose residual from the fit exceeds the
        `quantile` threshold of all residuals are disabled. This is repeated up
        to `repeat` times, refitting after each removal. The loop stops early if
        no outliers are found or fewer than 4 options remain.
        """
        options = list(self.option_securities(converged=True))
        # first remove options with high bid/offer spread
        for option in options:
            spread = option.iv_bid_ask_spread()
            mid = option.iv_mid()
            if mid > 0:
                if spread / mid > bid_ask_spread:
                    option.disable()
            else:
                option.disable()
        # remove outliers based on residuals from a quadratic smile fit
        forward = float(self.forward.mid)
        for _ in range(repeat):
            options = list(self.option_securities(converged=True))
            if len(options) < 4:
                break
            log_m = np.array([np.log(float(o.meta.strike) / forward) for o in options])
            iv_mid = np.array([o.iv_mid() for o in options])
            coeffs = np.polyfit(log_m, iv_mid, 2)
            residuals = np.abs(iv_mid - np.polyval(coeffs, log_m))
            threshold = np.quantile(residuals, quantile)
            found = False
            for option, residual in zip(options, residuals):
                if residual > threshold:
                    option.disable()
                    found = True
            if not found:
                break


class VolSurface(BaseModel, Generic[S]):
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

    ref_date: datetime = Field(description="Reference date for the volatility surface")
    asset: str = Field(description="Underlying asset of the volatility surface")
    spot: SpotPrice[S] = Field(description="Spot price of the underlying asset")
    maturities: tuple[VolCrossSection[S], ...] = Field(
        description=(
            "Sorted tuple of "
            "[VolCrossSection][quantflow.options.surface.VolCrossSection], "
            "each containing the forward price and option prices for that maturity"
        )
    )
    day_counter: DayCounter = Field(
        default=default_day_counter,
        description=(
            "Day counter for time to maturity calculations, "
            "by default it uses Act/Act"
        ),
    )
    tick_size_forwards: DecimalNumber | None = Field(
        default=None,
        description="Tick size for rounding forward and spot prices - optional",
    )
    tick_size_options: DecimalNumber | None = Field(
        default=None,
        description="Tick size for rounding option prices - optional",
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
            ref_date=self.ref_date,
            inputs=list(
                s.inputs()
                for s in self.securities(
                    select=select, converged=converged, index=index
                )
            ),
        )

    def term_structure(self) -> pd.DataFrame:
        """Return the term structure of the volatility surface as a DataFrame"""
        return pd.DataFrame(
            cross.info_dict(self.ref_date, self.spot) for cross in self.maturities
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
            yield from self.maturities[index].option_prices(
                self.ref_date,
                select=select,
                initial_vol=initial_vol,
                converged=converged,
            )
        else:
            for maturity in self.maturities:
                yield from maturity.option_prices(
                    self.ref_date,
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
                k=d.moneyness,
                price=d.price,
                ttm=d.ttm,
                initial_sigma=d.implied_vol,
                call_put=d.call_put,
            )
        for option, implied_vol, converged in zip(
            d.options, result.values, result.converged
        ):
            option.implied_vol = float(implied_vol)
            option.converged = converged and not np.isnan(implied_vol)
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
        return black_price(k=d.moneyness, sigma=d.implied_vol, ttm=d.ttm, s=d.call_put)

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
        which contains the option prices and their corresponding moneyness,
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
        moneyness = []
        ttm = []
        price = []
        vol = []
        call_put = []
        for option in options:
            moneyness.append(float(option.moneyness))
            price.append(float(option.price_in_forward_space))
            ttm.append(float(option.ttm))
            vol.append(float(option.implied_vol))
            call_put.append(1 if option.option_type.is_call() else -1)
        return OptionArrays(
            options=options,
            moneyness=np.array(moneyness),
            price=np.array(price),
            ttm=np.array(ttm),
            implied_vol=np.array(vol),
            call_put=np.array(call_put),
        )

    def reset_convergence(self) -> None:
        """Reset the convergence flag for all options in the surface"""
        for option in self.option_prices(select=OptionSelection.all):
            option.converged = False

    def disable_outliers(
        self,
        *,
        bid_ask_spread: Annotated[
            float,
            Doc(
                "Maximum allowed bid/ask spread as a fraction of the mid "
                "implied volatility. A value of 0.3 means that options with a bid/ask"
                "spread greater than 30% of the mid implied volatility will be "
                "considered outliers and have their implied volatility convergence "
                "set to False",
            ),
        ] = 0.3,
        quantile: Annotated[float, Doc("Quantile for determining outliers")] = 0.99,
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
                bid_ask_spread=bid_ask_spread, quantile=quantile, repeat=repeat
            )
        return self

    def calibrate_forwards(
        self,
        *,
        max_spread_fraction: Annotated[
            float,
            Doc(
                "Maximum allowed forward bid-ask spread as a fraction of the mid "
                "price. Forwards exceeding this threshold are considered unreliable "
                "and replaced with a synthetic price derived from interpolated rates. "
                "A value of 0.05 flags forwards whose spread is more than 5% of mid."
            ),
        ] = 0.05,
    ) -> Self:
        """Replace forwards with wide bid-ask spreads with synthetic prices
        interpolated from the smooth rate term structure.

        For each maturity the implied continuous rate is computed as
        `r = log(F_mid / S) / T`. Maturities whose forward bid-ask spread
        exceeds `max_spread_fraction` of the mid are treated as unreliable.
        A piecewise-linear interpolation (with flat extrapolation at the
        boundaries) is fitted through the reliable `(T, r)` pairs, and the
        synthetic forward is:

        `F_synth = S * exp(r_interp * T)`

        The synthetic bid and ask are both set to this value, giving a
        zero spread, and the cross-section forward is replaced accordingly.
        Returns a new `VolSurface` instance leaving the original unchanged.
        """
        spot = self.spot.mid
        max_spread = to_decimal(max_spread_fraction)
        good_ttms: list[float] = []
        good_rates: list[float] = []
        bad_indices: list[int] = []

        for i, cross in enumerate(self.maturities):
            ttm = cross.ttm(self.ref_date)
            spread_frac = cross.forward_spread_fraction()
            rate = cross.forward_rate(self.ref_date, self.spot)
            if ttm > 0 and spread_frac <= max_spread:
                good_ttms.append(ttm)
                good_rates.append(float(rate.rate))
            else:
                bad_indices.append(i)

        if not good_ttms or not bad_indices:
            return self

        ttm_arr = np.array(good_ttms)
        rate_arr = np.array(good_rates)

        new_maturities = list(self.maturities)
        for i in bad_indices:
            cross = self.maturities[i]
            ttm = cross.ttm(self.ref_date)
            if ttm <= 0:
                continue
            r_synth = float(np.interp(ttm, ttm_arr, rate_arr))
            f_synth = to_decimal(float(spot) * math.exp(r_synth * ttm))
            new_fwd = cross.forward.model_copy(update=dict(bid=f_synth, ask=f_synth))
            new_maturities[i] = cross.model_copy(update=dict(forward=new_fwd))

        return self.model_copy(update=dict(maturities=tuple(new_maturities)))

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
        bid: Annotated[Decimal, Doc("Bid price for the option")] = ZERO,
        ask: Annotated[Decimal, Doc("Ask price for the option")] = ZERO,
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
            open_interest=normalize_decimal(open_interest),
            volume=normalize_decimal(volume),
            inverse=inverse,
        )
        option = OptionPrices(
            security=security,
            meta=meta,
            bid=OptionPrice(price=normalize_decimal(bid), meta=meta, side=Side.bid),
            ask=OptionPrice(price=normalize_decimal(ask), meta=meta, side=Side.ask),
        )
        if option_type.is_call():
            self.strikes[strike].call = option
        else:
            self.strikes[strike].put = option

    def cross_section(
        self,
        ref_date: Annotated[
            datetime | None, Doc("Reference date for the volatility surface")
        ] = None,
        previous_forward: Annotated[
            Decimal | None,
            Doc(
                "Previous forward price for the volatility surface "
                "Usaed by the implied forward calculation to replace missing "
                "or unreliable forwards"
            ),
        ] = None,
    ) -> VolCrossSection[S] | None:
        strikes = []
        implied_forwards = []
        for strike in sorted(self.strikes):
            sk = self.strikes[strike]
            if sk.call is None and sk.put is None:
                continue
            if implied_forward := sk.implied_forward():
                implied_forwards.append(implied_forward)
            strikes.append(sk)
        forward = self.forward
        if implied_forwards:
            ttm = self.day_counter.dcf(ref_date or utcnow(), self.maturity)
            forward = ImpliedFwdPrice.aggregate(
                implied_forwards,
                ttm,
                default=self.forward,
                previous_forward=previous_forward,
            )
        if forward is None or not forward.is_valid():
            return None
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


class GenericVolSurfaceLoader(BaseModel, Generic[S], arbitrary_types_allowed=True):
    """Helper class to build a volatility surface from a list of securities

    Use this class to add spot, forward and option securities with their prices
    and then call the `surface` method to build a `VolSurface` instance
    from the provided data.
    """

    asset: str = Field(default="", description="Name of the underlying asset")
    spot: SpotPrice[S] | None = Field(
        default=None, description="Spot price of the underlying asset"
    )
    maturities: dict[datetime, VolCrossSectionLoader[S]] = Field(
        default_factory=dict,
        description=(
            "Dictionary of maturities and their corresponding cross section loaders"
        ),
    )
    day_counter: DayCounter = Field(
        default=default_day_counter,
        description=(
            "Day counter for time to maturity calculations "
            "by default it uses Act/Act"
        ),
    )
    tick_size_forwards: DecimalNumber | None = Field(
        default=None,
        description="Tick size for rounding forward and spot prices - optional",
    )
    tick_size_options: DecimalNumber | None = Field(
        default=None, description="Tick size for rounding option prices - optional"
    )
    exclude_open_interest: DecimalNumber | None = Field(
        default=None,
        description="Exclude options with open interest at or below this value",
    )
    exclude_volume: DecimalNumber | None = Field(
        default=None, description="Exclude options with volume at or below this value"
    )

    def get_or_create_maturity(
        self, maturity: Annotated[datetime, Doc("Maturity date for the options")]
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
        bid: Annotated[Decimal, Doc("Bid price for the spot")] = ZERO,
        ask: Annotated[Decimal, Doc("Ask price for the spot")] = ZERO,
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
        bid: Annotated[Decimal, Doc("Bid price for the forward")] = ZERO,
        ask: Annotated[Decimal, Doc("Ask price for the forward")] = ZERO,
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
        bid: Annotated[Decimal, Doc("Bid price for the option")] = ZERO,
        ask: Annotated[Decimal, Doc("Ask price for the option")] = ZERO,
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

    def surface(
        self,
        ref_date: Annotated[
            datetime | None, Doc("Reference date for the volatility surface")
        ] = None,
    ) -> VolSurface[S]:
        """Build a volatility surface from the provided data"""
        if not self.spot or self.spot.mid == ZERO:
            raise ValueError("No spot price provided")
        maturities = []
        ref_date = ref_date or utcnow()
        previous_forward = self.spot.mid
        for maturity in sorted(self.maturities):
            if section := self.maturities[maturity].cross_section(
                ref_date=ref_date,
                previous_forward=previous_forward,
            ):
                previous_forward = section.forward.mid
                maturities.append(section)
        return VolSurface(
            asset=self.asset,
            ref_date=ref_date,
            spot=self.spot,
            maturities=tuple(maturities),
            day_counter=self.day_counter,
            tick_size_forwards=self.tick_size_forwards,
            tick_size_options=self.tick_size_options,
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
    loader = VolSurfaceLoader()
    for input in inputs.inputs:
        loader.add(input)
    return loader.surface(ref_date=inputs.ref_date)
