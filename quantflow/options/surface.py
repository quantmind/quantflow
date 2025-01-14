from __future__ import annotations

import enum
import warnings
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Generic, Iterator, NamedTuple, Protocol, Self, TypeVar

import numpy as np
import pandas as pd
from ccy.core.daycounter import ActAct, DayCounter

from quantflow.utils import plot
from quantflow.utils.dates import utcnow
from quantflow.utils.interest_rates import rate_from_spot_and_forward
from quantflow.utils.numbers import Number, sigfig, to_decimal

from .bs import black_price, implied_black_volatility
from .inputs import (
    ForwardInput,
    OptionInput,
    OptionSidesInput,
    SpotInput,
    VolSecurityType,
    VolSurfaceInput,
    VolSurfaceInputs,
)

INITIAL_VOL = 0.5
ZERO = Decimal("0")
default_day_counter = ActAct()


class VolSurfaceSecurity(Protocol):
    def vol_surface_type(self) -> VolSecurityType: ...


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


@dataclass
class Price(Generic[S]):
    security: S
    bid: Decimal
    ask: Decimal

    @property
    def mid(self) -> Decimal:
        return (self.bid + self.ask) / 2


@dataclass
class SpotPrice(Price[S]):
    open_interest: int = 0
    volume: int = 0

    def inputs(self) -> SpotInput:
        return SpotInput(bid=self.bid, ask=self.ask)


@dataclass
class FwdPrice(Price[S]):
    maturity: datetime
    open_interest: int = 0
    volume: int = 0

    def inputs(self) -> ForwardInput:
        return ForwardInput(
            bid=self.bid,
            ask=self.ask,
            maturity=self.maturity,
        )


@dataclass
class OptionPrice:
    price: Decimal
    """Price of the option divided by the forward price"""
    strike: Decimal
    """Strike price"""
    call: bool
    """True if call, False if put"""
    maturity: datetime
    """Maturity date"""
    forward: Decimal = ZERO
    """Forward price of the underlying"""
    implied_vol: float = 0
    """Implied Black volatility"""
    ttm: float = 0
    """Time to maturity in years"""
    side: str = "bid"
    """Side of the market"""
    converged: bool = True
    """Flag indicating if implied vol calculation converged"""

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
        call: bool = True,
    ) -> OptionPrice:
        """Create an option price

        mainly used for testing
        """
        ref_date = ref_date or utcnow()
        maturity = maturity or ref_date + timedelta(days=365)
        day_counter = day_counter or default_day_counter
        return cls(
            price=to_decimal(price),
            strike=to_decimal(strike),
            forward=to_decimal(forward or strike),
            implied_vol=implied_vol,
            call=call,
            maturity=maturity,
            ttm=day_counter.dcf(ref_date, maturity),
        )

    @property
    def moneyness(self) -> float:
        return float(np.log(float(self.strike / self.forward)))

    @property
    def price_bp(self) -> Decimal:
        return self.price * 10000

    @property
    def forward_price(self) -> Decimal:
        return self.forward * self.price

    @property
    def price_intrinsic(self) -> Decimal:
        if self.call:
            return max(self.forward - self.strike, ZERO) / self.forward
        else:
            return max(self.strike - self.forward, ZERO) / self.forward

    @property
    def price_time(self) -> Decimal:
        return self.price - self.price_intrinsic

    @property
    def call_price(self) -> Decimal:
        """call price

        use put-call parity to calculate the call price if a put
        """
        if self.call:
            return self.price
        else:
            return self.price + 1 - self.strike / self.forward

    @property
    def put_price(self) -> Decimal:
        """put price

        use put-call parity to calculate the put price if a call
        """
        if self.call:
            return self.price - 1 + self.strike / self.forward
        else:
            return self.price

    @property
    def option_type(self) -> str:
        return "call" if self.call else "put"

    def can_price(self, converged: bool, select: OptionSelection) -> bool:
        if self.price_time > ZERO and not np.isnan(self.implied_vol):
            if not self.converged and converged is True:
                return False
            if select == OptionSelection.best:
                return self.price_intrinsic == ZERO
            return True
        return False

    def inputs(self) -> OptionInput:
        return OptionInput(
            strike=self.strike,
            price=self.price,
            maturity=self.maturity,
            call=self.call,
        )

    def calculate_price(self) -> OptionPrice:
        self.price = Decimal(
            sigfig(
                black_price(
                    np.asarray(self.moneyness),
                    self.implied_vol,
                    self.ttm,
                    1 if self.call else -1,
                ).sum(),
                8,
            )
        )
        return self

    def _asdict(self) -> dict[str, Any]:
        return dict(
            strike=float(self.strike),
            forward=float(self.forward),
            moneyness=self.moneyness,
            moneyness_ttm=self.moneyness / np.sqrt(self.ttm),
            ttm=self.ttm,
            implied_vol=self.implied_vol,
            price=float(self.price),
            price_bp=float(self.price_bp),
            forward_price=float(self.forward_price),
            type=self.option_type,
            side=self.side,
        )


class OptionArrays(NamedTuple):
    options: list[OptionPrice]
    moneyness: np.ndarray
    price: np.ndarray
    ttm: np.ndarray
    implied_vol: np.ndarray
    call_put: np.ndarray


@dataclass
class OptionPrices(Generic[S]):
    security: S
    bid: OptionPrice
    ask: OptionPrice
    open_interest: int = 0
    volume: int = 0

    def prices(
        self,
        forward: Decimal,
        ttm: float,
        *,
        select: OptionSelection = OptionSelection.best,
        initial_vol: float = INITIAL_VOL,
        converged: bool = True,
    ) -> Iterator[OptionPrice]:
        for o in (self.bid, self.ask):
            o.forward = forward
            o.ttm = ttm
            if not o.implied_vol:
                o.implied_vol = initial_vol
            if o.can_price(converged, select):
                yield o

    def inputs(self) -> OptionSidesInput:
        return OptionSidesInput(
            bid=self.bid.inputs(),
            ask=self.ask.inputs(),
        )


@dataclass
class Strike(Generic[S]):
    """Option prices for a single strike"""

    strike: Decimal
    call: OptionPrices[S] | None = None
    put: OptionPrices[S] | None = None

    def option_prices(
        self,
        forward: Decimal,
        ttm: float,
        *,
        select: OptionSelection = OptionSelection.best,
        initial_vol: float = INITIAL_VOL,
        converged: bool = True,
    ) -> Iterator[OptionPrice]:
        if select != OptionSelection.put and self.call:
            yield from self.call.prices(
                forward,
                ttm,
                select=select,
                initial_vol=initial_vol,
                converged=converged,
            )
        if select != OptionSelection.call and self.put:
            yield from self.put.prices(
                forward,
                ttm,
                select=select,
                initial_vol=initial_vol,
                converged=converged,
            )


@dataclass
class VolCrossSection(Generic[S]):
    """Represents a cross section of a volatility surface at a specific maturity."""

    maturity: datetime
    """Maturity date of the cross section"""
    forward: FwdPrice[S]
    """Forward price of the underlying asset at the time of the cross section"""
    strikes: tuple[Strike[S], ...]
    """Tuple of sorted strikes and their corresponding option prices"""
    day_counter: DayCounter = default_day_counter
    """Day counter for time to maturity calculations - by default it uses Act/Act"""

    def ttm(self, ref_date: datetime) -> float:
        """Time to maturity in years"""
        return self.day_counter.dcf(ref_date, self.maturity)

    def info_dict(self, ref_date: datetime, spot: SpotPrice[S]) -> dict:
        """Return a dictionary with information about the cross section"""
        return dict(
            maturity=self.maturity,
            ttm=self.ttm(ref_date),
            forward=self.forward.mid,
            basis=self.forward.mid - spot.mid,
            rate_percent=rate_from_spot_and_forward(
                spot.mid, self.forward.mid, self.maturity - ref_date
            ).percent,
            open_interest=self.forward.open_interest,
            volume=self.forward.volume,
        )

    def option_prices(
        self,
        ref_date: datetime,
        *,
        select: OptionSelection = OptionSelection.best,
        initial_vol: float = INITIAL_VOL,
        converged: bool = True,
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

    def securities(self) -> Iterator[FwdPrice[S] | OptionPrices[S]]:
        """Return a list of all securities in the cross section"""
        yield self.forward
        for strike in self.strikes:
            if strike.call is not None:
                yield strike.call
            if strike.put is not None:
                yield strike.put


@dataclass
class VolSurface(Generic[S]):
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

    ref_date: datetime
    """Reference date for the volatility surface"""
    spot: SpotPrice[S]
    """Spot price of the underlying asset"""
    maturities: tuple[VolCrossSection[S], ...]
    """Sorted tuple of :class:`.VolCrossSection` for different maturities"""
    day_counter: DayCounter = default_day_counter
    """Day counter for time to maturity calculations - by default it uses Act/Act"""
    tick_size_forwards: Decimal | None = None
    """Tick size for rounding forward and spot prices - optional"""
    tick_size_options: Decimal | None = None
    """Tick size for rounding option prices - optional"""

    def securities(self) -> Iterator[SpotPrice[S] | FwdPrice[S] | OptionPrices[S]]:
        """Iterator over all securities in the volatility surface"""
        yield self.spot
        for maturity in self.maturities:
            yield from maturity.securities()

    def inputs(self) -> VolSurfaceInputs:
        return VolSurfaceInputs(
            ref_date=self.ref_date, inputs=list(s.inputs() for s in self.securities())
        )

    def term_structure(self, frequency: float = 0) -> pd.DataFrame:
        """Return the term structure of the volatility surface"""
        return pd.DataFrame(
            cross.info_dict(self.ref_date, self.spot) for cross in self.maturities
        )

    def trim(self, num_maturities: int) -> Self:
        """Create a new volatility surface with the last `num_maturities` maturities"""
        return replace(self, maturities=self.maturities[-num_maturities:])

    def option_prices(
        self,
        *,
        select: OptionSelection = OptionSelection.best,
        index: int | None = None,
        initial_vol: float = INITIAL_VOL,
        converged: bool = True,
    ) -> Iterator[OptionPrice]:
        "Iterator over selected option prices in the surface"
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
        select: OptionSelection = OptionSelection.best,
        index: int | None = None,
        converged: bool = True,
    ) -> list[OptionPrice]:
        "List of selected option prices in the surface"
        return list(self.option_prices(select=select, index=index, converged=converged))

    def bs(
        self,
        *,
        select: OptionSelection = OptionSelection.best,
        index: int | None = None,
        initial_vol: float = INITIAL_VOL,
    ) -> list[OptionPrice]:
        """calculate Black-Scholes implied volatility for all options
        in the surface

        :param select: the :class:`.OptionSelection` method
        :param index: Index of the cross section to use, if None use all
        :param initial_vol: Initial volatility for the root finding algorithm

        Some options may not converge, in this case the implied volatility is not
        calculated correctly and the option is marked as not converged.
        """
        d = self.as_array(
            select=select,
            index=index,
            initial_vol=initial_vol,
            converged=False,
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
            d.options, result.root, result.converged
        ):
            option.implied_vol = float(implied_vol)
            option.converged = converged
        return d.options

    def calc_bs_prices(
        self,
        *,
        select: OptionSelection = OptionSelection.best,
        index: int | None = None,
    ) -> np.ndarray:
        """calculate Black-Scholes prices for all options in the surface"""
        d = self.as_array(select=select, index=index)
        return black_price(k=d.moneyness, sigma=d.implied_vol, ttm=d.ttm, s=d.call_put)

    def options_df(
        self,
        *,
        select: OptionSelection = OptionSelection.best,
        index: int | None = None,
        initial_vol: float = INITIAL_VOL,
        converged: bool = True,
    ) -> pd.DataFrame:
        """Time frame of Black-Scholes call input data"""
        data = self.option_prices(
            select=select,
            index=index,
            initial_vol=initial_vol,
            converged=converged,
        )
        return pd.DataFrame([d._asdict() for d in data])

    def as_array(
        self,
        *,
        select: OptionSelection = OptionSelection.best,
        index: int | None = None,
        initial_vol: float = INITIAL_VOL,
        converged: bool = True,
    ) -> OptionArrays:
        """Organize option prices in a numpy arrays for black volatility calculation"""
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
            price.append(float(option.price))
            ttm.append(float(option.ttm))
            vol.append(float(option.implied_vol))
            call_put.append(1 if option.call else -1)
        return OptionArrays(
            options=options,
            moneyness=np.array(moneyness),
            price=np.array(price),
            ttm=np.array(ttm),
            implied_vol=np.array(vol),
            call_put=np.array(call_put),
        )

    def disable_outliers(self, quantile: float = 0.99, repeat: int = 2) -> VolSurface:
        for _ in range(repeat):
            option_prices = self.option_list()
            implied_vols = [o.implied_vol for o in option_prices]
            exclude_above = np.quantile(implied_vols, quantile)
            for option in option_prices:
                if option.implied_vol > exclude_above:
                    option.converged = False
        return self

    def plot(
        self,
        *,
        index: int | None = None,
        select: OptionSelection = OptionSelection.best,
        **kwargs: Any,
    ) -> Any:
        """Plot the volatility surface"""
        df = self.options_df(index=index, select=select)
        return plot.plot_vol_surface(df, **kwargs)

    def plot3d(
        self,
        *,
        select: OptionSelection = OptionSelection.best,
        **kwargs: Any,
    ) -> Any:
        """Plot the volatility surface"""
        df = self.options_df(select=select)
        return plot.plot_vol_surface_3d(df, **kwargs)


@dataclass
class VolCrossSectionLoader(Generic[S]):
    maturity: datetime
    forward: FwdPrice[S] | None = None
    """Forward price of the underlying asset at the time of the cross section"""
    strikes: dict[Decimal, Strike[S]] = field(default_factory=dict)
    """List of strikes and their corresponding option prices"""
    day_counter: DayCounter = default_day_counter

    def add_option(
        self,
        strike: Decimal,
        call: bool,
        security: S,
        bid: Decimal = ZERO,
        ask: Decimal = ZERO,
        open_interest: int = 0,
        volume: int = 0,
    ) -> None:
        if strike not in self.strikes:
            self.strikes[strike] = Strike(strike=strike)
        option = OptionPrices(
            security,
            bid=OptionPrice(
                price=bid,
                strike=strike,
                call=call,
                maturity=self.maturity,
                side="bid",
            ),
            ask=OptionPrice(
                price=ask,
                strike=strike,
                call=call,
                maturity=self.maturity,
                side="ask",
            ),
            open_interest=open_interest,
            volume=volume,
        )
        if call:
            self.strikes[strike].call = option
        else:
            self.strikes[strike].put = option

    def cross_section(self) -> VolCrossSection[S] | None:
        if self.forward is None or self.forward.mid == ZERO:
            return None
        strikes = []
        for strike in sorted(self.strikes):
            sk = self.strikes[strike]
            if sk.call is None and sk.put is None:
                continue
            strikes.append(sk)
        return (
            VolCrossSection(
                maturity=self.maturity,
                forward=self.forward,
                strikes=tuple(strikes),
                day_counter=self.day_counter,
            )
            if strikes
            else None
        )


@dataclass
class GenericVolSurfaceLoader(Generic[S]):
    """Helper class to build a volatility surface from a list of securities"""

    spot: SpotPrice[S] | None = None
    """Spot price of the underlying asset"""
    maturities: dict[datetime, VolCrossSectionLoader[S]] = field(default_factory=dict)
    """Dictionary of maturities and their corresponding cross section loaders"""
    day_counter: DayCounter = default_day_counter
    """Day counter for time to maturity calculations - by default it uses Act/Act"""
    tick_size_forwards: Decimal | None = None
    """Tick size for rounding forward and spot prices - optional"""
    tick_size_options: Decimal | None = None
    """Tick size for rounding option prices - optional"""

    def get_or_create_maturity(self, maturity: datetime) -> VolCrossSectionLoader[S]:
        if maturity not in self.maturities:
            self.maturities[maturity] = VolCrossSectionLoader(
                maturity=maturity,
                day_counter=self.day_counter,
            )
        return self.maturities[maturity]

    def add_spot(
        self,
        security: S,
        bid: Decimal = ZERO,
        ask: Decimal = ZERO,
        open_interest: int = 0,
        volume: int = 0,
    ) -> None:
        """Add a spot to the volatility surface loader"""
        if security.vol_surface_type() != VolSecurityType.spot:
            raise ValueError("Security is not a spot")
        self.spot = SpotPrice(
            security,
            bid=bid,
            ask=ask,
            open_interest=open_interest,
            volume=volume,
        )

    def add_forward(
        self,
        security: S,
        maturity: datetime,
        bid: Decimal = ZERO,
        ask: Decimal = ZERO,
        open_interest: int = 0,
        volume: int = 0,
    ) -> None:
        """Add a forward to the volatility surface loader"""
        if security.vol_surface_type() != VolSecurityType.forward:
            raise ValueError("Security is not a forward")
        self.get_or_create_maturity(maturity=maturity).forward = FwdPrice(
            security,
            bid=bid,
            ask=ask,
            maturity=maturity,
            open_interest=open_interest,
            volume=volume,
        )

    def add_option(
        self,
        security: S,
        strike: Decimal,
        maturity: datetime,
        call: bool,
        bid: Decimal = ZERO,
        ask: Decimal = ZERO,
        open_interest: int = 0,
        volume: int = 0,
    ) -> None:
        """Add an option to the volatility surface loader"""
        if security.vol_surface_type() != VolSecurityType.option:
            raise ValueError("Security is not an option")
        self.get_or_create_maturity(maturity=maturity).add_option(
            strike,
            call,
            security,
            bid=bid,
            ask=ask,
            open_interest=open_interest,
            volume=volume,
        )

    def surface(self, ref_date: datetime | None = None) -> VolSurface[S]:
        """Build a volatility surface from the provided data"""
        if not self.spot or self.spot.mid == ZERO:
            raise ValueError("No spot price provided")
        maturities = []
        for maturity in sorted(self.maturities):
            if section := self.maturities[maturity].cross_section():
                maturities.append(section)
        return VolSurface(
            ref_date=ref_date or utcnow(),
            spot=self.spot,
            maturities=tuple(maturities),
            day_counter=self.day_counter,
            tick_size_forwards=self.tick_size_forwards,
            tick_size_options=self.tick_size_options,
        )


class VolSurfaceLoader(GenericVolSurfaceLoader[VolSecurityType]):
    def add(self, input: VolSurfaceInput[Any]) -> None:
        """Add a volatility security input to the loader

        :params input: The input data for the security,
            it can be spot, forward or option
        """
        if isinstance(input, SpotInput):
            self.add_spot(VolSecurityType.spot, bid=input.bid, ask=input.ask)
        elif isinstance(input, ForwardInput):
            self.add_forward(
                VolSecurityType.forward,
                maturity=input.maturity,
                bid=input.bid,
                ask=input.ask,
            )
        elif isinstance(input, OptionSidesInput):
            self.add_option(
                VolSecurityType.option,
                strike=assert_same(input.bid.strike, input.ask.strike),
                call=assert_same(input.bid.call, input.ask.call),
                maturity=assert_same(input.bid.maturity, input.ask.maturity),
                bid=input.bid.price,
                ask=input.ask.price,
            )
        else:
            raise ValueError(f"Unknown input type {type(input)}")


def surface_from_inputs(inputs: VolSurfaceInputs) -> VolSurface[VolSecurityType]:
    loader = VolSurfaceLoader()
    for input in inputs.inputs:
        loader.add(input)
    return loader.surface(ref_date=inputs.ref_date)


def assert_same(a: Any, b: Any) -> Any:
    if a != b:
        raise ValueError(f"Values are not the same: {a} != {b}")
    return a
