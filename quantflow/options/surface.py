from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Generic, Iterator, Protocol, TypeVar

import numpy as np
import pandas as pd

from ..utils.interest_rates import rate_from_spot_and_forward
from .bs import implied_black_volatility
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


class VolSurfaceSecurity(Protocol):
    def vol_surface_type(self) -> VolSecurityType:
        ...


S = TypeVar("S", bound=VolSurfaceSecurity)


def time_to_maturity(maturity: datetime, ref_date: datetime) -> float:
    delta = maturity - ref_date
    return (delta.days + delta.seconds / 86400) / 365


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
    def inputs(self) -> SpotInput:
        return SpotInput(bid=self.bid, ask=self.ask)


@dataclass
class FwdPrice(Price[S]):
    maturity: datetime

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

    @property
    def moneyness(self) -> float:
        return float(np.log(float(self.strike / self.forward)))

    @property
    def price_bp(self) -> Decimal:
        return self.price * 10000

    @property
    def forward_price(self) -> Decimal:
        return self.forward * self.price

    def inputs(self) -> OptionInput:
        return OptionInput(
            strike=self.strike,
            price=self.price,
            maturity=self.maturity,
            call=self.call,
        )

    def _asdict(self) -> dict[str, Any]:
        return dict(
            strike=float(self.strike),
            forward=float(self.forward),
            moneyness=self.moneyness,
            ttm=self.ttm,
            implied_vol=self.implied_vol,
            price=float(self.price),
            price_bp=float(self.price_bp),
            forward_price=float(self.forward_price),
            call=self.call,
            side=self.side,
        )


@dataclass
class OptionPrices(Generic[S]):
    security: S
    bid: OptionPrice
    ask: OptionPrice

    def prices(
        self,
        forward: Decimal,
        ttm: float,
        initial_vol: float,
    ) -> Iterator[OptionPrice]:
        for o in (self.bid, self.ask):
            o.forward = forward
            o.ttm = ttm
            if not o.implied_vol:
                o.implied_vol = initial_vol
            yield o

    def inputs(self) -> OptionSidesInput:
        return OptionSidesInput(
            bid=self.bid.inputs(),
            ask=self.ask.inputs(),
        )


@dataclass
class Strike(Generic[S]):
    strike: Decimal
    call: OptionPrices[S] | None = None
    put: OptionPrices[S] | None = None

    def option_prices(
        self,
        forward: Decimal,
        ttm: float,
        call: bool = True,
        initial_vol: float = INITIAL_VOL,
    ) -> Iterator[OptionPrice]:
        if call and self.call:
            yield from self.call.prices(forward, ttm, initial_vol)
        if not call and self.put:
            yield from self.put.prices(forward, ttm, initial_vol)


@dataclass
class VolCrossSection(Generic[S]):
    maturity: datetime
    forward: FwdPrice[S]
    """Forward price of the underlying asset at the time of the cross section"""
    strikes: tuple[Strike[S], ...]
    """Tuple of sorted strikes and their corresponding option prices"""

    def info_dict(self, ref_date: datetime, spot: SpotPrice[S]) -> dict:
        return dict(
            maturity=self.maturity,
            ttm=time_to_maturity(self.maturity, ref_date),
            forward=self.forward.mid,
            basis=self.forward.mid - spot.mid,
            rate_percent=rate_from_spot_and_forward(
                spot.mid, self.forward.mid, self.maturity - ref_date
            ).percent,
        )

    def option_prices(
        self, ref_date: datetime, call: bool = True, initial_vol: float = INITIAL_VOL
    ) -> Iterator[OptionPrice]:
        ttm = time_to_maturity(self.maturity, ref_date)
        for s in self.strikes:
            yield from s.option_prices(
                self.forward.mid, ttm, call, initial_vol=initial_vol
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
    ref_date: datetime
    spot: SpotPrice[S]
    maturities: tuple[VolCrossSection[S], ...]

    def term_structure(self, frequency: float = 0) -> pd.DataFrame:
        """Return the term structure of the volatility surface"""
        return pd.DataFrame(
            cross.info_dict(self.ref_date, self.spot) for cross in self.maturities
        )

    def option_prices(
        self,
        *,
        call: bool = True,
        index: int | None = None,
        initial_vol: float = INITIAL_VOL,
    ) -> Iterator[OptionPrice]:
        "Iterator over all option prices in the surface"
        if index is not None:
            yield from self.maturities[index].option_prices(
                self.ref_date, call=call, initial_vol=initial_vol
            )
        else:
            for maturity in self.maturities:
                yield from maturity.option_prices(
                    self.ref_date, call=call, initial_vol=initial_vol
                )

    def bs(
        self,
        *,
        call: bool = True,
        index: int | None = None,
        initial_vol: float = INITIAL_VOL,
    ) -> VolSurface:
        """calculate Black-Scholes implied volatilities for all options
        in the surface"""
        options = list(
            self.option_prices(call=call, index=index, initial_vol=initial_vol)
        )
        moneyness = []
        ttm = []
        price = []
        vol = []
        for option in options:
            moneyness.append(float(option.moneyness))
            ttm.append(float(option.ttm))
            price.append(float(option.price))
            vol.append(float(option.implied_vol))
        result = implied_black_volatility(
            np.array(moneyness), np.array(ttm), np.array(price), np.array(vol)
        )
        for option, implied_vol in zip(options, result):
            option.implied_vol = implied_vol
        return self

    def options_df(
        self,
        *,
        call: bool = True,
        index: int | None = None,
        initial_vol: float = INITIAL_VOL,
    ) -> pd.DataFrame:
        """Time frame of Black-Scholes call input data"""
        data = self.option_prices(call=call, index=index, initial_vol=initial_vol)
        return pd.DataFrame([d._asdict() for d in data])

    def securities(self) -> Iterator[SpotPrice[S] | FwdPrice[S] | OptionPrices[S]]:
        yield self.spot
        for maturity in self.maturities:
            yield from maturity.securities()

    def inputs(self) -> VolSurfaceInputs:
        return VolSurfaceInputs(
            ref_date=self.ref_date, inputs=list(s.inputs() for s in self.securities())
        )


@dataclass
class VolCrossSectionLoader(Generic[S]):
    maturity: datetime
    forward: FwdPrice[S] | None = None
    """Forward price of the underlying asset at the time of the cross section"""
    strikes: dict[Decimal, Strike[S]] = field(default_factory=dict)
    """List of strikes and their corresponding option prices"""

    def add_option(
        self,
        strike: Decimal,
        call: bool,
        security: S,
        bid: Decimal = ZERO,
        ask: Decimal = ZERO,
    ) -> None:
        if strike not in self.strikes:
            self.strikes[strike] = Strike(strike=strike)
        option = OptionPrices(
            security,
            bid=OptionPrice(
                price=bid, strike=strike, call=call, maturity=self.maturity, side="bid"
            ),
            ask=OptionPrice(
                price=ask, strike=strike, call=call, maturity=self.maturity, side="ask"
            ),
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
                maturity=self.maturity, forward=self.forward, strikes=tuple(strikes)
            )
            if strikes
            else None
        )


@dataclass
class GenericVolSurfaceLoader(Generic[S]):
    """Helper class to build a volatility surface from a list of securities"""

    spot: SpotPrice[S] | None = None
    maturities: dict[datetime, VolCrossSectionLoader[S]] = field(default_factory=dict)

    def get_or_create_maturity(self, maturity: datetime) -> VolCrossSectionLoader[S]:
        if maturity not in self.maturities:
            self.maturities[maturity] = VolCrossSectionLoader(maturity=maturity)
        return self.maturities[maturity]

    def add_spot(self, security: S, bid: Decimal = ZERO, ask: Decimal = ZERO) -> None:
        if security.vol_surface_type() != VolSecurityType.spot:
            raise ValueError("Security is not a spot")
        self.spot = SpotPrice(security, bid=bid, ask=ask)

    def add_forward(
        self, security: S, maturity: datetime, bid: Decimal = ZERO, ask: Decimal = ZERO
    ) -> None:
        if security.vol_surface_type() != VolSecurityType.forward:
            raise ValueError("Security is not a forward")
        self.get_or_create_maturity(maturity=maturity).forward = FwdPrice(
            security, bid=bid, ask=ask, maturity=maturity
        )

    def add_option(
        self,
        security: S,
        strike: Decimal,
        maturity: datetime,
        call: bool,
        bid: Decimal = ZERO,
        ask: Decimal = ZERO,
    ) -> None:
        if security.vol_surface_type() != VolSecurityType.option:
            raise ValueError("Security is not an option")
        self.get_or_create_maturity(maturity=maturity).add_option(
            strike, call, security, bid=bid, ask=ask
        )

    def surface(self, ref_date: datetime | None = None) -> VolSurface[S]:
        if not self.spot or self.spot.mid == ZERO:
            raise ValueError("No spot price provided")
        maturities = []
        for maturity in sorted(self.maturities):
            if section := self.maturities[maturity].cross_section():
                maturities.append(section)
        return VolSurface(
            ref_date=ref_date or datetime.utcnow().replace(tzinfo=timezone.utc),
            spot=self.spot,
            maturities=tuple(maturities),
        )


class VolSurfaceLoader(GenericVolSurfaceLoader[VolSecurityType]):
    def add(self, input: VolSurfaceInput[Any]) -> None:
        if isinstance(input, SpotInput):
            self.add_spot(VolSecurityType.spot, bid=input.bid, ask=input.ask)
        elif isinstance(input, FwdPrice):
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
