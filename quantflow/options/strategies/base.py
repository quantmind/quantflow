from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path
from typing import ClassVar

from ccy.core.daycounter import DayCounter
from pydantic import BaseModel, Field
from typing_extensions import Annotated, Doc

from quantflow.options.inputs import OptionType
from quantflow.options.pricer import ModelOptionPrice, OptionPricer  # noqa: TC001
from quantflow.options.surface import default_day_counter

OPTIONS_DOCS_PATH = Path(__file__).parent.parent / "docs"


def load_description(filename: str) -> str:
    """Load a strategy description from a markdown file in this package."""
    return (OPTIONS_DOCS_PATH / filename).read_text(encoding="utf-8")


class StrategyLeg(BaseModel, frozen=True):
    """A single leg of an option strategy."""

    option_type: OptionType = Field(description="Call or put")
    quantity: float = Field(
        description="Signed quantity: positive for long, negative for short"
    )
    strike: float = Field(description="Absolute strike price")
    maturity: datetime = Field(description="Expiry date of the option")

    @classmethod
    def from_moneyness(
        cls,
        option_type: OptionType,
        moneyness: float,
        forward: float,
        maturity: datetime,
        quantity: float = 1.0,
    ) -> StrategyLeg:
        """Create a leg from a log-strike moneyness offset and forward price."""
        return cls(
            option_type=option_type,
            quantity=quantity,
            strike=forward * math.exp(moneyness),
            maturity=maturity,
        )


class StrategyPrice(BaseModel, frozen=True):
    """Priced result of an option strategy."""

    legs: tuple[ModelOptionPrice, ...] = Field(
        description="Priced legs of the strategy"
    )
    price: float = Field(description="Total price in forward space")
    delta: float = Field(description="Total delta")
    gamma: float = Field(description="Total gamma")


class Strategy(BaseModel, frozen=True):
    """Base class for option strategies.

    Subclasses define a `description` class variable loaded from a markdown file
    via `load_description`. The description is intended for AI agents.

    Legs are built via classmethods and passed directly to the constructor.
    """

    description: ClassVar[str] = ""

    legs: Annotated[
        tuple[StrategyLeg, ...], Doc("Option legs that make up the strategy")
    ]

    def price(
        self,
        pricer: Annotated[OptionPricer, Doc("Option pricer with a fitted model")],
        forward: Annotated[float, Doc("Forward price of the underlying")],
        ref_date: Annotated[datetime, Doc("Reference date for ttm calculation")],
        day_counter: Annotated[
            DayCounter, Doc("Day count convention")
        ] = default_day_counter,
    ) -> StrategyPrice:
        """Price the strategy and return aggregate price and Greeks."""
        priced: list[ModelOptionPrice] = []
        total_price = 0.0
        total_delta = 0.0
        total_gamma = 0.0

        for leg in self.legs:
            ttm = day_counter.dcf(ref_date, leg.maturity)
            leg_price = pricer.price(
                option_type=leg.option_type,
                strike=leg.strike,
                forward=forward,
                ttm=ttm,
            )
            priced.append(leg_price)
            total_price += leg.quantity * leg_price.price
            total_delta += leg.quantity * leg_price.delta
            total_gamma += leg.quantity * leg_price.gamma

        return StrategyPrice(
            legs=tuple(priced),
            price=total_price,
            delta=total_delta,
            gamma=total_gamma,
        )
