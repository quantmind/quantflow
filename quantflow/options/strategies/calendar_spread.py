from __future__ import annotations

from datetime import datetime
from typing import ClassVar

from typing_extensions import Self

from quantflow.options.inputs import OptionType

from .base import Strategy, StrategyLeg, load_description


class CalendarSpread(Strategy, frozen=True):
    """Same strike, same option type, two maturities.

    Long the far maturity, short the near maturity when quantity > 0.
    """

    description: ClassVar[str] = load_description("calendar_spread.md")

    @property
    def option_type(self) -> OptionType:
        """Option type of the calendar spread."""
        return self.legs[0].option_type

    @classmethod
    def create(
        cls,
        strike: float,
        near_maturity: datetime,
        far_maturity: datetime,
        option_type: OptionType,
        quantity: float = 1.0,
    ) -> Self:
        """Long far call, short near call at the same strike."""
        if near_maturity >= far_maturity:
            raise ValueError("Near maturity must be before far maturity.")
        return cls(
            legs=(
                StrategyLeg(
                    option_type=option_type,
                    quantity=quantity,
                    strike=strike,
                    maturity=far_maturity,
                ),
                StrategyLeg(
                    option_type=option_type,
                    quantity=-quantity,
                    strike=strike,
                    maturity=near_maturity,
                ),
            )
        )

    @classmethod
    def call(
        cls,
        strike: float,
        near_maturity: datetime,
        far_maturity: datetime,
        quantity: float = 1.0,
    ) -> Self:
        return cls.create(
            strike, near_maturity, far_maturity, OptionType.call, quantity
        )

    @classmethod
    def put(
        cls,
        strike: float,
        near_maturity: datetime,
        far_maturity: datetime,
        quantity: float = 1.0,
    ) -> Self:
        return cls.create(strike, near_maturity, far_maturity, OptionType.put, quantity)
