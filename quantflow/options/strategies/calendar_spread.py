from __future__ import annotations

from datetime import datetime
from typing import ClassVar

from typing_extensions import Self

from quantflow.options.inputs import OptionMetadata, OptionType
from quantflow.utils.numbers import Number, to_decimal

from .base import Strategy, StrategyError, StrategyLeg, load_description


class CalendarSpread(Strategy, frozen=True):
    """Same strike, same option type, two maturities.

    Long the far maturity, short the near maturity when quantity > 0.
    """

    description: ClassVar[str] = load_description("calendar_spread.md")

    @property
    def option_type(self) -> OptionType:
        """Option type of the calendar spread."""
        return self.legs[0].meta.option_type

    @classmethod
    def create(
        cls,
        strike: Number,
        near_maturity: datetime,
        far_maturity: datetime,
        option_type: OptionType,
        quantity: Number = 1.0,
    ) -> Self:
        """Long far option, short near option at the same strike."""
        if near_maturity >= far_maturity:
            raise StrategyError("Near maturity must be before far maturity.")
        strike_ = to_decimal(strike)
        q = to_decimal(quantity)
        return cls(
            legs=(
                StrategyLeg(
                    meta=OptionMetadata(
                        option_type=option_type,
                        strike=strike_,
                        maturity=far_maturity,
                    ),
                    quantity=q,
                ),
                StrategyLeg(
                    meta=OptionMetadata(
                        option_type=option_type,
                        strike=strike_,
                        maturity=near_maturity,
                    ),
                    quantity=-q,
                ),
            )
        )

    @classmethod
    def call(
        cls,
        strike: Number,
        near_maturity: datetime,
        far_maturity: datetime,
        quantity: Number = 1.0,
    ) -> Self:
        return cls.create(
            strike, near_maturity, far_maturity, OptionType.CALL, quantity
        )

    @classmethod
    def put(
        cls,
        strike: Number,
        near_maturity: datetime,
        far_maturity: datetime,
        quantity: Number = 1.0,
    ) -> Self:
        return cls.create(strike, near_maturity, far_maturity, OptionType.PUT, quantity)
