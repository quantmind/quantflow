from __future__ import annotations

from datetime import datetime
from typing import ClassVar

from typing_extensions import Self

from quantflow.options.inputs import OptionMetadata, OptionType
from quantflow.utils.numbers import Number, to_decimal

from .base import Strategy, StrategyError, StrategyLeg, load_description


class Spread(Strategy, frozen=True):
    description: ClassVar[str] = load_description("spread.md")

    @classmethod
    def call(
        cls,
        low_strike: Number,
        high_strike: Number,
        maturity: datetime,
        quantity: Number = 1.0,
    ) -> Self:
        """Long call at low_strike, short call at high_strike."""
        low = to_decimal(low_strike)
        high = to_decimal(high_strike)
        if low >= high:
            raise StrategyError("low_strike must be less than high_strike.")
        q = to_decimal(quantity)
        return cls(
            legs=(
                StrategyLeg(
                    meta=OptionMetadata(
                        option_type=OptionType.CALL,
                        strike=low,
                        maturity=maturity,
                    ),
                    quantity=q,
                ),
                StrategyLeg(
                    meta=OptionMetadata(
                        option_type=OptionType.CALL,
                        strike=high,
                        maturity=maturity,
                    ),
                    quantity=-q,
                ),
            )
        )

    @classmethod
    def put(
        cls,
        low_strike: Number,
        high_strike: Number,
        maturity: datetime,
        quantity: Number = 1.0,
    ) -> Self:
        """Long put at high_strike, short put at low_strike."""
        low = to_decimal(low_strike)
        high = to_decimal(high_strike)
        if low >= high:
            raise StrategyError("low_strike must be less than high_strike.")
        q = to_decimal(quantity)
        return cls(
            legs=(
                StrategyLeg(
                    meta=OptionMetadata(
                        option_type=OptionType.PUT,
                        strike=high,
                        maturity=maturity,
                    ),
                    quantity=q,
                ),
                StrategyLeg(
                    meta=OptionMetadata(
                        option_type=OptionType.PUT,
                        strike=low,
                        maturity=maturity,
                    ),
                    quantity=-q,
                ),
            )
        )
