from __future__ import annotations

from datetime import datetime
from typing import ClassVar

from typing_extensions import Self

from quantflow.options.inputs import OptionMetadata, OptionType
from quantflow.utils.numbers import Number, to_decimal

from .base import Strategy, StrategyError, StrategyLeg, load_description


class Strangle(Strategy, frozen=True):
    """Call and put at different OTM strikes.

    Long vol when quantity > 0, short vol when quantity < 0.
    """

    description: ClassVar[str] = load_description("strangle.md")

    @classmethod
    def from_strikes(
        cls,
        put_strike: Number,
        call_strike: Number,
        maturity: datetime,
        quantity: Number = 1.0,
    ) -> Self:
        """Create a strangle from absolute strikes."""
        put = to_decimal(put_strike)
        call = to_decimal(call_strike)
        if put >= call:
            raise StrategyError("Put strike must be less than call strike.")
        q = to_decimal(quantity)
        return cls(
            legs=(
                StrategyLeg(
                    meta=OptionMetadata(
                        option_type=OptionType.PUT,
                        strike=put,
                        maturity=maturity,
                    ),
                    quantity=q,
                ),
                StrategyLeg(
                    meta=OptionMetadata(
                        option_type=OptionType.CALL,
                        strike=call,
                        maturity=maturity,
                    ),
                    quantity=q,
                ),
            )
        )
