from __future__ import annotations

from datetime import datetime
from typing import ClassVar

from typing_extensions import Self

from quantflow.options.inputs import OptionMetadata, OptionType
from quantflow.utils.numbers import Number, to_decimal

from .base import Strategy, StrategyLeg, load_description


class Straddle(Strategy, frozen=True):
    """Call and put at the same strike.

    Long vol when quantity > 0, short vol when quantity < 0.
    """

    description: ClassVar[str] = load_description("straddle.md")

    @classmethod
    def create(cls, strike: Number, maturity: datetime, quantity: Number = 1.0) -> Self:
        """Create a straddle at a given absolute strike."""
        strike_ = to_decimal(strike)
        q = to_decimal(quantity)
        return cls(
            legs=(
                StrategyLeg(
                    meta=OptionMetadata(
                        option_type=OptionType.call,
                        strike=strike_,
                        maturity=maturity,
                    ),
                    quantity=q,
                ),
                StrategyLeg(
                    meta=OptionMetadata(
                        option_type=OptionType.put,
                        strike=strike_,
                        maturity=maturity,
                    ),
                    quantity=q,
                ),
            )
        )
