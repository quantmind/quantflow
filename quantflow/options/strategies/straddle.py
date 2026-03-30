from __future__ import annotations

from datetime import datetime
from typing import ClassVar

from typing_extensions import Self

from quantflow.options.inputs import OptionType

from .base import Strategy, StrategyLeg, load_description


class Straddle(Strategy, frozen=True):
    """Call and put at the same strike.

    Long vol when quantity > 0, short vol when quantity < 0.
    """

    description: ClassVar[str] = load_description("straddle.md")

    @classmethod
    def create(cls, strike: float, maturity: datetime, quantity: float = 1.0) -> Self:
        """Create a straddle at a given absolute strike."""
        return cls(
            legs=(
                StrategyLeg(
                    option_type=OptionType.call,
                    quantity=quantity,
                    strike=strike,
                    maturity=maturity,
                ),
                StrategyLeg(
                    option_type=OptionType.put,
                    quantity=quantity,
                    strike=strike,
                    maturity=maturity,
                ),
            )
        )
