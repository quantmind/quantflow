from __future__ import annotations

from datetime import datetime
from typing import ClassVar

from typing_extensions import Self

from quantflow.options.inputs import OptionType

from .base import Strategy, StrategyLeg, load_description


class Spread(Strategy, frozen=True):
    """Vertical spread: same option type, two strikes, same maturity.

    Long the spread when quantity > 0 (debit), short when quantity < 0 (credit).
    Call spread: long low strike, short high strike.
    Put spread: long high strike, short low strike.
    """

    description: ClassVar[str] = load_description("spread.md")

    @classmethod
    def call(
        cls,
        low_strike: float,
        high_strike: float,
        maturity: datetime,
        quantity: float = 1.0,
    ) -> Self:
        """Long call at low_strike, short call at high_strike."""
        return cls(
            legs=(
                StrategyLeg(
                    option_type=OptionType.call,
                    quantity=quantity,
                    strike=low_strike,
                    maturity=maturity,
                ),
                StrategyLeg(
                    option_type=OptionType.call,
                    quantity=-quantity,
                    strike=high_strike,
                    maturity=maturity,
                ),
            )
        )

    @classmethod
    def put(
        cls,
        low_strike: float,
        high_strike: float,
        maturity: datetime,
        quantity: float = 1.0,
    ) -> Self:
        """Long put at high_strike, short put at low_strike."""
        return cls(
            legs=(
                StrategyLeg(
                    option_type=OptionType.put,
                    quantity=quantity,
                    strike=high_strike,
                    maturity=maturity,
                ),
                StrategyLeg(
                    option_type=OptionType.put,
                    quantity=-quantity,
                    strike=low_strike,
                    maturity=maturity,
                ),
            )
        )
