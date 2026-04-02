from __future__ import annotations

from datetime import datetime
from typing import ClassVar

from typing_extensions import Self

from quantflow.options.inputs import OptionType

from .base import Strategy, StrategyLeg, load_description


class Strangle(Strategy, frozen=True):
    """Call and put at different OTM strikes.

    Long vol when quantity > 0, short vol when quantity < 0.
    """

    description: ClassVar[str] = load_description("strangle.md")

    @classmethod
    def from_moneyness(
        cls,
        forward: float,
        maturity: datetime,
        put_moneyness: float = -0.05,
        call_moneyness: float = 0.05,
        quantity: float = 1.0,
    ) -> Self:
        """Create a strangle from log-strike offsets from forward."""
        return cls(
            legs=(
                StrategyLeg.from_moneyness(
                    OptionType.put, put_moneyness, forward, maturity, quantity
                ),
                StrategyLeg.from_moneyness(
                    OptionType.call, call_moneyness, forward, maturity, quantity
                ),
            )
        )

    @classmethod
    def from_strikes(
        cls,
        put_strike: float,
        call_strike: float,
        maturity: datetime,
        quantity: float = 1.0,
    ) -> Self:
        """Create a strangle from absolute strikes."""
        if put_strike >= call_strike:
            raise ValueError("Put strike must be less than call strike.")
        return cls(
            legs=(
                StrategyLeg(
                    option_type=OptionType.put,
                    quantity=quantity,
                    strike=put_strike,
                    maturity=maturity,
                ),
                StrategyLeg(
                    option_type=OptionType.call,
                    quantity=quantity,
                    strike=call_strike,
                    maturity=maturity,
                ),
            )
        )
