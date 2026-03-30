from __future__ import annotations

from datetime import datetime
from typing import ClassVar

from typing_extensions import Self

from quantflow.options.inputs import OptionType

from .base import Strategy, StrategyLeg, load_description


def _option_type_for_moneyness(mid_moneyness: float) -> OptionType:
    """Select option type based on body moneyness for best liquidity.

    Calls for body above ATM, puts for body below ATM, calls at ATM.
    """
    return OptionType.put if mid_moneyness < 0 else OptionType.call


class Butterfly(Strategy, frozen=True):
    """Three-strike strategy: long wings, short body.

    Long butterfly when quantity > 0, short butterfly when quantity < 0.
    Can be constructed with calls or puts — both are equivalent by put-call parity.
    """

    description: ClassVar[str] = load_description("butterfly.md")

    @classmethod
    def from_moneyness(
        cls,
        forward: float,
        maturity: datetime,
        wing_moneyness: float = 0.05,
        mid_moneyness: float = 0.0,
        quantity: float = 1.0,
        option_type: OptionType | None = None,
    ) -> Self:
        """Create a butterfly from a wing offset and body moneyness.

        If option_type is not specified, it is selected automatically based on
        the body moneyness for best liquidity.
        """
        ot = option_type or _option_type_for_moneyness(mid_moneyness)
        return cls(
            legs=(
                StrategyLeg.from_moneyness(
                    ot, mid_moneyness - wing_moneyness, forward, maturity, quantity
                ),
                StrategyLeg.from_moneyness(
                    ot, mid_moneyness, forward, maturity, -2.0 * quantity
                ),
                StrategyLeg.from_moneyness(
                    ot, mid_moneyness + wing_moneyness, forward, maturity, quantity
                ),
            )
        )

    @classmethod
    def from_strikes(
        cls,
        low_strike: float,
        mid_strike: float,
        high_strike: float,
        maturity: datetime,
        forward: float,
        quantity: float = 1.0,
        option_type: OptionType | None = None,
    ) -> Self:
        """Create a butterfly from absolute strikes.

        If option_type is not specified, it is selected automatically based on
        the body moneyness for best liquidity.
        """
        import math

        ot = option_type or _option_type_for_moneyness(math.log(mid_strike / forward))
        return cls(
            legs=(
                StrategyLeg(
                    option_type=ot,
                    quantity=quantity,
                    strike=low_strike,
                    maturity=maturity,
                ),
                StrategyLeg(
                    option_type=ot,
                    quantity=-2.0 * quantity,
                    strike=mid_strike,
                    maturity=maturity,
                ),
                StrategyLeg(
                    option_type=ot,
                    quantity=quantity,
                    strike=high_strike,
                    maturity=maturity,
                ),
            )
        )
