from __future__ import annotations

from datetime import datetime
from typing import ClassVar

from typing_extensions import Self

from quantflow.options.inputs import OptionMetadata, OptionType
from quantflow.options.moneyness import log_strike_from_strike
from quantflow.utils.numbers import Number, to_decimal

from .base import Strategy, StrategyError, StrategyLeg, load_description


def _option_type_for_log_strike(mid_log_strike: float) -> OptionType:
    """Select option type based on body position relative to ATM.

    Calls for body above ATM, puts for body below ATM, calls at ATM.
    """
    return OptionType.PUT if mid_log_strike < 0 else OptionType.CALL


class Butterfly(Strategy, frozen=True):
    """Three-strike strategy: long wings, short body.

    Long butterfly when quantity > 0, short butterfly when quantity < 0.
    Can be constructed with calls or puts, both are equivalent by put-call parity.
    """

    description: ClassVar[str] = load_description("butterfly.md")

    @classmethod
    def from_strikes(
        cls,
        low_strike: Number,
        mid_strike: Number,
        high_strike: Number,
        maturity: datetime,
        forward: Number,
        quantity: Number = 1.0,
        option_type: OptionType | None = None,
    ) -> Self:
        """Create a butterfly from absolute strikes.

        If option_type is not specified, it is selected automatically based on
        the body position relative to the forward for best liquidity.
        """
        low = to_decimal(low_strike)
        mid = to_decimal(mid_strike)
        high = to_decimal(high_strike)
        fwd = to_decimal(forward)
        if not (low < mid < high):
            raise StrategyError(
                "Strikes must be strictly increasing: low < mid < high."
            )
        q = to_decimal(quantity)
        ot = option_type or _option_type_for_log_strike(
            log_strike_from_strike(mid, fwd)
        )
        return cls(
            legs=(
                StrategyLeg(
                    meta=OptionMetadata(
                        option_type=ot,
                        strike=low,
                        maturity=maturity,
                    ),
                    quantity=q,
                ),
                StrategyLeg(
                    meta=OptionMetadata(
                        option_type=ot,
                        strike=mid,
                        maturity=maturity,
                    ),
                    quantity=to_decimal(-2) * q,
                ),
                StrategyLeg(
                    meta=OptionMetadata(
                        option_type=ot,
                        strike=high,
                        maturity=maturity,
                    ),
                    quantity=q,
                ),
            )
        )
