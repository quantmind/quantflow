from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import ClassVar

from typing_extensions import Self

from quantflow.options.inputs import OptionMetadata, OptionType
from quantflow.utils.numbers import Number, to_decimal

from .base import Strategy, StrategyError, StrategyLeg, load_description

Positions = tuple[tuple[Decimal, Decimal], ...]


class Ladder(Strategy, frozen=True):
    description: ClassVar[str] = load_description("ladder.md")

    @property
    def option_type(self) -> OptionType:
        """Option type of the ladder."""
        return self.legs[0].meta.option_type

    @classmethod
    def _from_positions(
        cls, option_type: OptionType, maturity: datetime, positions: Positions
    ) -> Self:
        return cls(
            legs=tuple(
                StrategyLeg(
                    meta=OptionMetadata(
                        option_type=option_type,
                        strike=strike,
                        maturity=maturity,
                    ),
                    quantity=quantity,
                )
                for strike, quantity in positions
            )
        )

    @classmethod
    def call(
        cls,
        low_strike: Number,
        mid_strike: Number,
        high_strike: Number,
        maturity: datetime,
        quantity: Number = 1.0,
    ) -> Self:
        """Long call at low_strike, short calls at mid_strike and high_strike.

        When mid_strike equals high_strike the two short legs collapse onto a
        single leg of twice the quantity and the ladder becomes a 1 by 2 call
        ratio spread.
        """
        low = to_decimal(low_strike)
        mid = to_decimal(mid_strike)
        high = to_decimal(high_strike)
        if not (low < mid <= high):
            raise StrategyError(
                "Strikes must satisfy low_strike < mid_strike <= high_strike."
            )
        q = to_decimal(quantity)
        positions: Positions = (
            ((low, q), (mid, to_decimal(-2) * q))
            if mid == high
            else ((low, q), (mid, -q), (high, -q))
        )
        return cls._from_positions(OptionType.CALL, maturity, positions)

    @classmethod
    def put(
        cls,
        low_strike: Number,
        mid_strike: Number,
        high_strike: Number,
        maturity: datetime,
        quantity: Number = 1.0,
    ) -> Self:
        """Long put at high_strike, short puts at mid_strike and low_strike.

        When low_strike equals mid_strike the two short legs collapse onto a
        single leg of twice the quantity and the ladder becomes a 1 by 2 put
        ratio spread.
        """
        low = to_decimal(low_strike)
        mid = to_decimal(mid_strike)
        high = to_decimal(high_strike)
        if not (low <= mid < high):
            raise StrategyError(
                "Strikes must satisfy low_strike <= mid_strike < high_strike."
            )
        q = to_decimal(quantity)
        positions: Positions = (
            ((high, q), (mid, to_decimal(-2) * q))
            if low == mid
            else ((high, q), (mid, -q), (low, -q))
        )
        return cls._from_positions(OptionType.PUT, maturity, positions)
