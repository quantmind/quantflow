from __future__ import annotations

import math
from datetime import timedelta
from decimal import Decimal
from typing import NamedTuple

from .numbers import to_decimal


class Rate(NamedTuple):
    rate: Decimal = Decimal("0")
    frequency: int = 0

    @classmethod
    def from_number(cls, rate: float, frequency: int = 0) -> Rate:
        return cls(rate=round(to_decimal(rate), 7), frequency=frequency)

    @property
    def percent(self) -> Decimal:
        return round(100 * self.rate, 5)

    @property
    def bps(self) -> Decimal:
        return round(10000 * self.rate, 3)


def rate_from_spot_and_forward(
    spot: Decimal, forward: Decimal, maturity: timedelta, frequency: int = 0
) -> Rate:
    """Calculate rate from spot and forward

    Args:
        basis: basis point
        maturity: maturity in years
        frequency: number of payments per year - 0 for continuous compounding

    Returns:
        Rate
    """
    # use Act/365 for now
    ttm = maturity.days / 365
    if ttm <= 0:
        return Rate(frequency=frequency)
    if frequency == 0:
        return Rate.from_number(
            rate=math.log(forward / spot) / ttm, frequency=frequency
        )
    else:
        # TODO: implement this
        raise NotImplementedError
