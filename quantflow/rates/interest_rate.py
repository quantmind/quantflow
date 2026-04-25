from __future__ import annotations

import math
from datetime import datetime
from decimal import Decimal
from typing import Self

from ccy import DayCounter, Period
from pydantic import BaseModel, Field
from typing_extensions import Annotated, Doc

from ..utils.numbers import ONE, ZERO, Number, to_decimal

ROUND_RATE = 7


class Rate(BaseModel, arbitrary_types_allowed=True):
    """Class representing an interest rate with optional compounding frequency"""

    rate: Decimal = Field(
        default=ZERO, description="Interest rate as a decimal (e.g. 0.05 for 5%)"
    )
    day_counter: DayCounter = Field(
        default=DayCounter.ACTACT,
        description="Day count convention to use when calculating time to maturity",
    )
    frequency: Period | None = Field(
        default=None,
        description=(
            "Compounding frequency, when None it is considered as "
            "continuous compounding"
        ),
    )

    @property
    def percent(self) -> Decimal:
        """Interest rate as a percentage"""
        return round(100 * self.rate, ROUND_RATE - 2)

    @property
    def bps(self) -> Decimal:
        """Interest rate as basis points, 1 bps = 0.01% = 0.0001 in decimal"""
        return round(10000 * self.rate, ROUND_RATE - 4)

    @classmethod
    def from_number(
        cls,
        rate: Annotated[Number, Doc("interest rate as a decimal (e.g. 0.05 for 5%)")],
        *,
        frequency: Annotated[
            Period | None,
            Doc(
                "Compounding frequency, when None it is considered as "
                "continuous compounding"
            ),
        ] = None,
        day_counter: Annotated[
            DayCounter, Doc("Day count convention to use")
        ] = DayCounter.ACTACT,
    ) -> Self:
        """Create a Rate instance from a Number"""
        return cls(
            rate=round(to_decimal(rate), ROUND_RATE),
            frequency=frequency,
            day_counter=day_counter,
        )

    @classmethod
    def from_spot_and_forward(
        cls,
        spot: Annotated[Decimal, Doc("Spot price of the underlying asset")],
        forward: Annotated[Decimal, Doc("Forward price of the underlying asset")],
        ref_date: Annotated[datetime, Doc("Reference date for the calculation")],
        maturity_date: Annotated[datetime, Doc("Maturity date for the calculation")],
        *,
        frequency: Annotated[
            Period | None,
            Doc(
                "Compounding frequency, when None it is considered as "
                "continuous compounding"
            ),
        ] = None,
        day_counter: Annotated[
            DayCounter, Doc("Day count convention to use")
        ] = DayCounter.ACTACT,
    ) -> Self:
        """Calculate rate from spot and forward"""
        # use Act/365 for now
        ttm = day_counter.dcf(ref_date, maturity_date)
        if ttm <= 0:
            return cls(frequency=frequency, day_counter=day_counter)
        if frequency is None:
            return cls.from_number(
                rate=math.log(float(forward / spot)) / ttm,
                day_counter=day_counter,
                frequency=frequency,
            )
        else:
            # TODO: implement this
            raise NotImplementedError("Discrete compounding is not implemented yet")

    def discount_factor(self, ref_date: datetime, maturity_date: datetime) -> Decimal:
        """Calculate discount factor from the rate"""
        ttm = self.day_counter.dcf(ref_date, maturity_date)
        if ttm <= 0:
            return ONE
        if self.frequency is None:
            return Decimal(math.exp(-float(self.rate) * ttm))
        else:
            raise NotImplementedError("Discrete compounding is not implemented yet")
