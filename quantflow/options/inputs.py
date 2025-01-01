from __future__ import annotations

import enum
from datetime import datetime
from decimal import Decimal
from typing import Generic, TypeVar

from pydantic import BaseModel

P = TypeVar("P")


class VolSecurityType(enum.StrEnum):
    """Type of security for the volatility surface"""

    spot = enum.auto()
    forward = enum.auto()
    option = enum.auto()

    def vol_surface_type(self) -> VolSecurityType:
        return self


class VolSurfaceInput(BaseModel, Generic[P]):
    bid: P
    ask: P


class OptionInput(BaseModel):
    price: Decimal
    strike: Decimal
    maturity: datetime
    call: bool


class SpotInput(VolSurfaceInput[Decimal]):
    security_type: VolSecurityType = VolSecurityType.spot


class ForwardInput(VolSurfaceInput[Decimal]):
    maturity: datetime
    security_type: VolSecurityType = VolSecurityType.forward


class OptionSidesInput(VolSurfaceInput[OptionInput]):
    security_type: VolSecurityType = VolSecurityType.option


class VolSurfaceInputs(BaseModel):
    ref_date: datetime
    inputs: list[ForwardInput | SpotInput | OptionSidesInput]
