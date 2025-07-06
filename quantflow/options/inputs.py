from __future__ import annotations

import enum
from datetime import datetime
from decimal import Decimal
from typing import TypeVar

from pydantic import BaseModel

from quantflow.utils.numbers import ZERO

P = TypeVar("P")


class Side(enum.StrEnum):
    """Side of the market"""

    bid = enum.auto()
    ask = enum.auto()


class OptionType(enum.StrEnum):
    """Type of option"""

    call = enum.auto()
    put = enum.auto()

    def is_call(self) -> bool:
        return self is OptionType.call

    def is_put(self) -> bool:
        return self is OptionType.put


class VolSecurityType(enum.StrEnum):
    """Type of security for the volatility surface"""

    spot = enum.auto()
    forward = enum.auto()
    option = enum.auto()

    def vol_surface_type(self) -> VolSecurityType:
        return self


class VolSurfaceInput(BaseModel):
    bid: Decimal
    ask: Decimal
    open_interest: Decimal = ZERO
    volume: Decimal = ZERO


class SpotInput(VolSurfaceInput):
    security_type: VolSecurityType = VolSecurityType.spot


class ForwardInput(VolSurfaceInput):
    maturity: datetime
    security_type: VolSecurityType = VolSecurityType.forward


class OptionInput(VolSurfaceInput):
    strike: Decimal
    maturity: datetime
    option_type: OptionType
    security_type: VolSecurityType = VolSecurityType.option


class VolSurfaceInputs(BaseModel):
    asset: str
    ref_date: datetime
    inputs: list[ForwardInput | SpotInput | OptionInput]
