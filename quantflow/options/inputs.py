from __future__ import annotations

import enum
from datetime import datetime
from typing import Self, TypeVar

from pydantic import BaseModel, Field

from quantflow.utils.numbers import ZERO, DecimalNumber

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

    def call_put(self) -> int:
        """Return 1 for call options and -1 for put options"""
        return 1 if self is OptionType.call else -1


class VolSecurityType(enum.StrEnum):
    """Type of security for the volatility surface"""

    spot = enum.auto()
    forward = enum.auto()
    option = enum.auto()


class VolSurfaceSecurity(BaseModel):
    def vol_surface_type(self) -> VolSecurityType:
        raise NotImplementedError("vol_surface_type must be implemented by subclasses")


class DefaultVolSecurity(VolSurfaceSecurity):
    security_type: VolSecurityType = Field(
        default=VolSecurityType.spot,
        description="Type of security for the volatility surface",
    )

    def vol_surface_type(self) -> VolSecurityType:
        return self.security_type

    @classmethod
    def spot(cls) -> Self:
        return cls(security_type=VolSecurityType.spot)

    @classmethod
    def forward(cls) -> Self:
        return cls(security_type=VolSecurityType.forward)

    @classmethod
    def option(cls) -> Self:
        return cls(security_type=VolSecurityType.option)


class VolSurfaceInput(BaseModel):
    """Base class for volatility surface inputs"""

    bid: DecimalNumber = Field(description="Bid price of the security")
    ask: DecimalNumber = Field(description="Ask price of the security")
    open_interest: DecimalNumber = Field(
        default=ZERO, description="Open interest of the security"
    )
    volume: DecimalNumber = Field(default=ZERO, description="Volume of the security")


class SpotInput(VolSurfaceInput):
    """Input data for a spot contract in the volatility surface"""

    security_type: VolSecurityType = Field(
        default=VolSecurityType.spot,
        description="Type of security for the volatility surface",
    )


class ForwardInput(VolSurfaceInput):
    """Input data for a forward contract in the volatility surface"""

    maturity: datetime = Field(description="Expiry date of the forward contract")
    security_type: VolSecurityType = Field(
        default=VolSecurityType.forward,
        description="Type of security for the volatility surface",
    )


class OptionInput(VolSurfaceInput):
    """Input data for an option in the volatility surface"""

    strike: DecimalNumber = Field(description="Strike price of the option")
    maturity: datetime = Field(description="Expiry date of the option")
    option_type: OptionType = Field(description="Type of the option - call or put")
    security_type: VolSecurityType = Field(
        default=VolSecurityType.option,
        description="Type of security for the volatility surface",
    )
    iv_bid: DecimalNumber | None = Field(
        default=None,
        description=(
            "Implied volatility based on the bid price as decimal number "
            "(e.g. 0.2 for 20%)"
        ),
    )
    iv_ask: DecimalNumber | None = Field(
        default=None,
        description=(
            "Implied volatility based on the ask price as decimal number "
            "(e.g. 0.2 for 20%)"
        ),
    )
    inverse: bool = Field(
        default=True,
        description=(
            "Whether the security is inverse (i.e. quoted in terms of the underlying) "
            "or not (i.e. quoted in terms of the quote currency)"
        ),
    )


class VolSurfaceInputs(BaseModel):
    """Class representing the inputs for a volatility surface"""

    asset: str = Field(description="Underlying asset of the volatility surface")
    ref_date: datetime = Field(description="Reference date for the volatility surface")
    inputs: list[ForwardInput | SpotInput | OptionInput] = Field(
        description="List of inputs for the volatility surface"
    )
