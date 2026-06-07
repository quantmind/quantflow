from __future__ import annotations

import enum
from datetime import datetime
from decimal import Decimal
from typing import Self, TypeVar

from pydantic import BaseModel, Field
from typing_extensions import Annotated, Doc

from quantflow.rates import AnyYieldCurve
from quantflow.utils.numbers import DecimalNumber
from quantflow.utils.price import PriceVolume

P = TypeVar("P")


class Side(enum.StrEnum):
    """Side of the market"""

    BID = enum.auto()
    ASK = enum.auto()


class OptionType(enum.StrEnum):
    """Type of option"""

    CALL = enum.auto()
    PUT = enum.auto()

    def is_call(self) -> bool:
        return self is OptionType.CALL

    def is_put(self) -> bool:
        return self is OptionType.PUT

    def call_put(self) -> int:
        """Return 1 for call options and -1 for put options"""
        return 1 if self is OptionType.CALL else -1


class OptionMetadata(BaseModel):
    """Represents the metadata of an option, including its strike, type, maturity,
    and other relevant information."""

    strike: DecimalNumber = Field(description="Strike price of the option")
    option_type: OptionType = Field(description="Type of the option, call or put")
    maturity: datetime = Field(description="Maturity date of the option")
    inverse: bool = Field(
        default=True,
        description=(
            "Whether the option is an inverse option (i.e. quoted in terms of the "
            "underlying) or not (i.e. quoted in terms of the quote currency)"
        ),
    )

    def is_in_the_money(self, forward: Decimal) -> bool:
        """Check if the option is in the money given the forward price"""
        if self.option_type.is_call():
            return self.strike < forward
        else:
            return self.strike > forward


class VolSecurityType(enum.StrEnum):
    """Type of security for the volatility surface"""

    SPOT = enum.auto()
    FORWARD = enum.auto()
    OPTION = enum.auto()


class VolSurfaceSecurity(BaseModel):
    """Base class for Volatility Surface Securities"""

    def vol_surface_type(self) -> VolSecurityType:
        raise NotImplementedError("vol_surface_type must be implemented by subclasses")

    @classmethod
    def forward(cls) -> Self:
        """Create a forward security for the volatility surface"""
        raise NotImplementedError("forward_input must be implemented by subclasses")


class DefaultVolSecurity(VolSurfaceSecurity):
    security_type: VolSecurityType = Field(
        default=VolSecurityType.SPOT,
        description="Type of security for the volatility surface",
    )

    def vol_surface_type(self) -> VolSecurityType:
        return self.security_type

    @classmethod
    def spot(cls) -> Self:
        return cls(security_type=VolSecurityType.SPOT)

    @classmethod
    def forward(cls) -> Self:
        return cls(security_type=VolSecurityType.FORWARD)

    @classmethod
    def option(cls) -> Self:
        return cls(security_type=VolSecurityType.OPTION)


class SpotInput(PriceVolume):
    """Input data for a spot contract in the volatility surface"""

    security_type: VolSecurityType = Field(
        default=VolSecurityType.SPOT,
        description="Type of security for the volatility surface",
    )


class ForwardInput(PriceVolume):
    """Input data for a forward contract in the volatility surface"""

    maturity: datetime = Field(description="Expiry date of the forward contract")
    security_type: VolSecurityType = Field(
        default=VolSecurityType.FORWARD,
        description="Type of security for the volatility surface",
    )


class OptionInput(PriceVolume, OptionMetadata):
    """Input data for an option in the volatility surface"""

    security_type: VolSecurityType = Field(
        default=VolSecurityType.OPTION,
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


VolSurfaceInput = Annotated[
    SpotInput | ForwardInput | OptionInput,
    Doc("Input data for a security in the volatility surface"),
]


class VolSurfaceInputs(BaseModel):
    """Class representing the inputs for a volatility surface"""

    asset: str = Field(description="Underlying asset of the volatility surface")
    asset_curve: AnyYieldCurve = Field(description="Asset yield curve")
    quote_curve: AnyYieldCurve = Field(description="Quote yield curve")
    inputs: list[ForwardInput | SpotInput | OptionInput] = Field(
        description="List of inputs for the volatility surface"
    )
