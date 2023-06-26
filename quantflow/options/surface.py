from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal


@dataclass
class Price:
    bid: Decimal
    ask: Decimal
    black_vol_bid: Decimal
    black_vol_ask: Decimal


@dataclass
class Strike:
    strike: Decimal
    call: Price
    put: Price


@dataclass
class VolCrossSection:
    forward: Decimal
    """Forward price of the underlying asset at the time of the cross section"""
    strikes: list[Strike]
    """List of strikes and their corresponding option prices"""


@dataclass
class VolSurface:
    maturities: list[VolCrossSection]
