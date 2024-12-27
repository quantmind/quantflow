import math
from decimal import Decimal
from enum import IntEnum, auto, unique

Number = Decimal | float | int | str
ZERO = Decimal(0)
ONE = Decimal(1)


@unique
class Rounding(IntEnum):
    ZERO = auto()
    UP = auto()
    DOWN = auto()


def to_decimal(value: Number) -> Decimal:
    return Decimal(str(value)) if not isinstance(value, Decimal) else value


def sigfig(value: Number, sig: int = 5) -> str:
    """round a number to the given significant digit"""
    return f"%.{sig}g" % to_decimal(value)


def normalize_decimal(d: Decimal) -> Decimal:
    return d.quantize(ONE) if d == d.to_integral() else d.normalize()


def round_to_step(
    amount_to_adjust: Number,
    rounding_precision: Number,
    rounding: Rounding = Rounding.ZERO,
) -> Decimal:
    amount = normalize_decimal(to_decimal(amount_to_adjust))
    precision = normalize_decimal(to_decimal(rounding_precision))
    # Quantize
    match rounding:
        case Rounding.ZERO:
            stepped_amount = precision * round(amount / precision)
        case Rounding.UP:
            stepped_amount = precision * math.ceil(amount / precision)
        case Rounding.DOWN:
            stepped_amount = precision * math.floor(amount / precision)
    return stepped_amount
