from decimal import Decimal

Number = Decimal | float | int | str
ZERO = Decimal(0)
ONE = Decimal(1)


def to_decimal(value: Number) -> Decimal:
    return Decimal(str(value)) if not isinstance(value, Decimal) else value


def sigfig(value: Number, sig: int = 5) -> str:
    """round a number to the given significant digit"""
    return f"%.{sig}g" % to_decimal(value)
