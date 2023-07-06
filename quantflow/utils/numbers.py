from decimal import Decimal

Number = Decimal | float | int | str
ZERO = Decimal(0)
ONE = Decimal(1)


def to_decimal(value: Number) -> Decimal:
    return Decimal(str(value)) if not isinstance(value, Decimal) else value
