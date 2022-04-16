from decimal import Decimal
from typing import Optional, Union

import numpy as np

Number = Decimal
Numbers = Union[int, float, np.number]
NumberType = Union[float, int, str, Number]
Vector = Union[int, float, complex, np.ndarray]


def as_number(num: Optional[NumberType] = None) -> Number:
    return Number(0 if num is None else str(num))


def as_float(num: Optional[NumberType] = None) -> float:
    return float(0 if num is None else num)


def as_array(n: Vector) -> np.ndarray:
    """Convert an input into an array"""
    if isinstance(n, int):
        return np.arange(n)
    else:
        return np.asarray(n)
