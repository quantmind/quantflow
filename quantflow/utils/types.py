from decimal import Decimal
from typing import Any, Optional, Union

import numpy as np
import numpy.typing as npt
import pandas as pd

Number = Decimal
Float = float | np.floating[Any]
Numbers = Union[int, Float, np.number]
NumberType = Union[float, int, str, Number]
Vector = Union[int, float, complex, np.ndarray, pd.Series]
FloatArray = npt.NDArray[np.floating[Any]]
IntArray = npt.NDArray[np.signedinteger[Any]]
FloatArrayLike = FloatArray | float


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
