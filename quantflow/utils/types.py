from decimal import Decimal
from typing import Annotated, Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from pydantic import PlainSerializer
from typing_extensions import TypeAlias

Number = Decimal
Float = float | np.floating[Any]
Numbers = int | Float | np.number
NumberType = float | int | str | Number
Vector: TypeAlias = int | float | complex | np.ndarray | pd.Series
FloatArray = Annotated[
    npt.NDArray[np.floating[Any]],
    PlainSerializer(lambda x: x.tolist(), return_type=list),
]
IntArray = npt.NDArray[np.signedinteger[Any]]
BoolArray = npt.NDArray[np.bool_]
FloatArrayLike = Annotated[
    Float | npt.NDArray[np.floating[Any]],
    PlainSerializer(lambda x: x.tolist() if isinstance(x, np.ndarray) else float(x)),
]


def as_number(num: NumberType | None = None) -> Number:
    return Number(0 if num is None else str(num))


def as_float(num: NumberType | None = None) -> float:
    return float(0 if num is None else num)


def as_array(n: Vector) -> np.ndarray:
    """Convert an input into an array"""
    if isinstance(n, int):
        return np.arange(n)
    else:
        return np.asarray(n)
