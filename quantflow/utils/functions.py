import math

import numpy as np
from scipy.integrate import quad

_factorial = [math.factorial(k) for k in range(51)]


@np.vectorize
def factorial(n: int) -> float:
    """Cached factorial function"""
    if n < 0:
        return np.inf
    return _factorial[n] if n < len(_factorial) else math.factorial(n)


def debye(n: int, x: float) -> float:
    xn = n * x ** (-n)
    return xn * quad(_debye, 0, x, args=(n,))[0]


def _debye(t: float, n: int) -> float:
    return t**n / (np.exp(t) - 1)
