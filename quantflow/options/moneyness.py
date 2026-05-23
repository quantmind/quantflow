"""Moneyness and log-strike conversion utilities.

This module provides the canonical conversions between absolute strikes,
log-strikes, and time-scaled moneyness used throughout the library.

See the [glossary](../glossary.md#moneyness) for definitions.
"""

from __future__ import annotations

import math
from decimal import Decimal

import numpy as np
from typing_extensions import Annotated, Doc

from quantflow.utils.numbers import to_decimal
from quantflow.utils.types import FloatArrayLike


def moneyness_from_log_strike(
    log_strike: Annotated[FloatArrayLike, Doc("Log-strike")],
    ttm: Annotated[FloatArrayLike, Doc("Time to maturity in years")],
) -> FloatArrayLike:
    r"""Convert log-strike $k$ to moneyness $m$.

    \begin{equation}
        m = \frac{k}{\sqrt{\tau}}
    \end{equation}
    """
    return log_strike / np.sqrt(ttm)


def log_strike_from_moneyness(
    moneyness: Annotated[FloatArrayLike, Doc("Time-scaled moneyness")],
    ttm: Annotated[FloatArrayLike, Doc("Time to maturity in years")],
) -> FloatArrayLike:
    r"""Convert time-scaled moneyness to log-strike.

    \begin{equation}
        k = m \sqrt{\tau}
    \end{equation}
    """
    return moneyness * np.sqrt(ttm)


def strike_from_moneyness(
    moneyness: Annotated[float, Doc("Time-scaled moneyness")],
    ttm: Annotated[float, Doc("Time to maturity in years")],
    forward: Annotated[float, Doc("Forward price of the underlying")],
) -> Decimal:
    r"""Convert time-scaled moneyness to an absolute strike price.

    \begin{equation}
        K = F \exp\left(m \sqrt{\tau}\right)
    \end{equation}
    """
    return to_decimal(forward * math.exp(moneyness * math.sqrt(ttm)))


def strike_from_log_strike(
    log_strike: Annotated[float, Doc("Log-strike")],
    forward: Annotated[float, Doc("Forward price of the underlying")],
) -> Decimal:
    r"""Convert log-strike to an absolute strike price.

    \begin{equation}
        K = F \exp(k)
    \end{equation}
    """
    return to_decimal(forward * math.exp(log_strike))


def log_strike_from_strike(
    strike: Annotated[float | Decimal, Doc("Absolute strike price")],
    forward: Annotated[float | Decimal, Doc("Forward price of the underlying")],
) -> float:
    r"""Convert an absolute strike to log-strike.

    \begin{equation}
        k = \ln\frac{K}{F}
    \end{equation}
    """
    return math.log(float(strike) / float(forward))
