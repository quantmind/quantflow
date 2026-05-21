from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike
from pydantic import Field
from scipy.optimize import least_squares
from typing_extensions import Self

from quantflow.sp.ou import Vasicek
from quantflow.sp.wiener import WienerProcess
from quantflow.utils.numbers import ZERO, DecimalNumber
from quantflow.utils.types import FloatArrayLike

from .yield_curve import YieldCurve


class VasicekCurve(YieldCurve):
    """Class representing a Vasicek yield curve"""

    curve_type: Literal["vasicek_curve"] = "vasicek_curve"
    rate: DecimalNumber = Field(description=r"Initial value $x_0$")
    kappa: DecimalNumber = Field(gt=ZERO, description=r"Mean reversion speed $\kappa$")
    theta: DecimalNumber = Field(description=r"Mean level $\theta$")
    sigma: DecimalNumber = Field(ge=ZERO, description=r"Volatility $\sigma$")

    def process(self) -> Vasicek:
        return Vasicek(
            rate=float(self.rate),
            kappa=float(self.kappa),
            theta=float(self.theta),
            bdlp=WienerProcess(sigma=float(self.sigma)),
        )

    def instanteous_forward_rate(self, ttm: FloatArrayLike) -> FloatArrayLike:
        r"""Calculate the instantaneous forward rate."""
        arr = np.asarray(ttm, dtype=float)
        ttma = np.maximum(arr, 0.0)
        kappa = float(self.kappa)
        theta = float(self.theta)
        sigma = float(self.sigma)
        rate = float(self.rate)
        s2 = sigma * sigma
        et = np.exp(-kappa * ttma)
        b = (1.0 - et) / kappa
        fwd = rate * et + theta * (1.0 - et) - s2 / (2.0 * kappa) * b * et
        return fwd if fwd.ndim > 0 else float(fwd)

    def discount_factor(self, ttm: FloatArrayLike) -> FloatArrayLike:
        r"""Calculate the discount factor for a given time to maturity."""
        arr = np.asarray(ttm, dtype=float)
        ttma = np.maximum(arr, 0.0)
        kappa = float(self.kappa)
        theta = float(self.theta)
        sigma = float(self.sigma)
        rate = float(self.rate)
        s2 = sigma * sigma
        b = (1.0 - np.exp(-kappa * ttma)) / kappa
        a = (theta - s2 / (2.0 * kappa * kappa)) * (b - ttma) + s2 * b * b / (
            4.0 * kappa
        )
        df = np.exp(a - rate * b)
        return df if df.ndim > 0 else float(df)

    @classmethod
    def calibrate(cls, ttm: ArrayLike, rates: ArrayLike) -> Self:
        """Fit the Vasicek curve to continuously compounded rates via least squares."""
        ttm_arr = np.asarray(ttm, dtype=float)
        rates_arr = np.asarray(rates, dtype=float)

        def residuals(params: np.ndarray) -> np.ndarray:
            curve = cls(
                rate=params[0], kappa=params[1], theta=params[2], sigma=params[3]
            )
            df = np.asarray(curve.discount_factor(ttm_arr), dtype=float)
            fitted = -np.log(df) / ttm_arr
            return fitted - rates_arr

        x0 = np.array([rates_arr[0], 1.0, rates_arr[-1], 0.01])
        result = least_squares(
            residuals, x0, bounds=([-1.0, 1e-4, -1.0, 0.0], [1.0, 50.0, 1.0, 1.0])
        )
        r, k, th, s = result.x
        return cls(rate=r, kappa=k, theta=th, sigma=s)
