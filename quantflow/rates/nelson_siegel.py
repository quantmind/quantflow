from __future__ import annotations

from decimal import Decimal

import numpy as np
from numpy.typing import ArrayLike
from pydantic import Field
from scipy.optimize import minimize_scalar
from typing_extensions import Annotated, Doc

from quantflow.utils.numbers import ONE, Number, to_decimal

from .yield_curve import YieldCurve


class NelsonSiegel(YieldCurve):
    r"""Class representing a Nelson-Siegel yield curve

    The Nelson-Siegel model is a popular parametric model for fitting
    the term structure of interest rates.
    It is defined by the following formula for the instantaneous forward rate:

    \begin{equation}
        f(\tau) = \beta_1 + \beta_2 e^{-\lambda \tau}
        + \beta_3 \lambda \tau e^{-\lambda \tau}
    \end{equation}

    where $\tau$ is the time to maturity, $\beta_1$ is the level parameter,
    $\beta_2$ is the slope parameter,
    $\beta_3$ is the curvature parameter and $\lambda$ is the decay factor.
    """

    beta1: Decimal = Field(..., description="Level parameter")
    beta2: Decimal = Field(..., description="Slope parameter")
    beta3: Decimal = Field(..., description="Curvature parameter")
    lambda_: Decimal = Field(..., description="Decay factor")

    def instanteous_forward_rate(self, ttm: Number) -> Decimal:
        ttmd = to_decimal(ttm)
        if ttmd <= 0:
            return self.beta1 + self.beta2
        else:
            tt = ttmd * self.lambda_
            et = (-tt).exp()
            return self.beta1 + self.beta2 * et + self.beta3 * tt * et

    def discount_factor(self, ttm: Number) -> Decimal:
        r"""Calculate the discount factor for a given time to maturity.

        The discount factor is calculated using the formula:

        \begin{align*}
            D(\tau) &= e^{-r(\tau) \tau} \\
            r(\tau) &= \beta_1 + \beta_2 \frac{1 - e^{-\lambda \tau}}
            {\lambda \tau} + \beta_3
            \left(\frac{1 - e^{-\lambda \tau}}{\lambda \tau}
              - e^{-\lambda \tau}\right)
        \end{align*}
        """
        ttmd = to_decimal(ttm)
        if ttmd <= 0:
            return ONE
        else:
            tt = ttmd * self.lambda_
            et = (-tt).exp()
            ett = (1 - et) / tt
            zero_coupon_rate = self.beta1 + self.beta2 * ett + self.beta3 * (ett - et)
            return (-zero_coupon_rate * ttmd).exp()

    @classmethod
    def fit(
        cls,
        ttm: Annotated[
            ArrayLike,
            Doc("times to maturity in years (1-D, length >= 3)"),
        ],
        rates: Annotated[
            ArrayLike, Doc("observed zero-coupon rates, same length as ttm")
        ],
        lambda_bounds: Annotated[
            tuple[float, float],
            Doc("search bounds for the decay parameter $\\lambda$"),
        ] = (0.01, 10.0),
    ) -> NelsonSiegel:
        r"""Fit a Nelson-Siegel curve to observed zero-coupon rates.

        Uses a profile OLS approach: for each candidate $\lambda$ the betas are
        solved exactly via least squares, so only a 1-D scalar minimisation over
        $\lambda$ is needed.
        """
        ttm_arr = np.asarray(ttm, dtype=float)
        rates_arr = np.asarray(rates, dtype=float)
        result = minimize_scalar(
            _rss,
            bounds=lambda_bounds,
            method="bounded",
            args=(ttm_arr, rates_arr),
        )
        lam: float = result.x
        b1, b2, b3 = _ols_betas(ttm_arr, rates_arr, lam)
        return cls(
            beta1=Decimal(str(round(b1, 10))),
            beta2=Decimal(str(round(b2, 10))),
            beta3=Decimal(str(round(b3, 10))),
            lambda_=Decimal(str(round(lam, 10))),
        )


def _design_matrix(ttm: np.ndarray, lam: float) -> np.ndarray:
    lt = lam * ttm
    with np.errstate(divide="ignore", invalid="ignore"):
        ett = np.where(lt > 1e-10, (1.0 - np.exp(-lt)) / lt, 1.0 - lt / 2.0)
    return np.column_stack([np.ones_like(ttm), ett, ett - np.exp(-lt)])


def _ols_betas(ttm: np.ndarray, rates: np.ndarray, lam: float) -> np.ndarray:
    betas, _, _, _ = np.linalg.lstsq(_design_matrix(ttm, lam), rates, rcond=None)
    return betas


def _rss(lam: float, ttm: np.ndarray, rates: np.ndarray) -> float:
    residuals = rates - _design_matrix(ttm, lam) @ _ols_betas(ttm, rates, lam)
    return float(np.dot(residuals, residuals))
