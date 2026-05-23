from __future__ import annotations

from decimal import Decimal
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike
from pydantic import Field
from scipy.optimize import Bounds, minimize_scalar
from typing_extensions import Annotated, Doc, Self

from quantflow.utils.types import FloatArray, FloatArrayLike, maybe_float

from .options import YieldCurveCalibration
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

    curve_type: Literal["nelson_siegel"] = "nelson_siegel"
    beta1: Decimal = Field(default=Decimal(0), description="Level parameter")
    beta2: Decimal = Field(default=Decimal(0), description="Slope parameter")
    beta3: Decimal = Field(default=Decimal(0), description="Curvature parameter")
    lambda_: Decimal = Field(default=Decimal(1), description="Decay factor")

    def calibrator(self) -> NelsonSiegelCalibration:
        """Return a [NelsonSiegelCalibration][...NelsonSiegelCalibration] wrapping
        this curve."""
        return NelsonSiegelCalibration(yield_curve=self)

    def instantaneous_forward_rate(self, ttm: FloatArrayLike) -> FloatArrayLike:
        b1, b2, b3, lam = (
            float(self.beta1),
            float(self.beta2),
            float(self.beta3),
            float(self.lambda_),
        )
        ttm_ = np.maximum(np.asarray(ttm, dtype=float), 0.0)
        lt = lam * ttm_
        et = np.exp(-lt)
        return maybe_float(b1 + b2 * et + b3 * lt * et)

    def discount_factor(self, ttm: FloatArrayLike) -> FloatArrayLike:
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
        ttma = np.maximum(np.asarray(ttm, dtype=float), 0.0)
        tt = ttma * float(self.lambda_)
        et = np.exp(-tt)
        with np.errstate(divide="ignore", invalid="ignore"):
            ett = np.where(tt > 1e-10, (1 - et) / tt, 1.0)
        zero_coupon_rate = (
            float(self.beta1) + float(self.beta2) * ett + float(self.beta3) * (ett - et)
        )
        df = np.exp(-zero_coupon_rate * ttma)
        return maybe_float(df)

    def jacobian(self, ttm: FloatArrayLike) -> FloatArray:
        r"""Analytical Jacobian of discount factors w.r.t. params.

        Params order: $[\beta_1, \beta_2, \beta_3, \lambda]$. Shape: (len(ttm), 4).
        """
        b2, b3, lam = float(self.beta2), float(self.beta3), float(self.lambda_)
        ttma = np.maximum(np.asarray(ttm, dtype=float), 0.0)
        lt = lam * ttma
        et = np.exp(-lt)
        with np.errstate(divide="ignore", invalid="ignore"):
            ett = np.where(lt > 1e-10, (1.0 - et) / lt, 1.0 - lt / 2.0)
        a_term = ttma * ett
        b_term = a_term - ttma * et
        zero_rate = float(self.beta1) + b2 * ett + b3 * (ett - et)
        df = np.exp(-zero_rate * ttma)
        with np.errstate(divide="ignore", invalid="ignore"):
            da_dlam = np.where(lam > 1e-10, ttma * et / lam - a_term / lam, 0.0)
        db_dlam = da_dlam + ttma**2 * et
        return np.column_stack(
            [
                -ttma * df,
                -a_term * df,
                -b_term * df,
                -df * (b2 * da_dlam + b3 * db_dlam),
            ]
        )

    @classmethod
    def calibrate(
        cls,
        ttm: Annotated[
            ArrayLike,
            Doc("times to maturity in years (1-D, length >= 3)"),
        ],
        rates: Annotated[
            ArrayLike, Doc("observed continuously compounded rates, same length as ttm")
        ],
        lambda_bounds: Annotated[
            tuple[float, float],
            Doc("search bounds for the decay parameter $\\lambda$"),
        ] = (0.01, 10.0),
    ) -> Self:
        r"""Fit a Nelson-Siegel curve to observed continuously compounded rates.

        Uses a profile OLS approach: for each candidate $\lambda$ the betas are
        solved exactly via least squares, so only a 1-D scalar minimisation over
        $\lambda$ is needed.

        Observations whose rates deviate by more than 3 robust standard deviations
        (MAD-scaled) from the median are excluded before fitting, making the result
        robust to a small number of bad parity observations.
        """
        ttm_arr = np.asarray(ttm, dtype=float)
        rates_arr = np.asarray(rates, dtype=float)
        mask = _mad_filter(rates_arr)
        fit_ttm = ttm_arr[mask] if mask.sum() >= 3 else ttm_arr
        fit_rates = rates_arr[mask] if mask.sum() >= 3 else rates_arr
        # Grid search to avoid local minima, then refine with bounded minimisation
        lo, hi = lambda_bounds
        grid = np.linspace(lo, hi, 50)
        rss_values = [_rss(lam, fit_ttm, fit_rates) for lam in grid]
        best_idx = int(np.argmin(rss_values))
        # Refine around the best grid point
        refine_lo = grid[max(best_idx - 1, 0)]
        refine_hi = grid[min(best_idx + 1, len(grid) - 1)]
        result = minimize_scalar(
            _rss,
            bounds=(refine_lo, refine_hi),
            method="bounded",
            args=(fit_ttm, fit_rates),
        )
        lam: float = result.x
        b1, b2, b3 = _ols_betas(fit_ttm, fit_rates, lam)
        return cls(
            beta1=Decimal(str(round(b1, 10))),
            beta2=Decimal(str(round(b2, 10))),
            beta3=Decimal(str(round(b3, 10))),
            lambda_=Decimal(str(round(lam, 10))),
        )


class NelsonSiegelCalibration(YieldCurveCalibration[NelsonSiegel]):
    """Calibration wrapper for a Nelson-Siegel yield curve."""

    beta_bounds: tuple[float, float] = Field(
        default=(-0.5, 0.5),
        description="Lower and upper bounds for beta parameters",
    )
    lambda_bounds: tuple[float, float] = Field(
        default=(0.01, 10.0),
        description="Lower and upper bounds for the decay parameter",
    )

    def get_params(self) -> FloatArray:
        ns = self.yield_curve
        return np.array(
            [float(ns.beta1), float(ns.beta2), float(ns.beta3), float(ns.lambda_)]
        )

    def set_params(self, params: FloatArray) -> None:
        b1, b2, b3, lam = params
        self.yield_curve.beta1 = Decimal(str(round(float(b1), 10)))
        self.yield_curve.beta2 = Decimal(str(round(float(b2), 10)))
        self.yield_curve.beta3 = Decimal(str(round(float(b3), 10)))
        self.yield_curve.lambda_ = Decimal(str(round(float(lam), 10)))

    def get_bounds(self) -> Bounds:
        lo, hi = self.beta_bounds
        lam_lo, lam_hi = self.lambda_bounds
        return Bounds([lo, lo, lo, lam_lo], [hi, hi, hi, lam_hi])

    def calibrate(
        self,
        ttm: Annotated[ArrayLike, Doc("Times to maturity in years.")],
        target: Annotated[
            ArrayLike, Doc("Target discount factors, same length as ttm.")
        ],
    ) -> NelsonSiegel:
        """Fit the curve using the fast profile-OLS solver.

        Drop times to maturity <= 1 day (if any) before fitting, as these are often
        dominated by noise and can cause instability in the fit.
        """
        ttm_ = np.asarray(ttm, dtype=float)
        mask = ttm_ >= 1 / 365
        rates = -np.log(np.asarray(target, dtype=float)[mask]) / ttm_[mask]
        ns = NelsonSiegel.calibrate(ttm_[mask], rates, lambda_bounds=self.lambda_bounds)
        self.yield_curve.beta1 = ns.beta1
        self.yield_curve.beta2 = ns.beta2
        self.yield_curve.beta3 = ns.beta3
        self.yield_curve.lambda_ = ns.lambda_
        return self.yield_curve


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


def _mad_filter(rates: np.ndarray, k: float = 3.0) -> np.ndarray:
    """Boolean mask: True for observations within k standard deviations (MAD-scaled)."""
    med = np.median(rates)
    mad = np.median(np.abs(rates - med))
    scale = max(mad / 0.6745, 1e-12)
    return np.abs(rates - med) <= k * scale
