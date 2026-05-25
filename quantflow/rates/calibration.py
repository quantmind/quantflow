from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, TypeVar

import ccy
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from pydantic import BaseModel, Field
from scipy.optimize import Bounds, least_squares
from typing_extensions import Annotated, Doc

from quantflow.utils.types import FloatArray

if TYPE_CHECKING:
    from .yield_curve import YieldCurve


Y = TypeVar("Y", bound="YieldCurve")


def tenor_to_years(label: str) -> float:
    """Convert a tenor label (e.g. "6m", "1y", "30d") to a year fraction
    using the 30/360 day convention of [ccy.Period][ccy.dates.period.Period]."""
    return ccy.period(label).totaldays / 360.0


def _dt_array(index: pd.Index) -> FloatArray:
    """Per-step time increments in years from a DatetimeIndex (ACT/365)."""
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError("rates index must be a pandas DatetimeIndex")
    if len(index) < 2:
        raise ValueError("at least two observations are required")
    return np.diff(index.asi8) / (1e9 * 86400.0 * 365.0)


def _to_continuous(rates: FloatArray, frequency: int | None) -> FloatArray:
    """Convert rates compounded at ``frequency`` periods per year to continuous."""
    if frequency is None:
        return rates
    if frequency <= 0:
        raise ValueError("frequency must be a positive integer")
    return frequency * np.log1p(rates / frequency)


class YieldCurveCalibration(BaseModel, Generic[Y]):
    yield_curve: Y = Field(..., description="Yield curve to be calibrated")

    @abstractmethod
    def get_params(self) -> FloatArray:
        """Current model parameters as a flat array (starting point for fit)"""

    @abstractmethod
    def set_params(self, params: FloatArray) -> None:
        """Update the yield curve from a flat parameter array"""

    @abstractmethod
    def get_bounds(self) -> Bounds:
        """Parameter bounds for the optimiser"""

    def __call__(self, ttm: FloatArray) -> FloatArray:
        """Discount factors for the given TTMs evaluated at current params.

        Called inside the optimiser hot loop: must not construct a new
        yield curve instance on every call.
        """
        return np.asarray(self.yield_curve.discount_factor(ttm), dtype=float)

    @abstractmethod
    def calibrate(
        self,
        ttm: Annotated[ArrayLike, Doc("Times to maturity in years.")],
        rates: Annotated[
            ArrayLike, Doc("Continuously compounded rates, same length as ttm.")
        ],
    ) -> Y:
        """Fit the yield curve to continuously compounded rates."""

    def calibrate_df(
        self,
        ttm: Annotated[ArrayLike, Doc("Times to maturity in years.")],
        target: Annotated[
            ArrayLike, Doc("Target discount factors, same length as ttm.")
        ],
    ) -> Y:
        """Fit the yield curve to target discount factors.

        Converts discount factors to continuously compounded rates then calls
        [calibrate][..calibrate].
        """
        ttm_ = np.asarray(ttm, dtype=float)
        rates = -np.log(np.asarray(target, dtype=float)) / ttm_
        return self.calibrate(ttm_, rates)

    def calibrate_historical_rates_dataframe(
        self,
        rates: Annotated[
            pd.DataFrame,
            Doc(
                "Historical zero rates with a DatetimeIndex and tenor column "
                "labels parsed by [ccy.Period][ccy.dates.period.Period] "
                "(e.g. ``'6m'``, ``'1y'``)."
            ),
        ],
        frequency: Annotated[
            int | None,
            Doc(
                "Compounding periods per year of the input rates. ``None`` "
                "(default) means continuously compounded."
            ),
        ] = None,
    ) -> Y:
        """Fit the yield curve from a historical panel of rates.

        Tenor column labels are parsed into times to maturity, per-step
        time increments are inferred from the DatetimeIndex (irregular
        spacing supported), and rates are converted to continuously
        compounded if a finite ``frequency`` is supplied. The actual fit
        is delegated to [calibrate_historical_rates][...calibrate_historical_rates],
        which subclasses override.
        """
        ttm = np.array([tenor_to_years(str(c)) for c in rates.columns], dtype=float)
        rates_arr = _to_continuous(np.asarray(rates.values, dtype=float), frequency)
        dt = _dt_array(rates.index)
        return self.calibrate_historical_rates(ttm, rates_arr, dt)

    def calibrate_historical_rates(
        self,
        ttm: Annotated[FloatArray, Doc("Times to maturity in years.")],
        rates: Annotated[
            FloatArray, Doc("Continuously compounded rates, same shape as ttm.")
        ],
        dt: Annotated[
            FloatArray,
            Doc("Time increments between observations, same length as rates."),
        ],
    ) -> Y:
        """Model-specific hook for historical rate calibration.

        Default implementation raises NotImplementedError. Subclasses with a
        stochastic short-rate dynamic override this method.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support historical rate calibration"
        )


@dataclass
class OptionsDiscountingCalibration:
    """Calibrate yield curves from option price parity data.

    The input data consists of arrays of call-put parity values, strikes, and times
    to maturity for a set of options on the same underlying. The calibration can be
    done jointly for both the asset and quote curves, or separately for one curve with
    the other fixed.
    """

    asset_curve: Annotated[
        YieldCurve | YieldCurveCalibration,
        Doc(
            "Yield curve for the underlying asset. An instance is treated as fixed; "
            "a YieldCurveCalibration will be calibrated from the parity data."
        ),
    ]
    quote_curve: Annotated[
        YieldCurve | YieldCurveCalibration,
        Doc(
            "Yield curve for the quote asset. An instance is treated as fixed; "
            "a YieldCurveCalibration will be calibrated from the parity data."
        ),
    ]
    cp: Annotated[FloatArray, Doc("(Call - Put) / Spot for each option pair")]
    strikes: Annotated[
        FloatArray, Doc("Strike / Spot for each option pair, same length as cp")
    ]
    ttm: Annotated[
        FloatArray,
        Doc("Time to maturity in years for each option pair, same length as cp"),
    ]

    def calibrate(self) -> tuple[YieldCurve, YieldCurve]:
        if isinstance(self.asset_curve, YieldCurveCalibration):
            if isinstance(self.quote_curve, YieldCurveCalibration):
                return self.joint_calibration(self.asset_curve, self.quote_curve)
            else:
                return self.asset_calibration(self.asset_curve, self.quote_curve)
        elif isinstance(self.quote_curve, YieldCurveCalibration):
            return self.quote_calibration(self.asset_curve, self.quote_curve)
        else:
            return self.asset_curve, self.quote_curve

    def joint_calibration(
        self,
        asset_cal: YieldCurveCalibration,
        quote_cal: YieldCurveCalibration,
    ) -> tuple[YieldCurve, YieldCurve]:
        """Calibrate both curves jointly from all parity observations."""
        pa = asset_cal.get_params()
        pq = quote_cal.get_params()
        has_jacobian = (
            asset_cal.yield_curve.jacobian(self.ttm) is not None
            and quote_cal.yield_curve.jacobian(self.ttm) is not None
        )
        n_a = len(pa)
        bounds = Bounds(
            np.concatenate([asset_cal.get_bounds().lb, quote_cal.get_bounds().lb]),
            np.concatenate([asset_cal.get_bounds().ub, quote_cal.get_bounds().ub]),
        )

        def residuals(params: np.ndarray) -> np.ndarray:
            asset_cal.set_params(params[:n_a])
            quote_cal.set_params(params[n_a:])
            da = asset_cal(self.ttm)
            dq = quote_cal(self.ttm)
            return self.cp - da + dq * self.strikes

        def jac(params: np.ndarray) -> FloatArray:
            asset_cal.set_params(params[:n_a])
            quote_cal.set_params(params[n_a:])
            ja = asset_cal.yield_curve.jacobian(self.ttm)
            jq = quote_cal.yield_curve.jacobian(self.ttm)
            if ja is None or jq is None:  # pragma: no cover
                raise TypeError("jacobian must not return None in joint calibration")
            return np.hstack([-ja, jq * self.strikes[:, None]])

        result = least_squares(
            residuals,
            np.concatenate([pa, pq]),
            jac=jac if has_jacobian else "2-point",
            bounds=bounds,
            method="trf",
        )
        asset_cal.set_params(result.x[:n_a])
        quote_cal.set_params(result.x[n_a:])
        return asset_cal.yield_curve, quote_cal.yield_curve

    def asset_calibration(
        self,
        asset_cal: YieldCurveCalibration,
        fixed_quote: YieldCurve,
    ) -> tuple[YieldCurve, YieldCurve]:
        """Calibrate only the asset curve; quote curve is fixed."""
        dq = np.asarray(fixed_quote.discount_factor(self.ttm), dtype=float)
        target_da = self.cp + dq * self.strikes
        return asset_cal.calibrate_df(self.ttm, target_da), fixed_quote

    def quote_calibration(
        self,
        fixed_asset: YieldCurve,
        quote_cal: YieldCurveCalibration,
    ) -> tuple[YieldCurve, YieldCurve]:
        """Calibrate only the quote curve; asset curve is fixed."""
        da = np.asarray(fixed_asset.discount_factor(self.ttm), dtype=float)
        target_dq = (da - self.cp) / self.strikes
        return fixed_asset, quote_cal.calibrate_df(self.ttm, target_dq)
