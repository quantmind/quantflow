from typing import cast

import numpy as np
from fastapi import APIRouter, Query
from pydantic import BaseModel, Field

from app.api.docs import load_description
from quantflow.rates import AnyYieldCurve, YieldCurve

rates_router = APIRouter()


class YieldCurveResponse(BaseModel):
    curve: AnyYieldCurve = Field(description="The fitted yield curve")
    ttm: list[float] = Field(
        description="List of time to maturities for the fitted curve"
    )
    rates: list[float] = Field(description="List of rates for the fitted curve")


@rates_router.get(
    "/yield-curve",
    summary="Fit and interpolate a yield curve",
    description=load_description("yield_curve.md"),
)
async def yield_curve(
    ttm: list[float] = Query(
        ..., description="List of time to maturities corresponding to the rates"
    ),
    rates: list[float] = Query(
        ..., description="List of rates to fit the Nelson-Siegel model"
    ),
    curve_type: str = Query(
        "nelson_siegel",
        description="Type of curve to fit",
        enum=list(YieldCurve.curve_types()),
    ),
    max_ttm: float | None = Query(
        None,
        description=(
            "Maximum time to maturity to consider for returning"
            " the fitted curve. If not provided, the curve will be returned for all "
            "time to maturities."
        ),
    ),
    num_points: int = Query(
        100,
        description=(
            "Number of points to return for the fitted curve. Only used if max_ttm is "
            "provided. Otherwise, the curve will be returned for all time "
            "to maturities."
        ),
    ),
) -> YieldCurveResponse:
    curve_class = YieldCurve.get_curve_class(curve_type)
    if curve_class is None:
        raise ValueError(f"Unsupported curve type: {curve_type}")
    calibrator = curve_class().calibrator()
    if calibrator is None:
        raise ValueError(f"Curve type {curve_type!r} does not support calibration")
    curve = cast(AnyYieldCurve, calibrator.calibrate(ttm, rates))
    if max_ttm is not None:
        ttm = list(np.geomspace(1 / 365, max_ttm, num_points))
    rates = [float(r) for r in np.atleast_1d(curve.continuously_compounded_rate(ttm))]
    return YieldCurveResponse(curve=curve, ttm=ttm, rates=rates)
