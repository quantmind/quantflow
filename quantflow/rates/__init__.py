from typing import Annotated, Union

from pydantic import Field

from .calibration import YieldCurveCalibration
from .cir import CIRCurve
from .interest_rate import Rate
from .interpolated import (
    InterpolatedLinearCurve,
    InterpolatedMonotonicCubicCurve,
    InterpolatedYieldCurve,
)
from .nelson_siegel import NelsonSiegelCurve
from .no_discount import NoDiscountCurve
from .vasicek import VasicekCurve
from .yield_curve import YieldCurve

__all__ = [
    "YieldCurve",
    "YieldCurveCalibration",
    "NoDiscountCurve",
    "CIRCurve",
    "InterpolatedYieldCurve",
    "InterpolatedLinearCurve",
    "InterpolatedMonotonicCubicCurve",
    "NelsonSiegelCurve",
    "VasicekCurve",
    "AnyYieldCurve",
    "Rate",
]

AnyYieldCurve = Annotated[
    Union[
        NoDiscountCurve,
        CIRCurve,
        InterpolatedLinearCurve,
        InterpolatedMonotonicCubicCurve,
        NelsonSiegelCurve,
        VasicekCurve,
    ],
    Field(discriminator="curve_type"),
]
"""Discriminated union of all concrete
[YieldCurve][quantflow.rates.yield_curve.YieldCurve] implementations.

Use this type for Pydantic fields that can hold any curve model, such as the
quote and asset curves of a
[VolSurface][quantflow.options.surface.VolSurface].

The `curve_type` discriminator selects the concrete class during validation,
so curves serialise to and from JSON without losing their type.
"""

YieldCurve.register_curve_types(
    NoDiscountCurve,
    CIRCurve,
    InterpolatedLinearCurve,
    InterpolatedMonotonicCubicCurve,
    NelsonSiegelCurve,
    VasicekCurve,
)
