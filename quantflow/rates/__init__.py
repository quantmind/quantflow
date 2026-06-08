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

YieldCurve.register_curve_types(
    NoDiscountCurve,
    CIRCurve,
    InterpolatedLinearCurve,
    InterpolatedMonotonicCubicCurve,
    NelsonSiegelCurve,
    VasicekCurve,
)
