from typing import Annotated, Union

from pydantic import Field

from .calibration import YieldCurveCalibration
from .cir import CIRCurve
from .interest_rate import Rate
from .interpolated import (
    InterpolatedYieldCurve,
    InterpolationType,
)
from .nelson_siegel import NelsonSiegel
from .no_discount import NoDiscount
from .vasicek import VasicekCurve
from .yield_curve import YieldCurve

__all__ = [
    "YieldCurve",
    "YieldCurveCalibration",
    "NoDiscount",
    "CIRCurve",
    "InterpolatedYieldCurve",
    "InterpolationType",
    "NelsonSiegel",
    "VasicekCurve",
    "AnyYieldCurve",
    "Rate",
]

AnyYieldCurve = Annotated[
    Union[NoDiscount, CIRCurve, InterpolatedYieldCurve, NelsonSiegel, VasicekCurve],
    Field(discriminator="curve_type"),
]

YieldCurve.register_curve_types(
    NoDiscount, CIRCurve, InterpolatedYieldCurve, NelsonSiegel, VasicekCurve
)
