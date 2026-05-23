from typing import Annotated, Union

from pydantic import Field

from .cir import CIRCurve
from .interest_rate import Rate
from .nelson_siegel import NelsonSiegel
from .options import YieldCurveCalibration
from .vasicek import VasicekCurve
from .yield_curve import NoDiscount, YieldCurve

__all__ = [
    "YieldCurve",
    "YieldCurveCalibration",
    "NoDiscount",
    "CIRCurve",
    "NelsonSiegel",
    "VasicekCurve",
    "AnyYieldCurve",
    "Rate",
]

AnyYieldCurve = Annotated[
    Union[NoDiscount, CIRCurve, NelsonSiegel, VasicekCurve],
    Field(discriminator="curve_type"),
]

YieldCurve.register_curve_types(NoDiscount, CIRCurve, NelsonSiegel, VasicekCurve)
