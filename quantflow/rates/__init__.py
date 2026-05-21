from typing import Annotated, Union

from pydantic import Field

from .interest_rate import Rate
from .nelson_siegel import NelsonSiegel
from .options import YieldCurveCalibration
from .vasicek import VasicekCurve
from .yield_curve import NoDiscount, YieldCurve

__all__ = [
    "YieldCurve",
    "YieldCurveCalibration",
    "NoDiscount",
    "NelsonSiegel",
    "VasicekCurve",
    "AnyYieldCurve",
    "Rate",
]

AnyYieldCurve = Annotated[
    Union[NoDiscount, NelsonSiegel, VasicekCurve],
    Field(discriminator="curve_type"),
]

YieldCurve.register_curve_types(NoDiscount, NelsonSiegel, VasicekCurve)
