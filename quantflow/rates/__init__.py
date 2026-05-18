from typing import Annotated, Union

from pydantic import Field

from .interest_rate import Rate
from .nelson_siegel import NelsonSiegel
from .vasicek import VasicekCurve
from .yield_curve import NoDiscount, YieldCurve

__all__ = [
    "YieldCurve",
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
