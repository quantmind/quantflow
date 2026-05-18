from decimal import Decimal
from typing import Literal

from pydantic import Field

from quantflow.sp.ou import Vasicek
from quantflow.sp.wiener import WienerProcess
from quantflow.utils.numbers import ONE, ZERO, DecimalNumber, Number, to_decimal

from .yield_curve import YieldCurve


class VasicekCurve(YieldCurve):
    """Class representing a Vasicek yield curve"""

    rate: DecimalNumber = Field(description=r"Initial value $x_0$")
    kappa: DecimalNumber = Field(gt=ZERO, description=r"Mean reversion speed $\kappa$")
    theta: DecimalNumber = Field(description=r"Mean level $\theta$")
    sigma: DecimalNumber = Field(ge=ZERO, description=r"Volatility $\sigma$")
    curve_type: Literal["vasicek"] = "vasicek"

    def process(self) -> Vasicek:
        return Vasicek(
            rate=float(self.rate),
            kappa=float(self.kappa),
            theta=float(self.theta),
            bdlp=WienerProcess(sigma=float(self.sigma)),
        )

    def discount_factor(self, ttm: Number) -> Decimal:
        r"""Calculate the discount factor for a given time to maturity."""
        ttmd = to_decimal(ttm)
        if ttmd <= ZERO:
            return ONE
        b = (ONE - (-self.kappa * ttmd).exp()) / self.kappa
        s2 = self.sigma * self.sigma
        a = (self.theta - s2 / (2 * self.kappa * self.kappa)) * (
            b - ttmd
        ) + s2 * b * b / (4 * self.kappa)
        return (a - self.rate * b).exp()
