from __future__ import annotations

from abc import ABC, abstractmethod
from decimal import Decimal
from typing import Any

from numpy.typing import ArrayLike
from pydantic import BaseModel
from typing_extensions import Annotated, Doc, Self

from quantflow.utils import plot
from quantflow.utils.numbers import ONE, ZERO


class YieldCurve(BaseModel, ABC, extra="forbid"):
    """Abstract base class for yield curves"""

    @abstractmethod
    def instanteous_forward_rate(self, ttm: float) -> Decimal:
        r"""Calculate the instantaneous forward rate for a given time to maturity

        The instantaneous forward rate is related to discount factor
        by the following formula:

        \begin{equation}
            f(\tau) = -\frac{\partial \ln D(\tau)}{\partial \tau}
        \end{equation}

        where $D(\tau)$ is the discount factor for a given time to maturity $\tau$.
        """

    @abstractmethod
    def discount_factor(self, ttm: float) -> Decimal:
        r"""Calculate the discount factor for a given time to maturity

        The discount factor is related to the instantaneous forward rate
        by the following formula:

        \begin{equation}
            D(\tau) = \exp{\left(-\int_0^\tau f(s) ds\right)}
        \end{equation}

        where $f(\tau)$ is the instantaneous forward rate for a given time to
        maturity $\tau$.
        """

    @classmethod
    @abstractmethod
    def calibrate(cls, ttm: ArrayLike, rates: ArrayLike) -> Self:
        """Fit the yield curve to continuously compounded rates.

        Parameters
        ----------
        ttm:
            Times to maturity in years.
        rates:
            Continuously compounded rates, same length as ttm (e.g. 0.05 for 5%).
        """

    def continuously_compounded_rate(self, ttm: float) -> Decimal:
        r"""Calculate the continuously compounded rate for a given time to maturity

        The continuously compounded rate is related to the discount factor
        by the following formula:

        \begin{equation}
            r(\tau) = -\frac{\ln D(\tau)}{\tau}
        \end{equation}

        where $D(\tau)$ is the discount factor for a given time to maturity $\tau$.
        """
        if ttm <= 0:
            return self.instanteous_forward_rate(0)
        else:
            return -self.discount_factor(float(ttm)).ln() / Decimal(ttm)

    def plot(
        self,
        ttm_max: Annotated[float, Doc("Maximum time to maturity in years")] = 10.0,
        n: Annotated[int, Doc("Number of points to evaluate")] = 200,
        **kwargs: Any,
    ) -> Any:
        """Plot the continuously compounded rate vs time to maturity.

        Requires plotly to be installed.
        """
        return plot.plot_yield_curve(self, ttm_max=ttm_max, n=n, **kwargs)


class NoDiscount(YieldCurve):
    """Flat yield curve with zero rates (discount factor is always 1)."""

    def instanteous_forward_rate(self, ttm: float) -> Decimal:
        return ZERO

    def discount_factor(self, ttm: float) -> Decimal:
        return ONE

    @classmethod
    def calibrate(cls, ttm: ArrayLike, rates: ArrayLike) -> Self:
        return cls()
