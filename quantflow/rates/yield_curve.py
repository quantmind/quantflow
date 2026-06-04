from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike
from pydantic import BaseModel, Field
from typing_extensions import Annotated, Doc

from quantflow.utils import plot
from quantflow.utils.dates import utcnow
from quantflow.utils.text import snake_case
from quantflow.utils.types import FloatArray, FloatArrayLike, maybe_float

if TYPE_CHECKING:
    from .calibration import YieldCurveCalibration


_CURVE_TYPES: dict[str, type[YieldCurve]] = {}
_TYPES_TO_NAMES: dict[type[YieldCurve], str] = {}


class YieldCurve(BaseModel, ABC, extra="forbid"):
    """Abstract base class for yield curves"""

    ref_date: datetime = Field(
        default_factory=utcnow,
        description="Reference date for the yield curve",
    )
    curve_type: str = Field(
        default="unknown",
        description=(
            "Type of the yield curve, used for serialization" " and discrimination"
        ),
    )

    @abstractmethod
    def instantaneous_forward_rate(self, ttm: FloatArrayLike) -> FloatArrayLike:
        r"""Calculate the instantaneous forward rate for a given time to maturity.

        The instantaneous forward rate is related to discount factor
        by the following formula:

        \begin{equation}
            f(\tau) = -\frac{\partial \ln D(\tau)}{\partial \tau}
        \end{equation}

        where $D(\tau)$ is the discount factor for a given time to maturity $\tau$.

        Accepts a scalar float or a float array. Returns a scalar float for scalar
        input and a numpy float array for array input.
        """

    @abstractmethod
    def discount_factor(self, ttm: FloatArrayLike) -> FloatArrayLike:
        r"""Calculate the discount factor for a given time to maturity.

        The discount factor is related to the instantaneous forward rate
        by the following formula:

        \begin{equation}
            D(\tau) = \exp{\left(-\int_0^\tau f(s) ds\right)}
        \end{equation}

        where $f(\tau)$ is the instantaneous forward rate for a given time to
        maturity $\tau$.

        Accepts a scalar float or a float array. Returns a scalar float for scalar
        input and a numpy float array for array input.
        """

    def calibrator(self) -> YieldCurveCalibration | None:
        """Return a calibration wrapper for this curve, or None if not available."""
        return None

    def jacobian(
        self, ttm: Annotated[FloatArrayLike, Doc("Times to maturity in years.")]
    ) -> FloatArray | None:
        """Analytical Jacobian of discount factors w.r.t. model parameters.

        Returns None if no analytical Jacobian is available (default).
        Shape when not None: (len(ttm), n_params).
        """
        return None

    def continuously_compounded_rate(
        self, ttm: Annotated[ArrayLike, Doc("Time to maturity in years")]
    ) -> FloatArrayLike:
        r"""Calculate the continuously compounded rate for a given time to maturity.

        The continuously compounded rate is related to the discount factor
        by the following formula:

        \begin{equation}
            r(\tau) = -\frac{\ln D(\tau)}{\tau}
        \end{equation}

        where $D(\tau)$ is the discount factor for a given time to maturity $\tau$.

        Accepts a scalar float or a float array. Returns a scalar float for scalar
        input and a numpy float array for array input.
        """
        ttm_ = np.asarray(ttm, dtype=float)
        df = np.asarray(self.discount_factor(ttm_), dtype=float)
        result = np.where(
            ttm_ <= 0, self.instantaneous_forward_rate(0.0), -np.log(df) / ttm_
        )
        return maybe_float(result)

    def rates(
        self,
        ttm: Annotated[ArrayLike, Doc("Time to maturity in years")],
        frequency: Annotated[
            int,
            Doc(
                "Compounding periods per year (e.g. 2 for semi-annual). "
                "Pass 0 for continuously compounded."
            ),
        ] = 2,
    ) -> FloatArrayLike:
        r"""Calculate zero rates compounded at the given frequency.

        The continuously compounded rate $r_c(\tau)$ is converted to a
        rate compounded $m$ times per year via:

        \begin{equation}
            r_m(\tau) = m\,(e^{r_c(\tau)/m} - 1)
        \end{equation}

        When ``frequency=0`` the result is continuously compounded (same as
        [continuously_compounded_rate][..continuously_compounded_rate]).
        """
        rc = self.continuously_compounded_rate(ttm)
        if frequency <= 0:
            return rc
        return frequency * np.expm1(rc / frequency)

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

    @classmethod
    def register_curve_types(cls, *curve_classes: type[YieldCurve]) -> None:
        """Register a yield curve subclass for deserialization."""
        for curve_cls in curve_classes:
            name = snake_case(curve_cls.__name__)
            if current_type := _CURVE_TYPES.pop(name, None):
                _TYPES_TO_NAMES.pop(current_type, None)
            _CURVE_TYPES[name] = curve_cls
            _TYPES_TO_NAMES[curve_cls] = name

    @classmethod
    def curve_types(cls) -> tuple[str, ...]:
        """Return the registered curve types."""
        return tuple(sorted(_CURVE_TYPES))

    @classmethod
    def get_curve_class(cls, curve_type: str) -> type[YieldCurve] | None:
        """Get the yield curve class for a given curve type."""
        return _CURVE_TYPES.get(curve_type)
