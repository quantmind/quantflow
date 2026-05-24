"""Exponentially Weighted Moving Average (EWMA) for time series smoothing.

EWMA is a simple and efficient smoothing technique that gives more weight to recent
observations while exponentially decreasing the weight of older observations.
"""

import math
from typing import Any, Self

from pydantic import BaseModel, Field, PrivateAttr

log2 = math.log(2)


class EWMA(BaseModel):
    r"""Exponentially Weighted Moving Average filter for time series data.

    This implementation uses the standard EWMA formula:

    \begin{equation}
        s_t = \alpha x_t + (1 - \alpha) s_{t-1}
    \end{equation}

    where $\alpha$ is the smoothing factor derived from the period parameter.
    The period represents the effective averaging window of the exponential decay,
    defined as the half-life $h$ divided by $\ln 2$:

    \begin{equation}
        p = \frac{h}{\ln 2}
    \end{equation}

    For an exponential decay with half-life $h$, the sum of all weights equals
    $h / \ln 2$, making the period the continuous-time equivalent of the number of
    observations in a simple moving average. The smoothing factor is:

    \begin{equation}
        \alpha = 1 - \exp\left(-\frac{1}{p}\right)
    \end{equation}

    where $p$ is the [period][.period]. This definition makes the period directly
    comparable to the period used in
    [SuperSmoother][quantflow.ta.supersmoother.SuperSmoother].

    If [tau][.tau] is provided, EWMA becomes asymmetric: for up-moves the update uses
    $\alpha \cdot \tau$, while for down-moves it uses $\alpha \cdot (1-\tau)$.
    This is useful when you want different reaction speeds to rising and falling values.

    ## Example

    ```python
    import pandas as pd
    ewma = EWMA(period=10)
    df = pd.DataFrame({"value": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
    df["ewma"] = df["value"].apply(EWMA(period=10).update)
    ```

    For online updates:
    ```python
    ewma = EWMA(period=10)
    for value in [1, 2, 3, 4, 5]:
        smoothed = ewma.update(value)
        print(smoothed)
    ```
    """

    period: int = Field(
        default=10,
        ge=1,
        description="Characteristic period for the smoothing filter (must be >= 1)",
    )
    tau: float | None = Field(
        default=None,
        ge=0,
        le=1,
        description=(
            "Optional asymmetry control. For increasing values use alpha*tau; "
            "for decreasing values use alpha*(1-tau)"
        ),
    )

    _count: int = PrivateAttr(default=0)
    _smoothed: float = PrivateAttr(default=0.0)
    _alpha: float = PrivateAttr(default=0.0)

    @property
    def current_value(self) -> float | None:
        """Get the most recent smoothed value, if available."""
        return self._smoothed if self._count > 0 else None

    @property
    def alpha(self) -> float:
        r"""Get the smoothing factor $0 < \alpha < 1$ used by the filter."""
        return self._alpha

    @property
    def half_life(self) -> float:
        r"""Get the half-life corresponding to the current period.

        \begin{equation}
            h = p \cdot \ln 2
        \end{equation}
        """
        return self.period * log2

    @classmethod
    def from_half_life(cls, half_life: float, tau: float | None = None) -> Self:
        r"""Create an EWMA using half-life semantics instead of period.

        The half-life represents the time for weight to decay to 0.5.

        \begin{equation}
            \alpha = 1 - \exp\left(-\frac{\ln(2)}{h}\right)
        \end{equation}
        """
        if half_life <= 0:
            raise ValueError("half_life must be greater than 0")
        period = int(round(half_life / log2))
        return cls(period=max(1, period), tau=tau)

    @classmethod
    def from_alpha(cls, alpha: float, tau: float | None = None) -> Self:
        r"""Create an EWMA directly from a specified alpha value.

        The period is computed as the inverse of:

        \begin{equation}
            \alpha = 1 - \exp\left(-\frac{1}{p}\right)
        \end{equation}
        """
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be between 0 and 1")
        period = int(round(-1.0 / math.log1p(-alpha)))
        return cls(period=max(1, period), tau=tau)

    def model_post_init(self, __context: Any) -> None:
        self._alpha = 1.0 - math.exp(-1.0 / self.period)

    def update(self, value: float) -> float:
        """Update the filter with a new value and return the smoothed result.

        Args:
            value: New data point to add to the filter

        Returns:
            Smoothed value using the EWMA algorithm
        """
        self._count += 1

        if self._count == 1:
            # Initialize with first value
            self._smoothed = value
        else:
            alpha = self._alpha
            if self.tau is not None:
                # Apply asymmetric smoothing if tau is set
                if value > self._smoothed:
                    alpha *= self.tau
                else:
                    alpha *= 1.0 - self.tau
            # Apply EWMA formula: S[t] = α * X[t] + (1 - α) * S[t-1]
            self._smoothed += alpha * (value - self._smoothed)

        return self._smoothed
