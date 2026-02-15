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

    $$
    \begin{equation}
        s_t = \alpha x_t + (1 - \alpha) s_{t-1}
    \end{equation}
    $$

    where $\alpha$ is the smoothing factor derived from the period parameter.
    To match SuperSmoother characteristics, the formula is:

    $$
    \begin{equation}
        \alpha = \frac{2}{\text{period} + 1}
    \end{equation}
    $$

    This provides frequency response similar to a simple moving average and
    approximates the smoothing characteristics of higher-order filters.

    ## Example

    ```python
    import pandas as pd
    ewma = EWMA(period=10)
    df = pd.DataFrame({"value": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
    df["ewma"] = df["value"].apply(ewma.update)
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
        description="Optional asymmetric smoothing filter",
    )

    _count: int = PrivateAttr(default=0)
    _smoothed: float = PrivateAttr(default=0.0)
    _alpha: float = PrivateAttr(default=0.0)

    @classmethod
    def from_half_life(cls, half_life: float, tau: float | None = None) -> Self:
        r"""Create an EWMA using half-life semantics instead of period.

        The half-life represents the time for weight to decay to 0.5.

        $$
        \alpha = 1 - \exp\left(-\frac{\ln(2)}{\text{half life}}\right)
        $$
        """
        # Calculate equivalent period that produces the desired half-life behavior
        # From: alpha = 1 - exp(-ln(2) / half_life)
        # And: alpha = 2 / (period + 1)
        # We solve: 2 / (period + 1) = 1 - exp(-ln(2) / half_life)
        alpha = 1.0 - math.exp(-log2 / half_life)
        period = int(2.0 / alpha - 1)
        return cls(period=max(1, period), tau=tau)

    def model_post_init(self, __context: Any) -> None:
        # Convert period to alpha using the standard formula:
        # alpha = 2 / (period + 1)
        # This matches the smoothing characteristics of an SMA with the same period
        self._alpha = 2.0 / (self.period + 1)

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

    @property
    def current_value(self) -> float | None:
        """Get the most recent smoothed value, if available."""
        return self._smoothed if self._count > 0 else None

    @property
    def alpha(self) -> float:
        """Get the smoothing factor (alpha) used by the filter."""
        return self._alpha
