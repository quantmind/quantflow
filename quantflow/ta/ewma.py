"""Exponentially Weighted Moving Average (EWMA) for time series smoothing.

EWMA is a simple and efficient smoothing technique that gives more weight to recent
observations while exponentially decreasing the weight of older observations.
"""

import math
from typing import Any

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

    where $\alpha$ is the smoothing factor derived from the period (half-life)
    parameter using the formula:

    $$
    \begin{equation}
        \alpha = 1 - e^{-\ln(2) / \text{period}}
    \end{equation}
    $$

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
        description="Half-life period: number of time steps for weight to decay to 0.5 (must be >= 1)",
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

    def model_post_init(self, __context: Any) -> None:
        # Convert half-life to alpha using the formula:
        # alpha = 1 - exp(-log(2) / half_life)
        # This ensures that after 'period' time steps, the weight is 0.5
        self._alpha = 1.0 - math.exp(-log2 / self.period)

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
