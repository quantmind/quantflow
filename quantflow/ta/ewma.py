"""Exponentially Weighted Moving Average (EWMA) for time series smoothing.

EWMA is a simple and efficient smoothing technique that gives more weight to recent
observations while exponentially decreasing the weight of older observations.
"""

from typing import Any

from pydantic import BaseModel, Field, PrivateAttr


class EWMA(BaseModel):
    """Exponentially Weighted Moving Average filter for time series data.

    This implementation uses the standard EWMA formula:
        S[t] = α * X[t] + (1 - α) * S[t-1]

    where α (alpha) is the smoothing factor derived from the period parameter.

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
        description="Number of periods for the smoothing filter (must be >= 1)",
    )

    _count: int = PrivateAttr(default=0)
    _smoothed: float = PrivateAttr(default=0.0)
    _alpha: float = PrivateAttr(default=0.0)

    def model_post_init(self, __context: Any) -> None:
        # Standard EWMA alpha: 2 / (period + 1)
        # This gives approximately the same weight distribution
        # as a simple moving average
        self._alpha = 2.0 / (self.period + 1.0)

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
            # Apply EWMA formula: S[t] = α * X[t] + (1 - α) * S[t-1]
            self._smoothed = self._alpha * value + (1.0 - self._alpha) * self._smoothed

        return self._smoothed

    @property
    def current_value(self) -> float | None:
        """Get the most recent smoothed value, if available."""
        return self._smoothed if self._count > 0 else None

    @property
    def alpha(self) -> float:
        """Get the smoothing factor (alpha) used by the filter."""
        return self._alpha
