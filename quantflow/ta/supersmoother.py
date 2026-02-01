from typing import Any

import numpy as np
from pydantic import BaseModel, Field, PrivateAttr
from typing_extensions import Annotated, Doc


class SuperSmoother(BaseModel):
    r"""SuperSmoother filter for time series data.

    This implementation uses a two-pole Butterworth filter with adaptive smoothing.
    The SuperSmoother filter is designed to remove high-frequency noise while
    preserving the underlying trend with minimal lag.
    The filter is defined by the following recurrence relation:

    $$
    y_t = c_1 \frac{x_t + x_{t-1}}{2} + c_2 y_{t-1} + c_3 y_{t-2}
    $$

    where the coefficients are calculated as:

    $$
    \begin{align}
        \lambda &= \frac{\pi \sqrt{2}}{N} \\
        a &= \exp(-\lambda) \\
        c_2 &= 2 a \cos(\lambda) \\
        c_3 &= -a^2 \\
        c_1 &= 1 - c_2 - c_3
    \end{align}
    $$

    where $N$ is the period.

    ## Example

    ```python
    import pandas as pd
    smoother = SuperSmoother(period=10)
    df = pd.DataFrame({"value": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
    df["smoothed"] = df["value"].apply(smoother.update)
    ```

    For online updates:

    ```python
    smoother = SuperSmoother(period=10)
    for value in [1, 2, 3, 4, 5]:
        smoothed = smoother(value)
        print(smoothed)
    ```
    """

    period: int = Field(default=10, ge=2)
    """Number of periods for the smoothing filter (must be >= 2)"""
    _prev_value: float | None = PrivateAttr(default=None)
    _prev_smooth1: float | None = PrivateAttr(default=None)
    _prev_smooth2: float | None = PrivateAttr(default=None)
    _c1: float = PrivateAttr(default=0.0)
    _c2: float = PrivateAttr(default=0.0)
    _c3: float = PrivateAttr(default=0.0)

    def model_post_init(self, __context: Any) -> None:
        self._compute_coefficients()

    def raw_value(self) -> float | None:
        """Get the most recent raw input value, if available."""
        return self._prev_value

    def value(self) -> float | None:
        """Get the most recent smoothed value, if available."""
        return self._prev_smooth1

    def update(
        self,
        value: Annotated[float, Doc("New data point to add to the filter")],
    ) -> float:
        """Update the filter with a new value and return the smoothed result."""
        if self._prev_value is None:
            self._prev_value = value
            return value
        elif self._prev_smooth1 is None:
            # Second value, use simple average
            smoothed = (value + self._prev_value) / 2.0
            self._prev_smooth2 = self._prev_value
            self._prev_smooth1 = smoothed
        else:
            # Calculate the average of current and previous value
            avg = (value + self._prev_value) / 2.0
            # Apply filter equation
            smoothed = (
                self._c1 * avg
                + self._c2 * self._prev_smooth1  # type: ignore
                + self._c3 * self._prev_smooth2  # type: ignore
            )
            # Update state for next iteration
            self._prev_smooth2 = self._prev_smooth1
            self._prev_smooth1 = smoothed
        self._prev_value = value
        return smoothed

    def _compute_coefficients(self) -> None:
        # Calculate the cutoff frequency
        # For a period of N, cutoff = 1.414 * pi / N (empirical optimal value)
        cutoff = np.sqrt(2) * np.pi / self.period
        # Two-pole Butterworth filter coefficients
        a = np.exp(-cutoff)
        b = 2 * a * np.cos(cutoff)
        c2 = b
        c3 = -a * a
        c1 = 1 - c2 - c3
        self._c1 = c1
        self._c2 = c2
        self._c3 = c3
