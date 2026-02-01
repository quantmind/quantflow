from typing_extensions import Annotated, Doc
from pydantic import BaseModel, Field, PrivateAttr


class KalmanFilter(BaseModel):
    r"""One-dimensional Kalman filter for time series data.

    This implementation uses a simple 1D state-space model:

    $$
    \begin{align}
        x_t &= x_{t-1} + w_t, \quad w_t \sim \mathcal{N}(0, Q) \\
        z_t &= x_t + v_t, \quad v_t \sim \mathcal{N}(0, R)
    \end{align}
    $$

    The Kalman filter estimates the hidden state $x_t$ given noisy measurements $z_t$.
    The ratio $Q/R$ determines the smoothing behavior.
    """

    R: float = Field(default=1.0, gt=0.0, description="Measurement noise covariance")
    Q: float = Field(default=0.01, gt=0.0, description="Process noise covariance")

    _x: float | None = PrivateAttr(default=None)  # State estimate
    _P: float = PrivateAttr(default=1.0)  # Error covariance
    _K: float = PrivateAttr(default=0.0)  # Kalman Gain

    def value(self) -> float | None:
        """Get the most recent smoothed value (state estimate), if available."""
        return self._x

    def update(
        self,
        value: Annotated[float, Doc("New noisy measurement to update the filter")],
    ) -> float:
        """Update the filter with a new value and return the smoothed result."""
        # Initialize on first update
        if self._x is None:
            self._x = value
            self._P = self.R
            return value

        # Prediction step
        # x_pred = x_prev (Random walk model)
        # P_pred = P_prev + Q
        x_pred = self._x
        P_pred = self._P + self.Q

        # Update step
        # K = P_pred / (P_pred + R)
        # x_new = x_pred + K * (measurement - x_pred)
        # P_new = (1 - K) * P_pred
        K = P_pred / (P_pred + self.R)
        x_new = x_pred + K * (value - x_pred)
        P_new = (1 - K) * P_pred

        # Update state
        self._x = x_new
        self._P = P_new
        self._K = K

        return x_new

    @property
    def error_covariance(self) -> float:
        """Current estimated error covariance."""
        return self._P

    @property
    def kalman_gain(self) -> float:
        """Most recent Kalman gain."""
        return self._K
