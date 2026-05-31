from __future__ import annotations

from decimal import Decimal
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike
from pydantic import Field, PrivateAttr
from scipy.optimize import Bounds, least_squares, minimize
from typing_extensions import Annotated, Doc

from quantflow.sp.ou import Vasicek
from quantflow.sp.wiener import WienerProcess
from quantflow.ta.kalman import KalmanFilter, LinearGaussianModel, MeanAndCov
from quantflow.utils.numbers import ZERO, DecimalNumber
from quantflow.utils.types import FloatArray, FloatArrayLike, maybe_float

from .calibration import YieldCurveCalibration
from .yield_curve import YieldCurve


class VasicekCurve(YieldCurve):
    r"""Yield curve derived from the Vasicek short-rate model.

    The Vasicek model describes the short rate as a mean-reverting
    Ornstein-Uhlenbeck process:

    \begin{equation}
        dr_t = \kappa(\theta - r_t)\, dt + \sigma\, dW_t
    \end{equation}

    The model admits a closed-form discount factor; see
    [discount_factor][.discount_factor].

    Throughout, the auxiliary quantity is:

    \begin{equation}
        B(\tau) = \frac{1 - e^{-\kappa\tau}}{\kappa}
    \end{equation}
    """

    curve_type: Literal["vasicek_curve"] = "vasicek_curve"
    rate: DecimalNumber = Field(
        default=Decimal("0.05"), description=r"Initial value $x_0$"
    )
    kappa: DecimalNumber = Field(
        default=Decimal("1.0"), gt=ZERO, description=r"Mean reversion speed $\kappa$"
    )
    theta: DecimalNumber = Field(
        default=Decimal("0.05"), description=r"Mean level $\theta$"
    )
    sigma: DecimalNumber = Field(
        default=Decimal("0.01"), ge=ZERO, description=r"Volatility $\sigma$"
    )

    def calibrator(self) -> VasicekCurveCalibration:
        """Return a [VasicekCurveCalibration][...VasicekCurveCalibration] wrapping
        this curve."""
        return VasicekCurveCalibration(yield_curve=self)

    def process(self) -> Vasicek:
        """Return the underlying [Vasicek][quantflow.sp.ou.Vasicek] process
        corresponding to this curve."""
        return Vasicek(
            rate=float(self.rate),
            kappa=float(self.kappa),
            theta=float(self.theta),
            bdlp=WienerProcess(sigma=float(self.sigma)),
        )

    def instantaneous_forward_rate(self, ttm: FloatArrayLike) -> FloatArrayLike:
        r"""Calculate the instantaneous forward rate for the Vasicek model.

        \begin{equation}
            f(\tau) = r_0\, e^{-\kappa\tau}
                + \theta(1 - e^{-\kappa\tau})
                - \frac{\sigma^2}{2\kappa}\, B(\tau)\, e^{-\kappa\tau}
        \end{equation}
        """
        arr = np.asarray(ttm, dtype=float)
        ttma = np.maximum(arr, 0.0)
        kappa = float(self.kappa)
        theta = float(self.theta)
        sigma = float(self.sigma)
        rate = float(self.rate)
        s2 = sigma * sigma
        et = np.exp(-kappa * ttma)
        b = (1.0 - et) / kappa
        fwd = rate * et + theta * (1.0 - et) - s2 / (2.0 * kappa) * b * et
        return fwd if fwd.ndim > 0 else float(fwd)

    def discount_factor(self, ttm: FloatArrayLike) -> FloatArrayLike:
        r"""Calculate the discount factor using the Vasicek closed-form solution.

        \begin{equation}
            D(\tau) = e^{A(\tau) - B(\tau)\, r_0}
        \end{equation}

        where $A(\tau)$ and $B(\tau)$ are the affine coefficients given by
        [affine_coefficients][.affine_coefficients].
        """
        arr = np.asarray(ttm, dtype=float)
        ttma = np.maximum(arr, 0.0)
        kappa = float(self.kappa)
        theta = float(self.theta)
        sigma = float(self.sigma)
        rate = float(self.rate)
        s2 = sigma * sigma
        b = (1.0 - np.exp(-kappa * ttma)) / kappa
        a = (theta - s2 / (2.0 * kappa * kappa)) * (b - ttma) + s2 * b * b / (
            4.0 * kappa
        )
        df = np.exp(a - rate * b)
        return df if df.ndim > 0 else float(df)

    def affine_coefficients(
        self, ttm: FloatArrayLike
    ) -> tuple[FloatArrayLike, FloatArrayLike]:
        r"""Return the affine coefficients $A(\tau)$ and $B(\tau)$
        of the log discount factor.

        \begin{equation}
            \log D(\tau) = A(\tau) - B(\tau)\, r_0
        \end{equation}

        where

        \begin{equation}
        \begin{aligned}
            B(\tau) &= \frac{1 - e^{-\kappa\tau}}{\kappa}, \\
            A(\tau) &= \left(\theta - \frac{\sigma^2}{2\kappa^2}\right)
                \bigl(B(\tau) - \tau\bigr)
                + \frac{\sigma^2 B(\tau)^2}{4\kappa}.
        \end{aligned}
        \end{equation}
        """
        arr = np.asarray(ttm, dtype=float)
        ttma = np.maximum(arr, 0.0)
        kappa = float(self.kappa)
        theta = float(self.theta)
        sigma = float(self.sigma)
        s2 = sigma * sigma
        et = np.exp(-kappa * ttma)
        b = (1.0 - et) / kappa
        a = (theta - s2 / (2.0 * kappa * kappa)) * (b - ttma) + s2 * b * b / (
            4.0 * kappa
        )
        return maybe_float(a), maybe_float(b)

    def jacobian(self, ttm: FloatArrayLike) -> FloatArray | None:
        r"""Analytical Jacobian of discount factors w.r.t.
        $[r_0, \kappa, \theta, \sigma]$. Returns shape (len(ttm), 4).
        """
        arr = np.asarray(ttm, dtype=float)
        ttma = np.maximum(arr, 0.0)
        kappa = float(self.kappa)
        theta = float(self.theta)
        sigma = float(self.sigma)
        rate = float(self.rate)
        s2 = sigma * sigma
        et = np.exp(-kappa * ttma)
        b = (1.0 - et) / kappa
        a = (theta - s2 / (2.0 * kappa * kappa)) * (b - ttma) + s2 * b * b / (
            4.0 * kappa
        )
        d = np.exp(a - rate * b)

        # \partial D/\partial r0
        d_rate = -b * d

        # \partial D/\partial \kappa
        db_k = (ttma * et * kappa - (1.0 - et)) / (kappa * kappa)
        da_k = (
            s2 / (kappa**3) * (b - ttma)
            + (theta - s2 / (2.0 * kappa * kappa)) * db_k
            + s2 * b * db_k / (2.0 * kappa)
            - s2 * b * b / (4.0 * kappa * kappa)
        )
        d_kappa = d * (da_k - rate * db_k)

        # \partial D/\partial \theta
        d_theta = d * (b - ttma)

        # \partial D/\partial \sigma
        da_s = (-sigma / (kappa * kappa)) * (b - ttma) + sigma * b * b / (2.0 * kappa)
        d_sigma = d * da_s

        return np.column_stack([d_rate, d_kappa, d_theta, d_sigma])


class VasicekCurveCalibration(YieldCurveCalibration[VasicekCurve]):
    """Calibration wrapper for a [VasicekCurve][..VasicekCurve] yield curve."""

    _filtered_short_rate: FloatArray | None = PrivateAttr(default=None)

    @property
    def filtered_short_rate(self) -> FloatArray:
        """Kalman-filtered short rate at each observation date.

        Populated by [calibrate_historical_rates][.calibrate_historical_rates];
        accessing it before a historical fit raises an error.
        """
        if self._filtered_short_rate is None:
            raise AttributeError("run calibrate_historical_rates first")
        return self._filtered_short_rate

    def get_params(self) -> FloatArray:
        c = self.yield_curve
        return np.array([float(c.rate), float(c.kappa), float(c.theta), float(c.sigma)])

    def set_params(self, params: FloatArray) -> None:
        rate, kappa, theta, sigma = params
        self.yield_curve.rate = Decimal(str(round(float(rate), 10)))
        self.yield_curve.kappa = Decimal(str(round(float(kappa), 10)))
        self.yield_curve.theta = Decimal(str(round(float(theta), 10)))
        self.yield_curve.sigma = Decimal(str(round(float(sigma), 10)))

    def get_bounds(self) -> Bounds:
        return Bounds([-1.0, 1e-4, -1.0, 0.0], [1.0, 1000.0, 1.0, 1.0])

    def calibrate(
        self,
        ttm: Annotated[ArrayLike, Doc("Times to maturity in years.")],
        rates: Annotated[
            ArrayLike, Doc("Continuously compounded rates, same length as ttm.")
        ],
    ) -> VasicekCurve:
        """Fit the Vasicek curve to continuously compounded rates via least squares."""
        ttm_arr = np.asarray(ttm, dtype=float)
        rates_arr = np.asarray(rates, dtype=float)

        def residuals(params: np.ndarray) -> np.ndarray:
            self.set_params(params)
            df = np.asarray(self.yield_curve.discount_factor(ttm_arr), dtype=float)
            return -np.log(df) / ttm_arr - rates_arr

        def jac(params: np.ndarray) -> FloatArray:
            self.set_params(params)
            df = np.asarray(self.yield_curve.discount_factor(ttm_arr), dtype=float)
            jac_d = np.asarray(self.yield_curve.jacobian(ttm_arr), dtype=float)
            return -jac_d / (df * ttm_arr)[:, None]

        x0 = np.array([rates_arr[0], 1.0, rates_arr[-1], 0.01])
        result = least_squares(
            residuals,
            jac=jac,
            x0=x0,
            bounds=([-1.0, 1e-4, -1.0, 0.0], [1.0, 1000.0, 1.0, 1.0]),
        )
        self.set_params(result.x)
        return self.yield_curve

    def calibrate_historical_rates(
        self,
        ttm: Annotated[FloatArray, Doc("Times to maturity in years, shape (n,).")],
        rates: Annotated[
            FloatArray,
            Doc("Continuously compounded rates, shape (T, n) (time by maturity)."),
        ],
        dt: Annotated[
            FloatArray,
            Doc("Per-step time increments in years, shape (T-1,); assumed uniform."),
        ],
    ) -> VasicekCurve:
        r"""Fit Vasicek by maximum likelihood with a Kalman filter on the panel.

        The short rate $r_t$ is the latent state of a
        [LinearGaussianModel][quantflow.ta.kalman.LinearGaussianModel]. Under a
        uniform time step $\Delta t$ the exact discretization of the
        Ornstein-Uhlenbeck dynamics is the Gaussian AR(1):

        \begin{equation}
            \begin{aligned}
                r_t &= \theta(1 - \phi) + \phi\, r_{t-1} + \varepsilon_t \\
                \phi &= e^{-\kappa \Delta t} \\
                \varepsilon_t &\sim N\left(0,
                    \tfrac{\sigma^2}{2\kappa}(1 - \phi^2)\right)
            \end{aligned}
        \end{equation}

        Centring the state on $\theta$ (filtering $z_t = r_t - \theta$) cancels
        the AR(1) drift, and the affine yields $y_i = (B_i r_t - A_i)/\tau_i$
        become linear in $z_t$ once their constant part is subtracted. The plain
        zero-intercept linear-Gaussian filter then applies.

        The negative log-likelihood is minimised over
        $(\kappa, \theta, \sigma, h)$, where $h$ is the observation noise
        standard deviation, then the curve's rate is set to the final filtered
        short rate.
        """
        rates = np.asarray(rates, dtype=float)
        ttm = np.asarray(ttm, dtype=float)
        dt = np.asarray(dt, dtype=float)
        if dt.size and not np.allclose(dt, dt[0], rtol=1e-2):
            raise ValueError(
                "calibrate_historical_rates assumes a uniform time step; "
                "the observation dates are not equally spaced"
            )
        step = float(dt[0]) if dt.size else 1.0

        def filtered(
            kappa: float, theta: float, sigma: float, h: float
        ) -> tuple[list[MeanAndCov], float]:
            self.set_params(np.array([0.0, kappa, theta, sigma]))
            a, b = self.yield_curve.affine_coefficients(ttm)
            A, B = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
            phi = np.exp(-kappa * step)
            q = sigma * sigma * (1.0 - phi * phi) / (2.0 * kappa)
            offset = (B * theta - A) / ttm  # constant part of the affine yields
            model = LinearGaussianModel(
                F=np.array([[phi]]),
                Q=np.array([[q]]),
                H=B / ttm,
                R=h * h * np.eye(len(ttm)),
                mu0=np.zeros(1),
                cov0=np.array([[sigma * sigma / (2.0 * kappa)]]),
            )
            return KalmanFilter(model=model, data=rates - offset).filter()

        theta0 = float(np.mean(rates))
        short = rates[:, int(np.argmin(ttm))]
        sigma0 = max(float(np.std(np.diff(short)) / np.sqrt(step)), 1e-4)
        h0 = max(float(np.std(rates - rates.mean(axis=0))) / 10.0, 1e-4)
        x0 = np.array([np.log(0.5), theta0, np.log(sigma0), np.log(h0)])

        def neg_loglik(x: np.ndarray) -> float:
            _, ll = filtered(
                float(np.exp(x[0])),
                float(x[1]),
                float(np.exp(x[2])),
                float(np.exp(x[3])),
            )
            return -ll

        result = minimize(neg_loglik, x0, method="Nelder-Mead")
        kappa = float(np.exp(result.x[0]))
        theta = float(result.x[1])
        sigma = float(np.exp(result.x[2]))
        h = float(np.exp(result.x[3]))
        states, _ = filtered(kappa, theta, sigma, h)
        # filtered short rate path: z_t + theta
        short_rate = theta + np.array([float(s.mean.item()) for s in states])
        self._filtered_short_rate = short_rate
        self.set_params(np.array([float(short_rate[-1]), kappa, theta, sigma]))
        return self.yield_curve
