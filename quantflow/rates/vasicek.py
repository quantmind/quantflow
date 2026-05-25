from __future__ import annotations

from decimal import Decimal
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike
from pydantic import Field
from scipy.optimize import Bounds, least_squares, minimize
from typing_extensions import Annotated, Doc

from quantflow.sp.ou import Vasicek
from quantflow.sp.wiener import WienerProcess
from quantflow.utils.numbers import ZERO, DecimalNumber
from quantflow.utils.types import FloatArray, FloatArrayLike

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

        The discount factor is:

        \begin{equation}
            D(\tau) = e^{A(\tau) - B(\tau)\, r_0}
        \end{equation}

        where:

        \begin{equation}
            A(\tau) = \left(\theta - \frac{\sigma^2}{2\kappa^2}\right)
                \bigl(B(\tau) - \tau\bigr)
                + \frac{\sigma^2 B(\tau)^2}{4\kappa}
        \end{equation}
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

        # ∂D/∂r0
        d_rate = -b * d

        # ∂D/∂κ
        db_k = (ttma * et * kappa - (1.0 - et)) / (kappa * kappa)
        da_k = (
            s2 / (kappa**3) * (b - ttma)
            + (theta - s2 / (2.0 * kappa * kappa)) * db_k
            + s2 * b * db_k / (2.0 * kappa)
            - s2 * b * b / (4.0 * kappa * kappa)
        )
        d_kappa = d * (da_k - rate * db_k)

        # ∂D/∂θ
        d_theta = d * (b - ttma)

        # ∂D/∂σ
        da_s = (-sigma / (kappa * kappa)) * (b - ttma) + sigma * b * b / (2.0 * kappa)
        d_sigma = d * da_s

        return np.column_stack([d_rate, d_kappa, d_theta, d_sigma])


class VasicekCurveCalibration(YieldCurveCalibration[VasicekCurve]):
    """Calibration wrapper for a [VasicekCurve][..VasicekCurve] yield curve."""

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
        ttm: FloatArray,
        rates: FloatArray,
        dt: FloatArray,
    ) -> VasicekCurve:
        r"""Fit Vasicek by maximum likelihood via a Kalman filter on the panel.

        The short rate $r_t$ is treated as a latent state evolving under the
        exact-discretization Vasicek dynamics; yields are linear in $r_t$ and
        observed with i.i.d. Gaussian noise of variance $h^2$. The negative
        log-likelihood is minimised over $(\kappa, \theta, \sigma, h)$, then
        the curve's rate is set to the final filtered short rate.
        """
        theta0 = float(np.mean(rates))
        short_idx = int(np.argmin(ttm))
        short = rates[:, short_idx]
        sigma0 = max(float(np.std(np.diff(short)) / np.sqrt(np.mean(dt))), 1e-4)
        h0 = max(float(np.std(rates - rates.mean(axis=0))) / 10.0, 1e-4)
        x0 = np.array([np.log(0.5), theta0, np.log(sigma0), np.log(h0)])

        def neg_loglik(x: np.ndarray) -> float:
            self.set_params(
                np.array([0.0, float(np.exp(x[0])), float(x[1]), float(np.exp(x[2]))])
            )
            ll, _ = self.kalman_filter(ttm, rates, dt, float(np.exp(x[3])))
            return -ll

        result = minimize(neg_loglik, x0, method="Nelder-Mead")
        kappa = float(np.exp(result.x[0]))
        theta = float(result.x[1])
        sigma = float(np.exp(result.x[2]))
        h = float(np.exp(result.x[3]))
        self.set_params(np.array([0.0, kappa, theta, sigma]))
        _, r_last = self.kalman_filter(ttm, rates, dt, h)
        self.set_params(np.array([r_last, kappa, theta, sigma]))
        return self.yield_curve

    def kalman_filter(
        self,
        ttm: FloatArray,
        rates: FloatArray,
        dt: FloatArray,
        h: float,
    ) -> tuple[float, float]:
        r"""Kalman log-likelihood and final filtered short rate at the
        current ``yield_curve`` parameters.

        The Vasicek discount factor (see
        [discount_factor][...VasicekCurve.discount_factor]) is affine in
        the short rate,

        \begin{equation}
            \log D(\tau) = A(\tau) - B(\tau)\, r_t,
        \end{equation}

        so $A(\tau)$ and $B(\tau)$ are recovered with two evaluations of
        [discount_factor][...VasicekCurve.discount_factor]:

        \begin{equation}
            A(\tau) = \log D(\tau)\big|_{r_t = 0},
            \qquad
            B(\tau) = A(\tau) - \log D(\tau)\big|_{r_t = 1}.
        \end{equation}

        The observation map is then $y(\tau) = (B(\tau)\, r_t - A(\tau))/\tau$.

        Latent-state dynamics come from the underlying
        [Vasicek][quantflow.sp.ou.Vasicek] process. Observation noise is
        i.i.d. Gaussian across maturities with variance $h^2$, so the
        rank-1 innovation covariance

        \begin{equation}
            F = h^2 I + P \, (B/\tau)(B/\tau)^\top
        \end{equation}

        is inverted in closed form via Sherman-Morrison.
        """
        curve = self.yield_curve
        saved_rate = curve.rate
        curve.rate = Decimal(0)
        log_d_at_zero = np.log(np.asarray(curve.discount_factor(ttm), dtype=float))
        curve.rate = Decimal(1)
        log_d_at_one = np.log(np.asarray(curve.discount_factor(ttm), dtype=float))
        curve.rate = saved_rate
        A = log_d_at_zero
        B = log_d_at_zero - log_d_at_one
        coef = B / ttm

        process = curve.process()
        theta = process.theta
        phi = np.exp(-process.kappa * dt)
        q = np.asarray(process.analytical_variance(dt), dtype=float)
        drift = theta * (1.0 - phi)

        cc = float(coef @ coef)
        var_obs = h * h
        r = theta
        p = process.bdlp.sigma**2 / (2.0 * process.kappa)
        n_obs, n_mat = rates.shape
        log_lik = -0.5 * n_obs * n_mat * np.log(2.0 * np.pi)
        for t in range(n_obs):
            if t > 0:
                phi_t = float(phi[t - 1])
                r = phi_t * r + float(drift[t - 1])
                p = phi_t * phi_t * p + float(q[t - 1])
            innov = rates[t] - (B * r - A) / ttm
            cv = float(coef @ innov)
            denom = var_obs + p * cc
            log_det = (n_mat - 1) * np.log(var_obs) + np.log(denom)
            quad = (float(innov @ innov) - p * cv * cv / denom) / var_obs
            log_lik -= 0.5 * (log_det + quad)
            r = r + p * cv / denom
            p = p * var_obs / denom
        return log_lik, r
