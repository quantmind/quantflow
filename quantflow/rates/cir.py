from __future__ import annotations

from decimal import Decimal
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike
from pydantic import Field
from scipy.optimize import Bounds, least_squares
from typing_extensions import Annotated, Doc

from quantflow.sp.cir import CIR
from quantflow.utils.numbers import ZERO, DecimalNumber
from quantflow.utils.types import FloatArray, FloatArrayLike, maybe_float

from .calibration import YieldCurveCalibration
from .yield_curve import YieldCurve


class CIRCurve(YieldCurve):
    r"""Yield curve derived from the Cox-Ingersoll-Ross short-rate model.

    The CIR model describes the short rate as a mean-reverting square-root
    diffusion:

    \begin{equation}
        dr_t = \kappa(\theta - r_t)\, dt + \sigma\sqrt{r_t}\, dW_t
    \end{equation}

    The model admits a closed-form discount factor; see
    [discount_factor][.discount_factor].

    Throughout, the auxiliary quantities are:

    \begin{equation}
    \begin{aligned}
        \gamma &= \sqrt{\kappa^2 + 2\sigma^2} \\
        d_e(\tau) &= (\gamma + \kappa) + (\gamma - \kappa)e^{-\gamma\tau}
    \end{aligned}
    \end{equation}
    """

    curve_type: Literal["cir_curve"] = "cir_curve"
    rate: DecimalNumber = Field(
        default=Decimal("0.05"), description=r"Initial short rate $r_0$"
    )
    kappa: DecimalNumber = Field(
        default=Decimal("1.0"), gt=ZERO, description=r"Mean reversion speed $\kappa$"
    )
    theta: DecimalNumber = Field(
        default=Decimal("0.05"), gt=ZERO, description=r"Long-run mean $\theta$"
    )
    sigma: DecimalNumber = Field(
        default=Decimal("0.1"), gt=ZERO, description=r"Volatility $\sigma$"
    )

    def calibrator(self) -> CIRCurveCalibration:
        """Return a [CIRCurveCalibration][...CIRCurveCalibration] wrapping
        this curve."""
        return CIRCurveCalibration(yield_curve=self)

    def process(self) -> CIR:
        """Return the underlying [CIR][quantflow.sp.cir.CIR] stochastic process."""
        return CIR(
            rate=float(self.rate),
            kappa=float(self.kappa),
            theta=float(self.theta),
            sigma=float(self.sigma),
        )

    def instantaneous_forward_rate(self, ttm: FloatArrayLike) -> FloatArrayLike:
        r"""Calculate the instantaneous forward rate for the CIR model.

        The forward rate is:

        \begin{equation}
            f(\tau) = -\frac{\partial}{\partial \tau}\ln D(\tau)
            = r_0 B'(\tau) - A'(\tau)
        \end{equation}

        where:

        \begin{equation}
        \begin{aligned}
            B'(\tau) &= \frac{4\gamma^2 e^{-\gamma\tau}}{d_e(\tau)^2} \\
            A'(\tau) &= \frac{2\kappa\theta}{\sigma^2}
                \left[-\frac{\gamma - \kappa}{2}
                + \frac{\gamma(\gamma - \kappa)e^{-\gamma\tau}}{d_e(\tau)}\right]
        \end{aligned}
        \end{equation}
        """
        arr = np.asarray(ttm, dtype=float)
        ttma = np.maximum(arr, 0.0)
        kappa = float(self.kappa)
        theta = float(self.theta)
        sigma = float(self.sigma)
        rate = float(self.rate)
        sigma2 = sigma * sigma
        gamma = np.sqrt(kappa * kappa + 2.0 * sigma2)
        gamma_kappa = gamma + kappa
        gamma_m_kappa = gamma - kappa
        emgt = np.exp(-gamma * ttma)
        de = gamma_kappa + gamma_m_kappa * emgt
        # dB/dτ = 4γ²e^{-γτ} / de²
        db = 4.0 * gamma * gamma * emgt / (de * de)
        # d(log_a)/dτ = (2κθ/σ²) * [-(γ-κ)/2 + γ(γ-κ)e^{-γτ}/de]
        kts = 2.0 * kappa * theta / sigma2
        d_log_a = kts * (-gamma_m_kappa / 2.0 + gamma * gamma_m_kappa * emgt / de)
        fwd = rate * db - d_log_a
        return maybe_float(fwd)

    def discount_factor(self, ttm: FloatArrayLike) -> FloatArrayLike:
        r"""Calculate the discount factor using the CIR closed-form solution.

        The discount factor is:

        \begin{equation}
            D(\tau) = e^{A(\tau) - B(\tau)\, r_0}
        \end{equation}

        The coefficients are:

        \begin{equation}
        \begin{aligned}
            B(\tau) &= \frac{2(1 - e^{-\gamma\tau})}{d_e(\tau)} \\
            A(\tau) &= \frac{2\kappa\theta}{\sigma^2}
                \ln\!\left(
                \frac{2\gamma\, e^{-(\gamma - \kappa)\tau/2}}{d_e(\tau)}
                \right)
        \end{aligned}
        \end{equation}
        """
        arr = np.asarray(ttm, dtype=float)
        ttma = np.maximum(arr, 0.0)
        kappa = float(self.kappa)
        theta = float(self.theta)
        sigma = float(self.sigma)
        rate = float(self.rate)
        sigma2 = sigma * sigma
        gamma = np.sqrt(kappa * kappa + 2.0 * sigma2)
        gamma_kappa = gamma + kappa
        gamma_m_kappa = gamma - kappa
        emgt = np.exp(-gamma * ttma)
        # d/e^{γτ} = (γ+κ) + (γ-κ)e^{-γτ}
        de = gamma_kappa + gamma_m_kappa * emgt
        b = 2.0 * (1.0 - emgt) / de
        # log(d) = γτ + log(de)
        log_a = (2.0 * kappa * theta / sigma2) * (
            np.log(2.0 * gamma) - 0.5 * gamma_m_kappa * ttma - np.log(de)
        )
        df = np.exp(log_a - b * rate)
        return maybe_float(df)

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
        sigma2 = sigma * sigma
        gamma = np.sqrt(kappa * kappa + 2.0 * sigma2)
        gm_k = gamma - kappa
        emgt = np.exp(-gamma * ttma)
        de = (gamma + kappa) + gm_k * emgt
        b = 2.0 * (1.0 - emgt) / de
        c = 2.0 * kappa * theta / sigma2
        f = np.log(2.0 * gamma) - 0.5 * gm_k * ttma - np.log(de)
        log_a = c * f
        d = np.exp(log_a - b * rate)

        # ∂D/∂r0
        d_rate = -b * d

        # ∂D/∂κ: use dγ/dκ = κ/γ
        gk = kappa / gamma
        d_de_k = (gk + 1.0) + (gk - 1.0) * emgt - gm_k * ttma * gk * emgt
        d_b_k = 2.0 * (ttma * gk * emgt * de - (1.0 - emgt) * d_de_k) / (de * de)
        d_f_k = kappa / (gamma * gamma) - 0.5 * (gk - 1.0) * ttma - d_de_k / de
        d_loga_k = (2.0 * theta / sigma2) * f + c * d_f_k
        d_kappa = d * (d_loga_k - rate * d_b_k)

        # ∂D/∂θ: only c = 2κθ/σ² depends on θ, so d(log D)/dθ = log_a / θ
        d_theta = d * log_a / theta

        # ∂D/∂σ: use dγ/dσ = 2σ/γ
        gs = 2.0 * sigma / gamma
        d_de_s = gs * (1.0 + emgt * (1.0 - gm_k * ttma))
        d_b_s = 2.0 * (ttma * gs * emgt * de - (1.0 - emgt) * d_de_s) / (de * de)
        d_f_s = 2.0 * sigma / (gamma * gamma) - sigma * ttma / gamma - d_de_s / de
        d_loga_s = (-2.0 * c / sigma) * f + c * d_f_s
        d_sigma = d * (d_loga_s - rate * d_b_s)

        return np.column_stack([d_rate, d_kappa, d_theta, d_sigma])


class CIRCurveCalibration(YieldCurveCalibration[CIRCurve]):
    """Calibration wrapper for a CIR yield curve."""

    def get_params(self) -> FloatArray:
        c = self.yield_curve
        kappa = float(c.kappa)
        theta = float(c.theta)
        sigma_ratio = float(c.sigma) / np.sqrt(2.0 * kappa * theta)
        return np.array([float(c.rate), kappa, theta, sigma_ratio])

    def set_params(self, params: FloatArray) -> None:
        rate, kappa, theta, sigma_ratio = params
        sigma = sigma_ratio * np.sqrt(2.0 * kappa * theta)
        self.yield_curve.rate = Decimal(str(round(float(rate), 10)))
        self.yield_curve.kappa = Decimal(str(round(float(kappa), 10)))
        self.yield_curve.theta = Decimal(str(round(float(theta), 10)))
        self.yield_curve.sigma = Decimal(str(round(float(sigma), 10)))

    def get_bounds(self) -> Bounds:
        return Bounds([0.0, 1e-4, 1e-6, 1e-6], [1.0, 1000.0, 1.0, 1.0])

    def calibrate(
        self,
        ttm: Annotated[ArrayLike, Doc("Times to maturity in years.")],
        rates: Annotated[
            ArrayLike, Doc("Continuously compounded rates, same length as ttm.")
        ],
    ) -> CIRCurve:
        r"""Fit the CIR curve to continuously compounded rates via least squares.

        The Feller condition is enforced by reparametrising $\sigma$ as

        \begin{equation}
            \sigma = \rho \sqrt{2\kappa\theta}, \quad \rho \in [0, 1]
        \end{equation}

        Since CIR requires non-negative rates, any negative input rates are
        floored to a small positive value before fitting.
        """
        ttm_arr = np.asarray(ttm, dtype=float)
        rates_arr = np.maximum(np.asarray(rates, dtype=float), 1e-6)

        def residuals(params: np.ndarray) -> np.ndarray:
            self.set_params(params)
            df = np.asarray(self.yield_curve.discount_factor(ttm_arr), dtype=float)
            fitted = -np.log(df) / ttm_arr
            return fitted - rates_arr

        def jac(params: np.ndarray) -> FloatArray:
            self.set_params(params)
            _, kappa, theta, sigma_ratio = params
            sigma = sigma_ratio * np.sqrt(2.0 * kappa * theta)
            df = np.asarray(self.yield_curve.discount_factor(ttm_arr), dtype=float)
            jac_d = np.asarray(self.yield_curve.jacobian(ttm_arr), dtype=float)
            d_sigma = jac_d[:, 3]
            jac_d[:, 1] += d_sigma * sigma / (2.0 * kappa)
            jac_d[:, 2] += d_sigma * sigma / (2.0 * theta)
            jac_d[:, 3] = d_sigma * np.sqrt(2.0 * kappa * theta)
            return -jac_d / (df * ttm_arr)[:, None]

        x0 = np.array([rates_arr[0], 1.0, rates_arr[-1], 0.5])
        result = least_squares(
            residuals,
            jac=jac,
            x0=x0,
            bounds=([0.0, 1e-4, 1e-6, 1e-4], [1.0, 1000.0, 1.0, 1.0]),
        )
        self.set_params(result.x)
        return self.yield_curve
