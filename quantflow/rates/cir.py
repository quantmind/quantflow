from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike
from pydantic import Field
from scipy.optimize import least_squares
from typing_extensions import Self

from quantflow.sp.cir import CIR
from quantflow.utils.numbers import ZERO, DecimalNumber
from quantflow.utils.types import FloatArray, FloatArrayLike, maybe_float

from .yield_curve import YieldCurve


class CIRCurve(YieldCurve):
    r"""Yield curve derived from the Cox-Ingersoll-Ross short-rate model.

    The CIR model describes the short rate as a mean-reverting square-root
    diffusion:

    \begin{equation}
        dr_t = \kappa(\theta - r_t)\, dt + \sigma\sqrt{r_t}\, dW_t
    \end{equation}

    The closed-form discount factor is:

    \begin{equation}
        D(\tau) = A(\tau)\, e^{-B(\tau)\, r_0}
    \end{equation}

    where

    \begin{equation}
    \begin{aligned}
        \gamma &= \sqrt{\kappa^2 + 2\sigma^2} \\
        B(\tau) &= \frac{2(e^{\gamma\tau} - 1)}
            {2\gamma + (\gamma + \kappa)(e^{\gamma\tau} - 1)} \\
        A(\tau) &= \left(\frac{2\gamma\, e^{(\gamma+\kappa)\tau/2}}
            {2\gamma + (\gamma + \kappa)(e^{\gamma\tau} - 1)}
            \right)^{2\kappa\theta/\sigma^2}
    \end{aligned}
    \end{equation}
    """

    curve_type: Literal["cir_curve"] = "cir_curve"
    rate: DecimalNumber = Field(description=r"Initial short rate $r_0$")
    kappa: DecimalNumber = Field(gt=ZERO, description=r"Mean reversion speed $\kappa$")
    theta: DecimalNumber = Field(gt=ZERO, description=r"Long-run mean $\theta$")
    sigma: DecimalNumber = Field(gt=ZERO, description=r"Volatility $\sigma$")

    def process(self) -> CIR:
        """Return the underlying CIR stochastic process."""
        return CIR(
            rate=float(self.rate),
            kappa=float(self.kappa),
            theta=float(self.theta),
            sigma=float(self.sigma),
        )

    def instantaneous_forward_rate(self, ttm: FloatArrayLike) -> FloatArrayLike:
        r"""Calculate the instantaneous forward rate for the CIR model."""
        arr = np.asarray(ttm, dtype=float)
        ttma = np.maximum(arr, 0.0)
        kappa = float(self.kappa)
        theta = float(self.theta)
        sigma = float(self.sigma)
        rate = float(self.rate)
        sigma2 = sigma * sigma
        gamma = np.sqrt(kappa * kappa + 2.0 * sigma2)
        gamma_kappa = gamma + kappa
        egt = np.exp(gamma * ttma)
        d = 2.0 * gamma + gamma_kappa * (egt - 1.0)
        # Use -d/dt ln D(t) = r0 * dB/dt - (2*kappa*theta/sigma2) * d/dt ln(A)
        egt_m1 = egt - 1.0
        db = (2.0 * gamma * egt * d - 2.0 * egt_m1 * gamma_kappa * gamma * egt) / (
            d * d
        )
        # d/dt ln(A) = (2*kappa*theta/sigma2) * d/dt ln(numerator/d)
        # numerator = 2*gamma*exp((gamma+kappa)*t/2)
        # d/dt ln(num) = (gamma+kappa)/2
        # d/dt ln(d) = gamma_kappa * gamma * egt / d
        d_ln_a = gamma_kappa / 2.0 - gamma_kappa * gamma * egt / d
        kts = 2.0 * kappa * theta / sigma2
        fwd = rate * db + kts * d_ln_a
        return maybe_float(fwd)

    def discount_factor(self, ttm: FloatArrayLike) -> FloatArrayLike:
        r"""Calculate the discount factor using the CIR closed-form solution."""
        arr = np.asarray(ttm, dtype=float)
        ttma = np.maximum(arr, 0.0)
        kappa = float(self.kappa)
        theta = float(self.theta)
        sigma = float(self.sigma)
        rate = float(self.rate)
        sigma2 = sigma * sigma
        gamma = np.sqrt(kappa * kappa + 2.0 * sigma2)
        gamma_kappa = gamma + kappa
        egt = np.exp(gamma * ttma)
        d = 2.0 * gamma + gamma_kappa * (egt - 1.0)
        b = 2.0 * (egt - 1.0) / d
        log_a = (2.0 * kappa * theta / sigma2) * (
            np.log(2.0 * gamma) + 0.5 * gamma_kappa * ttma - np.log(d)
        )
        df = np.exp(log_a - b * rate)
        return maybe_float(df)

    def jacobian(self, ttm: FloatArrayLike) -> FloatArray | None:
        """Analytical Jacobian of discount factors w.r.t. [rate, kappa, theta, sigma].

        Returns shape (len(ttm), 4).
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
        egt = np.exp(gamma * ttma)
        d = 2.0 * gamma + gamma_kappa * (egt - 1.0)
        b = 2.0 * (egt - 1.0) / d
        log_a = (2.0 * kappa * theta / sigma2) * (
            np.log(2.0 * gamma) + 0.5 * gamma_kappa * ttma - np.log(d)
        )
        df = np.exp(log_a - b * rate)
        # dD/d(rate) = -B * D
        d_rate = -b * df
        n = arr.shape[0] if arr.ndim > 0 else 1
        jac = np.column_stack([d_rate.reshape(n)])
        return jac.reshape(n, 1) if arr.ndim > 0 else jac

    @classmethod
    def calibrate(cls, ttm: ArrayLike, rates: ArrayLike) -> Self:
        """Fit the CIR curve to continuously compounded rates via least squares.

        The Feller condition is enforced by reparametrising sigma as
        sigma = ratio * sqrt(2 * kappa * theta), with ratio in [0, 1].
        """
        ttm_arr = np.asarray(ttm, dtype=float)
        rates_arr = np.asarray(rates, dtype=float)

        def residuals(params: np.ndarray) -> np.ndarray:
            rate, kappa, theta, sigma_ratio = params
            sigma = sigma_ratio * np.sqrt(2.0 * kappa * theta)
            curve = cls(rate=rate, kappa=kappa, theta=theta, sigma=sigma)
            df = np.asarray(curve.discount_factor(ttm_arr), dtype=float)
            fitted = -np.log(df) / ttm_arr
            return fitted - rates_arr

        x0 = np.array([rates_arr[0], 1.0, rates_arr[-1], 0.5])
        result = least_squares(
            residuals,
            x0,
            bounds=([0.0, 1e-4, 1e-6, 1e-4], [1.0, 50.0, 1.0, 1.0]),
        )
        r, k, th, sr = result.x
        s = sr * np.sqrt(2.0 * k * th)
        return cls(rate=r, kappa=k, theta=th, sigma=s)
