from __future__ import annotations

from decimal import Decimal
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike
from pydantic import Field, PrivateAttr
from scipy.optimize import Bounds, least_squares, minimize
from typing_extensions import Annotated, Doc

from quantflow.dists import MvNormal
from quantflow.sp.cir import CIR
from quantflow.ta.kalman import StateSpaceModel, UnscentedKalmanFilter
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

        \begin{equation}
            D(\tau) = e^{A(\tau) - B(\tau)\, r_0}
        \end{equation}

        where $A(\tau)$ and $B(\tau)$ are the
        [affine_coefficients][..affine_coefficients].
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
            B(\tau) &= \frac{2(1 - e^{-\gamma\tau})}{d_e(\tau)}, \\
            A(\tau) &= \frac{2\kappa\theta}{\sigma^2}
                \ln\!\left(
                \frac{2\gamma\, e^{-(\gamma - \kappa)\tau/2}}{d_e(\tau)}
                \right).
        \end{aligned}
        \end{equation}
        """
        arr = np.asarray(ttm, dtype=float)
        ttma = np.maximum(arr, 0.0)
        kappa = float(self.kappa)
        theta = float(self.theta)
        sigma = float(self.sigma)
        sigma2 = sigma * sigma
        gamma = np.sqrt(kappa * kappa + 2.0 * sigma2)
        gamma_m_kappa = gamma - kappa
        emgt = np.exp(-gamma * ttma)
        de = (gamma + kappa) + gamma_m_kappa * emgt
        b = 2.0 * (1.0 - emgt) / de
        log_a = (2.0 * kappa * theta / sigma2) * (
            np.log(2.0 * gamma) - 0.5 * gamma_m_kappa * ttma - np.log(de)
        )
        return maybe_float(log_a), maybe_float(b)

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


class CIRStateSpaceModel(StateSpaceModel, arbitrary_types_allowed=True):
    r"""State-space model for the Cox-Ingersoll-Ross short rate.

    The latent state is the short rate $r_t$, with dynamics taken from the
    embedded [CIRCurve][..CIRCurve]. The transition has a linear conditional
    mean but a state-dependent conditional variance, the signature of the
    square-root diffusion. Both are the one-step
    [analytical_mean][quantflow.sp.cir.CIR.analytical_mean] and
    [analytical_variance][quantflow.sp.cir.CIR.analytical_variance] of the CIR
    process started from $r_{t-1}$.

    Because the conditional variance depends on the state, the constant-$Q$
    linear [KalmanFilter][quantflow.ta.kalman.KalmanFilter] does not apply; the
    model is filtered with the
    [UnscentedKalmanFilter][quantflow.ta.kalman.UnscentedKalmanFilter], which
    only needs the conditional mean and covariance.

    The observations are the affine zero rates
    $y_i = (B_i r_t - A_i)/\tau_i$ with Gaussian measurement noise of standard
    deviation $h$.
    """

    curve: CIRCurve = Field(description="CIR curve supplying the short-rate dynamics.")
    ttm: FloatArray = Field(
        description="Times to maturity of the observed rates, shape $(n_y,)$."
    )
    dt: float = Field(gt=0, description=r"Time step $\Delta t$ in years.")
    h: float = Field(gt=0, description="Observation noise standard deviation $h$.")

    _process: CIR = PrivateAttr(default_factory=CIR)
    _affine: tuple[FloatArray, FloatArray] = PrivateAttr(
        default_factory=lambda: (np.zeros(0), np.zeros(0))
    )

    def model_post_init(self, context: object) -> None:
        """Cache the CIR process and the affine coefficients
        $A(\\tau_i)$, $B(\\tau_i)$."""
        self._process = self.curve.process()
        a, b = self.curve.affine_coefficients(self.ttm)
        self._affine = (np.asarray(a, dtype=float), np.asarray(b, dtype=float))

    def get_px0(self) -> MvNormal:
        r"""Stationary (equilibrium) distribution of the short rate, used as the
        prior for the latent initial state."""
        p = self._process
        return MvNormal(
            mean=np.array([p.equilibrium_mean]),
            cov=np.array([[p.equilibrium_variance]]),
        )

    def get_px(self, t: int, xp: FloatArray) -> MvNormal:
        r"""Transition $p(r_t \mid r_{t-1})$ from the CIR conditional moments
        over one time step."""
        self._process.rate = max(float(np.atleast_1d(xp)[0]), 0.0)
        mean = float(self._process.analytical_mean(self.dt))
        var = float(self._process.analytical_variance(self.dt))
        return MvNormal(mean=np.array([mean]), cov=np.array([[var]]))

    def get_py(self, t: int, xp: FloatArray, x: FloatArray) -> MvNormal:
        r"""Observation distribution of the affine zero rates given $r_t$."""
        a, b = self._affine
        r = float(np.atleast_1d(x)[0])
        mean = (b * r - a) / self.ttm
        cov = self.h * self.h * np.eye(self.ttm.size)
        return MvNormal(mean=mean, cov=cov)


class CIRCurveCalibration(YieldCurveCalibration[CIRCurve]):
    """Calibration wrapper for a CIR yield curve."""

    _filtered_short_rate: FloatArray | None = PrivateAttr(default=None)

    @property
    def filtered_short_rate(self) -> FloatArray:
        """Unscented-Kalman-filtered short rate at each observation date.

        Populated by [calibrate_historical_rates][.calibrate_historical_rates];
        accessing it before a historical fit raises an error.
        """
        if self._filtered_short_rate is None:
            raise AttributeError("run calibrate_historical_rates first")
        return self._filtered_short_rate

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
    ) -> CIRCurve:
        r"""Fit CIR by maximum likelihood with an unscented Kalman filter.

        The short rate $r_t$ is the latent state of a
        [CIRStateSpaceModel][..CIRStateSpaceModel]. Its exact transition has a
        linear conditional mean but a state-dependent conditional variance, so
        the panel is filtered with the
        [UnscentedKalmanFilter][quantflow.ta.kalman.UnscentedKalmanFilter]
        rather than the exact linear filter used for Vasicek.

        The negative log-likelihood is minimised over
        $(\kappa, \theta, \rho, h)$, where $h$ is the observation noise standard
        deviation and the Feller condition is enforced by reparametrising
        $\sigma = \rho \sqrt{2\kappa\theta}$ with $\rho \in (0, 1)$. The curve's
        rate is then set to the final filtered short rate.
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

        def unpack(x: np.ndarray) -> tuple[float, float, float, float]:
            # x = (log kappa, log theta, logit rho, log h); enforce Feller via
            # sigma = rho * sqrt(2 kappa theta) with rho in (0, 1)
            kappa = float(np.exp(x[0]))
            theta = float(np.exp(x[1]))
            rho = 1.0 / (1.0 + np.exp(-x[2]))
            sigma = float(rho * np.sqrt(2.0 * kappa * theta))
            return kappa, theta, sigma, float(np.exp(x[3]))

        def filtered(
            kappa: float, theta: float, sigma: float, h: float
        ) -> UnscentedKalmanFilter:
            curve = CIRCurve(
                rate=Decimal(str(round(theta, 10))),
                kappa=Decimal(str(round(kappa, 10))),
                theta=Decimal(str(round(theta, 10))),
                sigma=Decimal(str(round(sigma, 10))),
            )
            model = CIRStateSpaceModel(curve=curve, ttm=ttm, dt=step, h=h)
            return model.unscented_filter(rates)

        theta0 = max(float(np.mean(rates)), 1e-4)
        short = rates[:, int(np.argmin(ttm))]
        short_mean = max(float(np.mean(short)), 1e-4)
        sigma0 = max(float(np.std(np.diff(short)) / np.sqrt(step * short_mean)), 1e-4)
        rho0 = min(max(sigma0 / np.sqrt(2.0 * 0.5 * theta0), 0.01), 0.99)
        h0 = max(float(np.std(rates - rates.mean(axis=0))) / 10.0, 1e-5)
        x0 = np.array(
            [np.log(0.5), np.log(theta0), np.log(rho0 / (1.0 - rho0)), np.log(h0)]
        )

        def neg_loglik(x: np.ndarray) -> float:
            return -filtered(*unpack(x)).filter()

        result = minimize(neg_loglik, x0, method="Nelder-Mead")
        kappa, theta, sigma, h = unpack(result.x)
        ukf = filtered(kappa, theta, sigma, h)
        ukf.filter()
        short_rate = np.array([float(s.mean.item()) for s in ukf.states])
        self._filtered_short_rate = short_rate
        c = self.yield_curve
        c.rate = Decimal(str(round(float(short_rate[-1]), 10)))
        c.kappa = Decimal(str(round(kappa, 10)))
        c.theta = Decimal(str(round(theta, 10)))
        c.sigma = Decimal(str(round(sigma, 10)))
        return c
