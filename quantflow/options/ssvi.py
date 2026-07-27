from __future__ import annotations

from collections.abc import Sequence
from decimal import Decimal
from typing import Any, Self

import numpy as np
from numpy.typing import ArrayLike
from pydantic import BaseModel, Field, PrivateAttr
from scipy.interpolate import PchipInterpolator
from scipy.optimize import least_squares
from typing_extensions import Annotated, Doc

from quantflow.options.surface import VolSurface
from quantflow.utils.numbers import ONE, ZERO, DecimalNumber, to_decimal
from quantflow.utils.types import FloatArray, FloatArrayLike, maybe_float


class VarianceCurve(BaseModel, extra="forbid"):
    r"""Monotone interpolated total ATM variance curve.

    The curve stores total ATM variances $\theta_i = w(0, \tau_i)$ at strictly
    increasing maturities $\tau_i$. Static arbitrage requires the total ATM
    variance curve to be non decreasing:

    \begin{equation}
        \frac{d\theta}{d\tau} \geq 0
    \end{equation}

    The nodes are validated to satisfy this condition, then interpolated with a
    shape preserving cubic Hermite spline (PCHIP). For non decreasing nodes,
    PCHIP is monotone between nodes and therefore preserves the condition.
    """

    ttm: list[DecimalNumber] = Field(
        description="Times to maturity in years, strictly increasing",
    )
    theta: list[DecimalNumber] = Field(
        description=(
            "Total ATM variances at each maturity, same length as ttm, positive "
            "and non decreasing"
        ),
    )

    _ttm: FloatArray = PrivateAttr(default_factory=lambda: np.empty(0))
    _theta: FloatArray = PrivateAttr(default_factory=lambda: np.empty(0))

    def model_post_init(self, context: object) -> None:
        """Cache nodes and validate the no calendar arbitrage condition."""
        if len(self.ttm) != len(self.theta):
            raise ValueError("ttm and theta must have equal length")
        if not self.ttm:
            raise ValueError("at least one variance curve node is required")
        ttm = np.array([float(t) for t in self.ttm], dtype=float)
        theta = np.array([float(t) for t in self.theta], dtype=float)
        if np.any(ttm <= 0):
            raise ValueError("ttm nodes must be strictly positive")
        if np.any(np.diff(ttm) <= 0):
            raise ValueError("ttm nodes must be strictly increasing")
        if np.any(theta <= 0):
            raise ValueError("theta nodes must be strictly positive")
        if np.any(np.diff(theta) < 0):
            raise ValueError("theta nodes must be non decreasing")
        self._ttm = ttm
        self._theta = theta

    def total_variance(
        self,
        ttm: Annotated[ArrayLike, Doc("Time to maturity in years, scalar or array")],
    ) -> FloatArrayLike:
        r"""Interpolated total ATM variance $\theta(\tau)$."""
        tau = np.maximum(np.asarray(ttm, dtype=float), 0.0)
        if self._ttm.size < 2:
            theta = np.full(np.shape(tau), self._theta[0])
        else:
            clamped = np.clip(tau, self._ttm[0], self._ttm[-1])
            theta = PchipInterpolator(self._ttm, self._theta)(clamped)
        return maybe_float(np.asarray(theta, dtype=float))

    def derivative(
        self,
        ttm: Annotated[ArrayLike, Doc("Time to maturity in years, scalar or array")],
    ) -> FloatArrayLike:
        r"""Derivative $d\theta / d\tau$ of the interpolated variance curve."""
        tau = np.maximum(np.asarray(ttm, dtype=float), 0.0)
        if self._ttm.size < 2:
            derivative = np.zeros_like(tau)
        else:
            clamped = np.clip(tau, self._ttm[0], self._ttm[-1])
            inside = (tau >= self._ttm[0]) & (tau <= self._ttm[-1])
            derivative = PchipInterpolator(self._ttm, self._theta).derivative()(clamped)
            derivative = np.where(inside, derivative, 0.0)
        return maybe_float(np.asarray(derivative, dtype=float))

    def is_non_decreasing(self) -> bool:
        r"""True if the variance nodes satisfy $d\theta / d\tau \geq 0$."""
        return bool(np.all(np.diff(self._theta) >= 0))


class SSVI(BaseModel, extra="forbid"):
    r"""Surface SVI (SSVI) parametrisation of the implied volatility surface,
    introduced by
    [Gatheral and Jacquier (2014)](../../bibliography.md#gatheral_jacquier).

    The SSVI parametrisation expresses the total implied variance
    $w(k) = \sigma^2(k) \cdot \tau$ as a function of log-strike
    $k = \log(K/F)$ and the total ATM variance $\theta_\tau = w(0, \tau)$:

    \begin{equation}
        w(k, \tau) = \frac{\theta_\tau}{2}\left[1 + \rho \varphi(\theta_\tau) k
        + \sqrt{\left(\varphi(\theta_\tau) k + \rho\right)^2 + 1 - \rho^2}\right]
    \end{equation}

    The shape function $\varphi$ uses the power law form:

    \begin{equation}
        \varphi(\theta_\tau) =
        \frac{\eta}{\theta_\tau^\gamma (1 + \theta_\tau)^{1 - \gamma}}
    \end{equation}

    Each instance represents the surface through shared parameters $\rho$,
    $\eta$ and $\gamma$ together with a monotone total ATM variance curve
    $\theta(\tau)$. A single maturity slice is represented by a one node
    [VarianceCurve][.VarianceCurve].

    Use [fit_surface][.fit_surface] to jointly calibrate the global parameters
    and variance curve across several maturities. A surface built this way is
    free of static arbitrage
    provided each slice satisfies
    [no_butterfly_arbitrage][.no_butterfly_arbitrage] and the ATM variance
    curve is non decreasing in maturity, $d\theta / d\tau \geq 0$.
    """

    rho: Decimal = Field(
        gt=-ONE,
        lt=ONE,
        description=(
            "Correlation parameter controlling the skew of the smile. "
            "Negative values produce a left-skewed smile (typical for equities), "
            "positive values produce a right skew. Must satisfy $|\\rho| < 1$."
        ),
    )
    eta: Decimal = Field(
        gt=ZERO,
        description=(
            "Level of the power law shape function $\\varphi$. "
            "Larger values steepen the smile away from the money. "
            "Must be strictly positive."
        ),
    )
    gamma: Decimal = Field(
        gt=ZERO,
        le=ONE,
        description=(
            "Exponent of the power law shape function $\\varphi$, controlling "
            "how the smile curvature decays as the at-the-money total variance "
            "grows. Must lie in $(0, 1]$."
        ),
    )
    variance_curve: VarianceCurve = Field(
        description=(
            "Monotone interpolated total ATM variance curve "
            "$\\theta(\\tau) = \\sigma_{ATM}^2 \\cdot \\tau$."
        ),
    )

    @property
    def theta(self) -> Decimal:
        """Total ATM variance of the first curve node."""
        return self.variance_curve.theta[0]

    def phi(
        self,
        ttm: Annotated[ArrayLike, Doc("Time to maturity in years, scalar or array")],
    ) -> FloatArrayLike:
        r"""Power law shape function $\varphi(\theta(\tau))$."""
        theta = np.asarray(self.variance_curve.total_variance(ttm), dtype=float)
        eta = float(self.eta)
        gamma = float(self.gamma)
        phi = eta / (theta**gamma * (1 + theta) ** (1 - gamma))
        return maybe_float(np.asarray(phi, dtype=float))

    def total_variance(
        self,
        k: Annotated[ArrayLike, Doc("Log-moneyness log(K/F), scalar or array")],
        ttm: Annotated[ArrayLike, Doc("Time to maturity in years, scalar or array")],
    ) -> FloatArrayLike:
        r"""Total implied variance $w(k, \tau)$.

        Returns an array broadcast from the shapes of $k$ and $\tau$.
        """
        k_arr = np.asarray(k, dtype=float)
        theta = np.asarray(self.variance_curve.total_variance(ttm), dtype=float)
        rho = float(self.rho)
        phi = np.asarray(self.phi(ttm), dtype=float)
        pk = phi * k_arr
        w = 0.5 * theta * (1 + rho * pk + np.sqrt((pk + rho) ** 2 + 1 - rho**2))
        return maybe_float(np.asarray(w, dtype=float))

    def iv(
        self,
        k: Annotated[ArrayLike, Doc("Log-moneyness log(K/F), scalar or array")],
        ttm: Annotated[ArrayLike, Doc("Time to maturity in years, scalar or array")],
    ) -> FloatArrayLike:
        r"""Implied volatility $\sigma(k, \tau) = \sqrt{w(k, \tau) / \tau}$.

        Returns an array of the same shape as $k$. The SSVI total variance is
        strictly positive for $|\rho| < 1$, so no clipping is required.
        """
        tau = np.asarray(ttm, dtype=float)
        return maybe_float(np.asarray(np.sqrt(self.total_variance(k, ttm) / tau)))

    def no_butterfly_arbitrage(
        self,
        ttm: Annotated[
            float | None,
            Doc("Optional maturity to check. All curve nodes are checked when omitted"),
        ] = None,
    ) -> bool:
        r"""True if the slice satisfies the sufficient conditions for absence
        of butterfly arbitrage.

        The conditions, from Theorem 4.2 of
        [Gatheral and Jacquier (2014)](../../bibliography.md#gatheral_jacquier), are:

        \begin{equation}
        \begin{aligned}
            \theta \varphi(\theta) (1 + |\rho|) &< 4 \\
            \theta \varphi(\theta)^2 (1 + |\rho|) &\leq 4
        \end{aligned}
        \end{equation}
        """
        rho = abs(float(self.rho))
        ttms = [ttm] if ttm is not None else [float(t) for t in self.variance_curve.ttm]
        for tau in ttms:
            theta = float(self.variance_curve.total_variance(tau))
            phi = float(self.phi(tau))
            if not (theta * phi * (1 + rho) < 4 and theta * phi * phi * (1 + rho) <= 4):
                return False
        return True

    def no_static_arbitrage(self) -> bool:
        """True if the surface satisfies the implemented static checks."""
        return self.variance_curve.is_non_decreasing() and self.no_butterfly_arbitrage()

    @classmethod
    def fit_surface(
        cls,
        slices: Annotated[
            Sequence[tuple[ArrayLike, ArrayLike, float]],
            Doc(
                "One (log-moneyness, implied volatilities, time to maturity) "
                "tuple per maturity slice"
            ),
        ],
    ) -> Self:
        r"""Jointly fit the SSVI surface to several maturity slices via
        non-linear least squares.

        The global parameters rho, eta and gamma are shared across all slices,
        while the maturity dependent total ATM variance is represented by a
        monotone [VarianceCurve][quantflow.options.ssvi.VarianceCurve]. The
        fitted variables are the shared parameters $\rho$, $\eta$ and $\gamma$,
        plus positive increments for the total ATM variance curve. The
        increment parameterisation guarantees $d\theta / d\tau \geq 0$.
        """
        if not slices:
            raise ValueError("at least one maturity slice is required")
        data = []
        thetas0 = []
        for k, iv, ttm in slices:
            k_arr = np.asarray(k, dtype=float)
            iv_arr = np.asarray(iv, dtype=float)
            if k_arr.size == 0 or iv_arr.size == 0:
                raise ValueError("k and iv must contain at least one quote")
            if k_arr.shape != iv_arr.shape:
                raise ValueError("k and iv must have the same shape")
            w_obs = iv_arr**2 * ttm
            data.append((k_arr, w_obs, float(ttm)))
            atm = float(np.interp(0.0, k_arr, w_obs))
            thetas0.append(max(atm, 1e-4))
        order = np.argsort([ttm for _, _, ttm in data])
        data = [data[i] for i in order]
        theta_nodes = np.array([thetas0[i] for i in order], dtype=float)
        theta0 = theta_nodes[0]
        increments0 = np.maximum(np.diff(theta_nodes), 0.0)
        ttms = [data[i][2] for i in range(len(data))]

        x0 = [0.0, 1.0, 0.5, theta0, *increments0]

        def theta_from_params(x: np.ndarray) -> np.ndarray:
            theta = np.empty(len(data), dtype=float)
            theta[0] = x[3]
            if theta.size > 1:
                theta[1:] = theta[0] + np.cumsum(x[4:])
            return theta

        def residuals(x: np.ndarray) -> np.ndarray:
            rho, eta, gamma = x[0], x[1], x[2]
            theta_nodes = theta_from_params(x)
            res = []
            for (k_arr, w_obs, ttm), theta in zip(data, theta_nodes):
                phi = eta / (theta**gamma * (1 + theta) ** (1 - gamma))
                pk = phi * k_arr
                w_fit = (
                    0.5 * theta * (1 + rho * pk + np.sqrt((pk + rho) ** 2 + 1 - rho**2))
                )
                # normalise by ttm so residuals are in variance (sigma^2) space,
                # otherwise short maturities carry tiny total variance and are
                # outvoted by long maturities in the joint least squares
                res.append((w_fit - w_obs) / ttm)
            return np.concatenate(res)

        n = len(data)
        result = least_squares(
            residuals,
            x0,
            bounds=(
                [-1.0 + 1e-6, 1e-6, 1e-6, 1e-8] + [0.0] * (n - 1),
                [1.0 - 1e-6, np.inf, 1.0, np.inf] + [np.inf] * (n - 1),
            ),
        )

        rho, eta, gamma = result.x[:3]
        theta = theta_from_params(result.x)
        return cls(
            rho=to_decimal(round(rho, 10)),
            eta=to_decimal(round(eta, 10)),
            gamma=to_decimal(round(gamma, 10)),
            variance_curve=VarianceCurve(
                ttm=[to_decimal(round(ttm, 10)) for ttm in ttms],
                theta=[to_decimal(round(value, 10)) for value in theta],
            ),
        )

    @classmethod
    def fit_vol_surface(
        cls,
        surface: Annotated[
            VolSurface[Any],
            Doc(
                "Volatility surface with calculated implied volatilities and "
                "converged options"
            ),
        ],
    ) -> Self:
        """Fit an SSVI model to a volatility surface."""
        slices = []
        for index, maturity in enumerate(surface.maturities):
            options = list(surface.option_prices(index=index, converged=True))
            if not options:
                continue
            log_strike = np.array([option.log_strike for option in options])
            order = np.argsort(log_strike)
            slices.append(
                (
                    log_strike[order],
                    np.array([option.iv for option in options])[order],
                    maturity.ttm(surface.ref_date),
                )
            )
        return cls.fit_surface(slices)
