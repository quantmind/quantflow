from __future__ import annotations

from collections.abc import Sequence
from decimal import Decimal

import numpy as np
from numpy.typing import ArrayLike
from pydantic import BaseModel, Field
from scipy.optimize import least_squares
from typing_extensions import Annotated, Doc

from quantflow.utils.numbers import ONE, ZERO, to_decimal


class SSVI(BaseModel, extra="forbid"):
    r"""Surface SVI (SSVI) parametrisation of the implied volatility smile,
    introduced by
    [Gatheral and Jacquier (2014)](../../bibliography.md#gatheral_jacquier).

    The SSVI parametrisation expresses the total implied variance
    $w(k) = \sigma^2(k) \cdot \tau$ as a function of log-strike
    $k = \log(K/F)$ and the at-the-money total variance $\theta$:

    \begin{equation}
        w(k) = \frac{\theta}{2}\left[1 + \rho \varphi(\theta) k
        + \sqrt{\left(\varphi(\theta) k + \rho\right)^2 + 1 - \rho^2}\right]
    \end{equation}

    The shape function $\varphi$ uses the power law form:

    \begin{equation}
        \varphi(\theta) = \frac{\eta}{\theta^\gamma (1 + \theta)^{1 - \gamma}}
    \end{equation}

    Each instance represents a single maturity slice through its $\theta$
    parameter, while $\rho$, $\eta$ and $\gamma$ are global parameters shared
    across the whole surface.

    Use [fit][.fit] to calibrate a single slice, or [fit_surface][.fit_surface]
    to jointly calibrate the global parameters across several maturities. A
    surface built this way is free of calendar spread arbitrage provided the
    fitted $\theta$ are non decreasing in maturity.
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
    theta: Decimal = Field(
        gt=ZERO,
        description=(
            "At-the-money total implied variance of the slice, "
            "$\\theta = \\sigma_{ATM}^2 \\cdot \\tau$. This is the only "
            "maturity-dependent parameter. Must be strictly positive."
        ),
    )

    def phi(self) -> float:
        r"""Power law shape function $\varphi(\theta)$ evaluated at the slice
        at-the-money total variance $\theta$."""
        theta = float(self.theta)
        eta = float(self.eta)
        gamma = float(self.gamma)
        return eta / (theta**gamma * (1 + theta) ** (1 - gamma))

    def total_variance(
        self,
        k: Annotated[ArrayLike, Doc("Log-moneyness log(K/F), scalar or array")],
    ) -> np.ndarray:
        r"""Total implied variance $w(k)$.

        Returns an array of the same shape as $k$.
        """
        k_arr = np.asarray(k, dtype=float)
        theta = float(self.theta)
        rho = float(self.rho)
        phi = self.phi()
        pk = phi * k_arr
        return 0.5 * theta * (1 + rho * pk + np.sqrt((pk + rho) ** 2 + 1 - rho**2))

    def iv(
        self,
        k: Annotated[ArrayLike, Doc("Log-moneyness log(K/F), scalar or array")],
        ttm: Annotated[float, Doc("Time to maturity in years")],
    ) -> np.ndarray:
        r"""Implied volatility $\sigma(k) = \sqrt{w(k) / \tau}$.

        Returns an array of the same shape as $k$. The SSVI total variance is
        strictly positive for $|\rho| < 1$, so no clipping is required.
        """
        return np.sqrt(self.total_variance(k) / ttm)

    def no_butterfly_arbitrage(self) -> bool:
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
        theta = float(self.theta)
        rho = abs(float(self.rho))
        phi = self.phi()
        return theta * phi * (1 + rho) < 4 and theta * phi * phi * (1 + rho) <= 4

    @classmethod
    def fit(
        cls,
        k: Annotated[ArrayLike, Doc("Log-moneyness log(K/F) for each option")],
        iv: Annotated[ArrayLike, Doc("Observed implied volatilities")],
        ttm: Annotated[float, Doc("Time to maturity in years")],
        gamma: Annotated[
            float,
            Doc(
                "Fixed power law exponent. A single slice only identifies the "
                "value of the shape function, not the split between eta and "
                "gamma, so gamma is held fixed"
            ),
        ] = 0.5,
    ) -> SSVI:
        """Fit an SSVI slice to observed implied volatilities via non-linear
        least squares.

        Minimises the sum of squared differences between observed and model
        total variances, fitting rho, eta and theta with gamma held fixed.
        """
        k_arr = np.asarray(k, dtype=float)
        iv_arr = np.asarray(iv, dtype=float)
        w_obs = iv_arr**2 * ttm

        atm_var = float(np.interp(0.0, k_arr, w_obs)) if k_arr.size else w_obs.mean()
        x0 = [0.0, 1.0, max(atm_var, 1e-4)]

        def residuals(x: list[float]) -> np.ndarray:
            rho, eta, theta = x
            phi = eta / (theta**gamma * (1 + theta) ** (1 - gamma))
            pk = phi * k_arr
            w_fit = 0.5 * theta * (1 + rho * pk + np.sqrt((pk + rho) ** 2 + 1 - rho**2))
            return w_fit - w_obs

        result = least_squares(
            residuals,
            x0,
            bounds=(
                [-1.0 + 1e-6, 1e-6, 1e-8],
                [1.0 - 1e-6, np.inf, np.inf],
            ),
        )
        rho, eta, theta = result.x
        return cls(
            rho=to_decimal(round(rho, 10)),
            eta=to_decimal(round(eta, 10)),
            gamma=to_decimal(round(gamma, 10)),
            theta=to_decimal(round(theta, 10)),
        )

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
    ) -> list[SSVI]:
        """Jointly fit the SSVI surface to several maturity slices via
        non-linear least squares.

        The global parameters rho, eta and gamma are shared across all slices,
        while each slice has its own at-the-money total variance theta.
        Returns one [SSVI][quantflow.options.ssvi.SSVI] instance per input
        slice, in the same order.
        """
        data = []
        thetas0 = []
        for k, iv, ttm in slices:
            k_arr = np.asarray(k, dtype=float)
            iv_arr = np.asarray(iv, dtype=float)
            w_obs = iv_arr**2 * ttm
            data.append((k_arr, w_obs))
            atm = float(np.interp(0.0, k_arr, w_obs)) if k_arr.size else w_obs.mean()
            thetas0.append(max(atm, 1e-4))

        x0 = [0.0, 1.0, 0.5, *thetas0]

        def residuals(x: np.ndarray) -> np.ndarray:
            rho, eta, gamma = x[0], x[1], x[2]
            res = []
            for (k_arr, w_obs), theta in zip(data, x[3:]):
                phi = eta / (theta**gamma * (1 + theta) ** (1 - gamma))
                pk = phi * k_arr
                w_fit = (
                    0.5 * theta * (1 + rho * pk + np.sqrt((pk + rho) ** 2 + 1 - rho**2))
                )
                res.append(w_fit - w_obs)
            return np.concatenate(res)

        n = len(data)
        result = least_squares(
            residuals,
            x0,
            bounds=(
                [-1.0 + 1e-6, 1e-6, 1e-6] + [1e-8] * n,
                [1.0 - 1e-6, np.inf, 1.0] + [np.inf] * n,
            ),
        )
        rho, eta, gamma = result.x[:3]
        return [
            cls(
                rho=to_decimal(round(rho, 10)),
                eta=to_decimal(round(eta, 10)),
                gamma=to_decimal(round(gamma, 10)),
                theta=to_decimal(round(theta, 10)),
            )
            for theta in result.x[3:]
        ]
