from __future__ import annotations

from decimal import Decimal

import numpy as np
from numpy.typing import ArrayLike
from pydantic import BaseModel, Field
from scipy.optimize import least_squares
from typing_extensions import Annotated, Doc

from quantflow.utils.numbers import ONE, ZERO, to_decimal


class SVI(BaseModel, extra="forbid"):
    r"""Gatheral's Stochastic Volatility Inspired (SVI) parametrisation of the
    implied volatility smile.

    The raw SVI parametrisation expresses the total implied variance
    $w(k) = \sigma^2(k) \cdot \tau$ as a function of log-strike
    $k = \log(K/F)$:

    \begin{equation}
        w(k) = a + b \left[\rho (k - m) + \sqrt{(k - m)^2 + \theta^2}\right]
    \end{equation}

    References:
        Gatheral, J. (2004). A parsimonious arbitrage-free implied volatility
        parametrization with application to the valuation of volatility
        derivatives. Global Derivatives, Madrid.

        Gatheral, J. and Jacquier, A. (2014). Arbitrage-free SVI volatility
        surfaces. Quantitative Finance, 14(1), 59-71.
    """

    a: Decimal = Field(
        description=(
            "Vertical shift of the smile: overall level of total implied variance. "
            "Must satisfy $a + b \\theta \\sqrt{1 - \\rho^2} \\geq 0$ to "
            "avoid negative variance."
        )
    )
    b: Decimal = Field(
        ge=ZERO,
        description=(
            "Angle between the left and right asymptotes of the smile. "
            "Controls the overall steepness of the wings. Must be non-negative."
        ),
    )
    rho: Decimal = Field(
        gt=-ONE,
        lt=ONE,
        description=(
            "Correlation parameter controlling the skew of the smile. "
            "Negative values produce a left-skewed smile (typical for equities), "
            "positive values produce a right skew. Must satisfy $|\\rho| < 1$."
        ),
    )
    m: Decimal = Field(
        description=(
            "Location parameter: the log-moneyness at the vertex of the smile. "
            "Shifts the smile horizontally. A value of zero centres the smile "
            "at the forward."
        )
    )
    theta: Decimal = Field(
        gt=ZERO,
        description=(
            "Smoothness parameter controlling the curvature of the smile around "
            "the vertex. Larger values produce a flatter region near $m$. "
            "Must be strictly positive."
        ),
    )

    def total_variance(
        self,
        k: Annotated[ArrayLike, Doc("Log-moneyness log(K/F), scalar or array")],
    ) -> np.ndarray:
        r"""Total implied variance $w(k)$.

        Returns an array of the same shape as $k$.
        """
        k_arr = np.asarray(k, dtype=float)
        a = float(self.a)
        b = float(self.b)
        rho = float(self.rho)
        m = float(self.m)
        theta = float(self.theta)
        km = k_arr - m
        return a + b * (rho * km + np.sqrt(km**2 + theta**2))

    def implied_vol(
        self,
        k: Annotated[ArrayLike, Doc("Log-moneyness log(K/F), scalar or array")],
        ttm: Annotated[float, Doc("Time to maturity in years")],
    ) -> np.ndarray:
        r"""Implied volatility $\sigma(k) = \sqrt{w(k) / \tau}$.

        Returns an array of the same shape as $k$. Values are set to zero where
        total variance is non-positive.
        """
        w = self.total_variance(k)
        return np.where(w > 0, np.sqrt(np.maximum(w, 0.0) / ttm), 0.0)

    @classmethod
    def fit(
        cls,
        k: Annotated[ArrayLike, Doc("Log-moneyness log(K/F) for each option")],
        iv: Annotated[ArrayLike, Doc("Observed implied volatilities")],
        ttm: Annotated[float, Doc("Time to maturity in years")],
    ) -> SVI:
        """Fit SVI smile to observed implied volatilities via non-linear least squares.

        Minimises the sum of squared differences between observed and model
        total variances.
        """
        k_arr = np.asarray(k, dtype=float)
        iv_arr = np.asarray(iv, dtype=float)
        w_obs = iv_arr**2 * ttm

        atm_var = float(np.interp(0.0, k_arr, w_obs)) if k_arr.size else w_obs.mean()
        x0 = [atm_var * 0.9, 0.1, 0.0, 0.0, 0.1]

        def residuals(x: list[float]) -> np.ndarray:
            a, b, rho, m, theta = x
            km = k_arr - m
            w_fit = a + b * (rho * km + np.sqrt(km**2 + theta**2))
            return w_fit - w_obs

        result = least_squares(
            residuals,
            x0,
            bounds=(
                [-np.inf, 0.0, -1.0 + 1e-6, -np.inf, 1e-6],
                [np.inf, np.inf, 1.0 - 1e-6, np.inf, np.inf],
            ),
        )
        a, b, rho, m, theta = result.x
        return cls(
            a=to_decimal(round(a, 10)),
            b=to_decimal(round(b, 10)),
            rho=to_decimal(round(rho, 10)),
            m=to_decimal(round(m, 10)),
            theta=to_decimal(round(theta, 10)),
        )
