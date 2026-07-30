from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Self

import numpy as np
from numpy.typing import ArrayLike
from pydantic import BaseModel, Field, PrivateAttr
from scipy.optimize import minimize, minimize_scalar
from typing_extensions import Annotated, Doc

from quantflow.options.surface import VolSurface
from quantflow.utils.numbers import DecimalNumber, to_decimal
from quantflow.utils.types import FloatArray, FloatArrayLike, maybe_float

RHO_GRID_SIZE = 21
"""Number of correlation samples per search round in
[fit_surface][quantflow.options.ssvi.SSVI.fit_surface]."""

RHO_REFINEMENTS = 3
"""Number of correlation grid refinement rounds in
[fit_surface][quantflow.options.ssvi.SSVI.fit_surface]."""

RHO_LIMIT = 0.999
"""Largest correlation magnitude explored by the calibration."""


class SSVI(BaseModel, extra="forbid"):
    r"""eSSVI (extended Surface SVI) parametrisation of the implied
    volatility surface.

    The SSVI surface of
    [Gatheral and Jacquier (2014)](../../bibliography.md#gatheral_jacquier)
    is extended with a maturity dependent correlation, following
    [Hendriks and Martini (2019)](../../bibliography.md#hendriks_martini).
    Each maturity slice is described by three parameters stored at the node
    maturities:

    * the total ATM variance $\theta_\tau$
    * the curvature $\psi_\tau$
    * the correlation $\rho_\tau$

    The total implied variance $w(k) = \sigma^2(k) \cdot \tau$ at log-strike
    $k = \log(K/F)$ is

    \begin{equation}
        w(k, \tau) = \frac{\theta_\tau}{2}\left[1
        + \rho_\tau \varphi_\tau k
        + \sqrt{\left(\varphi_\tau k + \rho_\tau\right)^2
        + 1 - \rho_\tau^2}\right]
    \end{equation}

    where the shape function is the ratio of curvature and total ATM
    variance:

    \begin{equation}
        \varphi_\tau = \frac{\psi_\tau}{\theta_\tau}
    \end{equation}

    The parameters have direct smile interpretations. The total ATM variance
    sets the level of the slice, $w(0, \tau) = \theta_\tau$. The curvature
    and correlation set the slope and convexity of the total variance at the
    money:

    \begin{equation}
    \begin{aligned}
        \left.\frac{\partial w}{\partial k}\right|_{k=0}
            &= \rho_\tau \psi_\tau \\
        \left.\frac{\partial^2 w}{\partial k^2}\right|_{k=0}
            &= \frac{\psi_\tau^2 \left(1 - \rho_\tau^2\right)}{2 \theta_\tau}
    \end{aligned}
    \end{equation}

    The sign of the correlation therefore tilts the smile (negative for the
    left skew typical of equities) while the curvature scales both the tilt
    and the bend around the money.

    In the wings the total variance grows linearly,
    $w \to \frac{\psi_\tau (1 \pm \rho_\tau)}{2} |k|$ as $k \to \pm\infty$,
    so the curvature also sets the wing slopes. Lee's moment formula caps
    these slopes at 2, which is exactly the first butterfly condition
    $\psi_\tau (1 + |\rho_\tau|) < 4$ of
    [no_butterfly_arbitrage][.no_butterfly_arbitrage].

    The shape function acts as a moneyness rescaling: the log-strike enters
    the formula only through the product $\varphi_\tau k$, so $\varphi_\tau$
    measures how far a strike is from the money relative to the width of the
    smile at that maturity. Short maturities combine a small $\theta$ with a
    large $\varphi$ (the whole smile lives within a few percent of the
    forward), long maturities the opposite.

    Absence of calendar spread arbitrage requires both $\theta$ and $\psi$
    to be non decreasing in maturity, and the nodes are validated
    accordingly. Between nodes, the quantities $\theta$, $\psi$ and
    $\rho \psi$ are interpolated linearly, the natural interpolation of
    [Corbetta et al. (2019)](../../bibliography.md#ssvi_calibration), which
    preserves the absence of static arbitrage of the interpolated slices.
    Outside the node range the surface extrapolates flat.

    Use [fit_surface][.fit_surface] to calibrate the slice parameters with
    the star calibration algorithm of the same paper, which enforces the
    absence of butterfly and calendar spread arbitrage by construction.
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
    psi: list[DecimalNumber] = Field(
        description=(
            "Curvatures $\\psi_i = \\theta_i \\varphi_i$ at each maturity, "
            "same length as ttm, positive and non decreasing"
        ),
    )
    rho: list[DecimalNumber] = Field(
        description=(
            "Correlations at each maturity, same length as ttm, each strictly "
            "inside the interval (-1, 1)"
        ),
    )

    _ttm: FloatArray = PrivateAttr(default_factory=lambda: np.empty(0))
    _theta: FloatArray = PrivateAttr(default_factory=lambda: np.empty(0))
    _psi: FloatArray = PrivateAttr(default_factory=lambda: np.empty(0))
    _rho: FloatArray = PrivateAttr(default_factory=lambda: np.empty(0))

    def model_post_init(self, context: object) -> None:
        """Cache nodes and validate the no calendar arbitrage conditions."""
        if len(self.ttm) != len(self.theta):
            raise ValueError("ttm and theta must have equal length")
        if len(self.ttm) != len(self.psi):
            raise ValueError("ttm and psi must have equal length")
        if len(self.ttm) != len(self.rho):
            raise ValueError("ttm and rho must have equal length")
        if not self.ttm:
            raise ValueError("at least one surface node is required")
        ttm = np.array([float(t) for t in self.ttm], dtype=float)
        theta = np.array([float(t) for t in self.theta], dtype=float)
        psi = np.array([float(p) for p in self.psi], dtype=float)
        rho = np.array([float(r) for r in self.rho], dtype=float)
        if np.any(ttm <= 0):
            raise ValueError("ttm nodes must be strictly positive")
        if np.any(np.diff(ttm) <= 0):
            raise ValueError("ttm nodes must be strictly increasing")
        if np.any(theta <= 0):
            raise ValueError("theta nodes must be strictly positive")
        if np.any(np.diff(theta) < 0):
            raise ValueError("theta nodes must be non decreasing")
        if np.any(psi <= 0):
            raise ValueError("psi nodes must be strictly positive")
        if np.any(np.diff(psi) < 0):
            raise ValueError("psi nodes must be non decreasing")
        if np.any(np.abs(rho) >= 1):
            raise ValueError("rho nodes must satisfy |rho| < 1")
        self._ttm = ttm
        self._theta = theta
        self._psi = psi
        self._rho = rho

    def _interp(self, ttm: ArrayLike, values: FloatArray) -> FloatArray:
        tau = np.asarray(ttm, dtype=float)
        return np.asarray(np.interp(tau, self._ttm, values), dtype=float)

    def atm_variance(
        self,
        ttm: Annotated[ArrayLike, Doc("Time to maturity in years, scalar or array")],
    ) -> FloatArrayLike:
        r"""Interpolated total ATM variance $\theta(\tau)$."""
        return maybe_float(self._interp(ttm, self._theta))

    def curvature(
        self,
        ttm: Annotated[ArrayLike, Doc("Time to maturity in years, scalar or array")],
    ) -> FloatArrayLike:
        r"""Interpolated curvature $\psi(\tau)$."""
        return maybe_float(self._interp(ttm, self._psi))

    def correlation(
        self,
        ttm: Annotated[ArrayLike, Doc("Time to maturity in years, scalar or array")],
    ) -> FloatArrayLike:
        r"""Interpolated correlation $\rho(\tau)$.

        The interpolation is linear in $\rho \psi$ and $\psi$, so the
        correlation is their ratio rather than a direct linear interpolation.
        """
        rho_psi = self._interp(ttm, self._rho * self._psi)
        psi = self._interp(ttm, self._psi)
        return maybe_float(rho_psi / psi)

    def phi(
        self,
        ttm: Annotated[ArrayLike, Doc("Time to maturity in years, scalar or array")],
    ) -> FloatArrayLike:
        r"""Shape function $\varphi_\tau = \psi_\tau / \theta_\tau$."""
        return maybe_float(
            self._interp(ttm, self._psi) / self._interp(ttm, self._theta)
        )

    def total_variance(
        self,
        k: Annotated[ArrayLike, Doc("Log-moneyness log(K/F), scalar or array")],
        ttm: Annotated[ArrayLike, Doc("Time to maturity in years, scalar or array")],
    ) -> FloatArrayLike:
        r"""Total implied variance $w(k, \tau)$.

        Returns an array broadcast from the shapes of $k$ and $\tau$.
        """
        k_arr = np.asarray(k, dtype=float)
        theta = self._interp(ttm, self._theta)
        rho = np.asarray(self.correlation(ttm), dtype=float)
        pk = self._interp(ttm, self._psi) / theta * k_arr
        w = 0.5 * theta * (1 + rho * pk + np.sqrt((pk + rho) ** 2 + 1 - rho**2))
        return maybe_float(np.asarray(w, dtype=float))

    def iv(
        self,
        k: Annotated[ArrayLike, Doc("Log-moneyness log(K/F), scalar or array")],
        ttm: Annotated[ArrayLike, Doc("Time to maturity in years, scalar or array")],
    ) -> FloatArrayLike:
        r"""Implied volatility $\sigma(k, \tau) = \sqrt{w(k, \tau) / \tau}$.

        Returns an array of the same shape as $k$. The eSSVI total variance is
        strictly positive for $|\rho| < 1$, so no clipping is required.
        """
        tau = np.asarray(ttm, dtype=float)
        return maybe_float(np.asarray(np.sqrt(self.total_variance(k, ttm) / tau)))

    def no_butterfly_arbitrage(
        self,
        ttm: Annotated[
            float | None,
            Doc("Optional maturity to check. All nodes are checked when omitted"),
        ] = None,
    ) -> bool:
        r"""True if the slice satisfies the sufficient conditions for absence
        of butterfly arbitrage.

        The conditions, from Theorem 4.2 of
        [Gatheral and Jacquier (2014)](../../bibliography.md#gatheral_jacquier),
        expressed in terms of the curvature $\psi = \theta \varphi$, are:

        \begin{equation}
        \begin{aligned}
            \psi (1 + |\rho_\tau|) &< 4 \\
            \frac{\psi^2}{\theta} (1 + |\rho_\tau|) &\leq 4
        \end{aligned}
        \end{equation}
        """
        ttms = [ttm] if ttm is not None else [float(t) for t in self.ttm]
        for tau in ttms:
            theta = float(self.atm_variance(tau))
            psi = float(self.curvature(tau))
            rho = abs(float(self.correlation(tau)))
            if not (psi * (1 + rho) < 4 and psi * psi * (1 + rho) / theta <= 4):
                return False
        return True

    def no_calendar_arbitrage(self) -> bool:
        r"""True if the surface nodes satisfy the conditions for absence of
        calendar spread arbitrage.

        The conditions, necessary and sufficient from
        [Hendriks and Martini (2019)](../../bibliography.md#hendriks_martini),
        require $\theta$ and $\psi$ non decreasing (enforced by the node
        validation) together with, for consecutive maturities
        $\tau_1 < \tau_2$:

        \begin{equation}
            \left|\rho_2 \psi_2 - \rho_1 \psi_1\right| \leq \psi_2 - \psi_1
        \end{equation}

        Because $\theta$, $\psi$ and $\rho \psi$ interpolate linearly, node
        conditions extend to the whole interpolated surface (Section 5 of
        [Corbetta et al. (2019)](../../bibliography.md#ssvi_calibration)).
        """
        if self._psi.size < 2:
            return True
        psi = self._psi
        rho_psi = self._rho * self._psi
        return bool(np.all(np.abs(np.diff(rho_psi)) <= np.diff(psi) + 1e-9))

    def no_static_arbitrage(self) -> bool:
        """True if the surface satisfies the implemented static checks."""
        return self.no_butterfly_arbitrage() and self.no_calendar_arbitrage()

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
        weight_decay: Annotated[
            float,
            Doc(
                "Exponential decay rate of the quote weights with distance "
                "from the money, measured by the convexity adjusted "
                "moneyness. 0, the default, weights all quotes equally, "
                "larger values concentrate the fit at the money. "
                "Must be non negative"
            ),
        ] = 0.5,
    ) -> Self:
        r"""Calibrate the eSSVI surface slice by slice with the star
        calibration algorithm of
        [Corbetta et al. (2019)](../../bibliography.md#ssvi_calibration).

        Slices are sorted by maturity and calibrated going forward. The star
        calibration provides the starting point of each slice: the slice is
        anchored to the quote closest to the ATM forward, which eliminates
        the total ATM variance analytically ($\theta = \theta^* - \rho \psi
        k^*$), and the remaining pair $(\rho, \psi)$ is found by sampling the
        correlation on a progressively refined grid and, for each sample,
        minimising over the curvature with a bounded one dimensional search.
        The curvature bounds enforce the butterfly conditions and the
        calendar conditions with respect to the previously calibrated slice.

        The three parameters are then refined jointly with a direct search
        that releases the ATM anchor, letting all quotes set the level of
        the slice, while rejecting any candidate that violates the butterfly
        or calendar conditions, so the fitted surface remains free of static
        arbitrage by construction.

        The objective is the mean absolute implied variance error with an
        exponential weight per quote:

        \begin{equation}
            e^{-\lambda |d|} \quad \text{with} \quad
            d = \frac{k + \theta^* / 2}{\sqrt{\theta^*}}
        \end{equation}

        where $\lambda$ is the weight_decay parameter and $d$ is the
        [convexity adjusted moneyness](../../glossary.md#moneyness-convexity-adjusted)
        evaluated at the ATM total variance $\theta^*$.

        The weight is centered at $d = 0$, the median of the risk-neutral
        distribution, and concentrates the fit around the money where quotes
        are most reliable, while its slow exponential decay keeps far from
        the money quotes contributing to the fit.

        The error is measured in implied variance rather than implied
        volatility, which gives the smile wings, where the variance is
        largest, a stronger pull on the fit.
        """
        if not slices:
            raise ValueError("at least one maturity slice is required")
        data = []
        for k, iv, ttm in slices:
            k_arr = np.asarray(k, dtype=float)
            iv_arr = np.asarray(iv, dtype=float)
            if k_arr.size == 0 or iv_arr.size == 0:
                raise ValueError("k and iv must contain at least one quote")
            if k_arr.shape != iv_arr.shape:
                raise ValueError("k and iv must have the same shape")
            data.append((k_arr, iv_arr, float(ttm)))
        data.sort(key=lambda item: item[2])
        ttms = []
        thetas = []
        psis = []
        rhos = []
        prev: tuple[float, float, float] | None = None
        for k_arr, iv_arr, ttm in data:
            theta, psi, rho = _fit_slice(k_arr, iv_arr, ttm, prev, weight_decay)
            if prev is not None:
                # guard the non decreasing validation against rounding noise
                theta = max(theta, prev[0])
                psi = max(psi, prev[1])
            ttms.append(ttm)
            thetas.append(theta)
            psis.append(psi)
            rhos.append(rho)
            prev = (theta, psi, rho * psi)
        return cls(
            ttm=[to_decimal(round(value, 10)) for value in ttms],
            theta=[to_decimal(round(value, 10)) for value in thetas],
            psi=[to_decimal(round(value, 10)) for value in psis],
            rho=[to_decimal(round(value, 10)) for value in rhos],
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
        weight_decay: Annotated[
            float,
            Doc(
                "Exponential decay rate of the quote weights with distance "
                "from the money, see [fit_surface][..fit_surface]"
            ),
        ] = 0.0,
    ) -> Self:
        """Fit an eSSVI model to a volatility surface."""
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
        return cls.fit_surface(slices, weight_decay=weight_decay)


def _fit_slice(
    k: FloatArray,
    iv: FloatArray,
    ttm: float,
    prev: tuple[float, float, float] | None,
    weight_decay: float,
) -> tuple[float, float, float]:
    """Calibrate one eSSVI slice, returning (theta, psi, rho).

    An anchored star calibration provides an arbitrage free starting point,
    refined by a joint direct search of the three parameters. prev holds
    (theta, psi, rho*psi) of the previously calibrated slice and activates
    the calendar spread bounds when present.
    """
    var_obs = iv * iv
    w_obs = var_obs * ttm
    anchor = int(np.argmin(np.abs(k)))
    k_star = float(k[anchor])
    # average the quotes at the anchor strike (typically bid and ask)
    theta_star = float(np.mean(w_obs[np.isclose(k, k_star)]))
    if prev is not None:
        # repair a decreasing ATM variance in the data, which would otherwise
        # leave no feasible parameters for the slice
        theta_star = max(theta_star, prev[0])
    # convexity adjusted moneyness at the ATM volatility (minus d2)
    d = (k + theta_star / 2) / np.sqrt(theta_star)
    weight = np.exp(-weight_decay * np.abs(d))

    def variance_error(theta: float, psi: float, rho: float) -> float:
        pk = psi / theta * k
        w = 0.5 * theta * (1 + rho * pk + np.sqrt((pk + rho) ** 2 + 1 - rho**2))
        return float(np.mean(weight * np.abs(w / ttm - var_obs)))

    def objective(psi: float, rho: float) -> float:
        return variance_error(theta_star - rho * psi * k_star, psi, rho)

    def feasible(theta: float, psi: float, rho: float) -> bool:
        if theta <= 0 or psi <= 0 or abs(rho) > RHO_LIMIT:
            return False
        a = 1 + abs(rho)
        if psi * a >= 4 or psi * psi * a / theta > 4:
            return False
        if prev is not None:
            theta_prev, psi_prev, rho_psi_prev = prev
            if theta < theta_prev or psi < psi_prev:
                return False
            if abs(rho * psi - rho_psi_prev) > psi - psi_prev + 1e-12:
                return False
        return True

    def refine_objective(x: FloatArray) -> float:
        theta, psi, rho = (float(value) for value in x)
        if not feasible(theta, psi, rho):
            return np.inf
        return variance_error(theta, psi, rho)

    def psi_bounds(rho: float) -> tuple[float, float] | None:
        a = 1 + abs(rho)
        b = rho * k_star
        upper = min(
            4 / a,
            -2 * b / a + np.sqrt(4 * b * b / (a * a) + 4 * theta_star / a),
        )
        lower = 1e-8
        if prev is None:
            # keep theta strictly positive
            if b > 0:
                upper = min(upper, (1 - 1e-9) * theta_star / b)
        else:
            theta_prev, psi_prev, rho_psi_prev = prev
            lower = max(
                lower,
                psi_prev,
                (psi_prev - rho_psi_prev) / (1 - rho),
                (psi_prev + rho_psi_prev) / (1 + rho),
            )
            gap = theta_star - theta_prev
            if b > 0:
                upper = min(upper, gap / b)
            elif b < 0:
                lower = max(lower, gap / b)
            elif gap < 0:
                return None
        if lower > upper:
            return None
        return lower, upper

    def best_for_rho(rho: float) -> tuple[float, float] | None:
        bounds = psi_bounds(rho)
        if bounds is None:
            return None
        lower, upper = bounds
        if upper - lower < 1e-9:
            return objective(lower, rho), lower
        result = minimize_scalar(
            objective,
            args=(rho,),
            bounds=(lower, upper),
            method="bounded",
            options={"xatol": 1e-6},
        )
        return float(result.fun), float(result.x)

    best: tuple[float, float, float] | None = None
    grid = np.linspace(-RHO_LIMIT, RHO_LIMIT, RHO_GRID_SIZE)
    for _ in range(RHO_REFINEMENTS + 1):
        for rho in grid:
            found = best_for_rho(float(rho))
            if found is not None and (best is None or found[0] < best[0]):
                best = (found[0], found[1], float(rho))
        if best is None:
            raise ValueError(
                f"no arbitrage free eSSVI parameters for the slice at ttm={ttm}"
            )
        step = float(grid[1] - grid[0])
        grid = np.linspace(
            max(-RHO_LIMIT, best[2] - step),
            min(RHO_LIMIT, best[2] + step),
            RHO_GRID_SIZE,
        )
    assert best is not None
    _, best_psi, best_rho = best
    theta = theta_star - best_rho * best_psi * k_star
    # joint refinement releasing the ATM anchor, started from the feasible
    # star solution and kept only when it improves the objective
    refined = minimize(
        refine_objective,
        np.array([theta, best_psi, best_rho]),
        method="Nelder-Mead",
        options={"maxiter": 2000, "xatol": 1e-8, "fatol": 1e-10},
    )
    if np.isfinite(refined.fun) and refined.fun <= best[0]:
        theta, best_psi, best_rho = (float(value) for value in refined.x)
    return theta, best_psi, best_rho
