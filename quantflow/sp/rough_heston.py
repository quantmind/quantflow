from __future__ import annotations

from typing import Self

import numpy as np
from pydantic import Field
from scipy.special import gamma
from typing_extensions import Annotated, Doc

from quantflow.ta.paths import Paths
from quantflow.utils.types import FloatArrayLike, Vector

from .base import StochasticProcess1D
from .cir import CIR


class RoughHeston(StochasticProcess1D):
    r"""The rough Heston stochastic volatility model.

    The rough Heston model of
    [El Euch and Rosenbaum](../../bibliography.md#eleuch_rosenbaum) replaces the
    Markovian square-root variance of the classical
    [Heston][quantflow.sp.heston.Heston] model with a rough (fractional)
    square-root process. The variance has long memory and is not diffusive at
    short time scales (see
    [Volatility is rough](../../bibliography.md#gatheral_jaisson_rosenbaum)),
    which produces the steep short-maturity skew observed in equity and crypto
    option markets without resorting to jumps.

    Following the same time-changed convention as the classical
    [Heston][quantflow.sp.heston.Heston] model, the driving log-price $x_t$ is a
    Brownian motion time changed by the rough variance $v_t$,

    \begin{equation}
        \begin{aligned}
            d x_t &= \sqrt{v_t}\, d w^1_t \\
            v_t &= v_0 + \frac{1}{\Gamma(\alpha)} \int_0^t (t-s)^{\alpha-1}
                \kappa(\theta - v_s)\, ds
                + \frac{1}{\Gamma(\alpha)} \int_0^t (t-s)^{\alpha-1}
                \nu \sqrt{v_s}\, d w^2_s
        \end{aligned}
    \end{equation}

    with correlation $\rho\, dt = {\tt E}[d w^1 d w^2]$ and roughness index
    $\alpha = H + \tfrac{1}{2}$, where $H$ is the
    [Hurst exponent](../../glossary.md#hurst-exponent). The choice $H = 1/2$
    (so $\alpha = 1$) recovers the classical Heston model. The deterministic
    [convexity correction](../../theory/convexity_correction.md) that enforces
    the martingale property is applied by the pricing layer, exactly as for the
    other stochastic-volatility models.

    The characteristic function has no closed form. It is computed by solving a
    fractional Riccati equation numerically with a fractional Adams
    predictor-corrector scheme (see
    [characteristic_exponent][.characteristic_exponent]).
    """

    variance_process: CIR = Field(
        default_factory=CIR,
        description=(
            "The rough variance is a fractional extension of the "
            "[Cox-Ingersoll-Ross][quantflow.sp.cir.CIR] process, from which the "
            "mean-reversion speed, long-term variance, vol of vol and initial "
            "variance are taken"
        ),
    )
    rho: float = Field(
        default=0,
        ge=-1,
        le=1,
        description=(
            "Correlation between the Brownian motions, it provides "
            "the leverage effect and therefore the skewness of the distribution"
        ),
    )
    hurst: float = Field(
        default=0.1,
        gt=0,
        le=0.5,
        description=(
            "Hurst exponent H of the variance process, the roughness index is "
            "alpha = H + 1/2 in (1/2, 1]. Lower H means rougher paths and a "
            "steeper short-maturity skew. H = 1/2 recovers the classical Heston "
            "model"
        ),
    )
    adams_steps: int = Field(
        default=200,
        ge=10,
        description=(
            "Number of time steps of the fixed-step fractional Adams solver used "
            "for the Riccati equation on the interval [0, t]. The solver cost is "
            "quadratic in this value for each frequency, so increase it only when "
            "short-maturity accuracy requires it"
        ),
    )

    @property
    def alpha(self) -> float:
        """Roughness index of the variance process, $\\alpha = H + 1/2$."""
        return self.hurst + 0.5

    @classmethod
    def create(
        cls,
        *,
        rate: Annotated[float, Doc("Initial rate of the variance process")] = 1.0,
        vol: Annotated[
            float,
            Doc(
                "Volatility of the price process, normalized by the "
                "square root of time, as time tends to infinity "
                "(the long term standard deviation)"
            ),
        ] = 0.5,
        kappa: Annotated[
            float,
            Doc("Mean reversion speed for the variance process"),
        ] = 1,
        sigma: Annotated[
            float, Doc("Volatility of the variance process (a.k.a. the vol of vol)")
        ] = 0.8,
        rho: Annotated[
            float,
            Doc(
                "Correlation between the Brownian motions of the "
                "variance and price processes"
            ),
        ] = 0,
        theta: Annotated[
            float | None,
            Doc(
                "Long-term mean of the variance process. "
                r"If `None`, it defaults to the variance given by ${\tt vol}^2$."
            ),
        ] = None,
        hurst: Annotated[
            float,
            Doc("Hurst exponent H in (0, 1/2]; H = 1/2 recovers the Heston model"),
        ] = 0.1,
        adams_steps: Annotated[
            int, Doc("Number of steps of the fractional Adams Riccati solver")
        ] = 200,
    ) -> Self:
        r"""Create a rough Heston model.

        To understand the parameters lets introduce the following notation:

        \begin{align}
            {\tt var} &= {\tt vol}^2 \\
            v_0 &= {\tt rate}\cdot{\tt var}
        \end{align}
        """
        variance = vol * vol
        return cls(
            variance_process=CIR(
                rate=rate * variance,
                kappa=kappa,
                sigma=sigma,
                theta=theta if theta is not None else variance,
            ),
            rho=rho,
            hurst=hurst,
            adams_steps=adams_steps,
        )

    def characteristic_exponent(
        self,
        t: Annotated[FloatArrayLike, Doc("Time horizon (must be a scalar)")],
        u: Annotated[Vector, Doc("Characteristic exponent argument")],
    ) -> Vector:
        r"""Characteristic exponent of the rough Heston model.

        The characteristic function is semi-closed and given by

        \begin{equation}
            \Phi_{x_t, u} = \exp\left(
                \kappa\theta\, I^1 \psi(t)
                + v_0\, I^{1-\alpha} \psi(t)
            \right)
        \end{equation}

        where $I^1$ is the ordinary integral, $I^{1-\alpha}$ is the
        Riemann-Liouville fractional integral of order $1-\alpha$, and
        $\psi(\cdot) = \psi(u, \cdot)$ solves the fractional Riccati equation
        $\psi(u, 0) = 0$,

        \begin{equation}
            D^\alpha \psi(t) = -\tfrac{1}{2}u^2
                + (i u \rho \nu - \kappa)\, \psi(t)
                + \tfrac{1}{2}\nu^2\, \psi(t)^2
        \end{equation}

        with $\kappa$ the mean-reversion speed, $\theta$ the long-term variance,
        $\nu$ the vol of vol, $\rho$ the correlation and $v_0$ the initial
        variance. The equation is integrated with a fractional Adams
        predictor-corrector scheme on a grid of
        [adams_steps][..adams_steps] points.

        The characteristic exponent returned is $\phi = -\log \Phi$, consistent
        with the library convention $\Phi = e^{-\phi}$.
        """
        if np.ndim(t) != 0:
            raise ValueError(
                "RoughHeston.characteristic_exponent requires a scalar time "
                "horizon; the fractional Riccati equation is solved on a single "
                "interval [0, t]"
            )
        u_arr = np.atleast_1d(np.asarray(u, dtype=complex))
        psi = self._solve_fractional_riccati(float(t), u_arr)
        vp = self.variance_process
        dt = float(t) / self.adams_steps
        i1 = dt * (0.5 * psi[0] + psi[1:-1].sum(axis=0) + 0.5 * psi[-1])
        i_frac = self._fractional_integral_weights(1.0 - self.alpha) @ psi
        log_phi = vp.kappa * vp.theta * i1 + vp.rate * i_frac
        phi = -log_phi
        return phi[0] if np.ndim(u) == 0 else phi

    def _riccati_f(self, u: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Right-hand side of the fractional Riccati equation, vectorized over u."""
        vp = self.variance_process
        kappa = vp.kappa
        nu = vp.sigma
        return (
            -0.5 * u * u + (1j * u * self.rho * nu - kappa) * x + 0.5 * nu * nu * x * x
        )

    def _solve_fractional_riccati(self, t: float, u: np.ndarray) -> np.ndarray:
        """Fractional Adams (Diethelm-Ford-Freed) predictor-corrector solver.

        Returns the Riccati trajectory psi on the uniform grid of
        ``adams_steps + 1`` points, with shape ``(adams_steps + 1, u.size)``.
        Every operation is vectorized over the frequency grid ``u``.
        """
        alpha = self.alpha
        n = self.adams_steps
        h = t / n
        ha = h**alpha
        inv_ga1 = ha / gamma(alpha + 1.0)
        inv_ga2 = ha / gamma(alpha + 2.0)
        a1 = alpha + 1.0

        m = u.size
        psi = np.zeros((n + 1, m), dtype=complex)
        fh = np.zeros((n + 1, m), dtype=complex)
        fh[0] = self._riccati_f(u, psi[0])

        for k in range(n):
            j = np.arange(k + 1)
            # predictor (fractional Adams-Bashforth) weights
            b = (k + 1 - j) ** alpha - (k - j) ** alpha
            psi_pred = inv_ga1 * (b @ fh[: k + 1])
            f_pred = self._riccati_f(u, psi_pred)
            # corrector (fractional Adams-Moulton) weights
            p = k - j
            a = (p + 2) ** a1 + p**a1 - 2 * (p + 1) ** a1
            a[0] = k**a1 - (k - alpha) * (k + 1) ** alpha
            psi[k + 1] = inv_ga2 * (a @ fh[: k + 1] + f_pred)
            fh[k + 1] = self._riccati_f(u, psi[k + 1])
        return psi

    def _fractional_integral_weights(self, beta: float) -> np.ndarray:
        r"""Product-trapezoidal weights for the Riemann-Liouville integral.

        Returns the vector $c$ of length ``adams_steps + 1`` such that
        $I^\beta \psi(t) \approx c \cdot \psi$ on the solver grid. For
        $\beta = 0$ (i.e. $\alpha = 1$) the weights collapse to picking
        $\psi(t)$, recovering the Heston limit.
        """
        n = self.adams_steps
        b1 = beta + 1.0
        j = np.arange(n + 1)
        # middle-node weights (valid for 1 <= j <= n - 1); the (n - 1 - j) base
        # is negative only at j = n, whose weight is overwritten below, so clip
        # it to avoid a fractional power of a negative number
        low = np.clip(n - 1 - j, 0, None)
        c = (n + 1 - j) ** b1 + low**b1 - 2 * (n - j) ** b1
        # first node
        c[0] = (n - 1) ** b1 - (n - 1 - beta) * n**beta
        # last node (integrand endpoint)
        c[n] = 1.0
        h = 1.0 / n  # dt = t / n; the t**beta factor cancels with the psi grid step
        return (h**beta / gamma(beta + 2.0)) * c

    def sample(
        self,
        n: Annotated[int, Doc("Number of sample paths")],
        time_horizon: Annotated[float, Doc("Time horizon")] = 1,
        time_steps: Annotated[int, Doc("Number of discrete time steps")] = 100,
    ) -> Paths:
        dw1 = Paths.normal_draws(n, time_horizon, time_steps)
        dw2 = Paths.normal_draws(n, time_horizon, time_steps)
        return self.sample_from_draws(dw1, dw2)

    def sample_from_draws(
        self,
        draws: Annotated[
            Paths,
            Doc(
                "Pre-drawn standard normal increments for the variance Brownian motion"
            ),
        ],
        *args: Annotated[
            Paths,
            Doc(
                "Optional pre-drawn increments for the independent Brownian motion; "
                "new draws are generated if omitted"
            ),
        ],
    ) -> Paths:
        r"""Sample paths with an explicit Euler scheme on the Volterra equation.

        This is a simple approximate scheme: the fractional variance is
        discretized as a lower-triangular convolution of the power-law kernel
        $g(\tau) = \tau^{\alpha-1}/\Gamma(\alpha)$ with the Euler increments of
        the square-root process, and the driftless log-price integrates
        $d x = \sqrt{v}\, d w^1$ (the martingale convexity correction is applied
        by the pricing layer). It is intended for illustration only; the
        characteristic-function pricing path does not use it.
        """
        if args:
            path2 = args[0]
        else:
            path2 = Paths.normal_draws(draws.samples, draws.t, draws.time_steps)
        vp = self.variance_process
        alpha = self.alpha
        kappa = vp.kappa
        nu = vp.sigma
        theta = vp.theta
        v0 = vp.rate
        dt = draws.dt
        steps = draws.time_steps
        sqrt_dt = np.sqrt(dt)

        # power-law kernel weights g(k*dt) for k = 1, 2, ...
        lag = np.arange(1, steps + 1)
        g = (lag * dt) ** (alpha - 1.0) / gamma(alpha)

        db = draws.data * sqrt_dt  # variance Brownian increments
        dw_price = (
            self.rho * draws.data + np.sqrt(1.0 - self.rho * self.rho) * path2.data
        ) * sqrt_dt

        v = np.zeros(draws.data.shape)
        x = np.zeros(draws.data.shape)
        v[0] = v0
        # forcing increment at each past step: kappa(theta - v_j) dt + nu sqrt(v_j) dB_j
        forcing = np.zeros(draws.data.shape)
        for i in range(steps):
            vplus = np.clip(v[i], 0.0, None)
            forcing[i] = kappa * (theta - vplus) * dt + nu * np.sqrt(vplus) * db[i]
            # convolve the kernel with all forcing increments up to i:
            # v[i+1] = v0 + sum_{j=0}^{i} g_{i-j} * forcing[j]
            weights = g[i::-1]
            v[i + 1] = v0 + np.tensordot(weights, forcing[: i + 1], axes=(0, 0))
            x[i + 1] = x[i] + np.sqrt(vplus) * dw_price[i]
        return Paths(t=draws.t, data=x)
