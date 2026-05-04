from __future__ import annotations

from typing import Self

import numpy as np
from pydantic import Field
from typing_extensions import Annotated, Doc

from ..ta.paths import Paths
from ..utils.types import FloatArrayLike, Vector
from .base import StochasticProcess1D
from .ou import GammaOU


class BNS(StochasticProcess1D):
    r"""Barndorff-Nielson & Shephard ([BNS](../../bibliography.md#bns)) stochastic
    volatility model.

    This is a stochastic volatility model where the variance process is given by a
    non-Gaussian Ornstein-Uhlenbeck process driven by a pure-jump Lévy process.
    The BNS model is defined by the following system of SDEs:

    \begin{equation}
        \begin{aligned}
            dx_t &= \sqrt{v_t} dw_t + \rho dz_{\kappa t} \\
            dv_t &= -\kappa v_t dt + dz_{\kappa t}
        \end{aligned}
    \end{equation}

    The model is flexible and can capture various stylized facts of financial markets,
    such as volatility clustering and leverage effects.

    This implementation uses a [GammaOU][quantflow.sp.ou.GammaOU] process
    for the variance, which is a common choice in the BNS model.
    """

    variance_process: GammaOU = Field(
        default_factory=GammaOU.create, description="Variance process"
    )
    rho: float = Field(
        default=0,
        gt=-1,
        lt=1,
        description="Correlation between variance and price processes, in (-1, 1)",
    )

    @classmethod
    def create(cls, vol: float, kappa: float, decay: float, rho: float) -> Self:
        """Convenience constructor for BNS process with parameters
        of the variance process
        """
        return cls(
            variance_process=GammaOU.create(rate=vol * vol, kappa=kappa, decay=decay),
            rho=rho,
        )

    def characteristic_exponent(self, t: FloatArrayLike, u: Vector) -> Vector:
        r"""Characteristic exponent of the BNS process with Gamma-OU variance.

        \begin{equation}
            \phi(t, u) = B v_0
                - \lambda \left[
                    \frac{\kappa t \, A}{\beta - A}
                    + \frac{\beta}{\beta - A}
                    \log\frac{\beta - i u \rho + B}{\beta - i u \rho}
                \right]
        \end{equation}

        with

        \begin{equation}
            \begin{aligned}
                A &= i u \rho - \frac{u^2}{2 \kappa}, \\
                B &= \frac{u^2 (1 - e^{-\kappa t})}{2 \kappa}.
            \end{aligned}
        \end{equation}

        where $v_0$ is the initial variance, $\kappa$ is the mean-reversion speed,
        $\rho$ is the leverage parameter, and $(\lambda, \beta)$ are the intensity
        and exponential-jump rate of the background driving Lévy process.
        """
        return self._characteristic_exponent(t, u, 1.0)

    def _characteristic_exponent(
        self, t: FloatArrayLike, u: Vector, weight: float
    ) -> Vector:
        """BNS exponent with the diffusion variance multiplied by `weight`.

        With weight=1 this is the single-factor BNS exponent. With weight<1 it is
        the contribution of this factor when the price is driven by a single
        Brownian motion against a convex combination of OU variances (BNS2).
        """
        v = self.variance_process
        k = v.kappa
        beta = v.beta
        intensity = v.intensity
        v0 = v.rate
        rho = self.rho

        iur = 1j * u * rho
        u2w = u * u * weight
        a = iur - 0.5 * u2w / k
        b = 0.5 * u2w * (1 - np.exp(-k * t)) / k

        diffusion = b * v0
        bdlp = intensity * (
            k * t * a / (beta - a)
            + beta / (beta - a) * np.log((beta - iur + b) / (beta - iur))
        )
        return diffusion - bdlp

    def sample(self, n: int, time_horizon: float = 1, time_steps: int = 100) -> Paths:
        return self.sample_from_draws(Paths.normal_draws(n, time_horizon, time_steps))

    def sample_from_draws(self, path_dw: Paths, *args: Paths) -> Paths:
        path_dz = (
            args[0]
            if args
            else self.variance_process.bdlp.sample(
                path_dw.samples,
                self.variance_process.kappa * path_dw.t,
                path_dw.time_steps,
            )
        )
        v = self._variance_path(path_dz)
        # Price: x_t = integral sqrt(v) dW + rho * z_{kappa t}
        diffusion = np.sqrt(v[:-1] * path_dw.dt) * path_dw.data[:-1]
        paths = np.zeros_like(path_dw.data)
        paths[1:] = np.cumsum(diffusion, axis=0) + self.rho * path_dz.data[1:]
        return Paths(t=path_dw.t, data=paths)

    def _variance_path(self, path_dz: Paths) -> np.ndarray:
        """Simulate the OU variance path on the same grid as `path_dz`."""
        kappa = self.variance_process.kappa
        decay = np.exp(-kappa * path_dz.dt)
        dz = np.diff(path_dz.data, axis=0)
        v = np.zeros_like(path_dz.data)
        v[0] = self.variance_process.rate
        for i in range(path_dz.time_steps):
            v[i + 1] = v[i] * decay + dz[i]
        return v


class BNS2(StochasticProcess1D):
    r"""Two-factor Barndorff-Nielson & Shephard stochastic volatility model.

    The original multi-factor [BNS](/bibliography#bns) extension drives a single
    log-price with a single Brownian motion against a convex combination of
    independent Gamma-OU variances. With weight $w \in [0, 1]$ for the first
    factor:

    \begin{equation}
        \begin{aligned}
            \sigma^2_t &= w\, v^1_t + (1 - w)\, v^2_t \\
            dx_t &= \sigma_t\, dw_t
                + \rho_1\, dz^1_{\kappa_1 t}
                + \rho_2\, dz^2_{\kappa_2 t} \\
            dv^i_t &= -\kappa_i v^i_t\, dt + dz^i_{\kappa_i t}, \quad i = 1, 2
        \end{aligned}
    \end{equation}

    A fast and a slow factor combined this way add flexibility to the term
    structure of volatility while retaining the analytic tractability of BNS.
    """

    bns1: BNS = Field(default_factory=BNS, description="First BNS variance factor")
    bns2: BNS = Field(default_factory=BNS, description="Second BNS variance factor")
    weight: float = Field(
        default=0.5,
        ge=0,
        le=1,
        description=(
            "Weight $w$ of the first variance factor in the convex combination; "
            "the second factor receives weight $1 - w$"
        ),
    )

    def characteristic_exponent(
        self,
        t: Annotated[FloatArrayLike, Doc("Time horizon or array of evaluation times")],
        u: Annotated[Vector, Doc("Characteristic exponent argument")],
    ) -> Vector:
        r"""Characteristic exponent as the sum of two weighted BNS exponents.

        Conditional on the variance paths the diffusion is Gaussian with variance
        $w \int v^1_s\,ds + (1-w) \int v^2_s\,ds$. Independence of the two BDLPs
        then factorises the unconditional expectation into a product, giving

        \begin{equation}
            \phi_{x_t,u} = \phi^{(1)}_{x_t,u}\big|_{u^2 \mapsto w u^2}
                + \phi^{(2)}_{x_t,u}\big|_{u^2 \mapsto (1 - w) u^2}
        \end{equation}

        where the substitution applies only to the diffusion term ($u^2$) and
        leaves the leverage term ($i u \rho_i$) unchanged.
        """
        w = self.weight
        return self.bns1._characteristic_exponent(
            t, u, w
        ) + self.bns2._characteristic_exponent(t, u, 1.0 - w)

    def sample(
        self,
        n: Annotated[int, Doc("Number of sample paths")],
        time_horizon: Annotated[float, Doc("Time horizon")] = 1,
        time_steps: Annotated[int, Doc("Number of discrete time steps")] = 100,
    ) -> Paths:
        return self.sample_from_draws(Paths.normal_draws(n, time_horizon, time_steps))

    def sample_from_draws(
        self,
        path_dw: Annotated[Paths, Doc("Single Brownian motion driving both factors")],
        *args: Annotated[
            Paths,
            Doc("Optional pre-drawn BDLP paths for bns1 and bns2 (in that order)"),
        ],
    ) -> Paths:
        time_horizon = path_dw.t
        time_steps = path_dw.time_steps
        n = path_dw.samples
        path_dz1 = (
            args[0]
            if len(args) > 0
            else self.bns1.variance_process.bdlp.sample(
                n, self.bns1.variance_process.kappa * time_horizon, time_steps
            )
        )
        path_dz2 = (
            args[1]
            if len(args) > 1
            else self.bns2.variance_process.bdlp.sample(
                n, self.bns2.variance_process.kappa * time_horizon, time_steps
            )
        )
        v1 = self.bns1._variance_path(path_dz1)
        v2 = self.bns2._variance_path(path_dz2)
        w = self.weight
        sigma2 = w * v1 + (1.0 - w) * v2
        diffusion = np.sqrt(sigma2[:-1] * path_dw.dt) * path_dw.data[:-1]
        paths = np.zeros_like(path_dw.data)
        paths[1:] = (
            np.cumsum(diffusion, axis=0)
            + self.bns1.rho * path_dz1.data[1:]
            + self.bns2.rho * path_dz2.data[1:]
        )
        return Paths(t=time_horizon, data=paths)
