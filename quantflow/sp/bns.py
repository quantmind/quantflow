from __future__ import annotations

from typing import Self

import numpy as np
from pydantic import Field

from ..ta.paths import Paths
from ..utils.types import FloatArrayLike, Vector
from .base import StochasticProcess1D
from .ou import GammaOU


class BNS(StochasticProcess1D):
    r"""Barndorff-Nielson & Shephard ([BNS](/bibliography#bns)) stochastic
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
        v = self.variance_process
        k = v.kappa
        beta = v.beta
        intensity = v.intensity
        v0 = v.rate
        rho = self.rho

        iur = 1j * u * rho
        u2 = u * u
        a = iur - 0.5 * u2 / k
        b = 0.5 * u2 * (1 - np.exp(-k * t)) / k

        diffusion = b * v0
        bdlp = intensity * (
            k * t * a / (beta - a)
            + beta / (beta - a) * np.log((beta - iur + b) / (beta - iur))
        )
        return diffusion - bdlp

    def sample(self, n: int, time_horizon: float = 1, time_steps: int = 100) -> Paths:
        return self.sample_from_draws(Paths.normal_draws(n, time_horizon, time_steps))

    def sample_from_draws(self, path_dw: Paths, *args: Paths) -> Paths:
        kappa = self.variance_process.kappa
        time_steps = path_dw.time_steps
        time_horizon = path_dw.t
        dt = path_dw.dt

        if args:
            path_dz = args[0]
        else:
            # BDLP runs at kappa-rescaled time, so sample over [0, kappa*T]
            path_dz = self.variance_process.bdlp.sample(
                path_dw.samples, kappa * time_horizon, time_steps
            )

        # Variance via the OU recursion v_{i+1} = v_i * e^{-kappa dt} + Delta z_{i+1}
        decay = np.exp(-kappa * dt)
        dz = np.diff(path_dz.data, axis=0)
        v = np.zeros_like(path_dz.data)
        v[0] = self.variance_process.rate
        for i in range(time_steps):
            v[i + 1] = v[i] * decay + dz[i]

        # Price: x_t = integral sqrt(v) dW + rho * z_{kappa t}
        diffusion = np.sqrt(v[:-1] * dt) * path_dw.data[:-1]
        paths = np.zeros_like(path_dw.data)
        paths[1:] = np.cumsum(diffusion, axis=0) + self.rho * path_dz.data[1:]
        return Paths(t=time_horizon, data=paths)
