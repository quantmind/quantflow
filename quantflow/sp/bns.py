from __future__ import annotations

from typing import Self

import numpy as np
from pydantic import Field
from scipy.special import xlogy

from ..ta.paths import Paths
from ..utils.types import FloatArrayLike, Vector
from .base import Im, StochasticProcess1D
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
        return -self._zeta(t, 0.5 * Im * u * u, self.rho * u)

    def sample(self, n: int, time_horizon: float = 1, time_steps: int = 100) -> Paths:
        return self.sample_from_draws(Paths.normal_draws(n, time_horizon, time_steps))

    def sample_from_draws(self, path_dw: Paths, *args: Paths) -> Paths:
        if args:
            path_dz = args[0]
        else:
            # generate the background driving process samples if not provided
            path_dz = self.variance_process.bdlp.sample(
                path_dw.samples, path_dw.t, path_dw.time_steps
            )
        dt = path_dw.dt
        # sample the activity rate process
        v = self.variance_process.sample_from_draws(path_dz)
        # create the time-changed Brownian motion
        dw = path_dw.data * np.sqrt(v.data * dt)
        paths = np.zeros(dw.shape)
        paths[1:] = np.cumsum(dw[:-1], axis=0) + path_dz.data
        return Paths(t=path_dw.t, data=paths)

    # Internal characteristics function methods (see docs)

    def _zeta(self, t: Vector, a: Vector, b: Vector) -> Vector:
        k = self.variance_process.kappa
        c = a * (1 - np.exp(-k * t)) / k
        g = (a + b) / self.variance_process.beta
        return Im * c * self.variance_process.rate - self.variance_process.intensity * (
            self._i(b + c, g) - self._i(b, g)
        )

    def _i(self, x: Vector, g: Vector) -> Vector:
        k = self.variance_process.kappa
        beta = self.variance_process.beta
        l1 = xlogy(k - Im * g, x + Im * beta)
        l2 = xlogy(g / (g + Im * k) / k, beta * g / k - x)
        return l1 + l2
