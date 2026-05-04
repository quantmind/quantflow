from __future__ import annotations

from abc import abstractmethod
from typing import Generic, Self

import numpy as np
from pydantic import Field
from scipy.stats import gamma, norm

from ..ta.paths import Paths
from ..utils.distributions import Exponential
from ..utils.types import Float, FloatArrayLike, Vector
from .base import IntensityProcess, StochasticProcess1D
from .poisson import CompoundPoissonProcess, D
from .weiner import WeinerProcess


class Vasicek(StochasticProcess1D):
    r"""Gaussian OU process, also known as the
    [Vasicek model](https://en.wikipedia.org/wiki/Vasicek_model).

    Historically, the Vasicek model was used to model the short rate, but it can be
    used to model any process that reverts to a mean level at a rate proportional to
    the difference between the current level and the mean level. Unlike intensity
    processes, the Vasicek process can take negative values, so it is not constrained
    to a positive domain.

    \begin{equation}
        dx_t = \kappa (\theta - x_t) dt + \sigma dw_t
    \end{equation}

    where $\kappa$ is the mean reversion speed, $\theta$ is the mean level, and
    $\sigma$ is the volatility of the process. The solution to the SDE is given by

    \begin{equation}
        x_t = x_0 e^{-\kappa t} + \theta (1 - e^{-\kappa t}) +
            \sigma \int_0^t e^{-\kappa (t-s)} dw_s
    \end{equation}
    """

    rate: float = Field(default=1.0, description=r"Initial value $x_0$")
    kappa: float = Field(
        default=1.0, gt=0, description=r"Mean reversion speed $\kappa$"
    )
    theta: float = Field(default=1.0, description=r"Mean level $\theta$")
    bdlp: WeinerProcess = Field(
        default_factory=WeinerProcess,
        description="Background driving Wiener process",
    )

    def characteristic_exponent(self, t: FloatArrayLike, u: Vector) -> Vector:
        r"""The characteristic exponent of the Vasicek process is given by

        \begin{equation}
            \phi_{x_t}(u) = - iu \mathbb{E}[x_t] + \frac{1}{2} u^2 \text{Var}[x_t]
        \end{equation}
        """
        mu = self.analytical_mean(t)
        var = self.analytical_variance(t)
        return u * (-1j * mu + 0.5 * var * u)

    def sample(self, n: int, time_horizon: float = 1, time_steps: int = 100) -> Paths:
        paths = Paths.normal_draws(n, time_horizon, time_steps)
        return self.sample_from_draws(paths)

    def sample_from_draws(self, draws: Paths, *args: Paths) -> Paths:
        kappa = self.kappa
        theta = self.theta
        dt = draws.dt
        sdt = self.bdlp.sigma * np.sqrt(dt)
        paths = np.zeros(draws.data.shape)
        paths[0, :] = self.rate
        for t in range(draws.time_steps):
            x = paths[t, :]
            dx = kappa * (theta - x) * dt + sdt * draws.data[t, :]
            paths[t + 1, :] = x + dx
        return Paths(t=draws.t, data=paths)

    def ekt(self, t: FloatArrayLike) -> FloatArrayLike:
        return np.exp(-self.kappa * t)

    def analytical_mean(self, t: FloatArrayLike) -> FloatArrayLike:
        r"""The analytical mean of the Vasicek process is given by

        \begin{equation}
            \begin{aligned}
            \mathbb{E}[x_t] &= x_0 e^{-\kappa t} + \theta (1 - e^{-\kappa t}) \\
            &= \theta \qquad \text{as } t \to \infty
            \end{aligned}
        \end{equation}
        """
        ekt = self.ekt(t)
        return self.rate * ekt + self.theta * (1 - ekt)

    def analytical_variance(self, t: FloatArrayLike) -> FloatArrayLike:
        r"""The analytical variance of the Vasicek process is given by

        \begin{equation}
            \begin{aligned}
            \text{Var}[x_t] &= \frac{\sigma^2}{2\kappa} (1 - e^{-2\kappa t}) \\
            &= \frac{\sigma^2}{2\kappa} \qquad \text{as } t \to \infty
            \end{aligned}
        \end{equation}
        """
        ekt = self.ekt(2 * t)
        return 0.5 * self.bdlp.sigma2 * (1 - ekt) / self.kappa

    def analytical_pdf(self, t: FloatArrayLike, x: FloatArrayLike) -> FloatArrayLike:
        return norm.pdf(x, loc=self.analytical_mean(t), scale=self.analytical_std(t))

    def analytical_cdf(self, t: FloatArrayLike, x: FloatArrayLike) -> FloatArrayLike:
        return norm.cdf(x, loc=self.analytical_mean(t), scale=self.analytical_std(t))


class NGOU(IntensityProcess, Generic[D]):
    r"""The general definition of a non-Gaussian OU process is defined as
    the solution to the SDE

    \begin{equation}
        dx_t = -\kappa x_t dt + dZ_{\kappa t}
    \end{equation}

    where $Z_{\kappa t}$ is a pure jump Lévy process, also known as the background
    driving Lévy process (BDLP).
    The process is mean-reverting with mean reversion speed $\kappa>0$.

    The unusual timing $dZ_{\kappa t}$ is deliberately chosen so that it will turn
    out that whatever the value of $\kappa$, the marginal distribution of $x_t$
    will be unchanged.

    The solution to the SDE is given by

    \begin{equation}
        x_t = x_0 e^{-\kappa t} + \int_0^t e^{-\kappa (t-s)} dZ_{\kappa s}
    \end{equation}
    """

    bdlp: CompoundPoissonProcess[D] = Field(
        description=(
            "Background driving Lévy process is a "
            "[CompoundPoissonProcess][quantflow.sp.poisson.CompoundPoissonProcess] "
            "with jump distribution D"
        ),
    )

    @property
    def intensity(self) -> float:
        """The intensity of the NGOU process is the intensity of the background
        driving Lévy process"""
        return self.bdlp.intensity

    @abstractmethod
    def characteristic_exponent(self, t: FloatArrayLike, u: Vector) -> Vector:
        r"""Characteristic exponent of a non-Gaussian OU process driven by a
        Lévy process $Z$ with characteristic exponent $\phi_Z$:

        \begin{equation}
            \phi_{x_t, u} = - iu\, x_0\, e^{-\kappa t}
                + \int_{u e^{-\kappa t}}^{u} \frac{\phi_Z(v)}{v}\,dv
        \end{equation}
        """

    def sample_from_draws(self, draws: Paths, *args: Paths) -> Paths:
        raise NotImplementedError


class GammaOU(NGOU[Exponential]):
    """Gamma OU process is a non-Gaussian OU process where the background
    driving Lévy process is a compound Poisson process with exponential jumps"""

    @property
    def beta(self) -> float:
        return self.bdlp.jumps.decay

    @classmethod
    def create(cls, rate: float = 1, decay: float = 1, kappa: float = 1) -> Self:
        """Convenience constructor for GammaOU process"""
        return cls(
            rate=rate,
            kappa=kappa,
            bdlp=CompoundPoissonProcess[Exponential](
                intensity=rate * decay, jumps=Exponential(decay=decay)
            ),
        )

    def characteristic_exponent(self, t: FloatArrayLike, u: Vector) -> Vector:
        r"""Closed form of the [NGOU][quantflow.sp.ou.NGOU] characteristic
        exponent when the BDLP is a compound Poisson process with intensity
        $\lambda$ and exponential jumps of decay $\beta$:

        \begin{equation}
            \phi_{x_t, u} =
                - iu\, x_0\, e^{-\kappa t}
                + \lambda \log\!\left(
                    \frac{\beta - iu}{\beta - iu\, e^{-\kappa t}}
                \right)
        \end{equation}

        Derivation. The characteristic function of the
        [Exponential][quantflow.utils.distributions.Exponential.characteristic]
        jumps is

        \begin{equation}
            \Phi_J(v) = \frac{\beta}{\beta - iv},
        \end{equation}

        so the BDLP unit-time characteristic exponent
        ([CompoundPoissonProcess][quantflow.sp.poisson.CompoundPoissonProcess]
        at $t = 1$) is

        \begin{equation}
            \phi_Z(v) = \lambda\bigl(1 - \Phi_J(v)\bigr)
                = -\frac{i v\, \lambda}{\beta - iv},
        \end{equation}

        and $\phi_Z(v)/v = -i\lambda / (\beta - iv)$. Substituting
        $w = \beta - iv$ (so $dv = i\, dw$) in the NGOU integral gives

        \begin{equation}
            \int_{u e^{-\kappa t}}^{u} \frac{\phi_Z(v)}{v}\, dv
                = \lambda \int_{\beta - iu\, e^{-\kappa t}}^{\beta - iu}
                    \frac{dw}{w}
                = \lambda \log\!\left(
                    \frac{\beta - iu}{\beta - iu\, e^{-\kappa t}}
                \right),
        \end{equation}

        which combined with the drift term $-iu\, x_0\, e^{-\kappa t}$ yields
        the closed form above.
        """
        b = self.beta
        iu = 1j * u
        c1 = iu * np.exp(-self.kappa * t)
        c0 = self.intensity * np.log((b - c1) / (b - iu))
        return -c0 - c1 * self.rate

    def integrated_log_laplace(self, t: FloatArrayLike, u: Vector) -> Vector:
        r"""Log-Laplace transform of the time-integrated Gamma-OU process.

        \begin{equation}
        \begin{aligned}
            \phi(t, u) &= \log E\!\left[e^{-u \int_0^t x_s\, ds}\right] \\
            &= -u\, x_0\, \frac{1 - e^{-\kappa t}}{\kappa}
              + \frac{\lambda}{u + \beta\kappa}
              \!\left[\beta\kappa\,
                \log\!\frac{\beta\kappa + u(1 - e^{-\kappa t})}{\beta\kappa}
                - u\kappa t\right]
        \end{aligned}
        \end{equation}

        where $x_0$ is the initial rate, $\lambda$ is the BDLP intensity and
        $\beta$ the exponential jump decay.
        """
        kappa = self.kappa
        b = self.beta
        muk = -u / kappa
        ekt = np.exp(-kappa * t)
        c1 = muk * (1 - ekt)
        c0 = self.intensity * (
            b * np.log(b / (muk + (b - muk) / ekt)) / (muk - b) - kappa * t
        )
        return c0 + c1 * self.rate

    def sample(self, n: int, time_horizon: float = 1, time_steps: int = 100) -> Paths:
        dt = time_horizon / time_steps
        jump_process = self.bdlp
        paths = np.zeros((time_steps + 1, n))
        paths[0, :] = self.rate
        for p in range(n):
            arrivals = jump_process.arrivals(self.kappa * time_horizon)
            jumps = jump_process.sample_jumps(len(arrivals))
            pp = paths[:, p]
            i = 1
            for arrival, jump in zip(arrivals, jumps):
                arrival /= self.kappa
                while i * dt < arrival:
                    i = self._advance(i, pp, dt)
                if i <= time_steps:
                    i = self._advance(i, pp, dt, arrival, jump)
            while i <= time_steps:
                i = self._advance(i, pp, dt)
        return Paths(t=time_horizon, data=paths)

    def _advance(
        self,
        i: int,
        pp: np.ndarray,
        dt: Float,
        arrival: Float = 0,
        jump: Float = 0,
    ) -> int:
        x = pp[i - 1]
        kappa = self.kappa
        # step i fills pp[i] from pp[i-1] over the interval [(i-1) dt, i dt]
        t0 = (i - 1) * dt
        t1 = i * dt
        a = arrival or t1
        pp[i] = x - kappa * x * (a - t0) - kappa * (x + jump) * (t1 - a) + jump
        return i + 1

    def analytical_mean(self, t: FloatArrayLike) -> FloatArrayLike:
        r"""Analytical mean of the Gamma-OU process at time $t$.

        \begin{equation}
            E[x_t] = x_0\, e^{-\kappa t}
                   + \frac{\lambda}{\beta}\bigl(1 - e^{-\kappa t}\bigr)
        \end{equation}

        which converges to the stationary mean $\lambda / \beta$ as
        $t \to \infty$.
        """
        ekt = self.ekt(t)
        return self.rate * ekt + (self.intensity / self.beta) * (1 - ekt)

    def analytical_variance(self, t: FloatArrayLike) -> FloatArrayLike:
        r"""Analytical variance of the Gamma-OU process at time $t$.

        \begin{equation}
            \mathrm{Var}(x_t) = \frac{\lambda}{\beta^2}
                \bigl(1 - e^{-2\kappa t}\bigr)
        \end{equation}

        which converges to the stationary variance $\lambda / \beta^2$ as
        $t \to \infty$.
        """
        return self.intensity * (1 - self.ekt(2 * t)) / (self.beta * self.beta)

    def analytical_pdf(self, t: FloatArrayLike, x: FloatArrayLike) -> FloatArrayLike:
        """Stationary marginal density of the Gamma-OU process. The transient
        density is not available in closed form; use `pdf_from_characteristic`
        for finite $t$.
        """
        return gamma.pdf(x, self.intensity, scale=1 / self.beta)
