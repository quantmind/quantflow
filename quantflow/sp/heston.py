from __future__ import annotations

from typing import Generic, Self

import numpy as np
from pydantic import Field
from typing_extensions import Annotated, Doc

from quantflow.ta.paths import Paths
from quantflow.utils.types import FloatArrayLike, Vector

from .base import StochasticProcess1D
from .cir import CIR
from .jump_diffusion import JumpDiffusion
from .poisson import CompoundPoissonProcess, D


class Heston(StochasticProcess1D):
    r"""The Heston stochastic volatility model

    The classical square-root stochastic volatility model of Heston (1993)
    can be regarded as a standard Brownian motion $x_t$ time changed by a CIR
    activity rate process.

    \begin{align}
        d x_t &= d w^1_t \\
        d v_t &= \kappa (\theta - v_t) dt + \nu \sqrt{v_t} dw^2_t \\
        \rho dt &= {\tt E}[dw^1 dw^2]
    \end{align}
    """

    variance_process: CIR = Field(
        default_factory=CIR,
        description=(
            "The variance process is a [Cox-Ingersoll-Ross][quantflow.sp.cir.CIR] "
            "process which is guaranteed to be positive if the Feller condition is "
            "satisfied"
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
            Doc(
                "Mean reversion speed for the variance process, the lower the "
                "more pronounced the volatility clustering and therefore the fatter "
                "the tails of the distribution of the price process"
            ),
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
                r"If `None`, it defaults to the variance given by ${\tt var}$"
                " the long term variance described above."
            ),
        ] = None,
    ) -> Self:
        r"""Create an Heston model.

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
        )

    def characteristic_exponent(
        self,
        t: Annotated[FloatArrayLike, Doc("Time horizon or array of evaluation times")],
        u: Annotated[Vector, Doc("Characteristic exponent argument")],
    ) -> Vector:
        r"""Characteristic exponent of the Heston model in closed form.

        Define the correlation-adjusted drift and the characteristic frequency:

        \begin{equation}
            \tilde{\kappa} = \kappa - i u \nu \rho, \qquad
            \gamma = \sqrt{\tilde{\kappa}^2 + u^2 \nu^2}
        \end{equation}

        Then the exponent is

        \begin{equation}
            \phi_{x_t,u} = \frac{\kappa\theta}{\nu^2}
                \left[2 \ln c_{t,u} + (\gamma - \tilde{\kappa})\,t\right]
                + v_0\, b_{t,u}
        \end{equation}

        where

        \begin{align*}
            c_{t,u} &= \frac{\gamma - \tfrac{1}{2}(\gamma - \tilde{\kappa})
                (1 - e^{-\gamma t})}{\gamma} \\
            b_{t,u} &= \frac{u^2 (1 - e^{-\gamma t})}
                {(\gamma + \tilde{\kappa}) + (\gamma - \tilde{\kappa})\,e^{-\gamma t}}
        \end{align*}

        and $\nu$ is the vol of vol, $\kappa$ the mean-reversion speed,
        $\theta$ the long-term variance, $\rho$ the correlation, and $v_0$
        the initial variance.
        """
        eta = self.variance_process.sigma
        eta2 = eta * eta
        theta_kappa = self.variance_process.theta * self.variance_process.kappa
        # adjusted drift
        kappa = self.variance_process.kappa - 1j * u * eta * self.rho
        u2 = u * u
        gamma = np.sqrt(kappa * kappa + u2 * eta2)
        egt = np.exp(-gamma * t)
        c = (gamma - 0.5 * (gamma - kappa) * (1 - egt)) / gamma
        b = u2 * (1 - egt) / ((gamma + kappa) + (gamma - kappa) * egt)
        a = theta_kappa * (2 * np.log(c) + (gamma - kappa) * t) / eta2
        return a + b * self.variance_process.rate

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
        path1: Annotated[
            Paths,
            Doc("Pre-drawn standard normal increments for the first Brownian motion"),
        ],
        *args: Annotated[
            Paths,
            Doc(
                "Optional pre-drawn increments for the second Brownian motion; "
                "new draws are generated if omitted"
            ),
        ],
    ) -> Paths:
        if args:
            path2 = args[0]
        else:
            path2 = Paths.normal_draws(path1.samples, path1.t, path1.time_steps)
        dz = path1.data
        dw = self.rho * dz + np.sqrt(1 - self.rho * self.rho) * path2.data
        v = self.variance_process.sample_from_draws(path1)
        dx = dw * np.sqrt(v.data * path1.dt)
        paths = np.zeros(dx.shape)
        paths[1:] = np.cumsum(dx[:-1], axis=0)
        return Paths(t=path1.t, data=paths)


class HestonJ(Heston, Generic[D]):
    r"""A generic Heston stochastic volatility model with jumps

    The Heston model with jumps is an extension of the classical square-root
    stochastic volatility model of Heston (1993) with the addition of jump
    processes. The jumps are modeled via a compound Poisson process

    \begin{align}
        d x_t &= d w^1_t + d N_t\\
        d v_t &= \kappa (\theta - v_t) dt + \nu \sqrt{v_t} dw^2_t \\
        \rho dt &= {\tt E}[dw^1 dw^2]
    \end{align}

    This model is generic and therefore allows for different types of jumps
    distributions **D**.

    The Bates model is obtained by using the
    [Normal][quantflow.utils.distributions.Normal] distribution for the jump sizes.
    """

    jumps: CompoundPoissonProcess[D] = Field(description="Jump process")

    @classmethod
    def create(  # type: ignore [override]
        cls,
        jump_distribution: Annotated[
            type[D],
            Doc(
                "The distribution of jump size (currently only"
                " [Normal][quantflow.utils.distributions.Normal] and"
                " [DoubleExponential][quantflow.utils.distributions.DoubleExponential]"
                " are supported)"
            ),
        ],
        *,
        rate: Annotated[
            float, Doc("Define the initial value of the variance process")
        ] = 1.0,
        vol: Annotated[
            float,
            Doc(
                "The standard deviation of the price process, normalized by the"
                " square root of time, as time tends to infinity"
                " (the long term standard deviation)"
            ),
        ] = 0.5,
        kappa: Annotated[
            float, Doc("The mean reversion speed for the variance process")
        ] = 1,
        sigma: Annotated[float, Doc("The volatility of the variance process")] = 0.8,
        rho: Annotated[
            float,
            Doc(
                "The correlation between the Brownian motions of the"
                " variance and price processes"
            ),
        ] = 0,
        theta: Annotated[
            float | None,
            Doc(
                r"The long-term mean of the variance process, if `None`, it"
                r" defaults to the diffusion variance given by ${\tt var}_d$"
            ),
        ] = None,
        jump_intensity: Annotated[
            float, Doc("The average number of jumps per year")
        ] = 100,
        jump_fraction: Annotated[
            float, Doc("The fraction of variance due to jumps (between 0 and 1)")
        ] = 0.1,
        jump_asymmetry: Annotated[
            float, Doc("The asymmetry of the jump distribution (0 for symmetric jumps)")
        ] = 0.0,
    ) -> HestonJ[D]:
        r"""Create an Heston model with
        [DoubleExponential][quantflow.utils.distributions.DoubleExponential] jumps.

        To understand the parameters lets introduce the following notation:

        \begin{align}
            {\tt var} &= {\tt vol}^2 \\
            {\tt var}_j &= {\tt var} \cdot {\tt jump\_fraction} \\
            {\tt var}_d &= {\tt var} - {\tt var}_j \\
            v_0 &= {\tt rate}\cdot{\tt var}_d
        \end{align}
        """
        jd = JumpDiffusion.create(
            jump_distribution,
            vol=vol,
            jump_intensity=jump_intensity,
            jump_fraction=jump_fraction,
            jump_asymmetry=jump_asymmetry,
        )
        total_variance = vol * vol
        diffusion_variance = total_variance * (1 - jump_fraction)
        return cls(
            variance_process=CIR(
                rate=rate * diffusion_variance,
                kappa=kappa,
                sigma=sigma,
                theta=theta if theta is not None else diffusion_variance,
            ),
            rho=rho,
            jumps=jd.jumps,
        )

    def characteristic_exponent(
        self,
        t: Annotated[FloatArrayLike, Doc("Time horizon or array of evaluation times")],
        u: Annotated[
            Vector, Doc("Characteristic exponent argument (imaginary frequency)")
        ],
    ) -> Vector:
        r"""Characteristic exponent as the sum of the Heston and jump exponents.

        \begin{equation}
            \phi_{x_t,u} = \phi^{\text{Heston}}_{x_t,u} + \phi^{\text{jumps}}_{x_t,u}
        \end{equation}
        """
        return super().characteristic_exponent(
            t, u
        ) + self.jumps.characteristic_exponent(t, u)

    def sample_from_draws(self, path1: Paths, *args: Paths) -> Paths:
        diffusion = super().sample_from_draws(path1, *args)
        jump_path = self.jumps.sample(
            diffusion.samples, diffusion.t, diffusion.time_steps
        )
        return Paths(t=diffusion.t, data=diffusion.data + jump_path.data)


class DoubleHeston(StochasticProcess1D):
    r"""Double Heston stochastic volatility model.

    Two independent [Heston][quantflow.sp.heston.Heston] processes drive a single
    log-price:

    \begin{align}
        d x_t &= \sqrt{v^1_t}\,d w^1_t + \sqrt{v^2_t}\,d w^3_t \\
        d v^i_t &= \kappa_i (\theta_i - v^i_t) dt + \nu_i \sqrt{v^i_t}\,d w^{2i}_t \\
        \rho_i\,dt &= {\tt E}[d w^{2i-1} d w^{2i}] \quad i = 1, 2
    \end{align}

    Because the two components are independent, the characteristic exponent is the sum
    of the two individual Heston exponents.
    """

    heston1: Heston = Field(
        default_factory=Heston, description="First Heston variance process"
    )
    heston2: Heston = Field(
        default_factory=Heston, description="Second Heston variance process"
    )

    def characteristic_exponent(
        self,
        t: Annotated[FloatArrayLike, Doc("Time horizon or array of evaluation times")],
        u: Annotated[
            Vector, Doc("Characteristic exponent argument (imaginary frequency)")
        ],
    ) -> Vector:
        r"""Characteristic exponent as the sum of two independent Heston exponents.

        \begin{equation}
            \phi_{x_t,u} = \phi^{(1)}_{x_t,u} + \phi^{(2)}_{x_t,u}
        \end{equation}
        """
        return self.heston1.characteristic_exponent(
            t, u
        ) + self.heston2.characteristic_exponent(t, u)

    def sample(
        self,
        n: Annotated[int, Doc("Number of sample paths")],
        time_horizon: Annotated[float, Doc("Time horizon")] = 1,
        time_steps: Annotated[int, Doc("Number of discrete time steps")] = 100,
    ) -> Paths:
        dw1 = Paths.normal_draws(n, time_horizon, time_steps)
        dw2 = Paths.normal_draws(n, time_horizon, time_steps)
        dw3 = Paths.normal_draws(n, time_horizon, time_steps)
        dw4 = Paths.normal_draws(n, time_horizon, time_steps)
        return self.sample_from_draws(dw1, dw2, dw3, dw4)

    def sample_from_draws(
        self,
        path1: Annotated[Paths, Doc("First Brownian motion draws for heston1")],
        *args: Annotated[
            Paths,
            Doc(
                "args[0]: second BM draws for heston1; "
                "args[1], args[2]: first and second BM draws for heston2"
            ),
        ],
    ) -> Paths:
        paths1 = self.heston1.sample_from_draws(path1, args[0])
        paths2 = self.heston2.sample_from_draws(args[1], args[2])
        return Paths(t=path1.t, data=paths1.data + paths2.data)


class DoubleHestonJ(DoubleHeston, Generic[D]):
    r"""Double Heston stochastic volatility model with jumps.

    Extends [DoubleHeston][quantflow.sp.heston.DoubleHeston] by replacing the first
    (short-maturity) Heston process with a
    [HestonJ][quantflow.sp.heston.HestonJ] that carries a jump component.
    Jumps are assigned to the short end because they fade away at longer maturities:

    \begin{equation}
        \phi_{x_t,u} = \phi^{(1)}_{x_t,u} + \phi^{\text{jumps}}_{x_t,u}
            + \phi^{(2)}_{x_t,u}
    \end{equation}

    where $\phi^{(1)}$ and $\phi^{\text{jumps}}$ are both provided by the
    [HestonJ][quantflow.sp.heston.HestonJ] first process.
    """

    heston1: HestonJ[D] = Field(  # type: ignore[assignment]
        description="First (short-maturity) Heston process with jumps"
    )
