from __future__ import annotations

from pydantic import Field

from .base import StochasticProcess1D
from .ou import GammaOU


class BNS(StochasticProcess1D):
    """Barndorff-Nielson--Shephard (BNS) model.

    THe classical square-root stochastic volatility model of Heston (1993)
    can be regarded as a standard Brownian motion $x_t$ time changed by a CIR
    activity rate process.

    .. math::

        d x_t = d w^1_t \\
        d v_t = (a - \kappa v_t) dt + \nu \sqrt{v_t} dw^2_t
        \rho dt = \E[dw^1 dw^2]
    """

    variance_process: GammaOU = Field(
        default_factory=GammaOU, description="Variance process"
    )
    rho: float = Field(default=0, ge=-1, le=1, description="Correlation")
