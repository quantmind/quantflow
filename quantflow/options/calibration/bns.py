from __future__ import annotations

from typing import Generic, TypeVar

import numpy as np
from scipy.optimize import Bounds

from quantflow.sp.bns import BNS

from .base import VolModelCalibration

B = TypeVar("B", bound=BNS)


class BNSCalibration(VolModelCalibration[B], Generic[B]):
    r"""Calibration of the [BNS][quantflow.sp.bns.BNS] stochastic volatility model.

    The parameter vector is `[v0, theta, kappa, beta, rho]` where

    - `v0` is the initial variance ($v_0 = \text{variance\_process.rate}$)
    - `theta` is the long-run variance ($\theta = \lambda / \beta$)
    - `kappa` is the mean-reversion speed of the variance process
    - `beta` is the exponential decay of the BDLP jump-size distribution
    - `rho` is the leverage parameter (correlation between jumps in variance and
      jumps in log-price)

    The BDLP intensity is set as $\lambda = \theta \beta$ so that the stationary
    mean of the variance process equals $\theta$, mirroring the Heston
    parameterisation. The Gamma-OU variance process is positive by construction,
    so no Feller-style penalty is needed.
    """

    def get_bounds(self) -> Bounds:
        vol_range = self.implied_vol_range()
        vol_lb = 0.5 * vol_range.lb[0]
        vol_ub = 1.5 * vol_range.ub[0]
        v2 = vol_lb**2
        v2u = vol_ub**2
        return Bounds(
            [v2, v2, 1e-3, 1.0, -0.9],
            [v2u, v2u, np.inf, np.inf, 0.0],
        )

    def get_params(self) -> np.ndarray:
        vp = self.model.variance_process
        theta = vp.intensity / vp.beta
        return np.asarray([vp.rate, theta, vp.kappa, vp.beta, self.model.rho])

    def set_params(self, params: np.ndarray) -> None:
        vp = self.model.variance_process
        vp.rate = params[0]
        vp.kappa = params[2]
        vp.bdlp.jumps.decay = params[3]
        vp.bdlp.intensity = params[1] * params[3]
        self.model.rho = params[4]
