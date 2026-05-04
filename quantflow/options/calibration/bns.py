from __future__ import annotations

from typing import Generic, TypeVar

import numpy as np
from scipy.optimize import Bounds

from quantflow.sp.bns import BNS, BNS2

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
            [v2u, v2u, np.inf, np.inf, 0.9],
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


B2 = TypeVar("B2", bound=BNS2)


class BNS2Calibration(VolModelCalibration[B2], Generic[B2]):
    r"""Calibration of the [BNS2][quantflow.sp.bns.BNS2] two-factor BNS model.

    Following the BNS superposition-of-OU construction, both factors share
    the same Gamma stationary marginal: only the mean-reversion timescales
    and the leverage parameters differ between the fast and slow factors.
    The parameter vector has nine entries:

    `[v01, v02, theta, beta, kappa2, kappa_delta, rho1, rho2, w]`

    | Symbol | Description |
    |---|---|
    | `v01`, `v02` | Initial variances of the two factors |
    | `theta` | Long-run variance shared by both factors ($\theta = \lambda / \beta$) |
    | `beta` | Exponential decay of the BDLP jump-size distribution (shared) |
    | `kappa2` | Mean-reversion speed of the slow factor |
    | `kappa_delta` | Excess speed of the fast factor ($\kappa_1 - \kappa_2$) |
    | `rho1`, `rho2` | Leverage of the two factors, free in $[-0.9, 0.9]$ |
    | `w` | Weight of the first variance factor in the convex combination |

    Tying $(\theta, \beta)$ removes the degeneracy between the two
    marginal-distribution parameters and the timescales: the long-dated smile
    pins down a single stationary variance distribution, while the term
    structure of vol identifies the two relaxation speeds. The leverages
    $\rho_1, \rho_2$ stay independent because the empirical equity skew
    flattens with maturity, which a single shared leverage cannot reproduce.

    The user-supplied initial model still seeds the fit: pick distinct
    timescales for `bns1` and `bns2` (and consider opposite-sign leverages) so
    the optimiser starts away from the single-factor collapse. Any difference
    in `(theta, beta)` between the two seed factors is averaged when building
    the starting parameter vector.
    """

    def get_bounds(self) -> Bounds:
        vol_range = self.implied_vol_range()
        vol_lb = 0.5 * vol_range.lb[0]
        vol_ub = 1.5 * vol_range.ub[0]
        v2 = vol_lb**2
        v2u = vol_ub**2
        return Bounds(
            #  v01, v02, theta, beta, kappa2, kappa_delta, rho1, rho2, w
            [v2, v2, v2, 1.0, 1e-3, 1e-4, -0.9, -0.9, 0.0],
            [v2u, v2u, v2u, np.inf, 5.0, np.inf, 0.9, 0.9, 1.0],
        )

    def get_params(self) -> np.ndarray:
        vp1 = self.model.bns1.variance_process
        vp2 = self.model.bns2.variance_process
        theta1 = vp1.intensity / vp1.beta
        theta2 = vp2.intensity / vp2.beta
        theta = 0.5 * (theta1 + theta2)
        beta = 0.5 * (vp1.beta + vp2.beta)
        kappa_delta = max(vp1.kappa - vp2.kappa, 1e-4)
        return np.asarray(
            [
                vp1.rate,
                vp2.rate,
                theta,
                beta,
                vp2.kappa,
                kappa_delta,
                self.model.bns1.rho,
                self.model.bns2.rho,
                self.model.weight,
            ]
        )

    def set_params(self, params: np.ndarray) -> None:
        v01, v02, theta, beta, kappa2, kappa_delta, rho1, rho2, w = params
        vp1 = self.model.bns1.variance_process
        vp2 = self.model.bns2.variance_process
        vp1.rate = v01
        vp2.rate = v02
        vp1.bdlp.jumps.decay = beta
        vp2.bdlp.jumps.decay = beta
        intensity = theta * beta
        vp1.bdlp.intensity = intensity
        vp2.bdlp.intensity = intensity
        vp2.kappa = kappa2
        vp1.kappa = kappa2 + kappa_delta
        self.model.bns1.rho = rho1
        self.model.bns2.rho = rho2
        self.model.weight = w
