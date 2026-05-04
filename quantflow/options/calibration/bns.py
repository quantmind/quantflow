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

    The parameter vector is

    `[v01, theta1, kappa_delta, beta1, rho1, v02, theta2, kappa2, beta2, rho2, w]`

    where `kappa1 = kappa2 + kappa_delta` with `kappa_delta > 0`, enforcing that
    the first (short-maturity) factor mean-reverts faster than the second, and
    `w` is the convex-combination weight of the first variance factor. The same
    $(v_0, \theta)$ parameterisation as
    [BNSCalibration][quantflow.options.calibration.bns.BNSCalibration] is used
    for each factor: the BDLP intensity is set as $\lambda_i = \theta_i \beta_i$
    so the stationary mean of $v^i$ equals $\theta_i$.

    Both leverage parameters are free in $[-0.9, 0.9]$: a positive $\rho_i$
    produces up-jumps in the log-price that lift the OTM call wing, while a
    negative one produces equity-style downside skew. The joint fit relies on
    the user-supplied initial parameters: pick distinct timescales for `bns1`
    and `bns2` (and consider opposite-sign leverages) to give the optimiser a
    meaningful two-factor starting point.

    TODO: improve this calibration. The 11-parameter fit is slow (finite-diff
    Jacobian dominates) and tends to collapse the two timescales into a near
    single-factor solution unless the initial conditions force them apart.
    Candidate improvements: analytic Jacobian of the characteristic exponent,
    a smarter warm start that does not bias the kappas to merge, and tighter
    bounds on `kappa1` and `beta_i`.
    """

    def get_bounds(self) -> Bounds:
        vol_range = self.implied_vol_range()
        vol_lb = 0.5 * vol_range.lb[0]
        vol_ub = 1.5 * vol_range.ub[0]
        v2 = vol_lb**2
        v2u = vol_ub**2
        return Bounds(
            [v2, v2, 1e-4, 1.0, -0.9, v2, v2, 1e-3, 1.0, -0.9, 0.0],
            [v2u, v2u, np.inf, np.inf, 0.9, v2u, v2u, 5.0, np.inf, 0.9, 1.0],
        )

    def get_params(self) -> np.ndarray:
        vp1 = self.model.bns1.variance_process
        vp2 = self.model.bns2.variance_process
        kappa_delta = max(vp1.kappa - vp2.kappa, 1e-4)
        theta1 = vp1.intensity / vp1.beta
        theta2 = vp2.intensity / vp2.beta
        return np.asarray(
            [
                vp1.rate,
                theta1,
                kappa_delta,
                vp1.beta,
                self.model.bns1.rho,
                vp2.rate,
                theta2,
                vp2.kappa,
                vp2.beta,
                self.model.bns2.rho,
                self.model.weight,
            ]
        )

    def set_params(self, params: np.ndarray) -> None:
        vp1 = self.model.bns1.variance_process
        vp1.rate = params[0]
        vp1.bdlp.jumps.decay = params[3]
        vp1.bdlp.intensity = params[1] * params[3]
        self.model.bns1.rho = params[4]
        vp2 = self.model.bns2.variance_process
        vp2.rate = params[5]
        vp2.kappa = params[7]
        vp2.bdlp.jumps.decay = params[8]
        vp2.bdlp.intensity = params[6] * params[8]
        self.model.bns2.rho = params[9]
        vp1.kappa = vp2.kappa + params[2]  # kappa2 + kappa_delta
        self.model.weight = params[10]
