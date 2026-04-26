from __future__ import annotations

from typing import Generic, TypeVar

import numpy as np
from pydantic import Field
from scipy.optimize import Bounds

from quantflow.sp.heston import Heston, HestonJ
from quantflow.sp.jump_diffusion import D

from .calibration import VolModelCalibration

H = TypeVar("H", bound=Heston)


class HestonCalibration(VolModelCalibration[H], Generic[H]):
    """Calibration of the [Heston][quantflow.sp.heston.Heston] model.

    Also serves as the base class for Heston-with-jumps calibration, providing
    the Feller condition penalty and the core variance-process parameter handling.
    """

    feller_penalize: float = Field(
        default=1000.0,
        description=(
            "Penalty weight for violating the Feller condition "
            "$2\\kappa\\theta \\geq \\sigma^2$. Applied during the Nelder-Mead "
            "stage. Set to 0 to disable."
        ),
    )

    def get_bounds(self) -> Bounds:
        vol_range = self.implied_vol_range()
        vol_lb = 0.5 * vol_range.lb[0]
        vol_ub = 1.5 * vol_range.ub[0]
        return Bounds(
            [vol_lb**2, vol_lb**2, 0.0, 0.0, -0.9],
            [vol_ub**2, vol_ub**2, np.inf, np.inf, 0.0],
        )

    def get_params(self) -> np.ndarray:
        return np.asarray(
            [
                self.model.variance_process.rate,
                self.model.variance_process.theta,
                self.model.variance_process.kappa,
                self.model.variance_process.sigma,
                self.model.rho,
            ]
        )

    def set_params(self, params: np.ndarray) -> None:
        self.model.variance_process.rate = params[0]
        self.model.variance_process.theta = params[1]
        self.model.variance_process.kappa = params[2]
        self.model.variance_process.sigma = params[3]
        self.model.rho = params[4]

    def penalize(self) -> float:
        """Penalty for violating the Feller condition"""
        kt = 2 * self.model.variance_process.kappa * self.model.variance_process.theta
        neg = max(self.model.variance_process.sigma2 - kt, 0.0)
        return self.feller_penalize * neg * neg


class HestonJCalibration(HestonCalibration[HestonJ[D]], Generic[D]):
    """Calibration of the [HestonJ][quantflow.sp.heston.HestonJ] model with jumps.

    Extends [HestonCalibration][quantflow.options.heston_calibration.HestonCalibration]
    by appending jump parameters to the parameter vector and bounds.
    """

    def get_bounds(self) -> Bounds:
        base = super().get_bounds()
        vol_range = self.implied_vol_range()
        vol_lb = 0.5 * vol_range.lb[0]
        vol_ub = 1.5 * vol_range.ub[0]
        lower = list(base.lb) + [1.0, (0.01 * vol_lb) ** 2]
        upper = list(base.ub) + [np.inf, (0.5 * vol_ub) ** 2]
        try:
            self.model.jumps.jumps.asymmetry()
            lower.append(-2.0)
            upper.append(2.0)
        except NotImplementedError:
            pass
        return Bounds(lower, upper)

    def get_params(self) -> np.ndarray:
        params = list(super().get_params()) + [
            self.model.jumps.intensity,
            self.model.jumps.jumps.variance(),
        ]
        try:
            params.append(self.model.jumps.jumps.asymmetry())
        except NotImplementedError:
            pass
        return np.asarray(params)

    def set_params(self, params: np.ndarray) -> None:
        super().set_params(params)
        self.model.jumps.intensity = params[5]
        self.model.jumps.jumps.set_variance(params[6])
        try:
            self.model.jumps.jumps.set_asymmetry(params[7])
        except IndexError:
            pass
