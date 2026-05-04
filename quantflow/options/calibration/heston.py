from __future__ import annotations

from typing import Generic, TypeVar

import numpy as np
from pydantic import Field
from scipy.optimize import Bounds, OptimizeResult

from quantflow.sp.heston import DoubleHeston, DoubleHestonJ, Heston, HestonJ
from quantflow.sp.jump_diffusion import D

from ..pricer import OptionPricer
from .base import VolModelCalibration

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
        neg = min(self.model.variance_process.feller_condition, 0.0)
        return self.feller_penalize * neg * neg


class HestonJCalibration(HestonCalibration[HestonJ[D]], Generic[D]):
    """Calibration of the [HestonJ][quantflow.sp.heston.HestonJ] model with jumps.

    Extends [HestonCalibration][quantflow.options.calibration.heston.HestonCalibration]
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


DH = TypeVar("DH", bound=DoubleHeston)


class DoubleHestonCalibration(VolModelCalibration[DH], Generic[DH]):
    """Calibration of the [DoubleHeston][quantflow.sp.heston.DoubleHeston] model.

    The parameter vector is
    `[rate1, theta1, kappa_delta, sigma1, rho1, rate2, theta2, kappa2, sigma2, rho2]`
    where `kappa1 = kappa2 + kappa_delta` with `kappa_delta > 0`, enforcing that
    the first (short-maturity) process always mean-reverts faster than the second.

    The Feller penalty is applied independently to both variance processes.
    A warm start fits each process independently to its natural maturity range
    before the joint optimisation.
    """

    feller_penalize: float = Field(
        default=1000.0,
        description=(
            "Penalty weight for violating the Feller condition "
            "$2\\kappa\\theta \\geq \\sigma^2$. Applied during the L-BFGS-B "
            "stage. Set to 0 to disable."
        ),
    )
    ttm_split: float | None = Field(
        default=None,
        gt=0,
        description=(
            "TTM threshold in years separating short-maturity options (fitted to "
            "heston1) from long-maturity options (fitted to heston2) during warm "
            "start. Defaults to the median TTM across all calibration options."
        ),
    )

    def maturity_split(self) -> float:
        """TTM split to use for warm start: explicit value or median of option TTMs."""
        if self.ttm_split is not None:
            return self.ttm_split
        ttms = sorted({v.ttm for v in self.options.values()})
        return ttms[len(ttms) // 2]

    def get_bounds(self) -> Bounds:
        vol_range = self.implied_vol_range()
        vol_lb = 0.5 * vol_range.lb[0]
        vol_ub = 1.5 * vol_range.ub[0]
        v2 = vol_lb**2
        v2u = vol_ub**2
        return Bounds(
            [v2, v2, 1e-4, 0.0, -0.9, v2, v2, 0.0, 0.0, -0.9],
            [v2u, v2u, np.inf, np.inf, 0.0, v2u, v2u, 5.0, np.inf, 0.0],
        )

    def get_params(self) -> np.ndarray:
        vp1 = self.model.heston1.variance_process
        vp2 = self.model.heston2.variance_process
        kappa_delta = max(vp1.kappa - vp2.kappa, 1e-4)
        return np.asarray(
            [
                vp1.rate,
                vp1.theta,
                kappa_delta,
                vp1.sigma,
                self.model.heston1.rho,
                vp2.rate,
                vp2.theta,
                vp2.kappa,
                vp2.sigma,
                self.model.heston2.rho,
            ]
        )

    def set_params(self, params: np.ndarray) -> None:
        vp1 = self.model.heston1.variance_process
        vp1.rate = params[0]
        vp1.theta = params[1]
        vp1.sigma = params[3]
        self.model.heston1.rho = params[4]
        vp2 = self.model.heston2.variance_process
        vp2.rate = params[5]
        vp2.theta = params[6]
        vp2.kappa = params[7]
        vp2.sigma = params[8]
        self.model.heston2.rho = params[9]
        vp1.kappa = vp2.kappa + params[2]  # kappa2 + kappa_delta

    def feller_residuals(self) -> list[float]:
        """Extra residual terms penalising Feller violations for both processes.

        Appended to the main residual vector so the TRF stage also sees the
        constraint, not just the L-BFGS-B stage.
        """
        w = self.feller_penalize**0.5
        neg1 = min(self.model.heston1.variance_process.feller_condition, 0.0)
        neg2 = min(self.model.heston2.variance_process.feller_condition, 0.0)
        return [w * neg1, w * neg2]

    def penalize(self) -> float:
        """Feller penalty applied independently to both variance processes"""
        neg1 = min(self.model.heston1.variance_process.feller_condition, 0.0)
        neg2 = min(self.model.heston2.variance_process.feller_condition, 0.0)
        return self.feller_penalize * (neg1 * neg1 + neg2 * neg2)

    def residuals(self, params: np.ndarray) -> np.ndarray:
        return np.append(super().residuals(params), self.feller_residuals())

    def warm_start(self) -> None:
        """Sequential single-Heston fits to initialise the joint optimisation.

        Fits heston2 to long-dated options (ttm > split) then heston1 to
        short-dated options (ttm <= split), where the split defaults to the
        median TTM across all calibration options.
        """
        split = self.maturity_split()
        long_options = {k: v for k, v in self.options.items() if v.ttm > split}
        short_options = {k: v for k, v in self.options.items() if v.ttm <= split}
        if long_options:
            h2 = Heston(
                variance_process=self.model.heston2.variance_process.model_copy(),
                rho=self.model.heston2.rho,
            )
            HestonCalibration(
                pricer=OptionPricer(model=h2),
                vol_surface=self.vol_surface,
                options=long_options,
            ).fit()
            self.model.heston2.variance_process = h2.variance_process
            self.model.heston2.rho = h2.rho
        if short_options:
            h1 = Heston(
                variance_process=self.model.heston1.variance_process.model_copy(),
                rho=self.model.heston1.rho,
            )
            HestonCalibration(
                pricer=OptionPricer(model=h1),
                vol_surface=self.vol_surface,
                options=short_options,
            ).fit()
            self.model.heston1.variance_process = h1.variance_process
            self.model.heston1.rho = h1.rho
        vp1 = self.model.heston1.variance_process
        vp2 = self.model.heston2.variance_process
        vp1.kappa = max(vp1.kappa, vp2.kappa + 1e-4)

    def fit(self) -> OptimizeResult:
        """Warm-start then joint two-stage fit."""
        self.warm_start()
        return super().fit()


class DoubleHestonJCalibration(DoubleHestonCalibration[DoubleHestonJ[D]], Generic[D]):
    """Calibration of the [DoubleHestonJ][quantflow.sp.heston.DoubleHestonJ] model.

    Extends
    [DoubleHestonCalibration][quantflow.options.calibration.heston.DoubleHestonCalibration]
    by appending the jump parameters of `heston1` to the parameter vector and bounds.

    Overrides `warm_start` to fit a full
    [HestonJCalibration][quantflow.options.calibration.heston.HestonJCalibration]
    to the short-dated options, so that the jump parameters are also initialised
    before the joint optimisation.
    """

    def get_bounds(self) -> Bounds:
        base = super().get_bounds()
        vol_range = self.implied_vol_range()
        vol_lb = 0.5 * vol_range.lb[0]
        vol_ub = 1.5 * vol_range.ub[0]
        lower = list(base.lb) + [1.0, (0.01 * vol_lb) ** 2]
        upper = list(base.ub) + [np.inf, (0.5 * vol_ub) ** 2]
        try:
            self.model.heston1.jumps.jumps.asymmetry()
            lower.append(-2.0)
            upper.append(2.0)
        except NotImplementedError:
            pass
        return Bounds(lower, upper)

    def get_params(self) -> np.ndarray:
        params = list(super().get_params()) + [
            self.model.heston1.jumps.intensity,
            self.model.heston1.jumps.jumps.variance(),
        ]
        try:
            params.append(self.model.heston1.jumps.jumps.asymmetry())
        except NotImplementedError:
            pass
        return np.asarray(params)

    def set_params(self, params: np.ndarray) -> None:
        super().set_params(params)
        self.model.heston1.jumps.intensity = params[10]
        self.model.heston1.jumps.jumps.set_variance(params[11])
        try:
            self.model.heston1.jumps.jumps.set_asymmetry(params[12])
        except IndexError:
            pass
