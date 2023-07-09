from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any, Generic, NamedTuple, Sequence, TypeVar

import numpy as np
from scipy.optimize import Bounds, OptimizeResult, minimize

from quantflow.sp.base import StochasticProcess
from quantflow.sp.heston import Heston

from .surface import OptionPrice, VolSurface

M = TypeVar("M", bound=StochasticProcess)


class ModelCalibrationEntryKey(NamedTuple):
    maturity: datetime
    strike: Decimal


@dataclass
class OptionEntry:
    """Entry for a single option"""

    options: list[OptionPrice] = field(default_factory=list)

    def implied_vol_range(self) -> Bounds:
        """Get the range of implied volatilities"""
        implied_vols = tuple(option.implied_vol for option in self.options)
        return Bounds(min(implied_vols), max(implied_vols))

    def price_range(self) -> Bounds:
        """Get the range of prices"""
        prices = tuple(option.call_price for option in self.options)
        return Bounds(min(prices), max(prices))


@dataclass
class VolModelCalibration(ABC, Generic[M]):
    """Calibration of a stochastic volatility model"""

    model: M
    vol_surface: VolSurface[Any]
    minimize_method: str | None = None
    options: dict[ModelCalibrationEntryKey, OptionEntry] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.vol_surface.bs()
        for option in self.vol_surface.option_prices():
            key = ModelCalibrationEntryKey(option.maturity, option.strike)
            if key not in self.options:
                entry = OptionEntry()
                self.options[key] = entry
            entry.options.append(option)

    @abstractmethod
    def get_params(self) -> np.ndarray:
        """Get the parameters of the model"""

    @abstractmethod
    def cost_function(self, params: np.ndarray) -> float:
        """Cost function to minimize"""

    def get_bounds(self) -> Sequence[Bounds] | None:
        """Get the bounds for the calibration"""
        return None

    def get_constraints(self) -> Sequence[dict[str, Any]] | None:
        """Get the constraints for the calibration"""
        return None

    def implied_vol_range(self) -> Bounds:
        """Get the range of implied volatilities"""
        return Bounds(
            min(option.implied_vol_range().lb for option in self.options.values()),
            max(option.implied_vol_range().ub for option in self.options.values()),
        )

    def fit(self) -> OptimizeResult:
        """Fit the model"""
        return minimize(
            self.cost_function,
            self.get_params(),
            method=self.minimize_method,
            bound=self.get_bounds(),
            constraints=self.get_constraints(),
        )


@dataclass
class HestonCalibration(VolModelCalibration[Heston]):
    """Calibration of a stochastic volatility model"""

    def get_bounds(self) -> Sequence[Bounds] | None:
        vol_range = self.implied_vol_range()
        return (
            Bounds(0.5 * vol_range.lb, 2 * vol_range.ub),
            Bounds(0.5 * vol_range.lb, 2 * vol_range.ub),
            Bounds(0.0, None),
            Bounds(0.0, None),
            Bounds(-0.9, 0.9),
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

    def cost_function(self, params: np.ndarray) -> float:
        self.model.variance_process.rate = params[0]
        self.model.variance_process.theta = params[1]
        self.model.variance_process.kappa = params[2]
        self.model.variance_process.sigma = params[3]
        self.model.rho = params[4]
        return 0.0
