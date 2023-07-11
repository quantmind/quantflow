from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from datetime import datetime
from decimal import Decimal
from typing import Any, Generic, NamedTuple, Sequence, TypeVar

import numpy as np
from scipy.optimize import Bounds, OptimizeResult, minimize

from quantflow.sp.base import StochasticProcess1D
from quantflow.sp.heston import Heston

from .pricer import OptionPricer
from .surface import OptionPrice, VolSurface

M = TypeVar("M", bound=StochasticProcess1D)


class ModelCalibrationEntryKey(NamedTuple):
    maturity: datetime
    strike: Decimal


@dataclass
class OptionEntry:
    """Entry for a single option"""

    ttm: float
    moneyness: float
    options: list[OptionPrice] = field(default_factory=list)
    _prince_range: Bounds | None = None

    def implied_vol_range(self) -> Bounds:
        """Get the range of implied volatilities"""
        implied_vols = tuple(option.implied_vol for option in self.options)
        return Bounds(min(implied_vols), max(implied_vols))

    def residual(self, price: float) -> float:
        """Calculate the residual for a given price

        when inside bid/offer, the residual is 0
        """
        return min(np.min(self.price_range().residual(price)), 0)

    def price_range(self) -> Bounds:
        """Get the range of prices"""
        if self._prince_range is None:
            prices = tuple(float(option.call_price) for option in self.options)
            self._prince_range = Bounds(min(prices), max(prices))
        return self._prince_range


@dataclass
class VolModelCalibration(ABC, Generic[M]):
    """Calibration of a stochastic volatility model"""

    pricer: OptionPricer[M]
    vol_surface: VolSurface[Any]
    minimize_method: str | None = None
    options: dict[ModelCalibrationEntryKey, OptionEntry] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.options:
            self.vol_surface.bs()
            for option in self.vol_surface.option_prices():
                key = ModelCalibrationEntryKey(option.maturity, option.strike)
                if key not in self.options:
                    entry = OptionEntry(ttm=option.ttm, moneyness=option.moneyness)
                    self.options[key] = entry
                entry.options.append(option)

    @abstractmethod
    def get_params(self) -> np.ndarray:
        """Get the parameters of the model"""

    @abstractmethod
    def cost_function(self, params: np.ndarray) -> float:
        """Cost function to minimize"""

    @property
    def model(self) -> M:
        """Get the model"""
        return self.pricer.model

    @property
    def implied_vols(self) -> np.ndarray:
        data: list[float] = []
        for entry in self.options.values():
            data.extend(option.implied_vol for option in entry.options)
        return np.asarray(data)

    def remove_implied_above(self, quantile: float = 0.95) -> VolModelCalibration:
        exclude_above = np.quantile(self.implied_vols, quantile)
        options = {}
        for key, entry in self.options.items():
            if entry.implied_vol_range().ub <= exclude_above:
                options[key] = entry
        return replace(self, options=options)

    def get_bounds(self) -> Bounds | None:
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
            bounds=self.get_bounds(),
            constraints=self.get_constraints(),
        )

    def calculate_cost(self) -> float:
        """Calculate the cost of the model"""
        cost = 0.0
        for entry in self.options.values():
            model_price = self.pricer.price(entry.ttm, entry.moneyness)
            if residual := entry.residual(model_price):
                cost += residual**2
        return cost


@dataclass
class HestonCalibration(VolModelCalibration[Heston]):
    """Calibration of a stochastic volatility model"""

    def get_bounds(self) -> Sequence[Bounds] | None:
        vol_range = self.implied_vol_range()
        vol_lb = 0.5 * vol_range.lb[0]
        vol_ub = 1.5 * vol_range.ub[0]
        return Bounds(
            [vol_lb, vol_lb, 0.0, 0.0, -0.9],
            [vol_ub, vol_ub, np.inf, np.inf, 0.9],
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
        self.pricer.reset()
        return self.calculate_cost()
