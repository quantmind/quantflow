from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from datetime import datetime
from decimal import Decimal
from typing import Any, Generic, NamedTuple, Sequence, TypeVar

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, OptimizeResult, minimize

from quantflow.sp.base import StochasticProcess1D
from quantflow.sp.heston import Heston
from quantflow.utils import plot

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
    moneyness_weight: float = 0.5
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
    def set_params(self, params: np.ndarray) -> None:
        """Set the parameters of the model"""

    @property
    def model(self) -> M:
        """Get the model"""
        return self.pricer.model

    @property
    def ref_date(self) -> datetime:
        """Get the reference date"""
        return self.vol_surface.ref_date

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

    def get_bounds(self) -> Bounds | None:  # pragma: no cover
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

    def cost_weight(self, ttm: float, moneyness: float) -> float:
        return np.exp(-self.moneyness_weight * moneyness)

    def penalize(self) -> float:
        """Penalize the cost function"""
        return 0.0

    def cost_function(self, params: np.ndarray) -> float:
        """Calculate the cost function from the model prices"""
        self.set_params(params)
        self.pricer.reset()
        cost = 0.0
        for entry in self.options.values():
            model_price = self.pricer.call_price(entry.ttm, entry.moneyness)
            if residual := entry.residual(model_price):
                weight = self.cost_weight(entry.ttm, entry.moneyness)
                cost += weight * residual**2
        return cost + self.penalize()

    def plot(
        self,
        index: int = 0,
        *,
        max_moneyness_ttm: float | None = 1.0,
        support: int = 51,
        **kwargs: Any,
    ) -> Any:
        """Plot the implied volatility for market and model prices"""
        cross = self.vol_surface.maturities[index]
        options = tuple(self.vol_surface.option_prices(index=index))
        cross = self.vol_surface.maturities[index]
        model = self.pricer.maturity(cross.ttm(self.ref_date))
        if max_moneyness_ttm is not None:
            model = model.max_moneyness_ttm(
                max_moneyness_ttm=max_moneyness_ttm, support=support
            )
        return plot.plot_vol_surface(
            pd.DataFrame([d._asdict() for d in options]),
            model=model.df,
            **kwargs,
        )


@dataclass
class HestonCalibration(VolModelCalibration[Heston]):
    """Calibration of a stochastic volatility model"""

    feller_penalize: float = 0.0

    def get_bounds(self) -> Sequence[Bounds] | None:
        vol_range = self.implied_vol_range()
        vol_lb = 0.5 * vol_range.lb[0]
        vol_ub = 1.5 * vol_range.ub[0]
        return Bounds(
            [vol_lb * vol_lb, vol_lb * vol_lb, 0.0, 0.0, -0.9],
            [vol_ub * vol_ub, vol_ub * vol_ub, np.inf, np.inf, 0],
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
        kt = 2 * self.model.variance_process.kappa * self.model.variance_process.theta
        neg = max(0.5 * self.model.variance_process.sigma2 - kt, 0)
        return self.feller_penalize * neg * neg
