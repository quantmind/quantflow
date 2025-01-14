from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any, Generic, NamedTuple, Sequence, TypeVar

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from scipy.optimize import Bounds, OptimizeResult, minimize

from quantflow.sp.base import StochasticProcess1D
from quantflow.sp.heston import Heston, HestonJ
from quantflow.sp.jump_diffusion import D
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


class VolModelCalibration(BaseModel, ABC, Generic[M], arbitrary_types_allowed=True):
    """Abstract class for calibration of a stochastic volatility model"""

    pricer: OptionPricer[M]
    """The :class:`.OptionPricer` for the model"""
    vol_surface: VolSurface[Any] = Field(repr=False)
    """The :class:`.VolSurface` to calibrate the model with"""
    minimize_method: str | None = None
    """The optimization method to use - if None, the default is used"""
    moneyness_weight: float = Field(default=0.0, ge=0.0)
    """The weight for penalize options with moneyness as it moves away from 0

    The weight is applied as exp(-moneyness_weight * moneyness), therefore
    a value of 0 won't penalize moneyness at all
    """
    ttm_weight: float = Field(default=0.0, ge=0.0, le=1.0)
    """The weight for penalize options with ttm as it approaches 0

    The weight is applied as `1 - ttm_weight*exp(-ttm)`, therefore
    a value of 0 won't penalize ttm at all, a value of 1 will penalize
    options with ttm->0 the most
    """
    options: dict[ModelCalibrationEntryKey, OptionEntry] = Field(
        default_factory=dict, repr=False
    )
    """The options to calibrate"""

    def model_post_init(self, _ctx: Any) -> None:
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
        """Get the parameters of the model

        Must be implemented by the subclass
        """

    @abstractmethod
    def set_params(self, params: np.ndarray) -> None:
        """Set the parameters of the model

        Must be implemented by the subclass
        """

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
        return self.model_copy(update=dict(options=options))

    def get_bounds(self) -> Bounds | None:  # pragma: no cover
        """Get the parameter bounds for the calibration"""
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
        """Calculate the weight for the cost function for
        a given time to maturity and moneyness"""
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


class HestonCalibration(VolModelCalibration[Heston]):
    """A :class:`.VolModelCalibration` for the :class:`.Heston`
    stochastic volatility model"""

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


class HestonJCalibration(VolModelCalibration[HestonJ[D]], Generic[D]):
    """A :class:`.VolModelCalibration` for the :class:`.HestonJ`
    stochastic volatility model with :class:`.DoubleExponential`
    jumps
    """

    feller_penalize: float = 0.0

    def get_bounds(self) -> Sequence[Bounds] | None:
        vol_range = self.implied_vol_range()
        vol_lb = 0.5 * vol_range.lb[0]
        vol_ub = 1.5 * vol_range.ub[0]
        lower = [
            (0.5 * vol_lb) ** 2,  # rate
            (0.5 * vol_lb) ** 2,  # theta
            0.001,  # kappa - mean reversion speed
            0.001,  # sigma - vol of vol
            -0.9,  # correlation
            1.0,  # jump intensity
            (0.01 * vol_lb) ** 2,  # jump variance
        ]
        upper = [
            (1.5 * vol_ub) ** 2,  # rate
            (1.5 * vol_ub) ** 2,  # theta
            np.inf,  # kappa
            np.inf,  # sigma
            0.0,  # correlation
            np.inf,  # jump intensity
            (0.5 * vol_ub) ** 2,  # jump variance
        ]
        try:
            self.model.jumps.jumps.asymmetry()
            lower.append(-2.0)  # jump asymmetry
            upper.append(2.0)
        except NotImplementedError:
            pass
        return Bounds(lower, upper)

    def get_params(self) -> np.ndarray:
        params = [
            self.model.variance_process.rate,
            self.model.variance_process.theta,
            self.model.variance_process.kappa,
            self.model.variance_process.sigma,
            self.model.rho,
            self.model.jumps.intensity,
            self.model.jumps.jumps.variance(),
        ]
        try:
            params.append(self.model.jumps.jumps.asymmetry())
        except NotImplementedError:
            pass
        return np.asarray(params)

    def set_params(self, params: np.ndarray) -> None:
        self.model.variance_process.rate = params[0]
        self.model.variance_process.theta = params[1]
        self.model.variance_process.kappa = params[2]
        self.model.variance_process.sigma = params[3]
        self.model.rho = params[4]
        self.model.jumps.intensity = params[5]
        self.model.jumps.jumps.set_variance(params[6])
        try:
            self.model.jumps.jumps.set_asymmetry(params[7])
        except IndexError:
            pass

    def penalize(self) -> float:
        kt = 2 * self.model.variance_process.kappa * self.model.variance_process.theta
        neg = max(0.5 * self.model.variance_process.sigma2 - kt, 0)
        return self.feller_penalize * neg * neg
