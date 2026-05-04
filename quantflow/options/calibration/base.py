from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from decimal import Decimal
from typing import Any, Generic, NamedTuple, TypeVar

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, PrivateAttr
from scipy.optimize import Bounds, OptimizeResult, least_squares, minimize

from quantflow.sp.base import StochasticProcess1D
from quantflow.utils import plot

from ..pricer import OptionPricerBase
from ..surface import OptionPrice, VolSurface

M = TypeVar("M", bound=StochasticProcess1D)


class ModelCalibrationEntryKey(NamedTuple):
    maturity: datetime
    strike: Decimal


class OptionEntry(BaseModel):
    """Entry for a single option in the calibration dataset.

    Each entry corresponds to a unique (maturity, strike) pair and holds the
    bid and ask sides as separate
    [OptionPrice][quantflow.options.surface.OptionPrice] objects.
    """

    ttm: float = Field(description="Time to maturity in years")
    log_strike: float = Field(description="Log-strike as log(K/F)")
    options: list[OptionPrice] = Field(default_factory=list)
    """Bid and ask option prices for this entry"""
    _mid_price: float | None = PrivateAttr(default=None)

    def implied_vol_range(self) -> Bounds:
        """Get the range of implied volatilities across bid and ask"""
        implied_vols = tuple(option.implied_vol for option in self.options)
        return Bounds(min(implied_vols), max(implied_vols))

    def mid_price(self) -> float:
        """Mid price as the average of bid and ask call prices"""
        if self._mid_price is None:
            prices = tuple(float(option.call_price) for option in self.options)
            self._mid_price = sum(prices) / len(prices)
        return self._mid_price

    def mid_iv(self) -> float:
        """Mid implied volatility as the average of bid and ask"""
        ivs = tuple(option.implied_vol for option in self.options)
        return sum(ivs) / len(ivs)


class VolModelCalibration(BaseModel, ABC, Generic[M]):
    """Abstract base class for calibration of a stochastic volatility model.

    Subclasses must implement `get_params`, `set_params`, and `get_bounds`.

    The two-stage `fit` method is provided here and works for any subclass:

    - Stage 1: Nelder-Mead on the scalar `cost_function` to find a good basin.
    - Stage 2: Levenberg-Marquardt (TRF) on the `residuals` vector for
      precise convergence with bound constraints.
    """

    pricer: OptionPricerBase = Field(
        description=(
            "The [OptionPricerBase][quantflow.options.pricer.OptionPricerBase]"
            " for the model"
        )
    )
    vol_surface: VolSurface[Any] = Field(
        repr=False,
        description=(
            "The [VolSurface][quantflow.options.surface.VolSurface]"
            " to calibrate the model with"
        ),
    )
    moneyness_weight: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Weight penalising options as moneyness moves away from 0."
            " Applied as `exp(-moneyness_weight * |moneyness|)`."
            " A value of 0 applies no penalisation."
        ),
    )
    options: dict[ModelCalibrationEntryKey, OptionEntry] = Field(
        default_factory=dict,
        repr=False,
        description="The options to calibrate",
    )

    def model_post_init(self, _ctx: Any) -> None:
        if not self.options:
            self.vol_surface.bs()
            for option in self.vol_surface.option_prices(converged=True):
                key = ModelCalibrationEntryKey(option.maturity, option.strike)
                if key not in self.options:
                    entry = OptionEntry(ttm=option.ttm, log_strike=option.log_strike)
                    self.options[key] = entry
                entry.options.append(option)

    @abstractmethod
    def get_params(self) -> np.ndarray:
        """Current model parameters as a flat array (starting point for fit)"""

    @abstractmethod
    def set_params(self, params: np.ndarray) -> None:
        """Apply a flat parameter array back to the model"""

    @abstractmethod
    def get_bounds(self) -> Bounds:
        """Parameter bounds for the optimiser"""

    @property
    def model(self) -> M:
        return self.pricer.model  # type: ignore[attr-defined]

    @property
    def ref_date(self) -> datetime:
        return self.vol_surface.ref_date

    @property
    def implied_vols(self) -> np.ndarray:
        data: list[float] = []
        for entry in self.options.values():
            data.extend(option.implied_vol for option in entry.options)
        return np.asarray(data)

    def implied_vol_range(self) -> Bounds:
        """Range of implied volatilities across all calibration options"""
        return Bounds(
            min(option.implied_vol_range().lb for option in self.options.values()),
            max(option.implied_vol_range().ub for option in self.options.values()),
        )

    def fit(self) -> OptimizeResult:
        """Two-stage fit: Nelder-Mead basin search then LM refinement.

        Stage 1 (Nelder-Mead): gradient-free minimisation of `cost_function`
        to reach the right basin of attraction.

        Stage 2 (TRF/LM): `scipy.optimize.least_squares` on the `residuals`
        vector with parameter bounds for precise convergence.
        """
        bounds = self.get_bounds()
        stage1 = minimize(
            self.cost_function,
            self.get_params(),
            method="L-BFGS-B",
            bounds=list(zip(bounds.lb, bounds.ub)),
        )
        result = least_squares(
            self.residuals,
            np.clip(stage1.x, bounds.lb, bounds.ub),
            method="trf",
            bounds=(bounds.lb, bounds.ub),
            ftol=1e-10,
            xtol=1e-10,
            gtol=1e-10,
            max_nfev=10000,
        )
        self.set_params(result.x)
        return result

    def cost_weight(self, ttm: float, log_strike: float) -> float:
        """Weight for a given time to maturity and log-strike"""
        moneyness = log_strike / np.sqrt(ttm)
        return np.exp(-self.moneyness_weight * abs(moneyness))

    def penalize(self) -> float:
        """Additional scalar penalty added to the cost function (default: 0)"""
        return 0.0

    def residuals(self, params: np.ndarray) -> np.ndarray:
        """Weighted price residuals: `weight * (model_price - mid_price)` per option"""
        self.set_params(params)
        self.pricer.reset()
        res = []
        with np.errstate(all="ignore"):
            for entry in self.options.values():
                model_price = self.pricer.call_price(entry.ttm, entry.log_strike)
                weight = self.cost_weight(entry.ttm, entry.log_strike)
                r = weight * (model_price - entry.mid_price())
                res.append(r if np.isfinite(r) else 1e6)
        return np.asarray(res)

    def cost_function(self, params: np.ndarray) -> float:
        """Scalar cost: sum of squared residuals plus any penalty"""
        r = self.residuals(params)
        return float(np.dot(r, r)) + self.penalize()

    def plot(
        self,
        index: int = 0,
        *,
        max_moneyness: float | None = 1.0,
        support: int = 51,
        **kwargs: Any,
    ) -> Any:
        """Plot implied volatility for market and model prices"""
        cross = self.vol_surface.maturities[index]
        options = tuple(self.vol_surface.option_prices(index=index, converged=True))
        model = self.pricer.maturity(cross.ttm(self.ref_date))
        if max_moneyness is not None:
            model = model.max_moneyness(max_moneyness=max_moneyness, support=support)
        return plot.plot_vol_surface(
            pd.DataFrame([d.info_dict() for d in options]),
            model=model.df,
            **kwargs,
        )

    def plot_maturities(
        self,
        *,
        max_moneyness: float | None = 1.0,
        support: int = 51,
        cols: int = 2,
        row_height: int = 400,
        showlegend: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Plot implied volatility for all maturities as a subplot grid"""
        plot.check_plotly()
        n = len(self.vol_surface.maturities)
        rows = (n + cols - 1) // cols
        titles = [
            cross.maturity.strftime("%Y-%m-%d") for cross in self.vol_surface.maturities
        ]
        fig = plot.make_subplots(rows=rows, cols=cols, subplot_titles=titles)
        fig.update_layout(height=rows * row_height, showlegend=showlegend)
        for i, cross in enumerate(self.vol_surface.maturities):
            row = i // cols + 1
            col = i % cols + 1
            options = tuple(self.vol_surface.option_prices(index=i, converged=True))
            model = self.pricer.maturity(cross.ttm(self.ref_date))
            if max_moneyness is not None:
                model = model.max_moneyness(
                    max_moneyness=max_moneyness, support=support
                )

            plot.plot_vol_surface(
                pd.DataFrame([d.info_dict() for d in options]),
                model=model.df,
                fig=fig,
                fig_params={"row": row, "col": col},
                **kwargs,
            )
        return fig
