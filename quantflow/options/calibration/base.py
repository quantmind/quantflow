from __future__ import annotations

import enum
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

from ..bs import implied_black_volatility
from ..pricer import OptionPricerBase
from ..surface import OptionPrice, VolSurface

M = TypeVar("M", bound=StochasticProcess1D)


class ResidualKind(enum.StrEnum):
    """How calibration residuals are measured against the market"""

    PRICE = enum.auto()
    """Residual on forward-space option prices: `weight * (model_price - mid_price)`"""

    IV = enum.auto()
    """Residual on Black implied volatilities: `weight * (model_iv - mid_iv)`,
    where `model_iv` is recovered from the model price by Black-Scholes
    inversion. Naturally well-scaled across moneyness, so `moneyness_weight`
    is usually unnecessary."""


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

    def iv_range(self) -> Bounds:
        """Get the range of implied volatilities across bid and ask"""
        ivs = tuple(option.iv for option in self.options)
        return Bounds(min(ivs), max(ivs))

    def mid_price(self) -> float:
        """Mid price as the average of bid and ask call prices"""
        prices = tuple(float(option.call_price) for option in self.options)
        return sum(prices) / len(prices)

    def mid_iv(self) -> float:
        """Mid implied volatility as the average of bid and ask"""
        ivs = tuple(option.iv for option in self.options)
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
    residual_kind: ResidualKind = Field(
        default=ResidualKind.PRICE,
        description=(
            "Kind of residual used by the calibration cost function."
            " `price` (default) measures the residual on forward-space"
            " option prices and applies the `moneyness_weight` cost weights;"
            " `iv` measures it on Black implied volatilities by inverting"
            " the model price. The `iv` residual is already in vol units"
            " and is naturally well-scaled across moneyness, so the"
            " `moneyness_weight` cost weights are not applied in that mode."
        ),
    )
    moneyness_weight: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Coefficient that up-weights wing options in the cost function."
            " Applied as `min(exp(moneyness_weight * moneyness**2),"
            " max_cost_weight)`, with `moneyness = log(K/F) / sqrt(ttm)`."
            " The quadratic form mimics the gaussian shape of `1/vega` and"
            " puts wing residuals on the same footing as ATM ones. A value"
            " of 0 applies no moneyness weighting; typical values are in"
            " `[0.1, 0.5]`."
        ),
    )
    max_cost_weight: float = Field(
        default=10.0,
        gt=0.0,
        description=(
            "Hard cap on the per-option cost weight, to prevent a single"
            " deep-wing option from dominating the loss when"
            " `moneyness_weight` is large."
        ),
    )
    options: dict[ModelCalibrationEntryKey, OptionEntry] = Field(
        default_factory=dict,
        repr=False,
        description="The options to calibrate",
    )
    _log_strikes: np.ndarray = PrivateAttr(default_factory=lambda: np.empty(0))
    _ttms: np.ndarray = PrivateAttr(default_factory=lambda: np.empty(0))
    _moneyness: np.ndarray = PrivateAttr(default_factory=lambda: np.empty(0))
    _mid_prices: np.ndarray = PrivateAttr(default_factory=lambda: np.empty(0))
    _mid_ivs: np.ndarray = PrivateAttr(default_factory=lambda: np.empty(0))

    def model_post_init(self, _ctx: Any) -> None:
        if not self.options:
            self.vol_surface.bs()
            for option in self.vol_surface.option_prices(converged=True):
                key = ModelCalibrationEntryKey(option.maturity, option.strike)
                if key not in self.options:
                    entry = OptionEntry(ttm=option.ttm, log_strike=option.log_strike)
                    self.options[key] = entry
                entry.options.append(option)
        entries = tuple(self.options.values())
        self._log_strikes = np.asarray([e.log_strike for e in entries])
        self._ttms = np.asarray([e.ttm for e in entries])
        self._moneyness = self._log_strikes / np.sqrt(self._ttms)
        self._mid_prices = np.asarray([e.mid_price() for e in entries])
        self._mid_ivs = np.asarray([e.mid_iv() for e in entries])

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
    def ivs(self) -> np.ndarray:
        data: list[float] = []
        for entry in self.options.values():
            data.extend(option.iv for option in entry.options)
        return np.asarray(data)

    def iv_range(self) -> Bounds:
        """Range of implied volatilities across all calibration options"""
        return Bounds(
            min(option.iv_range().lb for option in self.options.values()),
            max(option.iv_range().ub for option in self.options.values()),
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
        """Weight for a given time to maturity and log-strike.

        Up-weights wing options via `exp(moneyness_weight * moneyness**2)`,
        capped at `max_cost_weight`. The quadratic form mimics `1/vega`.
        """
        moneyness = log_strike / np.sqrt(ttm)
        weight = np.exp(self.moneyness_weight * moneyness * moneyness)
        return float(min(weight, self.max_cost_weight))

    def cost_weights(self) -> np.ndarray:
        """Vector of cost weights for all calibration options"""
        weights = np.exp(self.moneyness_weight * self._moneyness * self._moneyness)
        return np.minimum(weights, self.max_cost_weight)

    def penalize(self) -> float:
        """Additional scalar penalty added to the cost function (default: 0)"""
        return 0.0

    def residuals(self, params: np.ndarray) -> np.ndarray:
        """Weighted residuals per option, in price or implied-vol space.

        Controlled by `residual_kind`:

        - `price`: `weight * (model_price - mid_price)`
        - `iv`: `weight * (model_iv - mid_iv)`, where `model_iv` is the
          Black implied volatility of the model price.
        """
        self.set_params(params)
        self.pricer.reset()
        with np.errstate(all="ignore"):
            try:
                model_prices = self.pricer.call_prices(self._ttms, self._log_strikes)
            except ValueError:
                return np.full(self._log_strikes.shape, 1e6)
            if self.residual_kind is ResidualKind.IV:
                implied = implied_black_volatility(
                    self._log_strikes,
                    model_prices,
                    self._ttms,
                    initial_sigma=self._mid_ivs,
                    call_put=1,
                )
                # Fourier pricers can return prices outside the no-arb band
                # for deep-wing strikes, where Newton fails to invert. Mask
                # those points out (zero residual). If fewer than half of the
                # options invert successfully the parameter set is treated as
                # invalid and a large penalty is returned.
                ok = implied.converged
                if 2 * int(ok.sum()) < ok.size:
                    return np.full(self._log_strikes.shape, 1e6)
                r = np.where(ok, implied.values - self._mid_ivs, 0.0)
            else:
                r = self.cost_weights() * (model_prices - self._mid_prices)
        return np.where(np.isfinite(r), r, 1e6)

    def cost_function(self, params: np.ndarray) -> float:
        """Scalar cost: sum of squared residuals plus any penalty"""
        r = self.residuals(params)
        return float(np.dot(r, r)) + self.penalize()

    def _model_grid(
        self, ttm: float, max_moneyness: float, support: int
    ) -> pd.DataFrame:
        log_strikes = np.linspace(-max_moneyness, max_moneyness, support) * np.sqrt(ttm)
        return self.pricer.maturity(ttm).prices(log_strikes)

    def plot(
        self,
        index: int = 0,
        *,
        max_moneyness: float = 1.0,
        support: int = 51,
        **kwargs: Any,
    ) -> Any:
        """Plot implied volatility for market and model prices"""
        cross = self.vol_surface.maturities[index]
        options = tuple(self.vol_surface.option_prices(index=index, converged=True))
        return plot.plot_vol_surface(
            pd.DataFrame([d.info_dict() for d in options]),
            model=self._model_grid(cross.ttm(self.ref_date), max_moneyness, support),
            **kwargs,
        )

    def plot_maturities(
        self,
        *,
        max_moneyness: float = 1.0,
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
            plot.plot_vol_surface(
                pd.DataFrame([d.info_dict() for d in options]),
                model=self._model_grid(
                    cross.ttm(self.ref_date), max_moneyness, support
                ),
                fig=fig,
                fig_params={"row": row, "col": col},
                **kwargs,
            )
        return fig
