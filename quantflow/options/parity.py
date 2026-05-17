from __future__ import annotations

from decimal import Decimal
from typing import Any, Self

import numpy as np
from pydantic import BaseModel, Field
from scipy.optimize import lsq_linear

from quantflow.utils.numbers import ZERO, Number, to_decimal
from quantflow.utils.price import Price
from quantflow.utils.types import FloatArray


class DiscountPair(BaseModel, frozen=True):
    asset_discount: float = Field(
        description="Discount factor for the underlying asset"
    )
    quote_discount: float = Field(description="Discount factor for the option quote")


class PutCallParity(BaseModel, frozen=True):
    """A matched put-call parity at a single strike,
    used for discount curve calibration."""

    strike: Decimal = Field(description="Strike price")
    call: Price = Field(description="Call option bid/ask prices")
    put: Price = Field(description="Put option bid/ask prices")
    inverse: bool = Field(default=False, description="Whether the option is inverse")

    @property
    def bid(self) -> Decimal:
        """Lower bound of the call-put price difference"""
        return self.call.bid - self.put.ask

    @property
    def ask(self) -> Decimal:
        """Upper bound of the call-put price difference"""
        return self.call.ask - self.put.bid

    @property
    def mid(self) -> Decimal:
        """Midpoint of the call-put price difference"""
        return (self.bid + self.ask) / 2

    @property
    def spread(self) -> Decimal:
        """Bid-ask spread of the call-put price difference"""
        return self.ask - self.bid


class PutCallParities(BaseModel, frozen=True):
    """A collection of put-call parities for a given maturity"""

    parities: list[PutCallParity] = Field(description="List of put-call parities")
    spot: Decimal = Field(ge=ZERO, description="Spot price of the underlying asset")
    ttm: Decimal = Field(gt=ZERO, description="Time to maturity in years")
    inverse: bool = Field(default=False, description="Whether the options are inverse")

    @classmethod
    def from_parities(
        cls, parities: list[PutCallParity], spot: Number, ttm: Number
    ) -> Self:
        inverse = any(p.inverse for p in parities)
        return cls(
            parities=parities,
            spot=to_decimal(spot),
            ttm=to_decimal(ttm),
            inverse=inverse,
        )

    def regressand(self) -> FloatArray:
        """Calculate the regressand for put-call parity regression.

        For direct options, the regressand is (C - P) / S, while for inverse
        options it is simply c - p.
        """
        scale = self.spot if not self.inverse else Decimal(1)
        return np.asarray([float(p.mid / scale) for p in self.parities])

    def regressor(self) -> FloatArray:
        """Calculate the regressor for put-call parity regression,
        which is the strike price divided by the spot price.
        """
        return np.asarray([float(p.strike / self.spot) for p in self.parities])

    def fit_discounts(
        self,
        dq: float | None = None,
        da: float | None = None,
        min_rate_q: float = 0.0,
        min_rate_a: float = 0.0,
    ) -> DiscountPair | None:
        """Return the fitted discount factors, or None if the result is invalid.

        Both direct and inverse options satisfy the same normalized equation
        y = Da - (Dq/S) * K, where y = mid/S for direct and y = mid for inverse.

        When both known values are None a full OLS is run via constrained least squares.
        When one is provided the other is solved analytically as the mean over pairs.
        Discount factors are bounded by D <= exp(-min_rate * ttm), so min_rate=0
        enforces D <= 1 (non-negative rates).
        """
        if not self.parities:
            return None
        ys = self.regressand()
        xs = self.regressor()
        ttm = float(self.ttm)
        max_dq = float(np.exp(-min_rate_q * ttm))
        max_da = float(np.exp(-min_rate_a * ttm))
        if dq is not None:
            if da is not None:
                return DiscountPair(asset_discount=da, quote_discount=dq)
            da = float(np.mean(ys + dq * xs))
        elif da is not None:
            dq = float(np.mean((da - ys) / xs))
        else:
            A = np.column_stack([np.ones(len(xs)), -xs])
            result = lsq_linear(A, ys, bounds=([0, 0], [max_da, max_dq]))
            da, dq = float(result.x[0]), float(result.x[1])
        if not (0 < dq <= max_dq and 0 < da <= max_da):
            return None
        return DiscountPair(asset_discount=da, quote_discount=dq)

    def plot(
        self,
        dq: float | None = None,
        da: float | None = None,
        min_rate_q: float = 0.0,
        min_rate_a: float = 0.0,
    ) -> Any:
        """Plot the normalized put-call parity data and the fitted regression line."""
        from quantflow.utils.plot import check_plotly

        check_plotly()
        import plotly.graph_objects as go

        xs = self.regressor()
        ys = self.regressand()
        discounts = self.fit_discounts(
            dq=dq, da=da, min_rate_q=min_rate_q, min_rate_a=min_rate_a
        )
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=xs, y=ys, mode="markers", name="market", marker_size=10)
        )
        if discounts is not None:
            x_range = np.linspace(xs.min(), xs.max(), 100)
            y_fit = discounts.asset_discount - discounts.quote_discount * x_range
            fig.add_trace(go.Scatter(x=x_range, y=y_fit, mode="lines", name="fit"))
        y_label = "c - p" if self.inverse else "(C - P) / S"
        return fig.update_layout(xaxis_title="K / S", yaxis_title=y_label)
