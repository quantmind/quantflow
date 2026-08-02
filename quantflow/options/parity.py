from __future__ import annotations

from decimal import Decimal
from typing import Any, Self

import numpy as np
from pydantic import BaseModel, Field
from scipy.optimize import lsq_linear
from typing_extensions import Annotated, Doc

from quantflow.utils.numbers import ZERO, Number, to_decimal
from quantflow.utils.price import Price
from quantflow.utils.types import FloatArray


class DiscountPair(BaseModel, frozen=True):
    asset_discount: float = Field(
        description="Discount factor for the underlying asset"
    )
    quote_discount: float = Field(description="Discount factor for the option quote")


class PutCallParity(BaseModel, frozen=True):
    """A [put-call parity](../../glossary.md#put-call-parity) at a single strike

    used for forward and discount curve calibration."""

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
        r"""Return the fitted discount factors, or None if the result is invalid.

        Both direct and inverse options satisfy the same normalized equation

        \begin{equation}
            y = Da - K \frac{Dq}{S}
        \end{equation}

        where y = mid/S for direct and y = mid for inverse.

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

    def implied_forward(
        self,
        dq: float | None = None,
        da: float | None = None,
    ) -> float | None:
        """Implied forward price from put-call parity regression.

        Fits asset and quote discount factors from the put-call parity data and
        returns `spot * Da / Dq`. Returns None if the fit is invalid.
        """
        discounts = self.fit_discounts(dq=dq, da=da)
        if discounts is None:
            return None
        return float(self.spot) * discounts.asset_discount / discounts.quote_discount

    def weights(self) -> FloatArray:
        """Inverse bid-ask spread weights for the put-call parity regression.

        Pairs with a tighter parity spread receive a larger weight. A floor
        of one tenth of the median positive spread avoids infinite weights on
        zero spread pairs.
        """
        scale = Decimal(1) if self.inverse else self.spot
        spreads = np.asarray([float(p.spread / scale) for p in self.parities])
        positive = spreads[spreads > 0]
        floor = 0.1 * float(np.median(positive)) if positive.size else 1e-9
        return 1.0 / (spreads + max(floor, 1e-9))

    def calibrate_forward(
        self,
        *,
        band: Annotated[
            float,
            Doc(
                "Initial half width of the pair selection band, in units of "
                "convexity adjusted moneyness (standard deviations). "
                "The band widens automatically when it contains fewer than "
                "min_pairs pairs."
            ),
        ] = 1.0,
        min_pairs: Annotated[
            int, Doc("Minimum number of pairs; the band widens until reached")
        ] = 3,
        max_iterations: Annotated[
            int, Doc("Maximum number of forward refinement iterations")
        ] = 5,
        tol: Annotated[
            float, Doc("Relative tolerance on the forward for convergence")
        ] = 1e-5,
    ) -> float | None:
        """Calibrate the forward price from put-call parity.

        The forward is the zero crossing of the weighted parity regression.
        Pairs far from the money contain one deep in the money option whose
        quote carries little information, so the regression is restricted to
        pairs within `band` units of convexity adjusted moneyness.

        The moneyness requires the forward and the volatility, which are not
        known upfront. The algorithm therefore iterates: fit the crossing with
        all pairs, estimate the at the money volatility from the straddle
        nearest the crossing, select the pairs inside the band, refit, and
        repeat until the forward is stable.

        Returns the forward price, or None when fewer than two pairs are
        available or the regression is degenerate.
        """
        if len(self.parities) < 2:
            return None
        ys = self.regressand()
        xs = self.regressor()
        weights = self.weights()
        ttm = float(self.ttm)
        x0 = self._crossing(xs, ys, weights)
        if x0 is None:
            return None
        for _ in range(max_iterations):
            sigma_sqrt_ttm = self._straddle_vol(x0) * np.sqrt(ttm)
            moneyness = np.log(xs / x0) / sigma_sqrt_ttm + 0.5 * sigma_sqrt_ttm
            half_width = band
            selected = np.abs(moneyness) < half_width
            while selected.sum() < min(min_pairs, len(xs)):
                half_width *= 2
                selected = np.abs(moneyness) < half_width
            x1 = self._crossing(xs[selected], ys[selected], weights[selected])
            if x1 is None:
                break
            converged = abs(x1 - x0) <= tol * x0
            x0 = x1
            if converged:
                break
        return x0 * float(self.spot)

    def quote_discount(
        self,
        forward: Annotated[float, Doc("Forward price divided by the spot price")],
    ) -> float | None:
        """Quote discount factor with the forward held fixed.

        With the forward known, put-call parity has a single free parameter,
        the quote discount factor $D_q$:

        \\begin{equation}
            y = D_q \\left(f - x\\right)
        \\end{equation}

        where $y$ is the normalized call put difference, $x$ the strike over
        spot and $f$ the forward over spot. The parameter is estimated by
        weighted least squares over all pairs, with the weights of
        [weights][..weights]. Returns None when the estimate is not positive.
        """
        ys = self.regressand()
        xs = self.regressor()
        weights = self.weights()
        zs = forward - xs
        denominator = float(np.sum((weights * zs) ** 2))
        if denominator <= 0:
            return None
        dq = float(np.sum(weights**2 * ys * zs) / denominator)
        return dq if dq > 0 else None

    def _crossing(
        self, xs: FloatArray, ys: FloatArray, weights: FloatArray
    ) -> float | None:
        """Zero crossing of the weighted parity regression, or None when the
        slope is not negative."""
        slope, intercept = np.polyfit(xs, ys, 1, w=weights)
        if slope >= 0:
            return None
        return float(-intercept / slope)

    def _straddle_vol(self, x0: float) -> float:
        """Rough at the money volatility from the straddle nearest the
        crossing, used only to size the moneyness band."""
        scale = Decimal(1) if self.inverse else self.spot
        xs = self.regressor()
        pair = self.parities[int(np.argmin(np.abs(xs - x0)))]
        straddle = float((pair.call.mid + pair.put.mid) / scale)
        sigma = straddle / (0.7979 * np.sqrt(float(self.ttm)))
        return float(np.clip(sigma, 0.05, 5.0))

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
