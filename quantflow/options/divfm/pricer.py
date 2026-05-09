from __future__ import annotations

import numpy as np
from pydantic import Field

from quantflow.utils.marginal import Greeks, OptionPricingResult
from quantflow.utils.types import FloatArray, FloatArrayLike

from ..bs import black_call
from ..pricer import MaturityPricer, OptionPricerBase
from .weights import DIVFMWeights


class OptionPricingResultDIVFM(OptionPricingResult, arbitrary_types_allowed=True):
    """Lazy DIVFM call-pricing result.

    Evaluates the fitted IV surface on demand at any log-strike via the
    network weights and OLS factor loadings, then applies Black-Scholes.
    """

    weights: DIVFMWeights = Field(description="Trained DIVFM network weights")
    betas: FloatArray = Field(description="Daily OLS factor loadings")
    ttm: float = Field(description="Time to maturity in years")
    extra: FloatArray | None = Field(
        default=None,
        description="Day-level extra features, shape (1, extra_features)",
    )

    def call_price(self, log_strikes: FloatArrayLike) -> FloatArray:
        """Evaluate call prices at arbitrary log-strikes."""
        ks = np.asarray(log_strikes, dtype=np.float32).reshape(-1)
        moneyness_ttm = ks / np.float32(np.sqrt(self.ttm))
        ttm_arr = np.full(ks.shape, self.ttm, dtype=np.float32)
        extra_arr = (
            np.repeat(self.extra, len(ks), axis=0) if self.extra is not None else None
        )
        F = self.weights.forward(moneyness_ttm, ttm_arr, extra_arr)
        iv = np.clip(F @ self.betas, 1e-6, None)
        return np.asarray(black_call(ks, iv, ttm=self.ttm), dtype=float)

    def call_greeks(self, log_strike: float) -> Greeks:
        """Call price and Greeks at a given log-strike via finite differences"""
        eps = 1e-4
        ks = np.array([log_strike - eps, log_strike, log_strike + eps])
        prices = self.call_price(ks)
        price = float(prices[1])
        dc_dk = float((prices[2] - prices[0]) / (2 * eps))
        d2c_dk2 = float((prices[2] - 2 * prices[1] + prices[0]) / (eps * eps))
        return Greeks(price=price, delta=price - dc_dk, gamma=d2c_dk2 - dc_dk)


class DIVFMPricer(OptionPricerBase, arbitrary_types_allowed=True):
    r"""Option pricer based on the Deep Implied Volatility Factor Model (DIVFM).

    The IV surface on a given day is modelled as a linear combination of p
    fixed latent functions learned by a neural network:

    \begin{equation}
        \sigma_t\left(m, \tau\right) = f\left(m, \tau; theta\right) \dot \beta_t
    \end{equation}

    where M = log(K/F) / sqrt(tau) is the time-scaled moneyness, f is
    implemented by [DIVFMWeights][quantflow.options.divfm.weights.DIVFMWeights],
    and beta_t are daily coefficients computed in closed form via OLS.

    Call prices are derived from the IV surface via Black-Scholes.

    Usage
    -----
    1. Train a [DIVFMNetwork][quantflow.options.divfm.network.DIVFMNetwork]
       and call its ``to_weights()`` method to obtain a
       [DIVFMWeights][quantflow.options.divfm.weights.DIVFMWeights] instance.
    2. Construct this pricer with those weights.
    3. Call [calibrate][quantflow.options.divfm.pricer.DIVFMPricer.calibrate]
       with the day's observed implied volatilities to fit beta_t.
    4. Use [maturity][quantflow.options.pricer.OptionPricerBase.maturity],
       [price][quantflow.options.pricer.OptionPricerBase.price] etc. as normal.
    """

    weights: DIVFMWeights = Field(
        description=(
            "Extracted weights of the trained DIVFM network."
            " No torch dependency required at inference time"
        )
    )
    betas: FloatArray = Field(
        default_factory=lambda: np.zeros(5),
        description="Daily OLS factor loadings beta_t, shape (num_factors,)",
    )
    extra: FloatArray | None = Field(
        default=None,
        description=(
            "Current day's observable features X, shape (extra_features,)."
            " Broadcast across all grid points in _compute_maturity."
            " Set automatically by calibrate() when extra is provided"
        ),
    )

    def calibrate(
        self,
        moneyness_ttm: FloatArray,
        ttm: FloatArray,
        implied_vols: FloatArray,
        extra: FloatArray | None = None,
    ) -> None:
        """Fit daily OLS coefficients from observed implied volatilities.

        Given a set of options observed on a single day, computes the
        closed-form OLS estimate:

            beta_t = (F^T F)^{-1} F^T IV_t

        where F is the (N, p) matrix of factor values from the network.

        Parameters
        ----------
        moneyness_ttm:
            Shape (N,). Time-scaled moneyness M = log(K/F) / sqrt(tau).
        ttm:
            Shape (N,). Time-to-maturity tau in years.
        implied_vols:
            Shape (N,). Observed implied volatilities.
        extra:
            Shape (N, extra_features) or None. Additional features passed to
            the network (e.g. time-to-earnings-announcement).
        """
        extra_arr = np.asarray(extra, dtype=np.float32) if extra is not None else None
        F = self.weights.forward(
            np.asarray(moneyness_ttm, dtype=np.float32),
            np.asarray(ttm, dtype=np.float32),
            extra_arr,
        )
        self.betas = np.linalg.lstsq(F, implied_vols, rcond=None)[0]
        # Store the mean X across options as the day-level representative value
        # used when pricing on a grid in _compute_maturity
        self.extra = (
            extra_arr.mean(axis=0, keepdims=True) if extra_arr is not None else None
        )
        self.reset()

    def _compute_maturity(self, ttm: float) -> MaturityPricer:
        """Compute a MaturityPricer for the given TTM using the fitted IV surface."""
        return MaturityPricer(
            ttm=ttm,
            pricing=OptionPricingResultDIVFM(
                weights=self.weights,
                betas=self.betas,
                ttm=ttm,
                extra=self.extra,
            ),
            name="DIVFM",
        )
