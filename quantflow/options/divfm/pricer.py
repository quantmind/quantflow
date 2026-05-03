from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import Field

from quantflow.utils.types import FloatArray

from ..bs import black_call
from ..pricer import MaturityPricer, OptionPricerBase
from .weights import DIVFMWeights


class DIVFMPricer(OptionPricerBase):
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
    max_moneyness_ttm: float = Field(
        default=3.0,
        description="Max time-scaled moneyness |M| used to build the pricing grid",
    )
    n: int = Field(
        default=100, description="Number of grid points along the moneyness axis"
    )

    model_config = {"arbitrary_types_allowed": True}

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

    def _compute_maturity(self, ttm: float, **kwargs: Any) -> MaturityPricer:
        """Compute a MaturityPricer for the given TTM using the fitted IV surface."""
        M_grid = np.linspace(
            -self.max_moneyness_ttm, self.max_moneyness_ttm, self.n, dtype=np.float32
        )
        ttm_arr = np.full_like(M_grid, ttm)
        # Broadcast day-level X across all grid points
        extra_grid = (
            np.repeat(self.extra, self.n, axis=0) if self.extra is not None else None
        )
        F = self.weights.forward(M_grid, ttm_arr, extra_grid)

        iv_grid = np.clip(F @ self.betas, 1e-6, None)

        # Convert from time-scaled moneyness M to log-strike = log(K/F)
        log_strike = M_grid * np.sqrt(ttm)
        call = np.asarray(black_call(log_strike, iv_grid, ttm=ttm))

        return MaturityPricer(
            ttm=ttm,
            std=float(np.mean(iv_grid) * np.sqrt(ttm)),
            log_strike=log_strike,
            call=call,
            name="DIVFM",
        )
