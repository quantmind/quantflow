from __future__ import annotations

from typing import NamedTuple

import numpy as np
from pydantic import BaseModel, Field
from scipy.optimize import newton
from scipy.stats import norm
from typing_extensions import Annotated, Doc

from ..utils.types import BoolArray, Float, FloatArray, FloatArrayLike


class ImpliedVol(NamedTuple):
    """Result of implied volatility calculation"""

    value: Float
    """The implied volatility in decimals (0.2 for 20%)"""
    converged: bool
    """Whether the root finding algorithm converged"""


class ImpliedVols(NamedTuple):
    """Result of root finding algorithm"""

    values: FloatArray
    """Array of implied volatilities in decimals (0.2 for 20%)"""
    converged: BoolArray
    """Array indicating whether the root finding algorithm converged for each
    implied volatility"""

    def single(self) -> ImpliedVol:
        """Return the first implied volatility and convergence status a
        a single ImpliedVol"""
        if len(self.values) != 1 or len(self.converged) != 1:
            raise ValueError("Expected exactly one root and convergence status")
        return ImpliedVol(value=self.values[0], converged=self.converged[0])


class BlackSensitivities(BaseModel, frozen=True, arbitrary_types_allowed=True):
    """Black model sensitivities (Greeks) for a single option or array of options"""

    iv: FloatArrayLike = Field(description="Implied Black volatility (0.1 for 10%)")
    price: FloatArrayLike = Field(description="Option price in forward space")
    delta: FloatArrayLike = Field(
        description="Change in price per unit change in forward"
    )
    gamma: FloatArrayLike = Field(
        description="Change in delta per unit change in forward"
    )
    vega: FloatArrayLike = Field(
        description="Change in price per unit change in volatility"
    )
    volga: FloatArrayLike = Field(
        description="Change in vega per unit change in volatility"
    )
    vanna: FloatArrayLike = Field(
        description="Change in delta per unit change in volatility"
    )
    theta: FloatArrayLike = Field(
        description="Change in price per unit time (negative = time decay)"
    )

    @classmethod
    def calculate(
        cls,
        k: Annotated[FloatArrayLike, Doc("Vector or single value of log-strikes")],
        ttm: Annotated[FloatArrayLike, Doc("Time to maturity")],
        s: Annotated[FloatArrayLike, Doc("Call/put flag (1 for call, -1 for put)")],
        iv: Annotated[
            FloatArrayLike | None, Doc("Implied volatility (0.2 for 20%)")
        ] = None,
        price: Annotated[
            FloatArrayLike | None, Doc("Option price in forward space")
        ] = None,
    ) -> BlackSensitivities:
        r"""Calculate Black model sensitivities (Greeks) in forward space.
        Either the implied volatility `iv` or the option `price` must be provided.
        If both are provided, the implied volatility will be used.

        \begin{align}
            d_1 &= \frac{-k + \frac{1}{2}\sigma^2 t}{\sigma\sqrt{t}} \\
            \Delta &= N(d_1) - \frac{1-s}{2} \\
            \Gamma &= \frac{N'(d_1)}{\sigma\sqrt{t}} \\
            \nu &= N'(d_1)\sqrt{t} \\
            \text{volga} &= \nu \cdot \frac{d_1 d_2}{\sigma\sqrt{t}} \\
            \text{vanna} &= -\frac{N'(d_1) d_2}{\sigma\sqrt{t}} \\
            \Theta &= -\frac{N'(d_1)\sigma}{2\sqrt{t}}
        \end{align}
        """
        if iv is None:
            if price is None:
                raise ValueError(
                    "Either implied volatility or option price must be provided"
                )
            result = implied_black_volatility(
                k=k,
                price=price,
                ttm=ttm,
                initial_sigma=0.2,
                call_put=s,
            )
            iv = result.single().value if np.isscalar(price) else result.values
        else:
            price = black_price(k, iv, ttm, s)
        sig2 = iv * iv * ttm
        sig = np.sqrt(sig2)
        d1 = (-k + 0.5 * sig2) / sig
        d2 = d1 - sig
        npd1 = norm.pdf(d1)
        nd1 = norm.cdf(d1)
        vega = npd1 * np.sqrt(ttm)
        return cls(
            iv=iv,
            price=price,
            delta=nd1 - 0.5 * (1 - s),
            gamma=npd1 / sig,
            vega=vega,
            volga=vega * d1 * d2 / sig,
            vanna=-npd1 * d2 / sig,
            theta=-npd1 * iv / (2 * np.sqrt(ttm)),
        )


def black_call(
    k: Annotated[FloatArrayLike, Doc("Vector or single value of log-strikes")],
    sigma: Annotated[
        FloatArrayLike,
        Doc(
            "Corresponding vector or single value of implied volatilities "
            "(0.2 for 20%)"
        ),
    ],
    ttm: Annotated[
        FloatArrayLike, Doc("Corresponding vector or single value of Time to Maturity")
    ],
) -> FloatArrayLike:
    kk = np.asarray(k)
    return black_price(kk, np.asarray(sigma), np.asarray(ttm), np.ones(kk.shape))


def black_price(
    k: Annotated[
        FloatArrayLike,
        Doc("Vector or single value of log-strikes"),
    ],
    sigma: Annotated[
        FloatArrayLike,
        Doc(
            (
                "Corresponding vector or single value of "
                "implied volatilities (0.2 for 20%)"
            )
        ),
    ],
    ttm: Annotated[
        FloatArrayLike, Doc("Corresponding vector or single value of Time to Maturity")
    ],
    s: Annotated[
        FloatArrayLike,
        Doc(
            "Corresponding vector or single value of call/put flag "
            "(1 for call, -1 for put)"
        ),
    ],
) -> FloatArray:
    r"""Calculate the Black call/put option prices in forward terms
    from the following params

    $$
    \begin{align}
        c &= \frac{C}{F} = N(d1) - e^k N(d2) \\
        p &= \frac{P}{F} = -N(-d1) + e^k N(-d2) \\
        d1 &= \frac{-k + \frac{\sigma^2 t}{2}}{\sigma \sqrt{t}} \\
        d2 &= d1 - \sigma \sqrt{t}
    \end{align}
    $$

    The results are option prices divided by the forward price also known as
    option prices in forward terms. These are non-dimensional prices
    that can be easily converted to actual option prices by multiplying with the
    forward price of the underlying asset.
    """
    sig2 = sigma * sigma * ttm
    sig = np.sqrt(sig2)
    d1 = (-k + 0.5 * sig2) / sig
    d2 = d1 - sig
    return s * norm.cdf(s * d1) - s * np.exp(k) * norm.cdf(s * d2)


def black_delta(
    k: Annotated[
        FloatArrayLike,
        Doc("Vector or single value of log-strikes"),
    ],
    sigma: Annotated[
        FloatArrayLike,
        Doc(
            "Corresponding vector or single value of implied volatilities "
            "(0.2 for 20%)"
        ),
    ],
    ttm: Annotated[
        FloatArrayLike, Doc("Corresponding vector or single value of Time to Maturity")
    ],
    s: Annotated[
        FloatArrayLike,
        Doc(
            "Corresponding vector or single value of call/put flag "
            "(1 for call, -1 for put)"
        ),
    ],
) -> FloatArrayLike:
    r"""Calculate the Black call/put option delta from the moneyness,
    volatility and time to maturity.

    \begin{align}
        \delta_c &= \frac{\partial C}{\partial F} = N(d1) \\
        \delta_p &= \frac{\partial P}{\partial F} = N(d1) - 1
    \end{align}
    """
    sig2 = sigma * sigma * ttm
    sig = np.sqrt(sig2)
    d1 = (-k + 0.5 * sig2) / sig
    return norm.cdf(d1) - 0.5 * (1 - s)


def black_vega(
    k: Annotated[
        FloatArrayLike,
        Doc("Vector or single value of log-strikes"),
    ],
    sigma: Annotated[
        FloatArrayLike,
        Doc(
            "Corresponding vector or single value of implied volatilities (0.2 for 20%)"
        ),
    ],
    ttm: Annotated[
        FloatArrayLike, Doc("Corresponding vector or single value of Time to Maturity")
    ],
) -> FloatArrayLike:
    r"""Calculate the Black option vega from the log-strikes,
    volatility and time to maturity. The vega is the same for calls and puts.

    $$
    \begin{align}
        \nu &= \frac{\partial c}{\partial \sigma} \\
            &= \frac{\partial p}{\partial \sigma}\\
            &= N'(d1) \sqrt{t}
    \end{align}
    $$

    Same formula for both calls and puts.
    """
    sig2 = sigma * sigma * ttm
    sig = np.sqrt(sig2)
    d1 = (-k + 0.5 * sig2) / sig
    return norm.pdf(d1) * np.sqrt(ttm)


def implied_black_volatility(
    k: Annotated[
        FloatArrayLike,
        Doc("Vector or scalar of log strikes"),
    ],
    price: Annotated[
        FloatArrayLike,
        Doc(
            "Corresponding vector or scalar of option price in forward terms "
            "(price divided by forward price)"
        ),
    ],
    ttm: Annotated[
        FloatArrayLike,
        Doc("Corresponding vector or single value of Time to Maturity"),
    ],
    initial_sigma: Annotated[
        FloatArrayLike,
        Doc("Corresponding vector or single value of initial volatility"),
    ],
    call_put: Annotated[
        FloatArrayLike,
        Doc(
            "Corresponding vector or single value of call/put flag "
            "(1 for call, -1 for put)"
        ),
    ],
) -> ImpliedVols:
    """Calculate the implied black volatility via Newton's method

    It returns a [ImpliedVols][quantflow.options.bs.ImpliedVols] object which
    contains the implied volatility and convergence status.
    Implied volatility is in decimals (0.2 for 20%).
    """
    if not np.isscalar(k) and np.isscalar(initial_sigma):
        initial_sigma = np.full_like(k, initial_sigma)
    result = newton(
        lambda x: black_price(k, x, ttm, call_put) - price,
        initial_sigma,
        fprime=lambda x: black_vega(k, x, ttm),
        full_output=True,
    )
    if hasattr(result, "root"):
        return ImpliedVols(values=result.root, converged=result.converged)
    else:
        return ImpliedVols(
            values=np.asarray([result[0]]), converged=np.asarray([result[1]])
        )
