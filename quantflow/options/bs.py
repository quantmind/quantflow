import numpy as np
from scipy.optimize import RootResults, newton
from scipy.stats import norm
from typing_extensions import Annotated, Doc

from ..utils.types import FloatArray, FloatArrayLike


def black_call(
    k: Annotated[np.ndarray, Doc("Vector or single value of log-strikes")],
    sigma: Annotated[
        FloatArrayLike,
        Doc(
            "Corresponding vector or single value of implied volatilities (0.2 for 20%)"
        ),
    ],
    ttm: Annotated[
        FloatArrayLike, Doc("Corresponding vector or single value of Time to Maturity")
    ],
) -> np.ndarray:
    kk = np.asarray(k)
    return black_price(kk, np.asarray(sigma), np.asarray(ttm), np.ones(kk.shape))


def black_price(
    k: Annotated[np.ndarray, Doc("Vector of log-strikes")],
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
    s: Annotated[FloatArrayLike, Doc("Call/Put Flag (1 for call, -1 for put)")],
) -> np.ndarray:
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
    option prices in forward terms.
    """
    sig2 = sigma * sigma * ttm
    sig = np.sqrt(sig2)
    d1 = (-k + 0.5 * sig2) / sig
    d2 = d1 - sig
    return s * norm.cdf(s * d1) - s * np.exp(k) * norm.cdf(s * d2)


def black_delta(
    k: Annotated[np.ndarray, Doc("a vector of moneyness, see above")],
    sigma: Annotated[
        FloatArrayLike,
        Doc("a corresponding vector of implied volatilities (0.2 for 20%)"),
    ],
    ttm: Annotated[
        FloatArrayLike, Doc("Corresponding vector or single value of Time to Maturity")
    ],
    s: Annotated[
        FloatArrayLike,
        Doc("Call/Put vector or single value Flag (1 for call, -1 for put)"),
    ],
) -> np.ndarray:
    r"""Calculate the Black call/put option delta from the moneyness,
    volatility and time to maturity.

    $$
    \begin{align}
        \delta_c &= \frac{\partial C}{\partial F} = N(d1) \\
        \delta_p &= \frac{\partial P}{\partial F} = N(d1) - 1
    \end{align}
    $$
    """
    sig2 = sigma * sigma * ttm
    sig = np.sqrt(sig2)
    d1 = (-k + 0.5 * sig2) / sig
    return norm.cdf(d1) - 0.5 * (1 - s)


def black_vega(
    k: Annotated[np.ndarray, Doc("a vector of moneyness, see above")],
    sigma: Annotated[
        np.ndarray, Doc("corresponding vector of implied volatilities (0.2 for 20%)")
    ],
    ttm: Annotated[FloatArrayLike, Doc("Time to Maturity")],
) -> np.ndarray:
    r"""Calculate the Black option vega from the moneyness,
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
    k: Annotated[np.ndarray, Doc("Vector of log strikes")],
    price: Annotated[np.ndarray, Doc("Corresponding vector of option_price/forward")],
    ttm: Annotated[FloatArrayLike, Doc("Time to Maturity")],
    initial_sigma: Annotated[FloatArray, Doc("Initial Volatility")],
    call_put: Annotated[FloatArrayLike, Doc("Call/Put Flag")],
) -> RootResults:
    """Calculate the implied block volatility via Newton's method"""
    return newton(
        lambda x: black_price(k, x, ttm, call_put) - price,
        initial_sigma,
        fprime=lambda x: black_vega(k, x, ttm),
        full_output=True,
    )
