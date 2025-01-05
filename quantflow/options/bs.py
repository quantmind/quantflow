import numpy as np
from scipy.optimize import RootResults, newton
from scipy.stats import norm

from ..utils.types import FloatArray, FloatArrayLike


def black_call(
    k: FloatArrayLike, sigma: FloatArrayLike, ttm: FloatArrayLike
) -> np.ndarray:
    kk = np.asarray(k)
    return black_price(kk, np.asarray(sigma), np.asarray(ttm), np.ones(kk.shape))


def black_put(
    k: FloatArrayLike, sigma: FloatArrayLike, ttm: FloatArrayLike
) -> np.ndarray:
    kk = np.asarray(k)
    return black_price(kk, np.asarray(sigma), np.asarray(ttm), -np.ones(kk.shape))


def black_price(
    k: np.ndarray,
    sigma: FloatArrayLike,
    ttm: FloatArrayLike,
    s: FloatArrayLike,
) -> np.ndarray:
    r"""Calculate the Black call/put option prices in forward terms
    from the following params

    .. math::
        c &= \frac{C}{F} = N(d1) - e^k N(d2)

        p &= \frac{P}{F} = -N(-d1) + e^k N(-d2)

        d1 &= \frac{-k + \frac{\sigma^2 t}{2}}{\sigma \sqrt{t}}

        d2 &= d1 - \sigma \sqrt{t}

    :param k: a vector of :math:`\log{\frac{K}{F}}` also known as moneyness
    :param sigma: a corresponding vector of implied volatilities (0.2 for 20%)
    :param ttm: time to maturity
    :param s: the call/put flag, 1 for calls, -1 for puts

    The results are option prices divided by the forward price also known as
    option prices in forward terms.
    """
    sig2 = sigma * sigma * ttm
    sig = np.sqrt(sig2)
    d1 = (-k + 0.5 * sig2) / sig
    d2 = d1 - sig
    return s * norm.cdf(s * d1) - s * np.exp(k) * norm.cdf(s * d2)


def black_delta(
    k: np.ndarray,
    sigma: FloatArrayLike,
    ttm: FloatArrayLike,
    s: FloatArrayLike,
) -> np.ndarray:
    r"""Calculate the Black call/put option delta from the moneyness,
    volatility and time to maturity.

    .. math::
        \begin{align}
            \delta_c &= \frac{\partial C}{\partial F} = N(d1) \\
            \delta_p &= \frac{\partial P}{\partial F} = N(d1) - 1
        \end{align}

    :param k: a vector of moneyness, see above
    :param sigma: a corresponding vector of implied volatilities (0.2 for 20%)
    :param ttm: time to maturity
    :param s: the call/put flag, 1 for calls, -1 for puts
    """
    sig2 = sigma * sigma * ttm
    sig = np.sqrt(sig2)
    d1 = (-k + 0.5 * sig2) / sig
    return norm.cdf(d1) - 0.5 * (1 - s)


def black_vega(k: np.ndarray, sigma: np.ndarray, ttm: FloatArrayLike) -> np.ndarray:
    r"""Calculate the Black option vega from the moneyness,
    volatility and time to maturity.

    .. math::

        \nu = \frac{\partial c}{\partial \sigma} =
            \frac{\partial p}{\partial \sigma} = N'(d1) \sqrt{t}

    :param k: a vector of moneyness, see above
    :param sigma: a corresponding vector of implied volatilities (0.2 for 20%)
    :param ttm: time to maturity

    Same formula for both calls and puts.
    """
    sig2 = sigma * sigma * ttm
    sig = np.sqrt(sig2)
    d1 = (-k + 0.5 * sig2) / sig
    return norm.pdf(d1) * np.sqrt(ttm)


def implied_black_volatility(
    k: np.ndarray,
    price: np.ndarray,
    ttm: FloatArrayLike,
    initial_sigma: FloatArray,
    call_put: FloatArrayLike,
) -> RootResults:
    """Calculate the implied block volatility via Newton's method

    :param k: a vector of log(strikes/forward) also known as moneyness
    :param price: a corresponding vector of option_price/forward
    :param ttm: time to maturity
    :param initial_sigma: a vector of initial volatility guesses
    :param call_put: a vector of call/put flags, 1 for calls, -1 for puts
    """
    return newton(
        lambda x: black_price(k, x, ttm, call_put) - price,
        initial_sigma,
        fprime=lambda x: black_vega(k, x, ttm),
        full_output=True,
    )
