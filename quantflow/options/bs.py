import numpy as np
import numpy.typing as npt
from scipy.optimize import RootResults, newton
from scipy.stats import norm


def black_call(
    k: npt.ArrayLike, sigma: npt.ArrayLike, ttm: npt.ArrayLike
) -> np.ndarray:
    kk = np.asarray(k)
    return black_price(kk, np.asarray(sigma), np.asarray(ttm), np.ones(kk.shape))


def black_put(k: npt.ArrayLike, sigma: npt.ArrayLike, ttm: npt.ArrayLike) -> np.ndarray:
    kk = np.asarray(k)
    return black_price(kk, np.asarray(sigma), np.asarray(ttm), -np.ones(kk.shape))


def black_price(
    k: np.ndarray, sigma: np.ndarray, ttm: np.ndarray, s: np.ndarray
) -> np.ndarray:
    """Calculate the Black call option price from

    1) a vector of log(strikes/forward)
    2) a corresponding vector of implied volatilities (0.2 for 20%)
    3) time to maturity
    4) s as the call/put flag, 1 for calls, -1 for puts
    """
    sig2 = sigma * sigma * ttm
    sig = np.sqrt(sig2)
    d1 = (-k + 0.5 * sig2) / sig
    d2 = d1 - sig
    return s * norm.cdf(s * d1) - s * np.exp(k) * norm.cdf(s * d2)


def black_vega(k: np.ndarray, sigma: np.ndarray, ttm: np.ndarray) -> np.ndarray:
    """Calculate the Black option vega from the log strike,
    volatility and time to maturity.

    Same formula for both calls and puts.
    """
    sig2 = sigma * sigma * ttm
    sig = np.sqrt(sig2)
    d1 = (-k + 0.5 * sig2) / sig
    return norm.pdf(d1) * np.sqrt(ttm)


def implied_black_volatility(
    k: np.ndarray,
    price: np.ndarray,
    ttm: np.ndarray,
    initial_sigma: np.ndarray,
    call_put: np.ndarray,
) -> RootResults:
    """Calculate the implied block volatility from

    1) a vector of log(strikes/forward)
    2) a corresponding vector of call_price/forward
    3) time to maturity and
    4) initial volatility guess
    """
    return newton(
        lambda x: black_price(k, x, ttm, call_put) - price,
        initial_sigma,
        fprime=lambda x: black_vega(k, x, ttm),
        full_output=True,
    )
