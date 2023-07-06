import numpy as np
from scipy.optimize import newton
from scipy.stats import norm


def black_call(k: np.ndarray, sigma: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Calculate the Black call option price from the log strike,
    volatility and time to maturity
    """
    sig2 = sigma * sigma * t
    sig = np.sqrt(sig2)
    d1 = (-k + 0.5 * sig2) / sig
    d2 = d1 - sig
    return norm.cdf(d1) - np.exp(k) * norm.cdf(d2)


def black_vega(k: np.ndarray, sigma: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Calculate the Black call option vega from the log strike,
    volatility and time to maturity
    """
    sig2 = sigma * sigma * t
    sig = np.sqrt(sig2)
    d1 = (-k + 0.5 * sig2) / sig
    return norm.pdf(d1) * np.sqrt(t)


def implied_black_volatility(
    k: np.ndarray, price: np.ndarray, ttm: np.ndarray, initial_sigma: np.ndarray
) -> np.ndarray:
    """Calculate the implied block volatility from
    1) a vector of log(strikes/forward)
    2) a corresponding vector of call_price/forward
    3) time to maturity and
    4) initial volatility guess
    """
    return newton(
        lambda x: black_call(k, x, ttm) - price,
        initial_sigma,
        fprime=lambda x: black_vega(k, x, ttm),
    )
