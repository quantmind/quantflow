import numpy as np
from scipy.optimize import newton
from scipy.stats import norm

from .types import Vector


def black_call(k: Vector, sigma: float, t: float) -> Vector:
    """Calculate the Black call option price from the log strike,
    volatility and time to maturity
    """
    sig2 = sigma * sigma * t
    sig = np.sqrt(sig2)
    d1 = (-k + 0.5 * sig2) / sig
    d2 = d1 - sig
    return norm.cdf(d1) - np.exp(k) * norm.cdf(d2)


def black_vega(k: Vector, sigma: float, t: float) -> Vector:
    """Calculate the Black call option vega from the log strike,
    volatility and time to maturity
    """
    sig2 = sigma * sigma * t
    sig = np.sqrt(sig2)
    d1 = (-k + 0.5 * sig2) / sig
    return norm.pdf(d1) * np.sqrt(t)


def implied_black_volatility(
    k: Vector, price: Vector, t: float, initial_sigma: float = 0.5
) -> Vector:
    """Calculate the implied block volatility from
    1) a vector of log strikes/spot
    2) a corresponding vector of call prices
    3) time to maturity and
    4) initial volatility guess
    """
    sigma_0 = initial_sigma * np.ones(np.shape(price))
    return newton(
        lambda x: black_call(k, x, t) - price,
        sigma_0,
        fprime=lambda x: black_vega(k, x, t),
    )
