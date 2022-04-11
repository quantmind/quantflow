import numpy as np
from scipy.stats import norm

from .types import Vector


def black_call(k: Vector, sigma: float, t: float) -> Vector:
    """Black call option price"""
    sig2 = sigma * sigma * t
    sig = np.sqrt(sig2)
    d1 = (-k + 0.5 * sig2) / sig
    d2 = d1 - sig
    return norm.cdf(d1) - np.exp(k) * norm.cdf(d2)
