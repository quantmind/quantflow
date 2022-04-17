from typing import Any

import numpy as np
from scipy.optimize import minimize

from .types import Vector


def parkinson_estimator(high: Vector, low: Vector) -> Vector:
    """Parkinson volatility estimator"""
    return np.power(np.log(high) - np.log(low), 2) / np.log(2) / 4


def garman_klass_estimator(high: Vector, low: Vector, close: Vector) -> Vector:
    """Garman-Klass volatility estimator"""
    return 0.5 * np.power(np.log(high) - np.log(low), 2) - (
        2 * np.log(2) - 1
    ) * np.power(np.log(close), 2)


def akaike_information(p: np.ndarray, log_like: float) -> float:
    return 2 * p.shape[0] - 2 * log_like


class GarchEstimator:
    """GARCH 1,1 volatility estimator"""

    def __init__(self, y2: np.ndarray, p: np.ndarray):
        self.y2 = y2
        self.p = p

    @property
    def n(self) -> int:
        return self.y2.shape[0]

    @classmethod
    def returns(cls, y: np.ndarray, dt: float = 1.0) -> "GarchEstimator":
        y = np.asarray(y)
        y2 = y * y / dt
        return cls(y2, y2)

    @classmethod
    def pk(cls, y: np.ndarray, pk: np.ndarray, dt: float = 1.0) -> "GarchEstimator":
        y = np.asarray(y)
        y2 = y * y / dt
        return cls(y2, np.asarray(pk) / dt)

    def filter(self, p: np.ndarray) -> np.ndarray:
        w, a, b = p
        sig2 = np.zeros(self.n)
        for i in range(self.n):
            if i == 0:
                sig2[0] = w / (1 - a - b)
            else:
                sig2[i] = w + a * self.p[i - 1] + b * sig2[i - 1]
        return sig2

    def log_like(self, p: np.ndarray) -> float:
        """Log-likelihood of the GARCH model

        The sign is flipped because we want to maximize not minimize
        """
        sig2 = self.filter(p)
        return np.sum(np.log(sig2) + self.y2 / sig2)

    def fit(self, **options: Any) -> dict:
        p = np.array([0.01, 0.05, 0.94])
        r = minimize(self.log_like, p, bounds=((0.001, None),) * 3, options=options)
        if r.success:
            return dict(params=r.x, aic=akaike_information(r.x, -r.fun))
        raise RuntimeError(r)
