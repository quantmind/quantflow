from abc import ABC, abstractmethod
from decimal import Decimal
from math import isclose

import numpy as np
from pydantic import BaseModel, Field

from quantflow.utils.functions import debye
from quantflow.utils.numbers import ZERO
from quantflow.utils.types import FloatArray, FloatArrayLike


class Copula(BaseModel, ABC):
    """Bivariate copula probability distribution - Abstract class

    Sklar's theorem states that any multivariate joint distribution can be
    written in terms of univariate marginal-distribution functions and a
    copula which describes the dependence structure between the variables.
    """

    @abstractmethod
    def __call__(self, u: FloatArrayLike, v: FloatArrayLike) -> FloatArrayLike:
        """
        Computes the copula, given the cdf of two 1D processes
        """

    @abstractmethod
    def tau(self) -> float:
        """Kendall's tau - rank correlation parameter"""

    @abstractmethod
    def rho(self) -> float:
        """Spearman's rho - rank correlation parameter"""

    def jacobian(self, u: FloatArrayLike, v: FloatArrayLike) -> FloatArray:
        """
        Jacobian with respected to u, v, and the internal parameters
        parameters of the copula.
        Optional to implement.
        """
        raise NotImplementedError


class IndependentCopula(Copula):
    """
    No-op copula that keep the distributions independent.

    .. math::

        C(u,v) = uv
    """

    def __call__(self, u: FloatArrayLike, v: FloatArrayLike) -> FloatArrayLike:
        return u * v

    def tau(self) -> float:
        return 0.0

    def rho(self) -> float:
        return 0.0

    def jacobian(self, u: FloatArrayLike, v: FloatArrayLike) -> FloatArray:
        return np.array([v, u])


class FrankCopula(Copula):
    r"""
    Frank Copula with parameter :math:`\kappa`

    .. math::

        C(u, v) = -\frac{1}{\kappa}\log\left[1+\frac{\left(\exp\left(-\kappa
        u\right)-1\right)\left(\exp\left(-\kappa
        v\right)-1\right)}{\exp\left(-\kappa\right)-1}\right]
    """

    kappa: Decimal = Field(default=ZERO, description="Frank copula parameter")

    def __call__(self, u: FloatArrayLike, v: FloatArrayLike) -> FloatArrayLike:
        k = float(self.kappa)
        if isclose(k, 0.0):
            return u * v
        eu = np.exp(-k * u)
        ev = np.exp(-k * v)
        e = np.exp(-k)
        return -np.log(1 + (eu - 1) * (ev - 1) / (e - 1)) / k

    def tau(self) -> float:
        """Kendall's tau"""
        k = float(self.kappa)
        if isclose(k, 0.0):
            return 0
        return 1 + 4 * (debye(1, k) - 1) / k

    def rho(self) -> float:
        """Spearman's rho"""
        k = float(self.kappa)
        if isclose(k, 0.0):
            return 0
        return 1 - 12 * (debye(2, -k) - debye(1, -k)) / k

    def jacobian(self, u: FloatArrayLike, v: FloatArrayLike) -> FloatArray:
        k = float(self.kappa)
        if isclose(k, 0.0):
            return np.array([v, u, v * 0])
        eu = np.exp(-k * u)
        ev = np.exp(-k * v)
        e = np.exp(-k)
        x = (eu - 1) * (ev - 1) / (e - 1)
        c = -np.log(1 + x) / k
        xx = x / (1 + x)
        du = eu * (ev - 1) / (e - 1) / (1 + x)
        # du = eu * xx / (eu - 1)
        dv = eu * (eu - 1) / (e - 1) / (1 + x)
        # dv = ev * xx / (ev - 1)
        dk = (u * du + v * dv - e * xx / (e - 1) - c) / k
        return np.array([du, dv, dk])
