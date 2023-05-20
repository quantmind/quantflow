"""Weibull model"""
from functools import lru_cache
from math import floor
from typing import cast

import numpy as np
from numpy import arange, array, exp, power
from numpy import sum as npsum
from scipy.special import gamma, gammaln

from quantflow.utils.types import Vector

from .base import CountingProcess1D


class WeibullProcess(CountingProcess1D):
    """
    Weibull counting process, where the interarrival times are distributed
    according to the weibull distribution.

    This model is defined by the shape parameter 'c' and the intensity
    parameter lambda 'la' (which is similar in meaning to the Poisson model)

    .. attribute:: la

        Arrival rate of the process. Must be positive.

    .. attribute:: c

        The shape parameter of the interarrival times. Must be positive.

    .. attribute:: N

        The maximum number of terms to sum when computing the pdf.
        Larger values make the computation more precise, but slower.

    """

    __slots__ = ("la", "c", "N")

    def __init__(self, la: float, c: float, N: int) -> None:
        """Create a new Weibull model

        :param la: intensity
        :param c: shape parameter
        :param N: optional number of events to stop summations
        """
        self.la = la
        self.c = c
        self.N = N
        assert self.la > 0, "scale must be positive"
        assert self.c > 0, "shape must be positive"
        assert self.N > 0, "max series must be positive"

    def pdf(self, t: float, n: Vector) -> Vector:
        r"""
        Probability density function of the number of events at time t.

        .. math::

            \begin{align*}
            f_{X}\left(n\right) &
            =\sum_{i=n}^{N}\frac{\left(-1\right)^{n+i}\left(\lambda
            t^{c}\right)^{i}\alpha_{i,n}}{\Gamma\left(ci+1\right)}\\
            \alpha_{i,0} & =\frac{\Gamma\left(ci+1\right)}{\Gamma(i+1)}\\
            \alpha_{i,n+1} &
            =\sum_{m=n}^{i-1}\alpha_{m,n}\frac{\Gamma\left(ci-cm+1\right)}{\Gamma\left(i-m+1\right)}
            \end{align*}

        """
        if isinstance(n, np.ndarray):
            return np.array([self.pdf(t, i) for i in n.ravel()]).reshape(n.shape)
        elif isinstance(n, int):
            j = arange(n, self.N)
            return npsum(
                power(-1, j + n)
                * power(self.la * power(t, self.c), j)
                * self.alpha(n)
                / gamma(self.c * j + 1)
            )
        else:
            raise TypeError("n must be an integer or array of integers")

    def cdf(self, t: float, n: Vector) -> Vector:
        r"""
        CDF of the number of events at time t.

        Computed numerically from the :class:`pdf`.
        """
        if isinstance(n, np.ndarray):
            return np.array([self.cdf(t, i) for i in n.ravel()]).reshape(n.shape)
        elif isinstance(n, int):
            return cast(np.ndarray, self.pdf(t, np.arange(0, floor(n) + 1))).sum()
        else:
            raise TypeError("n must be an integer or array of integers")

    @lru_cache(maxsize=None)
    def alpha(self, n: int) -> np.ndarray:
        """Calculate the alpha coefficients for n events"""
        J = arange(n, self.N)
        if n == 0:
            return exp(gammaln(self.c * J + 1) - gammaln(J + 1))
        else:
            c = []
            a = self.alpha(n - 1)
            for j in J:
                m = arange(n - 1, j)
                jm = j - m
                c.append(
                    npsum(
                        a[m - n + 1] * exp(gammaln(self.c * jm + 1) - gammaln(jm + 1))
                    )
                )
            return array(c)
