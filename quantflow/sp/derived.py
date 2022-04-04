from functools import wraps
from math import floor

import numpy as np

from ..utils.types import Vector
from .base import CountingProcess1D, CountingProcess2D


def _clamp_edge_values(func):
    @wraps(func)
    def wrapped(self, t: float, n: Vector) -> Vector:
        if not isinstance(n, np.ndarray):
            if n < self.nmin or self.nmax < n:
                return 0.0

        result = func(self, t, n)
        if isinstance(result, np.ndarray):
            result[(n < self.nmin) & (n > self.nmax)] = 0.0

        return result

    return wrapped


class SumProcess(CountingProcess1D):
    r"""
    Generic process for the sum of stochastic process.

    It's not a very fast implementation and should be used with parsimony.

    .. attribute:: process

        The internal 2D process, the sum of whose rv this process computes.

    .. attribute:: nmin
    .. attribute:: nmax

        The pdf of the sum is computed numerically. The integer values of
        ``nmin`` and ``nmax`` need to be chosen appropriately so that
        ``self.process.pdf(t, (n - nmin, nmin))`` and ``self.process.pdf(t, (n
        - nmax, nmax))`` are close to zero for appropriate values of ``n``.

    """

    def __init__(self, process: CountingProcess2D, nmin: int, nmax: int):
        self.process = process
        self.nmin = nmin
        self.nmax = nmax

    @_clamp_edge_values
    def pdf(self, t: float, n: Vector) -> Vector:
        r"""
        Pdf of the sum process computed numerically using

        .. math::

            f_{X+Y}\left(n\right)=\sum_{i=n_{\min}}^{n_{\max}}f_{X,Y}\left(n-i,i\right)

        Where :math:`f_{X,Y}` is the pdf of :class:`process`.
        """
        return sum(self.process.pdf(t, (n - i, i)) for i in range(self.nmin, self.nmax))

    def cdf(self, t: float, n: Vector) -> Vector:
        r"""
        Cdf of the sum process computed numerically from the :class:`pdf` method.

        .. math::

            F_{X+Y}\left(n\right)=\sum_{i=n_{\min}}^{n}f_{X+Y}\left(i\right)
        """
        if isinstance(n, np.ndarray):
            return np.array([self.cdf(t, i) for i in n.ravel()]).reshape(n.shape)
        return self.pdf(t, np.arange(self.nmin, floor(n) + 1)).sum()


class DifferenceProcess(SumProcess):
    r"""
    Generic process for the difference of stochastic process.

    It's not a very fast implementation and should be used with parsimony.

    .. attribute:: process

        The internal 2D process, the difference of whose rv this process computes.

    .. attribute:: nmin
    .. attribute:: nmax

        The pdf of the sum is computed numerically. The integer values of
        ``nmin`` and ``nmax`` need to be chosen appropriately so that
        ``self.process.pdf(t, (n + nmin, nmin))`` and ``self.process.pdf(t, (n
        + nmax, nmax))`` are close to zero for appropriate values of ``n``.

    """

    @_clamp_edge_values
    def pdf(self, t: float, n: Vector) -> Vector:
        r"""
        Pdf of the difference process computed numerically using

        .. math::

            f_{X-Y}\left(n\right)=\sum_{i=n_{\min}}^{n_{\max}}f_{X,Y}\left(n+i,i\right)

        Where :math:`f_{X,Y}` is the pdf of :class:`process`.
        """

        return sum(self.process.pdf(t, (n + i, i)) for i in range(self.nmin, self.nmax))
