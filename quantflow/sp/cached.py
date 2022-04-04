from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..utils.types import Vector
from .base import CountingProcess1D, CountingProcess2D


def diff(arr, axis=0) -> np.ndarray:
    if arr.ndim == 1:
        return np.ediff1d(arr, to_begin=arr[0])
    elif arr.ndim == 2 and axis == 0:
        return np.vstack((arr[0], np.diff(arr, axis=0)))
    elif arr.ndim == 2 and axis == 1:
        return np.hstack((arr[:, :1], np.diff(arr, axis=1)))
    raise ValueError("Too many dimensions")


def pdf_from_cdf(cdf: np.ndarray) -> np.ndarray:
    if cdf.ndim == 1:
        return diff(cdf)
    elif cdf.ndim == 2:
        return diff(diff(cdf, axis=1), axis=0)
    raise ValueError("Too many dimensions")


def cdf_from_pdf(pdf: np.ndarray) -> np.ndarray:
    if pdf.ndim == 1:
        return np.cumsum(pdf)
    elif pdf.ndim == 2:
        return np.cumsum(np.cumsum(pdf, axis=0), axis=1)
    raise ValueError("Too many dimensions")


def sum_probabilities(pdf: np.ndarray) -> np.ndarray:
    x = [np.diag(pdf[::-1], i).sum() for i in range(-pdf.shape[0] + 1, 1)]
    return np.array(x)


def diff_probabilities(pdf: np.ndarray) -> np.ndarray:
    x = [np.diag(pdf, -i).sum() for i in range(-pdf.shape[0] + 1, pdf.shape[0])]
    return np.array(x)


class MatrixProcessConstructorMixin:
    def __init__(
        self,
        range: Tuple[int, int],
        pdf_cache: Optional[Dict[float, np.ndarray]] = None,
        cdf_cache: Optional[Dict[float, np.ndarray]] = None,
    ):
        assert pdf_cache or cdf_cache, "Either pdf_cache or cdf_cache need to be passed"
        self.pdf_cache = pdf_cache or {
            t: pdf_from_cdf(cdf) for t, cdf in cdf_cache.items()
        }
        self.cdf_cache = cdf_cache or {
            t: cdf_from_pdf(pdf) for t, pdf in pdf_cache.items()
        }
        self.range = range


class MatrixProcess1D(CountingProcess1D, MatrixProcessConstructorMixin):
    """
    A 1D counting process where the pdf and cdf are just hard coded values stored
    in a numpy array.

    .. attribute:: range

        A tuple containing the range of values `n` can take.

    .. attribute:: pdf_cache

        A dictionary where the keys are times and the values are 1D numpy
        arrays representing pdf of the values in the range :attr:`range`. If
        not passed in the constructor, it's computed from the
        :attr:`cdf_cache`.

    .. attribute:: cdf_cache

        A dictionary where the keys are times and the values are 1D numpy
        arrays representing cdf of the values in the range :attr:`range`. If
        not passed in the constructor, it's computed from the
        :attr:`pdf_cache`.
    """

    @classmethod
    def from_process(
        cls,
        process: CountingProcess1D,
        times: List[float],
        range: Tuple[int, int],
        call_cdf: bool = False,
    ) -> "MatrixProcess1D":
        if not call_cdf:
            return cls(
                range, pdf_cache={t: process.pdf(t, np.arange(*range)) for t in times}
            )
        else:
            return cls(
                range=range,
                pdf_cache={t: process.pdf(t, np.arange(*range)) for t in times},
                cdf_cache={t: process.cdf(t, np.arange(*range)) for t in times},
            )

    def pdf(self, t: float, n: Vector) -> Vector:
        """
        Returns the pdf of the values ``n`` at time ``t``.

        This just checks the values stored in :class:`pdf_cache`. Any
        value of ``t`` or ``n`` that is not stored in it will raise an ``IndexError``.
        """
        # need to assert values are within range otherwise negative values will
        # wrap around the cache and give weird silent errors
        if not np.all(n >= self.range[0]):
            raise IndexError
        return self.pdf_cache[t][n - self.range[0]]

    def cdf(self, t: float, n: Vector) -> Vector:
        """
        Returns the cdf of the values ``n`` at time ``t``.

        This just checks the values stored in :class:`cdf_cache`. Any
        value of ``t`` or ``n`` that is not stored in it will raise an ``IndexError``.
        """
        if not np.all(n >= self.range[0]):
            raise IndexError
        n = np.floor(n).astype(int)
        return self.cdf_cache[t][n - self.range[0]]

    def sample(self, n: int, t: float = 1, steps: int = 0) -> np.array:
        """require implementation"""
        raise NotImplementedError


class MatrixProcess2D(CountingProcess2D, MatrixProcessConstructorMixin):
    """
    A 2D counting process where the pdf and cdf are just hard coded values stored
    in a numpy array.

    Useful as an optimization in cases where the pdf and cdf of a process are
    expensive to compute. A good example is pricing all markets of a football
    event (although it might actually be slower if you're pricing few markets).

    Keep in mind you need to chose the values of :attr:`range` such that the
    probabilities of the sum are close to normalized, otherwise marginals, sum
    processes and difference processes can return inaccurate values.

    .. attribute:: range

        A tuple containing the range of values that each of the two random
        variables of the process can take. This means that ``n`` can take any
        of the ``(range[1] - range[0]) ** 2`` possible combinations of values.

    .. attribute:: pdf_cache

        A dictionary where the keys are times and the values are 1D numpy
        arrays representing pdf of the values in the range :attr:`range`. If
        not passed in the constructor, it's computed from the
        :attr:`cdf_cache`.

    .. attribute:: cdf_cache

        A dictionary where the keys are times and the values are 1D numpy
        arrays representing cdf of the values in the range :attr:`range`. If
        not passed in the constructor, it's computed from the
        :attr:`pdf_cache`.
    """

    @classmethod
    def from_process(
        cls,
        process: CountingProcess2D,
        times: List[float],
        range: Tuple[int, int],
        call_cdf: bool = False,
    ) -> "MatrixProcess2D":
        if hasattr(process, "to_matrix_process"):
            return process.to_matrix_process(times, range, call_cdf)

        n0, n1 = np.mgrid[range[0] : range[1], range[0] : range[1]]
        if not call_cdf:
            return cls(range, pdf_cache={t: process.pdf(t, (n0, n1)) for t in times})
        else:
            return cls(
                pdf_cache={t: process.pdf(t, (n0, n1)) for t in times},
                cdf_cache={t: process.cdf(t, (n0, n1)) for t in times},
                range=range,
            )

    def pdf(self, t: float, n: Tuple[Vector, Vector]) -> Vector:
        """
        Returns the pdf of the values ``n`` at time ``t``.

        This just checks the values stored in :class:`pdf_cache`. Any
        value of ``t`` or ``n`` that is not stored in it will raise an ``IndexError``.
        """
        if not np.all(n[0] >= self.range[0]) and np.all(n[1] >= self.range[0]):
            raise IndexError
        return self.pdf_cache[t][n[0] - self.range[0], n[1] - self.range[0]]

    def cdf(self, t: float, n: Tuple[Vector, Vector]) -> Vector:
        """
        Returns the cdf of the values ``n`` at time ``t``.

        This just checks the values stored in :class:`cdf_cache`. Any
        value of ``t`` or ``n`` that is not stored in it will raise an ``IndexError``.
        """
        if not np.all(n[0] >= self.range[0]) and np.all(n[1] >= self.range[0]):
            raise IndexError
        n0, n1 = np.floor(n).astype(int)
        return self.cdf_cache[t][n0 - self.range[0], n1 - self.range[0]]

    @lru_cache(maxsize=1)
    def marginals(self) -> Tuple[MatrixProcess1D, MatrixProcess1D]:
        """
        Returns the marginal distributions from the process.

        *Warning: this is a method that takes no arguments, a bug in sphynx is
        not documenting its signature properly.*

        These are instances of :class:`MatrixProcess1D` computed from
        :attr:`pdf_cache`.

        This method is cached so it's only computed on the first time it's called.
        """
        marginal_pdf_0 = {t: pdf.sum(1) for t, pdf in self.pdf_cache.items()}
        marginal_pdf_1 = {t: pdf.sum(0) for t, pdf in self.pdf_cache.items()}
        return (
            MatrixProcess1D(self.range, pdf_cache=marginal_pdf_0),
            MatrixProcess1D(self.range, pdf_cache=marginal_pdf_1),
        )

    @lru_cache(maxsize=1)
    def sum_process(self) -> MatrixProcess1D:
        """
        Returns the 1D process that represents the sum of the two random variables of
        the process.

        *Warning: this is a method that takes no arguments, a bug in sphynx is
        not documenting its signature properly.*

        It is an instance of :class:`MatrixProcess1D` computed from
        :attr:`pdf_cache` and will have a range of ``(range[0] * 2, range[1] +
        range[0])``.

        If the values of :attr:`pdf_cache` do not sum to close to 1 this will
        raise a ``ValueError`` as the probabilities of the sum cannot be
        computed accurately.

        This method is cached so it's only computed on the first time it's called.
        """
        if any(pdf_cache.sum() < 0.9 for pdf_cache in self.pdf_cache.values()):
            raise ValueError(
                "the pdf matrix does not have enough precision to compute "
                "the sum process"
            )

        return MatrixProcess1D(
            range=(self.range[0] * 2, self.range[1] + self.range[0]),
            pdf_cache={t: sum_probabilities(pdf) for t, pdf in self.pdf_cache.items()},
        )

    @lru_cache(maxsize=1)
    def difference_process(self) -> MatrixProcess1D:
        """
        Returns the 1D process that represents the difference of the two random
        variables of the process.

        *Warning: this is a method that takes no arguments, a bug in sphynx is
        not documenting its signature properly.*

        It is an instance of :class:`MatrixProcess1D` computed from
        :attr:`pdf_cache` and will have a range of ``(range[0] - range[1] + 1,
        range[1] - range[0])``.

        If the values of :attr:`pdf_cache` do not sum to close to 1 this will
        raise a ``ValueError`` as the probabilities of the difference cannot be
        computed accurately.

        This method is cached so it's only computed on the first time it's called.
        """
        if any(pdf_cache.sum() < 0.9 for pdf_cache in self.pdf_cache.values()):
            raise ValueError(
                "the pdf matrix does not have enough precision to "
                "compute the difference process"
            )

        return MatrixProcess1D(
            range=(self.range[0] - self.range[1] + 1, self.range[1] - self.range[0]),
            pdf_cache={t: diff_probabilities(pdf) for t, pdf in self.pdf_cache.items()},
        )

    def sample(self, n: int, t: float = 1, steps: int = 0) -> np.array:
        """require implementation"""
        raise NotImplementedError
