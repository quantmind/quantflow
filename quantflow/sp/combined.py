"""Bivariate Point process with Frank Copula."""
from typing import Sequence, Tuple

import numpy as np

from ..utils.param import Parameters
from ..utils.text import shift
from ..utils.types import Vector
from .base import CountingProcess1D, CountingProcess2D
from .cached import MatrixProcess2D
from .copula import Copula
from .derived import DifferenceProcess, SumProcess


class CombinedStochasticProcess(CountingProcess2D):
    """
    Combines two 1D into a 2D process with a copula.

    This is such that the CDF of the combined distribution F is defined as

    :math:`F(t, (i, j)) = C(G(t, i), H(t, j))`

    Where G and H are the marginal CDFs.

    .. attribute:: left

        1D Process representing one of the marginal processe.

    .. attribute:: right

        1D Process representing the other marginal processe.

    .. attribute:: copula

        A callable that takes to float arguments and return a values between 0 and 1n
    """

    def __init__(
        self, left: CountingProcess1D, right: CountingProcess1D, copula: Copula
    ) -> None:
        self.left = left
        self.right = right
        self.copula = copula

    @property
    def parameters(self) -> Parameters:
        p = Parameters()
        p.extend(self.left.parameters, prefix="left_")
        p.extend(self.right.parameters, prefix="right_")
        p.extend(self.copula.parameters)
        return p

    def cdf(self, t: float, n: Tuple[Vector, Vector]) -> Vector:
        """
        Returns the CDF of the combined process :math:`F(t, (i, j)) = C(G(t,
        i), H(t, j))` where ``C`` is the copula, and ``G`` and ``H`` are the
        CDF of the marginal distributions.
        """
        u = self.left.cdf(t, n[0])
        v = self.right.cdf(t, n[1])
        return self.copula(u, v)

    def cdf_square(self, t: float, n: int) -> np.array:
        support = np.arange(n)
        u = np.tile(self.left.cdf(t, support), (n, 1))
        v = np.tile(self.right.cdf(t, support).reshape((n, 1)), (1, n))
        return self.copula(u, v)

    def marginals(self) -> Tuple[CountingProcess1D, CountingProcess1D]:
        """
        Returns the marginal process of each of the two random variables of the
        process.

        Because the copula does not change the marginal distributions, this just returns
        the tuple ``(self.left, self.right)``.
        """
        return self.left, self.right

    def sum_process(self) -> SumProcess:
        """
        If this is not overloaded it returns a :class:`SumProcess` in the
        range ``(0, 11)``
        """

        return SumProcess(self, 0, 11)

    def difference_process(self) -> DifferenceProcess:
        """
        If this is not overloaded it returns a :class:`DifferenceProcess` in
        the range ``(-10, 11)``
        """

        return DifferenceProcess(self, -10, 11)

    def cdf_jacobian(self, t: float, n: Tuple[Vector, Vector]) -> np.ndarray:
        """
        If ``left`` has `n` parameters, right `m`, and the copula `d` then
        the resulting jacobian will be a numpy away with the zeroeth axis
        of size ``n + m + d``.
        """
        # NOTE: The shape of this needs fixing, maybe the signature of
        # the jacobians should have ndarray as return
        u = self.left.cdf(t, n[0])
        v = self.right.cdf(t, n[1])
        j = self.copula.jacobian(u, v)
        jac = (
            j[0] * self.left.cdf_jacobian(t, n[0]),
            j[1] * self.right.cdf_jacobian(t, n[1]),
            j[2:],
        )
        return np.concatenate(jac)

    def to_matrix_process(
        self, times: Sequence[float], range: Tuple[int, int], call_cdf: bool = False
    ) -> MatrixProcess2D:
        """
        Optimized creation of a MatrixProcess2D based on the combined process.

        The constructor ``MatrixProcess2D.from_process`` also calls this function.
        """

        def repeat_till_square(arr):
            return np.repeat(arr[np.newaxis], arr.size, axis=0)

        n = np.arange(*range)
        # these should be generators but pytest-cov does not recognize them due
        # to a bug https://github.com/nedbat/coveragepy/issues/515
        if call_cdf:
            cdf_left = [repeat_till_square(self.left.cdf(t, n)).T for t in times]
            cdf_right = [repeat_till_square(self.right.cdf(t, n)) for t in times]
        else:
            cdf_left = [
                repeat_till_square(self.left.pdf(t, n).cumsum()).T for t in times
            ]
            cdf_right = [
                repeat_till_square(self.right.pdf(t, n).cumsum()) for t in times
            ]
        cdf_cache = {
            t: self.copula(lcdf, rcdf)
            for t, lcdf, rcdf in zip(times, cdf_left, cdf_right)
        }
        return MatrixProcess2D(range=range, cdf_cache=cdf_cache)

    def __repr__(self):
        left = f"left={self.left}"
        right = f"right={self.right}"
        copula = f"copula={self.copula}"
        return (
            f"{type(self).__name__}(\n"
            f"{shift(left, 4)},\n"
            f"{shift(right, 4)},\n"
            f"{shift(copula, 4)},\n"
            ")"
        )
