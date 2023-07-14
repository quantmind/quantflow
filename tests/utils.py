from typing import cast

import numpy as np

from quantflow.sp.base import StochasticProcess1D
from quantflow.utils.marginal import Marginal1D


def characteristic_tests(m: Marginal1D):
    assert m.characteristic(0) == 1
    u = np.linspace(0, 10, 1000)
    # test boundedness
    assert np.all(np.abs(m.characteristic(u)) <= 1)
    # hermitian symmetry
    np.testing.assert_allclose(
        m.characteristic(u), cast(np.ndarray, m.characteristic(-u)).conj()
    )


def analytical_tests(pr: StochasticProcess1D, tol: float = 1e-3):
    t = np.linspace(0.1, 2, 20)
    m = pr.marginal(t)
    np.testing.assert_allclose(m.mean(), m.mean_from_characteristic(), tol)
    np.testing.assert_allclose(m.std(), m.std_from_characteristic(), tol)
    np.testing.assert_allclose(m.variance(), m.variance_from_characteristic(), tol)
