from typing import cast

import numpy as np

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
