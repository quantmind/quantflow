from decimal import Decimal
from math import isclose

import numpy as np

from quantflow.sp.copula import FrankCopula, IndependentCopula


def test_independent_copula():
    c = IndependentCopula()
    assert c.tau() == 0
    assert c.rho() == 0
    assert np.allclose(c.jacobian(0.3, 0.4), np.array([0.4, 0.3]))


def test_frank_copula():
    c = FrankCopula(kappa=Decimal("0.3"))
    assert c.kappa == Decimal("0.3")
    assert c.tau() > 0
    assert c.rho() < 0
    assert c.jacobian(0.3, 0.4).shape == (3,)

    c.kappa = 0
    assert c.tau() == 0
    assert c.rho() == 0
    assert np.allclose(c.jacobian(0.3, 0.4), np.array([0.4, 0.3, 0.0]))

    c = FrankCopula()
    assert isclose(c(11.0, 3.0), 33.0)
    assert isclose(c(11.0, 3.0), 33.0)
