from math import isclose

import numpy as np

from quantflow.sp.copula import FrankCopula, IndependentCopula


def test_independent_copula():
    c = IndependentCopula()
    assert c.tau() == 0
    assert c.rho() == 0
    assert np.allclose(c.jacobian(0.3, 0.4), np.array([0.4, 0.3]))


def test_frank_copula():
    c = FrankCopula(kappa=0.3)
    assert c.kappa == 0.3
    assert c.tau() > 0
    assert c.rho() < 0
    assert c.jacobian(0.3, 0.4).shape == (3,)

    c.kappa = 0
    assert c.tau() == 0
    assert c.rho() == 0
    assert np.allclose(c.jacobian(0.3, 0.4), np.array([0.4, 0.3, 0.0]))

    c = FrankCopula()
    assert isclose(c(11.0, 3.0), 33.0)


def test_frank_copula_call():
    c = FrankCopula(kappa=2.5)
    # boundaries: C(0, v) = 0, C(u, 0) = 0, C(1, v) = v, C(u, 1) = u
    assert isclose(c(0.0, 0.4), 0.0, abs_tol=1e-12)
    assert isclose(c(0.3, 0.0), 0.0, abs_tol=1e-12)
    assert isclose(c(1.0, 0.4), 0.4)
    assert isclose(c(0.3, 1.0), 0.3)
    # symmetry: C(u, v) = C(v, u)
    assert isclose(c(0.3, 0.7), c(0.7, 0.3))
    # bounded by Frechet: max(u+v-1, 0) <= C(u, v) <= min(u, v)
    assert max(0.3 + 0.4 - 1, 0) <= c(0.3, 0.4) <= min(0.3, 0.4)


def test_frank_copula_jacobian():
    c = FrankCopula(kappa=2.5)
    u, v = 0.3, 0.4
    h = 1e-6
    du, dv, dk = c.jacobian(u, v)
    # finite differences vs analytical partials
    du_num = (c(u + h, v) - c(u - h, v)) / (2 * h)
    dv_num = (c(u, v + h) - c(u, v - h)) / (2 * h)
    assert isclose(float(du), du_num, abs_tol=1e-7)
    assert isclose(float(dv), dv_num, abs_tol=1e-7)
    # dk: numerical via perturbing kappa
    c_up = FrankCopula(kappa=2.5 + 1e-6)
    c_dn = FrankCopula(kappa=2.5 - 1e-6)
    dk_num = (c_up(u, v) - c_dn(u, v)) / 2e-6
    assert isclose(float(dk), dk_num, abs_tol=1e-6)
