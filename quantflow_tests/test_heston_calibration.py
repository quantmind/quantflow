from __future__ import annotations

import numpy as np

from quantflow.options.calibration import (
    DoubleHestonCalibration,
    DoubleHestonJCalibration,
    HestonCalibration,
    HestonJCalibration,
)
from quantflow.options.pricer import OptionPricer
from quantflow.sp.heston import DoubleHeston, DoubleHestonJ, Heston, HestonJ
from quantflow.utils.distributions import DoubleExponential


def test_heston_calibration_get_set_and_penalize(vol_surface) -> None:
    cal = HestonCalibration(
        pricer=OptionPricer(model=Heston.create(vol=0.4, kappa=2.0, sigma=0.6, rho=-0.4)),
        vol_surface=vol_surface,
    )
    params = cal.get_params()
    assert len(params) == 5
    updated = np.array([0.1, 0.1, 1.0, 1.0, -0.2], dtype=float)
    cal.set_params(updated)
    assert np.allclose(cal.get_params(), updated)
    assert cal.penalize() >= 0.0
    bounds = cal.get_bounds()
    assert len(bounds.lb) == 5
    assert len(bounds.ub) == 5


def test_hestonj_calibration_get_set_and_bounds(vol_surface) -> None:
    model = HestonJ.create(
        DoubleExponential,
        vol=0.3,
        kappa=1.5,
        sigma=0.4,
        rho=-0.3,
        jump_fraction=0.2,
        jump_asymmetry=0.1,
    )
    cal = HestonJCalibration(pricer=OptionPricer(model=model), vol_surface=vol_surface)
    params = cal.get_params()
    cal.set_params(params)
    assert np.allclose(cal.get_params(), params)
    bounds = cal.get_bounds()
    assert len(bounds.lb) == len(bounds.ub) == len(params)


def test_double_heston_calibration_param_logic(vol_surface) -> None:
    model = DoubleHeston(
        heston1=Heston.create(vol=0.3, kappa=2.0, sigma=0.4, rho=-0.2),
        heston2=Heston.create(vol=0.25, kappa=1.0, sigma=0.3, rho=-0.4),
    )
    cal = DoubleHestonCalibration(pricer=OptionPricer(model=model), vol_surface=vol_surface)
    params = cal.get_params()
    assert len(params) == 10
    cal.set_params(params)
    assert cal.model.heston1.variance_process.kappa >= cal.model.heston2.variance_process.kappa
    assert cal.penalize() >= 0.0
    assert len(cal.feller_residuals()) == 2
    assert cal.maturity_split() > 0.0


def test_double_hestonj_calibration_get_set_and_bounds(vol_surface) -> None:
    model = DoubleHestonJ(
        heston1=HestonJ.create(
            DoubleExponential,
            vol=0.3,
            kappa=2.5,
            sigma=0.4,
            rho=-0.2,
            jump_fraction=0.2,
            jump_asymmetry=0.1,
        ),
        heston2=Heston.create(vol=0.25, kappa=1.2, sigma=0.3, rho=-0.4),
    )
    cal = DoubleHestonJCalibration(
        pricer=OptionPricer(model=model),
        vol_surface=vol_surface,
    )
    params = cal.get_params()
    cal.set_params(params)
    assert np.allclose(cal.get_params(), params)
    bounds = cal.get_bounds()
    assert len(bounds.lb) == len(bounds.ub) == len(params)
