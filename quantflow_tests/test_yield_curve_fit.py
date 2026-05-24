"""Tests for yield curve fitting with a realistic steep short-end fixture."""

from __future__ import annotations

import numpy as np

from quantflow.rates.cir import CIRCurve
from quantflow.rates.nelson_siegel import NelsonSiegel
from quantflow.rates.vasicek import VasicekCurve

_TTM = [
    0.0027397260273972603,
    0.019178082191780823,
    0.08333333333333333,
    0.25,
    0.5,
    1,
    2,
    3,
    5,
    7,
    10,
    20,
    30,
]

_RATES = [
    0.01192362562511078,
    0.01999548804041057,
    0.044,
    0.045,
    0.046,
    0.047,
    0.043,
    0.041,
    0.04,
    0.041,
    0.043,
    0.048,
    0.049,
]


def _rmse(curve, ttm, rates) -> float:
    ttm_arr = np.asarray(ttm, dtype=float)
    rates_arr = np.asarray(rates, dtype=float)
    df = np.asarray(curve.discount_factor(ttm_arr), dtype=float)
    fitted = -np.log(df) / ttm_arr
    return float(np.sqrt(np.mean((rates_arr - fitted) ** 2)))


def test_nelson_siegel_fit() -> None:
    ns = NelsonSiegel().calibrator().calibrate(_TTM, _RATES)
    rmse = _rmse(ns, _TTM, _RATES)
    assert rmse < 0.005, f"NS RMSE too large: {rmse:.6f}"


def test_vasicek_fit() -> None:
    v = VasicekCurve().calibrator().calibrate(_TTM, _RATES)
    rmse = _rmse(v, _TTM, _RATES)
    assert rmse < 0.005, f"Vasicek RMSE too large: {rmse:.6f}"


def test_cir_fit() -> None:
    c = CIRCurve().calibrator().calibrate(_TTM, _RATES)
    rmse = _rmse(c, _TTM, _RATES)
    assert rmse < 0.005, f"CIR RMSE too large: {rmse:.6f}"
