"""Tests for the interpolated yield curve (zero rate interpolation)."""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import cast

import numpy as np
import pytest
from pydantic import TypeAdapter, ValidationError

from quantflow.rates import (
    AnyYieldCurve,
    InterpolatedLinearCurve,
    InterpolatedMonotonicCubicCurve,
    InterpolatedYieldCurve,
)

_REF = datetime(2026, 6, 7, tzinfo=timezone.utc)
_TTM = np.array([0.25, 1.0, 2.0, 5.0, 10.0])
_RATES = [Decimal(r) for r in ("0.02", "0.025", "0.03", "0.035", "0.04")]
_RATES_F = np.array([float(r) for r in _RATES])
_YEAR = 365.0 * 86400.0
_CURVE_CLASSES = (InterpolatedLinearCurve, InterpolatedMonotonicCubicCurve)
_ADAPTER: TypeAdapter[AnyYieldCurve] = TypeAdapter(AnyYieldCurve)


def _dates(ttm: np.ndarray = _TTM) -> list[datetime]:
    return [_REF + timedelta(seconds=float(t) * _YEAR) for t in ttm]


def _curve(
    curve_class: type[InterpolatedYieldCurve] = InterpolatedMonotonicCubicCurve,
    rates: list[Decimal] = _RATES,
) -> InterpolatedYieldCurve:
    return curve_class(
        ref_date=_REF,
        anchor_dates=_dates(),
        anchor_rates=rates,
    )


@pytest.fixture(params=_CURVE_CLASSES)
def curve_class(
    request: pytest.FixtureRequest,
) -> type[InterpolatedYieldCurve]:
    return request.param


# ---------------------------------------------------------------------------
# Construction and derived state
# ---------------------------------------------------------------------------


def test_curve_types() -> None:
    assert InterpolatedLinearCurve().curve_type == "interpolated_linear_curve"
    assert (
        InterpolatedMonotonicCubicCurve().curve_type
        == "interpolated_monotonic_cubic_curve"
    )


def test_base_class_is_abstract() -> None:
    with pytest.raises(TypeError):
        InterpolatedYieldCurve(  # type: ignore[abstract]
            ref_date=_REF, anchor_dates=_dates(), anchor_rates=_RATES
        )


def test_empty_curve_is_uncalibrated() -> None:
    curve = InterpolatedLinearCurve()
    assert curve._ttm.size == 0
    assert curve._rates.size == 0


def test_anchor_rates_coerced_to_decimal() -> None:
    curve = _curve()
    assert all(isinstance(r, Decimal) for r in curve.anchor_rates)


def test_private_attrs_populated() -> None:
    curve = _curve()
    assert curve._ttm == pytest.approx(_TTM)
    assert curve._rates == pytest.approx(_RATES_F)


# ---------------------------------------------------------------------------
# Discount factor and rates
# ---------------------------------------------------------------------------


def test_discount_factor_at_zero_is_one(
    curve_class: type[InterpolatedYieldCurve],
) -> None:
    curve = _curve(curve_class)
    assert float(curve.discount_factor(0.0)) == pytest.approx(1.0)


def test_reprices_nodes_exactly(curve_class: type[InterpolatedYieldCurve]) -> None:
    curve = _curve(curve_class)
    fitted = curve.continuously_compounded_rate(_TTM)
    assert np.asarray(fitted) == pytest.approx(_RATES_F)


def test_discount_factor_matches_node_rates(
    curve_class: type[InterpolatedYieldCurve],
) -> None:
    curve = _curve(curve_class)
    for t, r in zip(_TTM, _RATES_F):
        assert float(curve.discount_factor(t)) == pytest.approx(math.exp(-r * t))


def test_scalar_input_returns_float(
    curve_class: type[InterpolatedYieldCurve],
) -> None:
    curve = _curve(curve_class)
    assert isinstance(curve.discount_factor(1.0), float)
    assert isinstance(curve.instantaneous_forward_rate(1.0), float)


def test_inherited_rates_method_not_shadowed() -> None:
    # the field is anchor_rates, so YieldCurve.rates() still works
    curve = _curve()
    semi = curve.rates(_TTM, frequency=2)
    cont = curve.continuously_compounded_rate(_TTM)
    assert np.all(np.asarray(semi) > 0)
    # discrete compounding sits just below continuous for positive rates
    assert np.all(np.asarray(semi) > np.asarray(cont) - 1e-9)


# ---------------------------------------------------------------------------
# Monotonicity and extrapolation
# ---------------------------------------------------------------------------


def test_discount_factor_monotone_decreasing(
    curve_class: type[InterpolatedYieldCurve],
) -> None:
    curve = _curve(curve_class)
    grid = np.linspace(0.0, 15.0, 400)
    df = np.asarray(curve.discount_factor(grid))
    assert np.all(np.diff(df) <= 1e-12)


def test_monotone_cubic_introduces_no_new_extrema() -> None:
    # a non-monotone rate profile that would make a natural cubic overshoot
    rates = [Decimal(r) for r in ("0.05", "0.01", "0.05", "0.01", "0.05")]
    curve = _curve(InterpolatedMonotonicCubicCurve, rates=rates)
    # the interpolated zero rate must stay within the envelope of the node values
    r = np.asarray(
        curve.continuously_compounded_rate(np.linspace(_TTM[0], _TTM[-1], 500))
    )
    rates_f = np.array([float(x) for x in rates])
    assert r.min() >= rates_f.min() - 1e-9
    assert r.max() <= rates_f.max() + 1e-9


def test_flat_rate_extrapolation(
    curve_class: type[InterpolatedYieldCurve],
) -> None:
    curve = _curve(curve_class)
    # beyond the last node the zero rate is held flat at the last node value
    r_last = _RATES_F[-1]
    assert float(curve.continuously_compounded_rate(15.0)) == pytest.approx(r_last)
    assert float(curve.continuously_compounded_rate(30.0)) == pytest.approx(r_last)
    # so is the forward rate, and the discount factor follows the flat rate
    assert float(curve.instantaneous_forward_rate(15.0)) == pytest.approx(r_last)
    assert float(curve.discount_factor(15.0)) == pytest.approx(math.exp(-r_last * 15.0))


def test_linear_rate_is_piecewise_linear() -> None:
    curve = _curve(InterpolatedLinearCurve)
    # within a single segment (1y..2y) the zero rate is linear in tau
    r1 = float(curve.continuously_compounded_rate(1.0))
    r2 = float(curve.continuously_compounded_rate(2.0))
    rmid = float(curve.continuously_compounded_rate(1.5))
    assert rmid == pytest.approx(0.5 * (r1 + r2))


def test_forward_rate_consistent_with_discount_factor(
    curve_class: type[InterpolatedYieldCurve],
) -> None:
    # f(t) = -d/dt ln D(t): check against a central finite difference
    curve = _curve(curve_class)
    t, h = 3.0, 1e-5
    g_plus = math.log(float(curve.discount_factor(t + h)))
    g_minus = math.log(float(curve.discount_factor(t - h)))
    numerical = -(g_plus - g_minus) / (2 * h)
    assert float(curve.instantaneous_forward_rate(t)) == pytest.approx(
        numerical, rel=1e-4
    )


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def test_json_round_trip_via_union(
    curve_class: type[InterpolatedYieldCurve],
) -> None:
    curve = _curve(curve_class)
    restored = _ADAPTER.validate_json(_ADAPTER.dump_json(cast(AnyYieldCurve, curve)))
    assert type(restored) is curve_class
    assert restored._ttm == pytest.approx(_TTM)
    assert np.asarray(restored.discount_factor(_TTM)) == pytest.approx(
        np.asarray(curve.discount_factor(_TTM))
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_length_mismatch_raises() -> None:
    with pytest.raises((ValidationError, ValueError)):
        InterpolatedLinearCurve(
            ref_date=_REF, anchor_dates=_dates()[:2], anchor_rates=_RATES
        )


def test_non_increasing_dates_raise() -> None:
    dates = _dates()
    dates[1], dates[2] = dates[2], dates[1]
    with pytest.raises((ValidationError, ValueError)):
        InterpolatedLinearCurve(ref_date=_REF, anchor_dates=dates, anchor_rates=_RATES)


def test_anchor_before_ref_date_raises() -> None:
    with pytest.raises((ValidationError, ValueError)):
        InterpolatedLinearCurve(
            ref_date=_REF, anchor_dates=[_REF], anchor_rates=[Decimal("0.02")]
        )


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def test_calibrate_from_ttm_reprices_exactly(
    curve_class: type[InterpolatedYieldCurve],
) -> None:
    target = _RATES_F * 1.1
    curve = curve_class().calibrator().calibrate(_TTM, target)
    assert type(curve) is curve_class
    assert np.asarray(curve.continuously_compounded_rate(_TTM)) == pytest.approx(target)
    assert all(isinstance(r, Decimal) for r in curve.anchor_rates)


def test_set_params_updates_rates() -> None:
    curve = _curve()
    calibrator = curve.calibrator()
    new_rates = _RATES_F * 0.5
    calibrator.set_params(new_rates)
    assert curve._rates == pytest.approx(new_rates)
    assert np.asarray(curve.continuously_compounded_rate(_TTM)) == pytest.approx(
        new_rates
    )
