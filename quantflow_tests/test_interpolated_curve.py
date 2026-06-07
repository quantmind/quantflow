"""Tests for the interpolated yield curve (log discount factor interpolation)."""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import numpy as np
import pytest
from pydantic import TypeAdapter, ValidationError

from quantflow.rates import AnyYieldCurve, InterpolatedYieldCurve, InterpolationType

_REF = datetime(2026, 6, 7, tzinfo=timezone.utc)
_TTM = np.array([0.25, 1.0, 2.0, 5.0, 10.0])
_RATES = [Decimal(r) for r in ("0.02", "0.025", "0.03", "0.035", "0.04")]
_RATES_F = np.array([float(r) for r in _RATES])
_YEAR = 365.0 * 86400.0
_ADAPTER: TypeAdapter[AnyYieldCurve] = TypeAdapter(AnyYieldCurve)


def _dates(ttm: np.ndarray = _TTM) -> list[datetime]:
    return [_REF + timedelta(seconds=float(t) * _YEAR) for t in ttm]


def _curve(
    interpolation_type: InterpolationType = InterpolationType.MONOTONE_CUBIC,
    rates: list[Decimal] = _RATES,
) -> InterpolatedYieldCurve:
    return InterpolatedYieldCurve(
        ref_date=_REF,
        anchor_dates=_dates(),
        anchor_rates=rates,
        interpolation_type=interpolation_type,
    )


@pytest.fixture(params=list(InterpolationType))
def interpolation_type(request: pytest.FixtureRequest) -> InterpolationType:
    return request.param


# ---------------------------------------------------------------------------
# Construction and derived state
# ---------------------------------------------------------------------------


def test_default_interpolation_is_linear() -> None:
    curve = InterpolatedYieldCurve(
        ref_date=_REF, anchor_dates=_dates(), anchor_rates=_RATES
    )
    assert curve.interpolation_type is InterpolationType.LINEAR


def test_anchor_rates_coerced_to_decimal() -> None:
    curve = _curve()
    assert all(isinstance(r, Decimal) for r in curve.anchor_rates)


def test_private_attrs_populated() -> None:
    curve = _curve()
    assert curve._ttm == pytest.approx(_TTM)
    assert curve._log_discount == pytest.approx(-_RATES_F * _TTM)


# ---------------------------------------------------------------------------
# Discount factor and rates
# ---------------------------------------------------------------------------


def test_discount_factor_at_zero_is_one(interpolation_type: InterpolationType) -> None:
    curve = _curve(interpolation_type)
    assert float(curve.discount_factor(0.0)) == pytest.approx(1.0)


def test_reprices_nodes_exactly(interpolation_type: InterpolationType) -> None:
    curve = _curve(interpolation_type)
    fitted = curve.continuously_compounded_rate(_TTM)
    assert np.asarray(fitted) == pytest.approx(_RATES_F)


def test_discount_factor_matches_node_rates(
    interpolation_type: InterpolationType,
) -> None:
    curve = _curve(interpolation_type)
    for t, r in zip(_TTM, _RATES_F):
        assert float(curve.discount_factor(t)) == pytest.approx(math.exp(-r * t))


def test_scalar_input_returns_float(interpolation_type: InterpolationType) -> None:
    curve = _curve(interpolation_type)
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
    interpolation_type: InterpolationType,
) -> None:
    curve = _curve(interpolation_type)
    grid = np.linspace(0.0, 15.0, 400)
    df = np.asarray(curve.discount_factor(grid))
    assert np.all(np.diff(df) <= 1e-12)


def test_monotone_cubic_introduces_no_new_extrema() -> None:
    # a non-monotone forward profile that would make a natural cubic overshoot
    rates = [Decimal(r) for r in ("0.05", "0.01", "0.05", "0.01", "0.05")]
    curve = _curve(InterpolationType.MONOTONE_CUBIC, rates=rates)
    g = np.log(np.asarray(curve.discount_factor(np.linspace(0.0, 10.0, 500))))
    # log discount factor must stay within the envelope of the node values
    rates_f = np.array([float(r) for r in rates])
    node_g = np.concatenate([[0.0], -rates_f * _TTM])
    assert g.min() >= node_g.min() - 1e-9
    assert g.max() <= node_g.max() + 1e-9


def test_flat_forward_extrapolation(interpolation_type: InterpolationType) -> None:
    curve = _curve(interpolation_type)
    f_last = float(curve.instantaneous_forward_rate(10.0))
    assert float(curve.instantaneous_forward_rate(15.0)) == pytest.approx(f_last)
    assert float(curve.instantaneous_forward_rate(30.0)) == pytest.approx(f_last)
    # discount factor extends consistently with the flat forward
    expected = float(curve.discount_factor(10.0)) * math.exp(-f_last * 5.0)
    assert float(curve.discount_factor(15.0)) == pytest.approx(expected)


def test_linear_forward_is_piecewise_constant() -> None:
    curve = _curve(InterpolationType.LINEAR)
    # within a single segment (1y..2y) the forward rate is constant
    f1 = float(curve.instantaneous_forward_rate(1.2))
    f2 = float(curve.instantaneous_forward_rate(1.8))
    assert f1 == pytest.approx(f2)


def test_forward_rate_consistent_with_discount_factor(
    interpolation_type: InterpolationType,
) -> None:
    # f(t) = -d/dt ln D(t): check against a central finite difference
    curve = _curve(interpolation_type)
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


def test_json_round_trip_via_union(interpolation_type: InterpolationType) -> None:
    curve = _curve(interpolation_type)
    restored = _ADAPTER.validate_json(_ADAPTER.dump_json(curve))
    assert type(restored) is InterpolatedYieldCurve
    assert restored.interpolation_type is interpolation_type
    assert restored._ttm == pytest.approx(_TTM)
    assert np.asarray(restored.discount_factor(_TTM)) == pytest.approx(
        np.asarray(curve.discount_factor(_TTM))
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_length_mismatch_raises() -> None:
    with pytest.raises((ValidationError, ValueError)):
        InterpolatedYieldCurve(
            ref_date=_REF, anchor_dates=_dates()[:2], anchor_rates=_RATES
        )


def test_non_increasing_dates_raise() -> None:
    dates = _dates()
    dates[1], dates[2] = dates[2], dates[1]
    with pytest.raises((ValidationError, ValueError)):
        InterpolatedYieldCurve(ref_date=_REF, anchor_dates=dates, anchor_rates=_RATES)


def test_anchor_before_ref_date_raises() -> None:
    with pytest.raises((ValidationError, ValueError)):
        InterpolatedYieldCurve(
            ref_date=_REF, anchor_dates=[_REF], anchor_rates=[Decimal("0.02")]
        )


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def test_calibrate_from_ttm_reprices_exactly() -> None:
    target = _RATES_F * 1.1
    curve = _curve().calibrator().calibrate(_TTM, target)
    assert np.asarray(curve.continuously_compounded_rate(_TTM)) == pytest.approx(target)
    assert all(isinstance(r, Decimal) for r in curve.anchor_rates)


def test_set_params_updates_log_discount() -> None:
    curve = _curve()
    calibrator = curve.calibrator()
    new_rates = _RATES_F * 0.5
    calibrator.set_params(new_rates)
    assert curve._log_discount == pytest.approx(-new_rates * _TTM)
    assert np.asarray(curve.continuously_compounded_rate(_TTM)) == pytest.approx(
        new_rates
    )
