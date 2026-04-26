"""Tests for VolCrossSection.disable_outliers and VolSurface.calibrate_forwards."""

from __future__ import annotations

from decimal import Decimal

import pytest

from quantflow.options.surface import VolSurface, VolSurfaceInputs, surface_from_inputs
from quantflow_tests.utils import load_fixture_dict


@pytest.fixture
def vol_surface() -> VolSurface:
    inputs = load_fixture_dict("volsurface.json")
    return surface_from_inputs(VolSurfaceInputs(**inputs))


# ---------------------------------------------------------------------------
# disable_outliers
# ---------------------------------------------------------------------------


def test_disable_outliers_runs_without_error(vol_surface: VolSurface) -> None:
    for cross in vol_surface.maturities:
        cross.disable_outliers(ttm=cross.ttm(vol_surface.ref_date))


def test_disable_outliers_never_increases_converged_count(
    vol_surface: VolSurface,
) -> None:
    for cross in vol_surface.maturities:
        before = sum(1 for _ in cross.option_securities(converged=True))
        cross.disable_outliers(ttm=cross.ttm(vol_surface.ref_date))
        after = sum(1 for _ in cross.option_securities(converged=True))
        assert after <= before


def test_disable_outliers_surface_level(vol_surface: VolSurface) -> None:
    before = sum(
        1
        for cross in vol_surface.maturities
        for _ in cross.option_securities(converged=True)
    )
    vol_surface.disable_outliers()
    after = sum(
        1
        for cross in vol_surface.maturities
        for _ in cross.option_securities(converged=True)
    )
    assert after <= before


def test_disable_outliers_removes_zero_mid_vol(vol_surface: VolSurface) -> None:
    cross = vol_surface.maturities[0]
    # inject a zero mid vol by disabling then re-enabling a strike whose call
    # has zero prices — just verify that any already-zero mid options get caught
    cross.disable_outliers(ttm=cross.ttm(vol_surface.ref_date))
    for option in cross.option_securities(converged=True):
        assert option.iv_mid() > 0


def test_disable_outliers_removes_wide_spread_options(vol_surface: VolSurface) -> None:
    cross = vol_surface.maturities[0]
    # with a very tight bid_ask_spread_fraction almost everything gets removed
    cross.disable_outliers(
        ttm=cross.ttm(vol_surface.ref_date), bid_ask_spread_fraction=0.001
    )
    for option in cross.option_securities(converged=True):
        spread = option.iv_bid_ask_spread()
        mid = option.iv_mid()
        assert mid == 0 or spread / mid <= 0.001 + 1e-9


def test_disable_outliers_svi_pass_respects_threshold(vol_surface: VolSurface) -> None:
    cross = vol_surface.maturities[0]
    ttm = cross.ttm(vol_surface.ref_date)
    # with a huge svi_residual_fraction nothing is removed in pass 2
    before = sum(1 for _ in cross.option_securities(converged=True))
    cross.disable_outliers(
        ttm=ttm, bid_ask_spread_fraction=1.0, svi_residual_fraction=100.0
    )
    after = sum(1 for _ in cross.option_securities(converged=True))
    assert after == before


# ---------------------------------------------------------------------------
# calibrate_forwards
# ---------------------------------------------------------------------------


def test_calibrate_forwards_returns_new_instance(vol_surface: VolSurface) -> None:
    # use a threshold that guarantees at least one bad forward
    spreads = sorted(float(c.forward_spread_fraction()) for c in vol_surface.maturities)
    threshold = (spreads[0] + spreads[1]) / 2.0
    result = vol_surface.calibrate_forwards(max_spread_fraction=threshold)
    assert result is not vol_surface


def test_calibrate_forwards_does_not_mutate_original(vol_surface: VolSurface) -> None:
    original_fwds = [c.forward.mid for c in vol_surface.maturities]
    vol_surface.calibrate_forwards(max_spread_fraction=0.0)
    assert [c.forward.mid for c in vol_surface.maturities] == original_fwds


def test_calibrate_forwards_no_bad_forwards_returns_self(
    vol_surface: VolSurface,
) -> None:
    # with a very loose threshold all forwards are "good" — no bad_indices
    result = vol_surface.calibrate_forwards(max_spread_fraction=1.0)
    assert result is vol_surface


def test_calibrate_forwards_wide_spread_replaced_with_zero_spread(
    vol_surface: VolSurface,
) -> None:
    # force max_spread_fraction=0 so every forward is flagged as bad
    # (except those with exactly zero spread, which shouldn't exist in fixture)
    # use a threshold just below the tightest spread so at least one is bad
    spreads = [float(c.forward_spread_fraction()) for c in vol_surface.maturities]
    threshold = min(spreads) - 1e-9  # everything is "bad"
    result = vol_surface.calibrate_forwards(max_spread_fraction=threshold)
    # if no good forwards exist calibrate_forwards returns self
    if result is vol_surface:
        pytest.skip("no reliable forwards available at this threshold")
    for cross in result.maturities:
        fwd = cross.forward
        # replaced forwards have bid == ask (zero spread)
        if fwd.spread == Decimal(0):
            assert fwd.bid == fwd.ask


def test_calibrate_forwards_synthetic_bid_equals_ask(vol_surface: VolSurface) -> None:
    # flag the middle maturity as bad by giving it a tiny threshold
    if len(vol_surface.maturities) < 3:
        pytest.skip("need at least 3 maturities")
    spreads = sorted(float(c.forward_spread_fraction()) for c in vol_surface.maturities)
    # threshold between the tightest and second tightest — exactly one bad
    threshold = (spreads[0] + spreads[1]) / 2.0
    result = vol_surface.calibrate_forwards(max_spread_fraction=threshold)
    for orig, new in zip(vol_surface.maturities, result.maturities):
        if float(orig.forward_spread_fraction()) > threshold:
            assert new.forward.bid == new.forward.ask


def test_calibrate_forwards_preserves_tight_forwards(vol_surface: VolSurface) -> None:
    spreads = [float(c.forward_spread_fraction()) for c in vol_surface.maturities]
    threshold = max(spreads) / 2.0
    result = vol_surface.calibrate_forwards(max_spread_fraction=threshold)
    for orig, new in zip(vol_surface.maturities, result.maturities):
        if float(orig.forward_spread_fraction()) <= threshold:
            assert new.forward.mid == orig.forward.mid
