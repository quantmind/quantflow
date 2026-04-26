"""Tests for VolCrossSection.disable_outliers and VolSurface.calibrate_forwards."""

from __future__ import annotations


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
