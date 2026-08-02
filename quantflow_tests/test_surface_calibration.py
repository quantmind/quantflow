"""Tests for GenericVolSurfaceLoader calibration methods.

Covers collect_put_call_parities, calibrate_curves, calibrate_spot, and
implied_forward_term_structure using the SPX fixture (non-inverse, matched
call/put pairs).
"""

from __future__ import annotations

from datetime import datetime
from typing import AsyncIterator
from unittest.mock import AsyncMock, patch

import pytest

from quantflow.data.deribit import Deribit
from quantflow.data.yahoo import Yahoo
from quantflow.options.surface import VolSurfaceLoader
from quantflow.rates.interpolated import InterpolatedMonotonicCubicCurve
from quantflow.rates.nelson_siegel import NelsonSiegelCurve
from quantflow.rates.no_discount import NoDiscountCurve
from quantflow_tests.utils import load_fixture_dict


@pytest.fixture
def btc_loader() -> VolSurfaceLoader:
    bundle = load_fixture_dict("deribit_btc.json.gz")
    return Deribit.loader_from_book(
        bundle["futures"],
        bundle["options"],
        bundle["instruments"],
        currency="btc",
        ref_date=datetime.fromisoformat(bundle["as_of"]),
    )


@pytest.fixture
def spx_chain() -> dict:
    return load_fixture_dict("yahoo_spx.json.gz")


@pytest.fixture
async def yahoo_cli(spx_chain: dict) -> AsyncIterator[Yahoo]:
    with patch.object(Yahoo, "option_chain", AsyncMock(return_value=spx_chain)):
        async with Yahoo() as cli:
            yield cli


@pytest.fixture
async def loader(yahoo_cli: Yahoo) -> VolSurfaceLoader:
    return await yahoo_cli.volatility_surface_loader("^SPX")


async def test_collect_put_call_parities_shapes(loader: VolSurfaceLoader) -> None:
    ttm, cp, strikes = loader.collect_put_call_parities()
    assert ttm.shape == cp.shape == strikes.shape
    assert len(ttm) > 0


async def test_collect_put_call_parities_ttm_positive(loader: VolSurfaceLoader) -> None:
    ttm, _cp, _strikes = loader.collect_put_call_parities()
    assert (ttm > 0).all()


async def test_collect_put_call_parities_strikes_positive(
    loader: VolSurfaceLoader,
) -> None:
    _ttm, _cp, strikes = loader.collect_put_call_parities()
    assert (strikes > 0).all()


async def test_calibrate_spot_returns_positive_value(loader: VolSurfaceLoader) -> None:
    implied = loader.calibrate_spot()
    assert implied is not None
    assert float(implied) > 0


async def test_calibrate_spot_close_to_original(loader: VolSurfaceLoader) -> None:
    original = loader.spot_price()
    implied = loader.calibrate_spot()
    assert implied is not None
    assert float(implied) == pytest.approx(float(original), rel=0.05)


async def test_calibrate_spot_no_short_maturities_returns_none(
    loader: VolSurfaceLoader,
) -> None:
    implied = loader.calibrate_spot(max_ttm=0.0)
    assert implied is None


async def test_calibrate_curves_asset_only(loader: VolSurfaceLoader) -> None:
    loader.calibrate_curves(asset_curve=NelsonSiegelCurve)
    assert isinstance(loader.asset_curve, NelsonSiegelCurve)


async def test_calibrate_curves_quote_only(loader: VolSurfaceLoader) -> None:
    loader.calibrate_curves(quote_curve=NelsonSiegelCurve)
    assert isinstance(loader.quote_curve, NelsonSiegelCurve)


async def test_calibrate_curves_joint(loader: VolSurfaceLoader) -> None:
    loader.calibrate_curves(
        asset_curve=NelsonSiegelCurve, quote_curve=NelsonSiegelCurve
    )
    assert isinstance(loader.asset_curve, NelsonSiegelCurve)
    assert isinstance(loader.quote_curve, NelsonSiegelCurve)


async def test_calibrate_curves_both_none_is_noop(loader: VolSurfaceLoader) -> None:
    original_asset = loader.asset_curve
    original_quote = loader.quote_curve
    loader.calibrate_curves()
    assert loader.asset_curve is original_asset
    assert loader.quote_curve is original_quote


async def test_calibrate_curves_no_discount_asset(loader: VolSurfaceLoader) -> None:
    # a curve without a calibrator is treated as fixed
    loader.calibrate_curves(quote_curve=NelsonSiegelCurve, asset_curve=NoDiscountCurve)
    assert isinstance(loader.quote_curve, NelsonSiegelCurve)
    assert isinstance(loader.asset_curve, NoDiscountCurve)


async def test_calibrate_curves_both_no_discount(loader: VolSurfaceLoader) -> None:
    loader.calibrate_curves(quote_curve=NoDiscountCurve, asset_curve=NoDiscountCurve)
    assert isinstance(loader.quote_curve, NoDiscountCurve)
    assert isinstance(loader.asset_curve, NoDiscountCurve)


async def test_calibrate_curves_joint_interpolated(loader: VolSurfaceLoader) -> None:
    # interpolated curves place one node per calibrated maturity
    loader.calibrate_curves(
        quote_curve=InterpolatedMonotonicCubicCurve,
        asset_curve=InterpolatedMonotonicCubicCurve,
    )
    assert isinstance(loader.quote_curve, InterpolatedMonotonicCubicCurve)
    assert isinstance(loader.asset_curve, InterpolatedMonotonicCubicCurve)
    nodes = len(loader.quote_curve.anchor_dates)
    assert 0 < nodes <= len(loader.maturities)
    # the calibrated curves evaluate cleanly
    assert float(loader.quote_curve.discount_factor(1.0)) > 0
    assert float(loader.asset_curve.discount_factor(1.0)) > 0


async def test_calibrate_curves_interpolated_quote_only(
    loader: VolSurfaceLoader,
) -> None:
    # analytic path, nodes averaged over duplicated times to maturity
    loader.calibrate_curves(quote_curve=InterpolatedMonotonicCubicCurve)
    assert isinstance(loader.quote_curve, InterpolatedMonotonicCubicCurve)
    assert float(loader.quote_curve.discount_factor(1.0)) > 0


# ---------------------------------------------------------------------------
# calibrate_forwards
# ---------------------------------------------------------------------------


def test_calibrate_forwards_close_to_futures(btc_loader: VolSurfaceLoader) -> None:
    results = btc_loader.calibrate_forwards()
    assert len(results) == len(btc_loader.maturities)
    for maturity, ttm, forward in results:
        assert ttm > 0
        section = btc_loader.maturities[maturity]
        assert section.parity_forward == forward
        future = float(section.forward.mid)  # type: ignore[union-attr]
        # the parity forward reproduces the listed future within 10 bp
        assert abs(float(forward) - future) / future < 0.001


async def test_calibrate_forwards_sets_parity_forward(
    loader: VolSurfaceLoader,
) -> None:
    results = loader.calibrate_forwards()
    assert results
    for maturity, _ttm, forward in results:
        assert float(forward) > 0
        assert loader.maturities[maturity].parity_forward == forward


def test_surface_prices_off_parity_forwards(btc_loader: VolSurfaceLoader) -> None:
    btc_loader.calibrate_forwards()
    surface = btc_loader.surface()
    for cross in surface.maturities:
        assert cross.parity_forward is not None
        assert cross.pricing_forward == cross.parity_forward
    frame = surface.term_structure()
    assert list(frame["implied_forward"]) == [
        cross.parity_forward for cross in surface.maturities
    ]


def test_calibrate_curves_reproduces_parity_forwards(
    btc_loader: VolSurfaceLoader,
) -> None:
    # with an interpolated asset curve the curve implied forward reproduces
    # the parity forward exactly at the nodes
    btc_loader.tick_size_forwards = None
    btc_loader.calibrate_curves(
        quote_curve=NelsonSiegelCurve,
        asset_curve=InterpolatedMonotonicCubicCurve,
        min_ttm=0,
    )
    for maturity, section in btc_loader.maturities.items():
        assert section.parity_forward is not None
        curve_forward = float(btc_loader.forward(maturity))
        assert curve_forward == pytest.approx(float(section.parity_forward), rel=1e-6)


async def test_implied_forward_term_structure_returns_entries(
    loader: VolSurfaceLoader,
) -> None:
    ts = loader.implied_forward_term_structure()
    assert len(ts) > 0


async def test_implied_forward_term_structure_ttm_positive(
    loader: VolSurfaceLoader,
) -> None:
    ts = loader.implied_forward_term_structure()
    for _mat, ttm, _fwd in ts:
        assert ttm > 0


async def test_implied_forward_term_structure_forward_positive(
    loader: VolSurfaceLoader,
) -> None:
    ts = loader.implied_forward_term_structure()
    for _mat, _ttm, fwd in ts:
        assert fwd > 0


async def test_implied_forward_term_structure_increases_with_ttm(
    loader: VolSurfaceLoader,
) -> None:
    ts = loader.implied_forward_term_structure()
    ttms = [ttm for _mat, ttm, _fwd in ts]
    assert ttms == sorted(ttms)
