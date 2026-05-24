"""Tests for GenericVolSurfaceLoader calibration methods.

Covers collect_put_call_parities, calibrate_curves, calibrate_spot, and
implied_forward_term_structure using the SPX fixture (non-inverse, matched
call/put pairs).
"""

from __future__ import annotations

from typing import AsyncIterator
from unittest.mock import AsyncMock, patch

import pytest

from quantflow.data.yahoo import Yahoo
from quantflow.options.surface import VolSurfaceLoader
from quantflow.rates.nelson_siegel import NelsonSiegel
from quantflow_tests.utils import load_fixture_dict


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
    loader.calibrate_curves(asset_curve=NelsonSiegel)
    assert isinstance(loader.asset_curve, NelsonSiegel)


async def test_calibrate_curves_quote_only(loader: VolSurfaceLoader) -> None:
    loader.calibrate_curves(quote_curve=NelsonSiegel)
    assert isinstance(loader.quote_curve, NelsonSiegel)


async def test_calibrate_curves_joint(loader: VolSurfaceLoader) -> None:
    loader.calibrate_curves(asset_curve=NelsonSiegel, quote_curve=NelsonSiegel)
    assert isinstance(loader.asset_curve, NelsonSiegel)
    assert isinstance(loader.quote_curve, NelsonSiegel)


async def test_calibrate_curves_both_none_is_noop(loader: VolSurfaceLoader) -> None:
    original_asset = loader.asset_curve
    original_quote = loader.quote_curve
    loader.calibrate_curves()
    assert loader.asset_curve is original_asset
    assert loader.quote_curve is original_quote


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
