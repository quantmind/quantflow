"""Tests for the Yahoo volatility surface loader."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import AsyncIterator
from unittest.mock import AsyncMock, patch

import pytest

from quantflow.data.yahoo import Yahoo

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def spx_chain() -> dict:
    return json.loads(gzip.decompress((FIXTURES / "yahoo_spx.json.gz").read_bytes()))


@pytest.fixture
async def yahoo_cli(spx_chain: dict) -> AsyncIterator[Yahoo]:
    with patch.object(Yahoo, "option_chain", AsyncMock(return_value=spx_chain)):
        async with Yahoo() as cli:
            yield cli


async def test_loader_builds_surface(yahoo_cli: Yahoo, spx_chain: dict) -> None:
    """Surface has one cross section per expiration in the fixture."""
    loader = await yahoo_cli.volatility_surface_loader("^SPX")
    surface = loader.surface()
    assert surface.asset == "^SPX"
    assert len(surface.maturities) == len(spx_chain["options"])
    assert surface.spot.mid > 0


async def test_loader_options_are_non_inverse(yahoo_cli: Yahoo) -> None:
    """Yahoo equity options are quoted in USD, so the loader marks them
    non-inverse."""
    loader = await yahoo_cli.volatility_surface_loader("^SPX")
    surface = loader.surface()
    options = list(surface.option_prices())
    assert options
    assert all(not o.meta.inverse for o in options)


async def test_loader_skips_zero_bid_ask(yahoo_cli: Yahoo, spx_chain: dict) -> None:
    """Contracts with zero or missing bid/ask are dropped."""
    raw_contracts = sum(
        len(e.get("calls", [])) + len(e.get("puts", [])) for e in spx_chain["options"]
    )
    loader = await yahoo_cli.volatility_surface_loader("^SPX")
    surface = loader.surface()
    loaded = sum(
        (1 if s.call else 0) + (1 if s.put else 0)
        for m in surface.maturities
        for s in m.strikes
    )
    assert loaded < raw_contracts
