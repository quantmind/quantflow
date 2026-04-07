"""Tests for the Deribit volatility surface loader."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, AsyncIterator
from unittest.mock import AsyncMock, patch

import pytest

from quantflow.data.deribit import Deribit

FIXTURES = Path(__file__).parent / "fixtures"


def load_fixture(name: str) -> list[dict]:
    return json.loads((FIXTURES / name).read_text())


@pytest.fixture
def futures() -> list[dict]:
    return load_fixture("deribit_futures.json")


@pytest.fixture
def options() -> list[dict]:
    return load_fixture("deribit_options.json")


@pytest.fixture
def instruments() -> list[dict]:
    return load_fixture("deribit_instruments.json")


@pytest.fixture
async def deribit_cli(
    futures: list[dict], options: list[dict], instruments: list[dict]
) -> AsyncIterator[Deribit]:
    async def get_path(path: str, **kw: Any) -> list[dict]:
        params = kw.get("params", {})
        if path == "public/get_instruments":
            return instruments
        if path == "public/get_book_summary_by_currency":
            return futures if params.get("kind") == "future" else options
        raise ValueError(f"Unexpected path: {path}")

    with patch.object(Deribit, "get_path", AsyncMock(side_effect=get_path)):
        async with Deribit() as cli:
            yield cli


async def test_loader_loads_known_options(deribit_cli: Deribit) -> None:
    """Options present in both book summary and instruments are loaded."""
    loader = await deribit_cli.volatility_surface_loader("btc")
    surface = loader.surface()
    # fixture has 2 strikes (70000 C+P, 75000 C) all on one maturity
    total_strikes = sum(len(m.strikes) for m in surface.maturities)
    assert total_strikes == 2


async def test_loader_skips_option_missing_from_instruments(
    deribit_cli: Deribit, options: list[dict]
) -> None:
    """Options absent from the instruments list are silently skipped."""
    ghost = "BTC-10APR26-67500-P"
    assert any(o["instrument_name"] == ghost for o in options)

    loader = await deribit_cli.volatility_surface_loader("btc")
    surface = loader.surface()
    all_strikes = {
        strike.strike for mat in surface.maturities for strike in mat.strikes
    }
    assert 67500 not in all_strikes


async def test_loader_skips_future_missing_from_instruments(
    deribit_cli: Deribit, futures: list[dict]
) -> None:
    """Futures absent from the instruments list are silently skipped."""
    assert any(f["instrument_name"] == "BTC-GHOST-26" for f in futures)

    loader = await deribit_cli.volatility_surface_loader("btc")
    assert loader is not None
