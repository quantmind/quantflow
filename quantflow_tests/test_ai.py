"""Unit tests for the quantflow MCP server tools."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest
from mcp.server.fastmcp import FastMCP

from quantflow.ai.tools import charts, crypto, fred, stocks, vault
from quantflow.ai.tools.base import McpTool
from quantflow.data.vault import Vault

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def text(result: Any) -> str:
    """Extract text from a call_tool result (tuple of (blocks, metadata))."""
    blocks = result[0] if isinstance(result, tuple) else result
    if blocks and hasattr(blocks[0], "text"):
        return blocks[0].text
    return str(result)


def raw(result: Any) -> Any:
    """Get the raw return value from call_tool result."""
    if isinstance(result, tuple) and len(result) > 1:
        return result[1].get("result")
    return result


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def vault_path(tmp_path: Path) -> Path:
    return tmp_path / ".vault"


@pytest.fixture
def mcp_tool(vault_path: Path) -> McpTool:
    return McpTool(vault=Vault(vault_path))


@pytest.fixture
def mock_fmp() -> AsyncMock:
    mock = AsyncMock()
    mock.__aenter__ = AsyncMock(return_value=mock)
    mock.__aexit__ = AsyncMock(return_value=False)
    return mock


@pytest.fixture
def mock_fred() -> AsyncMock:
    mock = AsyncMock()
    mock.__aenter__ = AsyncMock(return_value=mock)
    mock.__aexit__ = AsyncMock(return_value=False)
    return mock


@pytest.fixture
def vault_server(mcp_tool: McpTool) -> FastMCP:
    mcp = FastMCP("test-vault")
    vault.register(mcp, mcp_tool)
    return mcp


@pytest.fixture
def stocks_server(mcp_tool: McpTool) -> FastMCP:
    mcp = FastMCP("test-stocks")
    stocks.register(mcp, mcp_tool)
    return mcp


@pytest.fixture
def crypto_server(mcp_tool: McpTool) -> FastMCP:
    mcp = FastMCP("test-crypto")
    crypto.register(mcp, mcp_tool)
    return mcp


@pytest.fixture
def fred_server(mcp_tool: McpTool) -> FastMCP:
    mcp = FastMCP("test-fred")
    fred.register(mcp, mcp_tool)
    return mcp


@pytest.fixture
def charts_server(mcp_tool: McpTool) -> FastMCP:
    mcp = FastMCP("test-charts")
    charts.register(mcp, mcp_tool)
    return mcp


# ---------------------------------------------------------------------------
# Vault tools
# ---------------------------------------------------------------------------


async def test_vault_keys_empty(vault_server: FastMCP) -> None:
    result = await vault_server.call_tool("vault_keys", {})
    assert raw(result) == []


async def test_vault_add(vault_server: FastMCP, mcp_tool: McpTool) -> None:
    result = await vault_server.call_tool("vault_add", {"key": "fmp", "value": "abc"})
    assert "fmp" in text(result)
    assert mcp_tool.vault.get("fmp") == "abc"


async def test_vault_keys_after_add(vault_server: FastMCP) -> None:
    await vault_server.call_tool("vault_add", {"key": "fred", "value": "xyz"})
    result = await vault_server.call_tool("vault_keys", {})
    assert "fred" in raw(result)


async def test_vault_delete_existing(vault_server: FastMCP) -> None:
    await vault_server.call_tool("vault_add", {"key": "fmp", "value": "abc"})
    result = await vault_server.call_tool("vault_delete", {"key": "fmp"})
    assert "deleted" in text(result)


async def test_vault_delete_missing(vault_server: FastMCP) -> None:
    result = await vault_server.call_tool("vault_delete", {"key": "nope"})
    assert "not found" in text(result)


# ---------------------------------------------------------------------------
# Stock tools
# ---------------------------------------------------------------------------


async def test_stock_indices(
    stocks_server: FastMCP, mcp_tool: McpTool, mock_fmp: AsyncMock
) -> None:
    mcp_tool.vault.add("fmp", "test-key")
    mock_fmp.indices.return_value = [
        {"symbol": "^GSPC", "name": "S&P 500"},
        {"symbol": "^IXIC", "name": "NASDAQ Composite"},
    ]
    with patch("quantflow.ai.tools.base.FMP", return_value=mock_fmp):
        result = await stocks_server.call_tool("stock_indices", {})
    assert "^GSPC" in text(result)
    assert "S&P 500" in text(result)


async def test_stock_search(
    stocks_server: FastMCP, mcp_tool: McpTool, mock_fmp: AsyncMock
) -> None:
    mcp_tool.vault.add("fmp", "test-key")
    mock_fmp.search.return_value = [
        {
            "symbol": "AAPL",
            "name": "Apple Inc.",
            "currency": "USD",
            "stockExchange": "NASDAQ",
        },
    ]
    with patch("quantflow.ai.tools.base.FMP", return_value=mock_fmp):
        result = await stocks_server.call_tool("stock_search", {"query": "Apple"})
    assert "AAPL" in text(result)


async def test_stock_profile_found(
    stocks_server: FastMCP, mcp_tool: McpTool, mock_fmp: AsyncMock
) -> None:
    mcp_tool.vault.add("fmp", "test-key")
    mock_fmp.profile.return_value = [
        {
            "symbol": "AAPL",
            "companyName": "Apple Inc.",
            "description": "Tech company.",
            "price": 200.0,
        }
    ]
    with patch("quantflow.ai.tools.base.FMP", return_value=mock_fmp):
        result = await stocks_server.call_tool("stock_profile", {"symbol": "AAPL"})
    assert "Tech company" in text(result)
    assert "AAPL" in text(result)


async def test_stock_profile_not_found(
    stocks_server: FastMCP, mcp_tool: McpTool, mock_fmp: AsyncMock
) -> None:
    mcp_tool.vault.add("fmp", "test-key")
    mock_fmp.profile.return_value = []
    with patch("quantflow.ai.tools.base.FMP", return_value=mock_fmp):
        result = await stocks_server.call_tool("stock_profile", {"symbol": "FAKE"})
    assert "No profile" in text(result)


async def test_stock_prices(
    stocks_server: FastMCP, mcp_tool: McpTool, mock_fmp: AsyncMock
) -> None:
    mcp_tool.vault.add("fmp", "test-key")
    mock_fmp.prices.return_value = pd.DataFrame(
        [
            {
                "date": "2025-01-01",
                "open": 100,
                "high": 110,
                "low": 90,
                "close": 105,
                "volume": 1000,
            }
        ]
    )
    with patch("quantflow.ai.tools.base.FMP", return_value=mock_fmp):
        result = await stocks_server.call_tool("stock_prices", {"symbol": "AAPL"})
    assert "2025-01-01" in text(result)


async def test_stock_prices_empty(
    stocks_server: FastMCP, mcp_tool: McpTool, mock_fmp: AsyncMock
) -> None:
    mcp_tool.vault.add("fmp", "test-key")
    mock_fmp.prices.return_value = pd.DataFrame()
    with patch("quantflow.ai.tools.base.FMP", return_value=mock_fmp):
        result = await stocks_server.call_tool("stock_prices", {"symbol": "FAKE"})
    assert "No price data" in text(result)


# ---------------------------------------------------------------------------
# Crypto tools
# ---------------------------------------------------------------------------


async def test_crypto_instruments(crypto_server: FastMCP) -> None:
    mock_client = AsyncMock()
    mock_client.get_instruments.return_value = [
        MagicMock(__str__=lambda self: "BTC-SPOT")
    ]

    with patch("quantflow.ai.tools.crypto.Deribit") as MockDeribit:
        MockDeribit.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        MockDeribit.return_value.__aexit__ = AsyncMock(return_value=False)

        result = await crypto_server.call_tool(
            "crypto_instruments", {"currency": "BTC"}
        )
        assert "BTC" in text(result)


async def test_crypto_instruments_empty(crypto_server: FastMCP) -> None:
    mock_client = AsyncMock()
    mock_client.get_instruments.return_value = []

    with patch("quantflow.ai.tools.crypto.Deribit") as MockDeribit:
        MockDeribit.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        MockDeribit.return_value.__aexit__ = AsyncMock(return_value=False)

        result = await crypto_server.call_tool(
            "crypto_instruments", {"currency": "BTC"}
        )
        assert "No instruments" in text(result)


async def test_crypto_historical_volatility(crypto_server: FastMCP) -> None:
    mock_client = AsyncMock()
    mock_client.get_volatility.return_value = pd.DataFrame(
        [{"date": "2025-01-01", "volatility": 0.8}]
    )

    with patch("quantflow.ai.tools.crypto.Deribit") as MockDeribit:
        MockDeribit.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        MockDeribit.return_value.__aexit__ = AsyncMock(return_value=False)

        result = await crypto_server.call_tool(
            "crypto_historical_volatility", {"currency": "BTC"}
        )
        assert "volatility" in text(result)
        assert "2025-01-01" in text(result)


async def test_crypto_historical_volatility_empty(crypto_server: FastMCP) -> None:
    mock_client = AsyncMock()
    mock_client.get_volatility.return_value = pd.DataFrame()

    with patch("quantflow.ai.tools.crypto.Deribit") as MockDeribit:
        MockDeribit.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        MockDeribit.return_value.__aexit__ = AsyncMock(return_value=False)

        result = await crypto_server.call_tool(
            "crypto_historical_volatility", {"currency": "BTC"}
        )
        assert "No volatility data" in text(result)


async def test_crypto_term_structure(crypto_server: FastMCP, vol_surface: Any) -> None:
    mock_loader = MagicMock()
    mock_loader.surface.return_value = vol_surface
    mock_client = AsyncMock()
    mock_client.volatility_surface_loader.return_value = mock_loader

    with patch("quantflow.ai.tools.crypto.Deribit") as MockDeribit:
        MockDeribit.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        MockDeribit.return_value.__aexit__ = AsyncMock(return_value=False)

        result = await crypto_server.call_tool(
            "crypto_term_structure", {"currency": "ETH"}
        )
        assert "ttm" in text(result)


async def test_crypto_implied_volatility(
    crypto_server: FastMCP, vol_surface: Any
) -> None:
    mock_loader = MagicMock()
    mock_loader.surface.return_value = vol_surface
    mock_client = AsyncMock()
    mock_client.volatility_surface_loader.return_value = mock_loader

    with patch("quantflow.ai.tools.crypto.Deribit") as MockDeribit:
        MockDeribit.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        MockDeribit.return_value.__aexit__ = AsyncMock(return_value=False)

        result = await crypto_server.call_tool(
            "crypto_implied_volatility", {"currency": "ETH"}
        )
        assert "implied_vol" in text(result)


async def test_crypto_prices(
    crypto_server: FastMCP, mcp_tool: McpTool, mock_fmp: AsyncMock
) -> None:
    mcp_tool.vault.add("fmp", "test-key")
    mock_fmp.prices.return_value = pd.DataFrame(
        [
            {
                "date": "2025-01-01",
                "open": 90000,
                "high": 95000,
                "low": 88000,
                "close": 92000,
                "volume": 500,
            }
        ]
    )
    with patch("quantflow.ai.tools.base.FMP", return_value=mock_fmp):
        result = await crypto_server.call_tool("crypto_prices", {"symbol": "BTCUSD"})
    assert "close" in text(result)
    assert "2025-01-01" in text(result)


# ---------------------------------------------------------------------------
# FRED tools
# ---------------------------------------------------------------------------


async def test_fred_subcategories(
    fred_server: FastMCP, mcp_tool: McpTool, mock_fred: AsyncMock
) -> None:
    mcp_tool.vault.add("fred", "test-key")
    mock_fred.subcategories.return_value = {
        "categories": [{"id": "32991", "name": "Money, Banking, & Finance"}]
    }
    with patch("quantflow.ai.tools.base.Fred", return_value=mock_fred):
        result = await fred_server.call_tool("fred_subcategories", {})
    assert "Money" in text(result)


async def test_fred_subcategories_empty(
    fred_server: FastMCP, mcp_tool: McpTool, mock_fred: AsyncMock
) -> None:
    mcp_tool.vault.add("fred", "test-key")
    mock_fred.subcategories.return_value = {"categories": []}
    with patch("quantflow.ai.tools.base.Fred", return_value=mock_fred):
        result = await fred_server.call_tool("fred_subcategories", {})
    assert "No categories" in text(result)


async def test_fred_series(
    fred_server: FastMCP, mcp_tool: McpTool, mock_fred: AsyncMock
) -> None:
    mcp_tool.vault.add("fred", "test-key")
    mock_fred.series.return_value = {
        "seriess": [
            {
                "id": "GDP",
                "popularity": 90,
                "title": "Gross Domestic Product",
                "frequency": "Quarterly",
                "observation_start": "1947-01-01",
                "observation_end": "2025-01-01",
            }
        ]
    }
    with patch("quantflow.ai.tools.base.Fred", return_value=mock_fred):
        result = await fred_server.call_tool("fred_series", {"category_id": "106"})
    assert "GDP" in text(result)


async def test_fred_series_empty(
    fred_server: FastMCP, mcp_tool: McpTool, mock_fred: AsyncMock
) -> None:
    mcp_tool.vault.add("fred", "test-key")
    mock_fred.series.return_value = {"seriess": []}
    with patch("quantflow.ai.tools.base.Fred", return_value=mock_fred):
        result = await fred_server.call_tool("fred_series", {"category_id": "999"})
    assert "No series" in text(result)


async def test_fred_data(
    fred_server: FastMCP, mcp_tool: McpTool, mock_fred: AsyncMock
) -> None:
    mcp_tool.vault.add("fred", "test-key")
    mock_fred.serie_data.return_value = pd.DataFrame(
        [{"date": "2025-01-01", "value": 27000.0}]
    )
    with patch("quantflow.ai.tools.base.Fred", return_value=mock_fred):
        result = await fred_server.call_tool("fred_data", {"series_id": "GDP"})
    assert "value" in text(result)
    assert "2025-01-01" in text(result)


# ---------------------------------------------------------------------------
# Charts tools
# ---------------------------------------------------------------------------


async def test_ascii_chart(
    charts_server: FastMCP, mcp_tool: McpTool, mock_fmp: AsyncMock
) -> None:
    mcp_tool.vault.add("fmp", "test-key")
    mock_fmp.prices.return_value = pd.DataFrame(
        [
            {
                "date": f"2025-01-{i:02d}",
                "open": 100 + i,
                "high": 110 + i,
                "low": 90 + i,
                "close": 105 + i,
                "volume": 1000,
            }
            for i in range(1, 11)
        ]
    )
    with patch("quantflow.ai.tools.base.FMP", return_value=mock_fmp):
        result = await charts_server.call_tool("ascii_chart", {"symbol": "AAPL"})
    t = text(result)
    assert "AAPL" in t
    assert "High" in t
    assert "Low" in t


async def test_ascii_chart_empty(
    charts_server: FastMCP, mcp_tool: McpTool, mock_fmp: AsyncMock
) -> None:
    mcp_tool.vault.add("fmp", "test-key")
    mock_fmp.prices.return_value = pd.DataFrame()
    with patch("quantflow.ai.tools.base.FMP", return_value=mock_fmp):
        result = await charts_server.call_tool("ascii_chart", {"symbol": "FAKE"})
    assert "No price data" in text(result)
