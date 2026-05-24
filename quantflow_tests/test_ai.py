"""Tests for the quantflow MCP crypto tools."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest
from mcp.server.fastmcp import FastMCP

from app.api.mcp import create_mcp


def text(result: Any) -> str:
    """Extract text from a call_tool result."""
    blocks = result[0] if isinstance(result, tuple) else result
    if blocks and hasattr(blocks[0], "text"):
        return blocks[0].text
    return str(result)


@pytest.fixture
def mcp_server() -> FastMCP:
    return create_mcp()


async def test_crypto_instruments(mcp_server: FastMCP) -> None:
    mock_client = AsyncMock()
    mock_client.get_instruments.return_value = [
        MagicMock(__str__=lambda self: "BTC-SPOT")
    ]
    with patch("app.api.mcp.Deribit") as MockDeribit:
        MockDeribit.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        MockDeribit.return_value.__aexit__ = AsyncMock(return_value=False)
        result = await mcp_server.call_tool("crypto_instruments", {"currency": "BTC"})
    assert "BTC" in text(result)


async def test_crypto_instruments_empty(mcp_server: FastMCP) -> None:
    mock_client = AsyncMock()
    mock_client.get_instruments.return_value = []
    with patch("app.api.mcp.Deribit") as MockDeribit:
        MockDeribit.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        MockDeribit.return_value.__aexit__ = AsyncMock(return_value=False)
        result = await mcp_server.call_tool("crypto_instruments", {"currency": "BTC"})
    assert "No instruments" in text(result)


async def test_crypto_historical_volatility(mcp_server: FastMCP) -> None:
    mock_client = AsyncMock()
    mock_client.get_volatility.return_value = pd.DataFrame(
        [{"date": "2025-01-01", "volatility": 0.8}]
    )
    with patch("app.api.mcp.Deribit") as MockDeribit:
        MockDeribit.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        MockDeribit.return_value.__aexit__ = AsyncMock(return_value=False)
        result = await mcp_server.call_tool(
            "crypto_historical_volatility", {"currency": "BTC"}
        )
    assert "volatility" in text(result)
    assert "2025-01-01" in text(result)


async def test_crypto_historical_volatility_empty(mcp_server: FastMCP) -> None:
    mock_client = AsyncMock()
    mock_client.get_volatility.return_value = pd.DataFrame()
    with patch("app.api.mcp.Deribit") as MockDeribit:
        MockDeribit.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        MockDeribit.return_value.__aexit__ = AsyncMock(return_value=False)
        result = await mcp_server.call_tool(
            "crypto_historical_volatility", {"currency": "BTC"}
        )
    assert "No volatility data" in text(result)


async def test_crypto_term_structure(mcp_server: FastMCP, vol_surface: Any) -> None:
    mock_loader = MagicMock()
    mock_loader.surface.return_value = vol_surface
    mock_client = AsyncMock()
    mock_client.volatility_surface_loader.return_value = mock_loader
    with patch("app.api.mcp.Deribit") as MockDeribit:
        MockDeribit.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        MockDeribit.return_value.__aexit__ = AsyncMock(return_value=False)
        result = await mcp_server.call_tool(
            "crypto_term_structure", {"currency": "ETH"}
        )
    assert "ttm" in text(result)


async def test_crypto_implied_volatility(mcp_server: FastMCP, vol_surface: Any) -> None:
    mock_loader = MagicMock()
    mock_loader.surface.return_value = vol_surface
    mock_client = AsyncMock()
    mock_client.volatility_surface_loader.return_value = mock_loader
    with patch("app.api.mcp.Deribit") as MockDeribit:
        MockDeribit.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        MockDeribit.return_value.__aexit__ = AsyncMock(return_value=False)
        result = await mcp_server.call_tool(
            "crypto_implied_volatility", {"currency": "ETH"}
        )
    assert "iv" in text(result)


async def test_vol_surface_snapshot(
    mcp_server: FastMCP, vol_surface: Any, tmp_path: Path
) -> None:
    mock_loader = MagicMock()
    mock_loader.surface.return_value = vol_surface
    mock_client = AsyncMock()
    mock_client.volatility_surface_loader.return_value = mock_loader
    out = str(tmp_path / "snapshot.json")
    with patch("app.api.mcp.Deribit") as MockDeribit:
        MockDeribit.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        MockDeribit.return_value.__aexit__ = AsyncMock(return_value=False)
        result = await mcp_server.call_tool(
            "vol_surface_snapshot", {"currency": "ETH", "path": out}
        )
    assert "Saved" in text(result)
    assert Path(out).exists()
