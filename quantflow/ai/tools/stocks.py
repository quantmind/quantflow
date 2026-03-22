"""Stocks tools for the quantflow MCP server."""

from datetime import timedelta

import pandas as pd
from mcp.server.fastmcp import FastMCP

from quantflow.utils.dates import utcnow

from .base import McpTool


def register(mcp: FastMCP, tool: McpTool) -> None:

    @mcp.tool()
    async def stock_indices() -> str:
        """List available stock market indices."""
        async with tool.fmp() as client:
            data = await client.indices()
        return tool.rich(pd.DataFrame(data))

    @mcp.tool()
    async def stock_search(query: str) -> str:
        """Search for stocks by company name or symbol.

        Args:
            query: Company name or ticker symbol to search for
        """
        async with tool.fmp() as client:
            data = await client.search(query)

        df = pd.DataFrame(data, columns=["symbol", "name", "currency", "stockExchange"])
        return f"Search results for '{query}':\n{df.to_string(index=False)}"

    @mcp.tool()
    async def stock_profile(symbol: str) -> str:
        """Get company profile for a stock symbol.

        Args:
            symbol: Stock ticker symbol e.g. AAPL, MSFT
        """
        async with tool.fmp() as client:
            data = await client.profile(symbol)
        if not data:
            return f"No profile found for {symbol}"
        d = dict(data[0])
        description = d.pop("description", "") or ""
        lines = "\n".join(f"{k}: {v}" for k, v in d.items())
        return f"{description}\n\n{lines}".strip()

    @mcp.tool()
    async def stock_prices(symbol: str, frequency: str = "") -> str:
        """Get OHLC price history for a stock.

        Args:
            symbol: Stock ticker symbol e.g. AAPL, MSFT
            frequency: Data frequency - 1min, 5min, 15min, 30min, 1hour, 4hour,
                or empty for daily
        """
        async with tool.fmp() as client:
            df = await client.prices(symbol, frequency=frequency)
        if df.empty:
            return f"No price data for {symbol}"
        df = df[["date", "open", "high", "low", "close", "volume"]].sort_values("date")
        return f"Prices for {symbol}:\n{df.tail(50).to_string(index=False)}"

    @mcp.tool()
    async def sector_performance(period: str = "1d") -> str:
        """Get sector performance and PE ratios.

        Args:
            period: Time period - 1d, 1w, 1m, 3m, 6m, 1y (default: 1d)
        """
        from ccy import period as to_period
        from ccy.tradingcentres import prevbizday
        from fluid.utils.data import compact_dict

        async with tool.fmp() as client:
            to_date = utcnow().date()
            if period != "1d":
                from_date = to_date - timedelta(days=to_period(period).totaldays)
                sp = await client.sector_performance(
                    from_date=prevbizday(from_date, 0),
                    to_date=prevbizday(to_date, 0),
                    summary=True,
                )
            else:
                sp = await client.sector_performance()
            pe = await client.sector_pe(
                params=compact_dict(date=prevbizday(to_date, 0).isoformat())
            )

        from typing import cast

        import pandas as pd

        spd = cast(dict, sp)
        pes = {k["sector"]: round(float(k["pe"]), 3) for k in pe if k["sector"] in spd}
        rows = [
            {"sector": k, "performance": float(v), "pe": pes.get(k, float("nan"))}
            for k, v in spd.items()
        ]
        df = pd.DataFrame(rows).sort_values("performance", ascending=False)
        return f"Sector performance ({period}):\n{df.to_string(index=False)}"
