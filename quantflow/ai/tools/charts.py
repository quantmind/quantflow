"""Chart tools for the quantflow MCP server."""

from mcp.server.fastmcp import FastMCP

from .base import McpTool


def register(mcp: FastMCP, tool: McpTool) -> None:

    @mcp.tool()
    async def ascii_chart(symbol: str, frequency: str = "", height: int = 20) -> str:
        """Plot an ASCII candlestick chart for a stock or cryptocurrency.

        Args:
            symbol: Ticker symbol e.g. AAPL, BTCUSD, ETHUSD
            frequency: Data frequency - 1min, 5min, 15min, 30min, 1hour, 4hour,
                or empty for daily
            height: Chart height in terminal rows (default: 20)
        """
        import asciichartpy as ac

        async with tool.fmp() as client:
            df = await client.prices(symbol, frequency=frequency)
        if df.empty:
            return f"No price data for {symbol}"

        df = df.sort_values("date").tail(50)
        prices = df["close"].tolist()
        first_date = df["date"].iloc[0]
        last_date = df["date"].iloc[-1]
        low = min(prices)
        high = max(prices)
        last = prices[-1]

        chart = ac.plot(prices, {"height": height, "format": "{:8,.0f}"})
        return (
            f"{symbol} Close Price ({first_date} → {last_date})\n"
            f"High: {high:,.2f}  Low: {low:,.2f}  Last: {last:,.2f}\n\n"
            f"{chart}"
        )
