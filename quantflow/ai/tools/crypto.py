"""Crypto tools for the quantflow MCP server."""

from mcp.server.fastmcp import FastMCP

from quantflow.data.deribit import Deribit, InstrumentKind

from .base import McpTool


def register(mcp: FastMCP, tool: McpTool) -> None:

    @mcp.tool()
    async def crypto_instruments(currency: str, kind: str = "spot") -> str:
        """List available instruments for a cryptocurrency on Deribit.

        Args:
            currency: Cryptocurrency symbol e.g. BTC, ETH
            kind: Instrument kind - spot, future, option (default: spot)
        """
        async with Deribit() as client:
            data = await client.get_instruments(
                currency=currency, kind=InstrumentKind(kind)
            )
        if not data:
            return f"No instruments found for {currency} ({kind})"
        rows = "\n".join(str(d) for d in data[:20])
        return f"Instruments for {currency} ({kind}):\n{rows}"

    @mcp.tool()
    async def crypto_historical_volatility(currency: str) -> str:
        """Get historical volatility for a cryptocurrency from Deribit.

        Args:
            currency: Cryptocurrency symbol e.g. BTC, ETH
        """
        async with Deribit() as client:
            df = await client.get_volatility(currency)
        if df.empty:
            return f"No volatility data for {currency}"
        return f"Historical volatility for {currency}:\n{df.to_string(index=False)}"

    @mcp.tool()
    async def crypto_term_structure(currency: str) -> str:
        """Get the volatility term structure for a cryptocurrency from Deribit.

        Args:
            currency: Cryptocurrency symbol e.g. BTC, ETH
        """
        from quantflow.options.surface import VolSurface

        async with Deribit() as client:
            loader = await client.volatility_surface_loader(currency)
        vs: VolSurface = loader.surface()
        ts = vs.term_structure().round({"ttm": 4})
        return f"Term structure for {currency}:\n{ts.to_string(index=False)}"

    @mcp.tool()
    async def crypto_implied_volatility(currency: str, maturity_index: int = -1) -> str:
        """Get the implied volatility surface for a cryptocurrency from Deribit.

        Args:
            currency: Cryptocurrency symbol e.g. BTC, ETH
            maturity_index: Maturity index (-1 for all maturities)
        """
        from quantflow.options.surface import VolSurface

        async with Deribit() as client:
            loader = await client.volatility_surface_loader(currency)
        vs: VolSurface = loader.surface()
        index = None if maturity_index < 0 else maturity_index
        vs.bs(index=index)
        df = vs.options_df(index=index)
        df["implied_vol"] = df["implied_vol"].map("{:.2%}".format)
        return f"Implied volatility for {currency}:\n{df.to_string(index=False)}"

    @mcp.tool()
    async def crypto_prices(symbol: str, frequency: str = "") -> str:
        """Get OHLC price history for a cryptocurrency via FMP.

        Args:
            symbol: Cryptocurrency symbol e.g. BTCUSD
            frequency: Data frequency - 1min, 5min, 15min, 30min, 1hour, 4hour,
                or empty for daily
        """
        async with tool.fmp() as client:
            df = await client.prices(symbol, frequency=frequency)
        if df.empty:
            return f"No price data for {symbol}"
        df = df[["date", "open", "high", "low", "close", "volume"]].sort_values("date")
        return f"Prices for {symbol}:\n{df.tail(50).to_string(index=False)}"
