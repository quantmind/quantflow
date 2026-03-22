"""FRED tools for the quantflow MCP server."""

from mcp.server.fastmcp import FastMCP

from .base import McpTool


def register(mcp: FastMCP, tool: McpTool) -> None:

    @mcp.tool()
    async def fred_subcategories(category_id: str | None = None) -> str:
        """List FRED categories. Omit category_id to get top-level categories.

        Args:
            category_id: FRED category ID (optional, defaults to root)
        """
        from fluid.utils.data import compact_dict

        async with tool.fred() as client:
            data = await client.subcategories(
                params=compact_dict(category_id=category_id)
            )
        cats = data.get("categories", [])
        if not cats:
            return "No categories found"
        import pandas as pd

        df = pd.DataFrame(cats, columns=["id", "name"])
        return f"FRED categories:\n{df.to_string(index=False)}"

    @mcp.tool()
    async def fred_series(category_id: str) -> str:
        """List data series available in a FRED category.

        Args:
            category_id: FRED category ID
        """
        from fluid.utils.data import compact_dict

        async with tool.fred() as client:
            data = await client.series(params=compact_dict(category_id=category_id))
        series = data.get("seriess", [])
        if not series:
            return f"No series found for category {category_id}"
        import pandas as pd

        df = pd.DataFrame(
            series,
            columns=[
                "id",
                "popularity",
                "title",
                "frequency",
                "observation_start",
                "observation_end",
            ],
        ).sort_values("popularity", ascending=False)
        return f"FRED series for category {category_id}:\n{df.to_string(index=False)}"

    @mcp.tool()
    async def fred_data(
        series_id: str,
        length: int = 100,
        frequency: str = "d",
    ) -> str:
        """Fetch observations for a FRED data series.

        Args:
            series_id: FRED series ID e.g. GDP, UNRATE, DGS10
            length: Number of data points to return (default: 100)
            frequency: Frequency - d, w, bw, m, q, sa, a (default: d for daily)
        """
        async with tool.fred() as client:
            df = await client.serie_data(
                params=dict(
                    series_id=series_id,
                    limit=length,
                    frequency=frequency,
                    sort_order="desc",
                )
            )
        return f"FRED data for {series_id}:\n{df.to_string(index=False)}"
