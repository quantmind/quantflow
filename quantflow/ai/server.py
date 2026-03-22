"""Quantflow MCP server."""

from mcp.server.fastmcp import FastMCP

from quantflow.ai.tools import charts, crypto, fred, stocks, vault

from .tools.base import McpTool


def create_server() -> FastMCP:
    mcp = FastMCP("quantflow")
    tool = McpTool()
    vault.register(mcp, tool)
    crypto.register(mcp, tool)
    stocks.register(mcp, tool)
    fred.register(mcp, tool)
    charts.register(mcp, tool)
    return mcp


def main() -> None:
    server = create_server()
    server.run()


if __name__ == "__main__":
    main()
