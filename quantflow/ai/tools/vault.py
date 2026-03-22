"""Vault tools for the quantflow MCP server."""

from mcp.server.fastmcp import FastMCP

from .base import McpTool


def register(mcp: FastMCP, tool: McpTool) -> None:

    @mcp.tool()
    def vault_keys() -> list[str]:
        """List all API keys stored in the vault."""
        return tool.vault.keys()

    @mcp.tool()
    def vault_add(key: str, value: str) -> str:
        """Add or update an API key in the vault.

        Args:
            key: Key name e.g. fmp, fred
            value: API key value
        """
        tool.vault.add(key, value)
        return f"Key '{key}' saved to vault"

    @mcp.tool()
    def vault_delete(key: str) -> str:
        """Delete an API key from the vault.

        Args:
            key: Key name to delete
        """
        if tool.vault.delete(key):
            return f"Key '{key}' deleted from vault"
        return f"Key '{key}' not found in vault"
