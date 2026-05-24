# MCP Server

Quantflow exposes its crypto volatility tools as an [MCP](https://modelcontextprotocol.io) server, allowing AI clients such as Claude to query live Deribit data directly.

The server is hosted at:

```
https://quantflow.quantmind.com/mcp
```

It uses the [Streamable HTTP](https://modelcontextprotocol.io/specification/2025-03-26/basic/transports#streamable-http) transport. No API key is required — all tools use the public Deribit API.

## Claude Code

```bash
claude mcp add --transport http quantflow https://quantflow.quantmind.com/mcp
```

## Claude Desktop

Add to your Claude Desktop config (`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS):

```json
{
  "mcpServers": {
    "quantflow": {
      "type": "streamable-http",
      "url": "https://quantflow.quantmind.com/mcp"
    }
  }
}
```

## Available tools

| Tool | Description |
|---|---|
| `crypto_instruments` | List instruments for a currency on Deribit (spot, future, option) |
| `crypto_historical_volatility` | Historical volatility index from Deribit |
| `crypto_term_structure` | Volatility term structure across maturities |
| `crypto_implied_volatility` | Implied volatility surface (all maturities or a single one) |
| `vol_surface_snapshot` | Fetch a live vol surface and write it as a JSON file |
