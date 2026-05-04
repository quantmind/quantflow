# MCP Server

Quantflow exposes its data tools as an [MCP](https://modelcontextprotocol.io) server, allowing AI clients such as Claude to query market data, crypto volatility surfaces, and economic indicators directly.

Install with the `ai` and `data` extras:

```bash
pip install quantflow[ai,data]
```

## API keys

Store your API keys in `~/.quantflow/.vault`:

```
fmp=your-fmp-key
fred=your-fred-key
```

Or let the AI manage them for you via the `vault_add` tool once connected.

## Claude Code

```bash
claude mcp add quantflow -- uv run qf-mcp
```

## Claude Desktop

Add to your Claude Desktop config (`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS):

```json
{
  "mcpServers": {
    "quantflow": {
      "command": "uv",
      "args": ["run", "qf-mcp"]
    }
  }
}
```

## Available tools

| Tool | Description |
|---|---|
| `vault_keys` | List stored API keys |
| `vault_add` | Add or update an API key |
| `vault_delete` | Delete an API key |
| `stock_indices` | List stock market indices |
| `stock_search` | Search companies by name or symbol |
| `stock_profile` | Get company profile |
| `stock_prices` | Get OHLC price history |
| `sector_performance` | Sector performance and PE ratios |
| `crypto_instruments` | List Deribit instruments |
| `crypto_historical_volatility` | Historical volatility from Deribit |
| `crypto_term_structure` | Volatility term structure |
| `crypto_implied_volatility` | Implied volatility surface |
| `crypto_prices` | Crypto OHLC price history |
| `ascii_chart` | ASCII chart for any stock or crypto symbol |
| `fred_subcategories` | Browse FRED categories |
| `fred_series` | List series in a FRED category |
| `fred_data` | Fetch FRED observations |
