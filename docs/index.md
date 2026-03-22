# <a href="https://quantmind.github.io/quantflow"><img src="https://raw.githubusercontent.com/quantmind/quantflow/main/notebooks/assets/quantflow-light.svg" width=300 /></a>

[![PyPI version](https://badge.fury.io/py/quantflow.svg)](https://badge.fury.io/py/quantflow)
[![Python versions](https://img.shields.io/pypi/pyversions/quantflow.svg)](https://pypi.org/project/quantflow)
[![Python downloads](https://img.shields.io/pypi/dd/quantflow.svg)](https://pypi.org/project/quantflow)
[![build](https://github.com/quantmind/quantflow/actions/workflows/build.yml/badge.svg)](https://github.com/quantmind/quantflow/actions/workflows/build.yml)
[![codecov](https://codecov.io/gh/quantmind/quantflow/branch/main/graph/badge.svg?token=wkH9lYKOWP)](https://codecov.io/gh/quantmind/quantflow)

Quantitative analysis and pricing tools.

![btcvol](https://github.com/quantmind/quantflow/assets/144320/88ed85d1-c3c5-489c-ac07-21b036593214)

## Installation

```bash
pip install quantflow
```

## Modules

* [quantflow.ai](https://github.com/quantmind/quantflow/tree/main/quantflow/ai) MCP server for AI clients (requires `quantflow[ai,data]`)
* [quantflow.data](https://github.com/quantmind/quantflow/tree/main/quantflow/data) data APIs (requires `quantflow[data]`)
* [quantflow.options](https://github.com/quantmind/quantflow/tree/main/quantflow/options) option pricing and calibration
* [quantflow.sp](https://github.com/quantmind/quantflow/tree/main/quantflow/sp) stochastic process primitives
* [quantflow.ta](https://github.com/quantmind/quantflow/tree/main/quantflow/ta) timeseries analysis tools
* [quantflow.utils](https://github.com/quantmind/quantflow/tree/main/quantflow/utils) utilities and helpers

## Optional dependencies

* `data` — data retrieval: `pip install quantflow[data]`
* `ai` — MCP server for AI clients: `pip install quantflow[ai,data]`

## MCP Server

Quantflow exposes its data tools as an [MCP](https://modelcontextprotocol.io) server, allowing AI clients such as Claude to query market data, crypto volatility surfaces, and economic indicators directly.

Install with the `ai` and `data` extras:

```bash
pip install quantflow[ai,data]
```

### API keys

Store your API keys in `~/.quantflow/.vault`:

```
fmp=your-fmp-key
fred=your-fred-key
```

Or let the AI manage them for you via the `vault_add` tool once connected.

### Claude Code

```bash
claude mcp add quantflow -- uv run qf-mcp
```

### Claude Desktop

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

### Available tools

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
