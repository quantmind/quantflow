# <a href="https://quantmind.github.io/quantflow"><img src="https://raw.githubusercontent.com/quantmind/quantflow/main/docs/assets/logos/quantflow-lockup.svg" width=300 /></a>

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
* `ml` — training the Deep Implied Volatility model: `pip install quantflow[ml]`

## MCP Server

Quantflow exposes its data tools as an [MCP](https://modelcontextprotocol.io) server for AI clients.
See [MCP Server](https://quantflow.quantmind.com/mcp/) for setup and available tools.

## License

Released under the [BSD 3-Clause License](https://github.com/quantmind/quantflow/blob/main/LICENSE).

## Citation

If you use Quantflow in your research, please cite it using the metadata in [CITATION.cff](https://github.com/quantmind/quantflow/blob/main/CITATION.cff).

## License

Released under the [BSD 3-Clause License](https://github.com/quantmind/quantflow/blob/main/LICENSE).
