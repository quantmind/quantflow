# <a href="https://quantmind.github.io/quantflow"><img src="https://raw.githubusercontent.com/quantmind/quantflow/main/notebooks/assets/quantflow-light.svg" width=300 /></a>

[![PyPI version](https://badge.fury.io/py/quantflow.svg)](https://badge.fury.io/py/quantflow)
[![Python versions](https://img.shields.io/pypi/pyversions/quantflow.svg)](https://pypi.org/project/quantflow)
[![Python downloads](https://img.shields.io/pypi/dd/quantflow.svg)](https://pypi.org/project/quantflow)
[![build](https://github.com/quantmind/quantflow/actions/workflows/build.yml/badge.svg)](https://github.com/quantmind/quantflow/actions/workflows/build.yml)
[![codecov](https://codecov.io/gh/quantmind/quantflow/branch/main/graph/badge.svg?token=wkH9lYKOWP)](https://codecov.io/gh/quantmind/quantflow)

Quantitative analysis and pricing tools.

Documentation is available as [quantflow jupyter book](https://quantflow.quantmind.com).

## Installation

```bash
pip install quantflow
```

![btcvol](https://github.com/quantmind/quantflow/assets/144320/88ed85d1-c3c5-489c-ac07-21b036593214)


## Modules

* [quantflow.cli](https://github.com/quantmind/quantflow/tree/main/quantflow/cli) command line client (requires `quantflow[cli,data]`)
* [quantflow.data](https://github.com/quantmind/quantflow/tree/main/quantflow/data) data APIs (requires `quantflow[data]`)
* [quantflow.options](https://github.com/quantmind/quantflow/tree/main/quantflow/options) option pricing and calibration
* [quantflow.sp](https://github.com/quantmind/quantflow/tree/main/quantflow/sp) stochastic process primitives
* [quantflow.ta](https://github.com/quantmind/quantflow/tree/main/quantflow/ta) timeseries analysis tools
* [quantflow.utils](https://github.com/quantmind/quantflow/tree/main/quantflow/utils) utilities and helpers



## Command line tools

The command line tools are available when installing with the extra `cli` and `data` dependencies.

```bash
pip install quantflow[cli,data]
```

It is possible to use the command line tool `qf` to download data and run pricing and calibration scripts.
