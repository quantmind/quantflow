---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.7
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Quantflow

A library for quantitative analysis and pricing.

```{grid}
```{grid-item}
```{image} _static/heston.gif
:alt: Heston volatility surface
:width: 400px
```{grid-item}
```


This documentation is organized into a few major sections.
* [Theory](./theory/overview.md) cover some important concept used throughout the library, for the curious reader
* [Stochastic models](./models/overview.md) cover all the stochastic models supported and their use
* [Applications](./applications/overview.md) show case the real-world use cases
* [Examples](./examples/overview.md) random examples
* [API Reference](./api/index.rst) python API reference

## Installation

To install the library use
```
pip install quantflow
```


## Optional dependencies

Quantflow comes with two optional dependencies:

* `data` for data retrieval, to install it use
  ```
  pip install quantflow[data]
  ```
* `cli` for command line interface, to install it use
  ```
  pip install quantflow[data,cli]
  ```
