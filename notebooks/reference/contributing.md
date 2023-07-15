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

# Contributing

Welcome to `quantflow` repository! We are excited you are here and want to contribute.

## Getting Started

To get started with quantflow's codebase, take the following steps:

* Clone the repo
```
git clone git@github.com:quantmind/quantflow.git
```
* Install dev dependencies
```
make install-dev
```
* Run tests
```
make tests
```
* Run the jupyter notebook server during development
```
make notebook
```

+++

## Documentation

The documentation is built using [Jupyter book](https://jupyterbook.org/en/stable/intro.html) which supports an *extended version of Jupyter Markdown* called "MyST Markdown".
For information about the MyST syntax and how to use it, see
[the MyST-Parser documentation](https://myst-parser.readthedocs.io/en/latest/using/syntax.html).

To build the documentation website
```
make book
```
Navigate to the `notebook/_build/html` directory to find the `index.html` file you can open on your browser.
