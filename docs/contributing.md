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

## Documentation

Documentation is one of the areas where we most need help, and it is a great way to start contributing to the project. You do not need to be an expert in quantitative finance to make a difference.

There are several ways to contribute:

* **Reviewing**: read the existing pages (tutorials, theory pages, API reference) and report anything unclear, incorrect, or out of date. Opening an issue is valuable on its own; a PR with the fix is even better.
* **Adding context**: many pages would benefit from more background, intuition behind the models, or links to the [glossary](glossary.md) and [bibliography](bibliography.md).
* **Adding examples**: new tutorials and example scripts are always welcome. See the existing examples in `docs/examples/` and follow the structure of the pages under `docs/tutorials/`.

To preview the documentation locally run:

```
uv run mkdocs serve
```

Example scripts can be rebuilt with `make docs-examples`.

### Use of AI tools

AI assisted contributions are welcome: using AI tools to draft, review, or improve documentation is perfectly fine.

However, you are the author of your PR. You must understand every change you submit, be able to explain it, and respond to review feedback yourself. Please do not submit unreviewed AI output: if you cannot vouch for the content, do not open the PR.

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](https://github.com/quantmind/quantflow/blob/main/CODE_OF_CONDUCT.md). By participating, you are expected to uphold it.
