
.PHONY: help
help:
	@echo ================================================================================
	@fgrep -h "##" $(MAKEFILE_LIST) | fgrep -v fgrep | sed -e 's/\\$$//' | sed -e 's/##//'
	@echo ================================================================================

.PHONY: docs
docs:				## build documentation
	@cp docs/index.md readme.md
	@uv run ./dev/build-examples
	@uv run mkdocs build

.PHONY: docs-examples
docs-examples:			## Regenerate docs examples
	@uv run ./dev/build-examples

.PHONY: docs-serve
docs-serve:			## serve documentation
	@uv run mkdocs serve --livereload --watch quantflow --watch docs

.PHONY: install-dev
install-dev:			## Install development dependencies
	@./dev/install

.PHONY: lint
lint:				## Lint and fix
	@uv run ./dev/lint fix

.PHONY: lint-check
lint-check:			## Lint check only
	@uv run ./dev/lint

.PHONY: marimo
marimo:				## Run marimo for editing notebooks
	@./dev/marimo edit

.PHONY: outdated
outdated:			## Show outdated packages
	uv tree --outdated

.PHONY: publish
publish:			## Release to pypi
	@uv build
	@uv publish --token $(PYPI_TOKEN)

.PHONY: tests
tests:				## Unit tests
	@./dev/test

.PHONY: upgrade
upgrade:			## Upgrade dependencies
	uv lock --upgrade
