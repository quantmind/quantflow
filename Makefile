
.PHONY: help
help:
	@echo ================================================================================
	@fgrep -h "##" $(MAKEFILE_LIST) | fgrep -v fgrep | sed -e 's/\\$$//' | sed -e 's/##//'
	@echo ================================================================================


.PHONY: lint
lint:				## Lint and fix
	@uv run ./dev/lint fix


.PHONY: lint-check
lint-check:			## Lint check only
	@uv run ./dev/lint


.PHONY: install-dev
install-dev:			## Install development dependencies
	@./dev/install

.PHONY: marimo
marimo:				## Run marimo for editing notebooks
	@./dev/marimo edit

.PHONY: docs-png
docs-png:			## Regenerate PNG assets in docs/assets/ (requires Chrome via kaleido)
	@for f in docs/examples_png/*.py; do uv run python $$f; done

.PHONY: docs
docs:				## build documentation
	@cp docs/index.md readme.md
	@uv run ./dev/build-examples
	@uv run mkdocs build

.PHONY: docs-serve
docs-serve:			## serve documentation
	@uv run mkdocs serve --livereload --watch quantflow --watch docs

.PHONY: publish
publish:			## Release to pypi
	@uv build
	@uv publish --token $(PYPI_TOKEN)

.PHONY: tests
tests:				## Unit tests
	@./dev/test


.PHONY: outdated
outdated:			## Show outdated packages
	uv tree --outdated

.PHONY: upgrade
upgrade:			## Upgrade dependencies
	uv lock --upgrade
