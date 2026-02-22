
.PHONY: help
help:
	@echo ================================================================================
	@fgrep -h "##" $(MAKEFILE_LIST) | fgrep -v fgrep | sed -e 's/\\$$//' | sed -e 's/##//'
	@echo ================================================================================


.PHONY: lint
lint:				## Lint and fix
	@poetry run ./dev/lint fix


.PHONY: lint-check
lint-check:			## Lint check only
	@poetry run ./dev/lint


.PHONY: install-dev
install-dev:			## Install development dependencies
	@./dev/install

.PHONY: marimo
marimo:				## Run marimo for editing notebooks
	@./dev/marimo edit

.PHONY: docs
docs:				## build documentation
	@cp docs/index.md readme.md
	@poetry run mkdocs build

.PHONY: docs-serve
docs-serve:			## serve documentation
	@poetry run mkdocs serve --livereload --watch quantflow --watch docs

.PHONY: publish
publish:			## Release to pypi
	@poetry publish --build -u __token__ -p $(PYPI_TOKEN)

.PHONY: tests
tests:				## Unit tests
	@./dev/test


.PHONY: outdated
outdated:			## Show outdated packages
	poetry show -o -a
