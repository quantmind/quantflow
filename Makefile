
.PHONY: help
help:
	@echo ================================================================================
	@fgrep -h "##" $(MAKEFILE_LIST) | fgrep -v fgrep | sed -e 's/\\$$//' | sed -e 's/##//'
	@echo ================================================================================

.PHONY: app-serve
app-serve:			## serve app
	@MICRO_SERVICE_HOST=127.0.0.1 uv run python -m app

.PHONY: docs
docs:				## build documentation
	@cp docs/index.md readme.md
	@uv run ./dev/build-examples
	@uv run mkdocs build

.PHONY: docs-bib
docs-bib:			## Regenerate docs bibliography
	@uv run ./docs/bib2md.py

.PHONY: docs-examples
docs-examples:			## Regenerate docs examples
	@uv run ./dev/build-examples

.PHONY: docs-serve
docs-serve:			## serve docs, examples, and API with auto-reload
	@bash ./dev/docs-serve

.PHONY: frontend-build
frontend-build:			## build Observable frontend examples
	@rm -rf app/examples
	@npm --prefix app/frontend run build

.PHONY: frontend-serve
frontend-serve:			## serve Observable frontend with auto-reload
	@bash ./dev/frontend-serve

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

.PHONY: release
release:			## Tag current version (from pyproject.toml) and push
	$(eval VERSION := $(shell grep '^version' pyproject.toml | head -1 | sed 's/version = "\(.*\)"/\1/'))
	@read -p "Tagging with v$(VERSION), are you sure? [Y/n] " ans; \
	ans=$${ans:-Y}; \
	if [ "$$ans" = "Y" ] || [ "$$ans" = "y" ]; then \
		git tag -a v$(VERSION) -m "v$(VERSION)" && git push origin v$(VERSION); \
	else \
		echo "Aborted."; \
	fi

.PHONY: tests
tests:				## Unit tests
	@./dev/test

.PHONY: upgrade
upgrade:			## Upgrade dependencies
	uv lock --upgrade
