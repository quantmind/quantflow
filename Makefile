
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


.PHONY: notebook
notebook:			## Run Jupyter notebook server
	@poetry run ./dev/start-jupyter 9095


.PHONY: book
book:				## Build static jupyter {book}
	poetry run jupyter-book build notebooks --all
	cp notebooks/CNAME notebooks/_build/html/CNAME


.PHONY: nbconvert
nbconvert:			## Convert notebooks to myst markdown
	poetry run ./dev/nbconvert

.PHONY: nbsync
nbsync:				## Sync python myst notebooks to .ipynb files - needed for vs notebook development
	poetry run ./dev/nbsync

.PHONY: sphinx-config
sphinx-config:			## Build sphinx config
	poetry run jupyter-book config sphinx notebooks


.PHONY: sphinx
sphinx:
	poetry run sphinx-build notebooks path/to/book/_build/html -b html


.PHONY: publish
publish:			## Release to pypi
	@poetry publish --build -u __token__ -p $(PYPI_TOKEN)


.PHONY: publish-book
publish-book:			## publish the book to github pages
	poetry run ghp-import -n -p -f notebooks/_build/html


.PHONY: tests
tests:				## Unit tests
	@./dev/test


.PHONY: outdated
outdated:			## Show outdated packages
	poetry show -o -a
