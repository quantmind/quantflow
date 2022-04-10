

.PHONY: lint notebook book tests

help:
	@echo ================================================================================
	@fgrep -h "##" $(MAKEFILE_LIST) | fgrep -v fgrep | sed -e 's/\\$$//' | sed -e 's/##//'
	@echo ================================================================================

lint:				## Build architecture diagrams
	@poetry run ./dev/lint


install-dev:			## Install development dependencies
	@./dev/install

notebook:			## Run Jupyter notebook server
	@poetry run ./dev/start-jupyter 9090


book:				## Build static jupyter {book}
	poetry run jupyter-book build notebooks --all


tests:				## unit tests
	poetry run pytest
