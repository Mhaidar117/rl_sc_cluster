#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = rl_sc_cluster
PYTHON_INTERPRETER = python
VENV_PATH = venv
VENV_BIN = $(VENV_PATH)/bin

#################################################################################
# ENVIRONMENT SETUP                                                             #
#################################################################################

## Create virtual environment and install dependencies
.PHONY: venv
venv:
	$(PYTHON_INTERPRETER) -m venv $(VENV_PATH)
	@echo ">>> Virtual environment created at $(VENV_PATH)"
	@echo ">>> Installing dependencies..."
	@bash -c "source $(VENV_BIN)/activate && pip install --upgrade pip && pip install -r requirements.txt"
	@echo ">>> Setup complete! Activate with: source $(VENV_BIN)/activate"

## Remove virtual environment
.PHONY: clean-venv
clean-venv:
	rm -rf $(VENV_PATH)
	@echo ">>> Virtual environment removed"

#################################################################################
# CODE QUALITY                                                                  #
#################################################################################

## Lint using flake8, black, and isort
.PHONY: lint
lint:
	flake8 rl_sc_cluster_utils
	isort --check --diff rl_sc_cluster_utils
	black --check rl_sc_cluster_utils

## Format source code with black and isort
.PHONY: format
format:
	isort rl_sc_cluster_utils
	black rl_sc_cluster_utils

#################################################################################
# TESTING                                                                       #
#################################################################################

## Run all tests
.PHONY: test
test:
	$(PYTHON_INTERPRETER) -m pytest tests -v

## Run environment tests only
.PHONY: test-env
test-env:
	$(PYTHON_INTERPRETER) -m pytest tests/env_test -v

## Run tests with coverage report
.PHONY: test-cov
test-cov:
	$(PYTHON_INTERPRETER) -m pytest tests --cov=rl_sc_cluster_utils --cov-report=html --cov-report=term

#################################################################################
# DOCUMENTATION                                                                 #
#################################################################################

## Build documentation with mkdocs
.PHONY: docs
docs:
	cd docs && mkdocs build

## Serve documentation locally
.PHONY: docs-serve
docs-serve:
	cd docs && mkdocs serve

## Deploy documentation to GitHub Pages
.PHONY: docs-deploy
docs-deploy:
	cd docs && mkdocs gh-deploy

#################################################################################
# CLEANUP                                                                       #
#################################################################################

## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Deep clean (compiled files + build artifacts)
.PHONY: clean-all
clean-all: clean
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf docs/site/

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

## Make dataset
.PHONY: data
data:
	$(PYTHON_INTERPRETER) rl_sc_cluster_utils/dataset.py


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
