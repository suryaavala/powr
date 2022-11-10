.PHONY: setup-dev setup-prod install-poetry install-py-dev-req install-py-prod-req install-package clean install-git-hooks poetry-shell docker-dev-build docker-dev-shell lint-style lint-security lint-types lint-all test test-lint-all

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PROFILE = default
PROJECT_NAME = powr
PYTHON_INTERPRETER = python3
POETRY_VERSION = 1.2.2

#################################################################################
# COMMANDS                                                                      #
#################################################################################


####### SETUP ENV #######
## Setup Local development environment
setup-dev: install-poetry install-py-dev-req install-git-hooks poetry-shell dvc-pull

## Setup Production environment
setup-prod: install-poetry install-py-prod-req

## Install Poety
install-poetry:
	pip3 install poetry==$(POETRY_VERSION)

## Install all py requirements including dev for local development
install-py-dev-req:
	poetry install --all-extras

## Install only required py packages for prod deployment
install-py-prod-req:
	poetry install --only main

## Install current package
install-package:
	poetry install --only-root

## Clean up cache
clean: lint-all
	find . -type f -name "*.DS_Store" -ls -delete
	find . | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -rf
	find . | grep -E ".pytest_cache" | xargs rm -rf
	find . | grep -E ".ipynb_checkpoints" | xargs rm -rf
	find . | grep -E ".trash" | xargs rm -rf
	rm -f .coverage

## Install git pre-commit hooks
install-git-hooks:
	pre-commit install

## Load poetry shell
poetry-shell:
	poetry shell

## Build a docker image with local dev environment setup
docker-dev-build:
	docker build --target dev -t powr-dev --file docker/dev.Dockerfile .

## Load a docker development shell with local dev environment setup
docker-dev-shell: docker-dev-build
	docker run -it --entrypoint /bin/bash powr-dev:latest


####### Lint #######
## Lint using flake8
lint-style:
	black .
	flake8
	isort .

## security check using bandit
lint-security:
	bandit -l --recursive -x ./tests -r .

## type checks with mypy
lint-types:
	mypy $(PROJECT_NAME)

## checks code for linting, security and type errors
lint-all: lint-style lint-types lint-security


####### Tests #######
## unit tests code with pytest
test:
	pytest -m "not training"

## runs lint-all, pytest, coverage; use when testing locally
test-lint-all: lint-all test


####### DVC #######
.PHONY: dvc-pull
dvc-pull:
	dvc pull

## Add data to dvc & sync with remote
dvc-sync:
	dvc pull
	dvc push
	dvc status -r redundant -q
	dvc status -q


####### Dev Help #######
.PHONY: test-watch
## watch relevant directories for tests and run tests when changes are detected
test-watch:
	exclude_pattern=".*"
	include_pattern=".*\.py$$"
	fswatch -or1 -e "$exclude_pattern" -i "$include_pattern" --event=Updated powr tests | xargs -n1 -I{} make test

.PHONY: data-clean
## clean up data
elt-data:
	python3 main.py elt-data

.PHONY: generate-dataset
## generate dataset
generate-dataset:
	python3 main.py generate-dataset

.PHONY: train-model
## train model
train-model:
	python3 main.py train-model

.PHONY: predict-powr
## predict powr
predict-powr:
	python3 main.py predict-powr

.PHONY: show-pipeline
## show pipeline
show-pipeline:
	dvc dag

.PHONY: run-pipeline
## run pipeline
run-pipeline: show-pipeline
	dvc repro
	dvc push
#################################################################################
# PROJECT RULES                                                                 #
#################################################################################



#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
