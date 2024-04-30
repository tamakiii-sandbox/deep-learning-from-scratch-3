.PHONY: help test mypy format format-check

help:
	@cat $(firstword $(MAKEFILE_LIST))

test:
	$(error Not implemented yet)

mypy:
	mypy steps

format:
	black .

format-check:
	black --check .
