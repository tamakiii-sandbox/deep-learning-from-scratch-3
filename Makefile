.PHONY: help test check mypy format format-check

help:
	@cat $(firstword $(MAKEFILE_LIST))

test:
	$(error Not implemented yet)

check: \
	mypy \
	format-check

mypy:
	mypy steps

format:
	black .

format-check:
	black --check .
