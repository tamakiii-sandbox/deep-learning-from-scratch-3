.PHONY: help test format format-check

help:
	@cat $(firstword $(MAKEFILE_LIST))

test:
	mypy steps

format:
	black .

format-check:
	black --check .
