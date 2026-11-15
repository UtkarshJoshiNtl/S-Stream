.PHONY: install test

install:
	python3 -m pip install -e ./simulation

test:
	python3 -m pytest simulation/tests
