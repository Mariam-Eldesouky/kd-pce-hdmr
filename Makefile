# Usage: make install | make data | make test | make quick | make all
# On Windows the venv interpreter lives in .venv/Scripts; on Linux/macOS override:
#   make PY=.venv/bin/python test
PY ?= .venv/Scripts/python

.PHONY: install data test quick all

install:
	python -m venv .venv
	$(PY) -m pip install -r requirements.txt

data:
	$(PY) scripts/check_data.py

test:
	$(PY) -m pytest -q tests

quick:
	$(PY) scripts/run_all.py --dataset all --quick --seeds 0

all:
	$(PY) scripts/run_all.py --dataset all --seeds 0 1 2 3 4
