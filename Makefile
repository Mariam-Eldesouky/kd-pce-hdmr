# Usage: make install | make data | make test | make quick | make all | make figures
# Long runs on Windows: scriptsun_full_windows.bat (one job per dataset/seed, resumable).
# On Windows the venv interpreter lives in .venv/Scripts; on Linux/macOS override:
#   make PY=.venv/bin/python test
PY ?= .venv/Scripts/python

.PHONY: install data test quick all figures

install:
	python -m venv .venv
	$(PY) -m pip install -r requirements.txt

data:
	$(PY) scripts/check_data.py

test:
	$(PY) -m pytest -q tests

quick:
	$(PY) scripts/run_all.py --dataset all --quick --seeds 0
	$(PY) scripts/make_figures.py --quick

all:
	$(PY) scripts/run_all.py --dataset all --seeds 0 1 2 3 4
	$(PY) scripts/make_figures.py

figures:
	$(PY) scripts/make_figures.py
