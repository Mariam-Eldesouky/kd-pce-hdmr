# Usage: make install | make data | make test | make quick | make all | make analysis | make figures
# Long runs on Windows: scripts/run_full_windows.bat (one job per dataset/seed, resumable).
# On Windows the venv interpreter lives in .venv/Scripts; on Linux/macOS override:
#   make PY=.venv/bin/python test
PY ?= .venv/Scripts/python

.PHONY: install data test quick all analysis figures

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
	$(MAKE) analysis
	$(PY) scripts/make_figures.py

# Bioresponse follow-up analyses (need the full 5-seed results): screening ablation, Var/D diagnosis
analysis:
	$(PY) scripts/ablation_screening.py --dataset bioresponse --seeds 0 1 2 3 4
	$(PY) scripts/diagnose_var_ratio.py --dataset bioresponse --seeds 0 1 2 3 4

figures:
	$(PY) scripts/make_figures.py
