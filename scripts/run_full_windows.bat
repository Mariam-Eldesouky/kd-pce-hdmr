@echo off
REM Full experiment (Phase 5): 2 datasets x 5 seeds, one job at a time, then figures.
REM Run from the repository root in Command Prompt:   scripts\run_full_windows.bat
REM Each dataset/seed is a separate run with its own log in logs\. If a run fails or the PC
REM restarts, just start this file again: finished work is loaded or skipped, not recomputed
REM (teachers are loaded from disk; PCE and EBM grids resume from their *_partial.csv files).

setlocal EnableDelayedExpansion
set PYTHONWARNINGS=ignore
if not exist logs mkdir logs

for %%D in (magic bioresponse) do (
  for %%S in (0 1 2 3 4) do (
    echo [!date! !time!] start %%D seed %%S
    .venv\Scripts\python -u scripts\run_all.py --dataset %%D --seeds %%S > logs\phase5_%%D_seed%%S.log 2>&1
    if errorlevel 1 (
      echo [!date! !time!]   FAILED %%D seed %%S - see logs\phase5_%%D_seed%%S.log
    ) else (
      echo [!date! !time!]   done   %%D seed %%S
    )
  )
)

echo [!date! !time!] building figures
.venv\Scripts\python -u scripts\make_figures.py > logs\phase5_figures.log 2>&1
if errorlevel 1 (
  echo [!date! !time!]   FAILED figures - see logs\phase5_figures.log
) else (
  echo [!date! !time!]   done   figures
)
echo [!date! !time!] ALL FINISHED
