"""Print the README's headline tables (Markdown) from results/summary_<dataset>.csv.

    python scripts/results_table.py

Every number comes from the summary CSVs written by run_all.py / make_figures.py (mean +- std over seeds).
"""
import sys
from pathlib import Path

import pandas as pd

RESULTS = Path(__file__).resolve().parents[1] / "results"
DATASETS = {"magic": "MAGIC", "bioresponse": "Bioresponse"}


def get(s, section, item, metric):
    row = s[(s.section == section) & (s.item == item) & (s.metric == metric)]
    return row.iloc[0] if len(row) else None


def pm(s, section, item, metric, digits=3):
    r = get(s, section, item, metric)
    if r is None:
        return "–"
    return f"{r['mean']:.{digits}f} ± {r['std']:.{digits}f}" if pd.notna(r["std"]) else f"{r['mean']:.{digits}f}"


def selected(s, model):
    rows = s[(s.section == "selected_config") & (s.item == model)]
    return "; ".join(f"{r.metric} ({int(r['mean'])}/{int(r.n_seeds)})" for _, r in rows.iterrows())


def main():
    sys.stdout.reconfigure(encoding="utf-8")   # Windows consoles default to cp1252 (no τ, ±)
    summaries = {k: pd.read_csv(RESULTS / f"summary_{k}.csv") for k in DATASETS
                 if (RESULTS / f"summary_{k}.csv").exists()}
    n = {k: int(s["n_seeds"].max()) for k, s in summaries.items()}

    print("| dataset | model | selected config (seeds) | active terms | test R² (logit) | label agreement "
          "| off-manifold R² | accuracy vs y | AUC vs y |")
    print("|---|---|---|---|---|---|---|---|---|")
    for k, s in summaries.items():
        print(f"| {DATASETS[k]} | teacher (XGBoost) | – | – | – | – | – | {pm(s, 'teacher', 'test', 'accuracy')} "
              f"| {pm(s, 'teacher', 'test', 'auc')} |")
        for m in ["PCE", "EBM"]:
            print(f"| {DATASETS[k]} | {m} student | {selected(s, m)} | {pm(s, 'fidelity', m, 'n_terms', 1)} "
                  f"| {pm(s, 'fidelity', m, 'r2_logit_test')} | {pm(s, 'fidelity', m, 'label_agreement_test')} "
                  f"| {pm(s, 'fidelity', m, 'r2_logit_offman')} | {pm(s, 'fidelity', m, 'accuracy_vs_y_test')} "
                  f"| {pm(s, 'fidelity', m, 'auc_vs_y_test')} |")
    seeds = ", ".join(f"{DATASETS[k]} {v}" for k, v in n.items())
    print(f"\nmean ± std over seeds (number of seeds: {seeds}).\n")

    print("| dataset | pair | Kendall τ (all features) | Kendall τ (top-20 union) | top-5 overlap | top-10 overlap |")
    print("|---|---|---|---|---|---|")
    pairs = ["PCE_ST vs MC_ST", "EBM_total vs MC_ST", "SHAP vs MC_ST", "PCE_ST vs SHAP", "EBM_total vs SHAP",
             "PCE_ST vs EBM_total"]
    for k, s in summaries.items():
        for p in pairs:
            print(f"| {DATASETS[k]} | {p} | {pm(s, 'agreement', p, 'kendall_tau_all', 2)} "
                  f"| {pm(s, 'agreement', p, 'kendall_tau_top', 2)} | {pm(s, 'agreement', p, 'top5_overlap', 2)} "
                  f"| {pm(s, 'agreement', p, 'top10_overlap', 2)} |")

    print("\n| dataset | PCE grid compute (s) | EBM grid compute (s) | explain stage (s) |")
    print("|---|---|---|---|")
    for k, s in summaries.items():
        print(f"| {DATASETS[k]} | {pm(s, 'grid_compute', 'pce', 'seconds', 0)} "
              f"| {pm(s, 'grid_compute', 'ebm', 'seconds', 0)} | {pm(s, 'runtime', 'explain', 'seconds', 1)} |")


if __name__ == "__main__":
    main()
