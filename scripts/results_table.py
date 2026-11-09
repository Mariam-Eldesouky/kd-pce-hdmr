"""Print the README's tables (Markdown) from results/summary_<dataset>*.csv.

    python scripts/results_table.py

Every number comes from the summary CSVs written by run_all.py / make_figures.py and the two analysis
scripts (ablation_screening.py, diagnose_var_ratio.py); mean +- std over seeds.
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kdpce.plots import METHOD_LABELS, primary_ebm  # noqa: E402

RESULTS = ROOT / "results"
DATASETS = {"magic": "MAGIC", "bioresponse": "Bioresponse"}


def get(s, section, item, metric):
    row = s[(s.section == section) & (s.item == item) & (s.metric == metric)]
    return row.iloc[0] if len(row) else None


def pm(s, section, item, metric, digits=3):
    r = get(s, section, item, metric)
    if r is None:
        return "–"
    return f"{r['mean']:.{digits}f} ± {r['std']:.{digits}f}" if pd.notna(r["std"]) else f"{r['mean']:.{digits}f}"


def pm_wide(df, row, col, digits=3):
    return f"{df.loc[row, f'{col}_mean']:.{digits}f} ± {df.loc[row, f'{col}_std']:.{digits}f}"


def selected(s, model):
    rows = s[(s.section == "selected_config") & (s.item == model)]
    return "; ".join(f"{r.metric} ({int(r['mean'])}/{int(r.n_seeds)})" for _, r in rows.iterrows())


def main():
    sys.stdout.reconfigure(encoding="utf-8")   # Windows consoles default to cp1252 (no τ, ±)
    summaries = {k: pd.read_csv(RESULTS / f"summary_{k}.csv") for k in DATASETS
                 if (RESULTS / f"summary_{k}.csv").exists()}
    n = {k: int(s["n_seeds"].max()) for k, s in summaries.items()}

    print("### Fidelity\n")
    print("| dataset | model | selected config (seeds) | active terms | test R² | test RMSE | label agreement "
          "| off-manifold R² | off-manifold RMSE | accuracy vs y | AUC vs y |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for k, s in summaries.items():
        print(f"| {DATASETS[k]} | teacher (XGBoost) | – | – | – | – | – | – | – | {pm(s, 'teacher', 'test', 'accuracy')} "
              f"| {pm(s, 'teacher', 'test', 'auc')} |")
        for m in ["PCE", "EBM"]:
            print(f"| {DATASETS[k]} | {m} student | {selected(s, m)} | {pm(s, 'fidelity', m, 'n_terms', 1)} "
                  f"| {pm(s, 'fidelity', m, 'r2_logit_test')} | {pm(s, 'fidelity', m, 'rmse_logit_test')} "
                  f"| {pm(s, 'fidelity', m, 'label_agreement_test')} "
                  f"| {pm(s, 'fidelity', m, 'r2_logit_offman')} | {pm(s, 'fidelity', m, 'rmse_logit_offman')} "
                  f"| {pm(s, 'fidelity', m, 'accuracy_vs_y_test')} | {pm(s, 'fidelity', m, 'auc_vs_y_test')} |")
    seeds = ", ".join(f"{DATASETS[k]} {v}" for k, v in n.items())
    print(f"\nR² and RMSE are on teacher logits; mean ± std over seeds (number of seeds: {seeds}).\n")

    print("### Teacher logit variance (R² = 1 − MSE / Var(teacher logit))\n")
    print("| dataset | Var(teacher logit), test | Var(teacher logit), off-manifold |")
    print("|---|---|---|")
    for k, s in summaries.items():
        print(f"| {DATASETS[k]} | {pm(s, 'teacher_logit_variance', 'test', 'var_teacher_logit', 2)} "
              f"| {pm(s, 'teacher_logit_variance', 'offmanifold', 'var_teacher_logit', 2)} |")

    print("\n### Agreement of explanations\n")
    print("| dataset | pair | Kendall τ (all features) | Kendall τ (top-20 union) | top-5 overlap | top-10 overlap |")
    print("|---|---|---|---|---|---|")
    for k, s in summaries.items():
        ebm = primary_ebm(k)
        pairs = ["PCE_ST vs MC_ST", f"{ebm} vs MC_ST", "SHAP vs MC_ST", "PCE_ST vs SHAP", f"{ebm} vs SHAP",
                 f"PCE_ST vs {ebm}"]
        for p in pairs:
            a, b = p.split(" vs ")
            label = f"{METHOD_LABELS[a]} vs {METHOD_LABELS[b]}".replace("$", "").replace("|", "\\|")  # Markdown
            print(f"| {DATASETS[k]} | {label} | {pm(s, 'agreement', p, 'kendall_tau_all', 2)} "
                  f"| {pm(s, 'agreement', p, 'kendall_tau_top', 2)} | {pm(s, 'agreement', p, 'top5_overlap', 2)} "
                  f"| {pm(s, 'agreement', p, 'top10_overlap', 2)} |")

    f = RESULTS / "summary_bioresponse_ablation_screening.csv"
    if f.exists():
        ab = pd.read_csv(f).set_index("screening")
        print("\n### Ablation (Bioresponse): screening of interaction features\n")
        print("| screening | CV R² (pool) | test R² | test RMSE | off-manifold R² | off-manifold RMSE "
              "| τ top-20 vs SHAP | top-10 vs SHAP | τ top-20 vs MC S_T | top-10 vs MC S_T |")
        print("|---|---|---|---|---|---|---|---|---|---|")
        for m, label in [("shap", "teacher SHAP on the pool (main study)"),
                         ("lasso_first_order", "first-order LASSO PCE on the pool")]:
            if m in ab.index:
                print(f"| {label} | {pm_wide(ab, m, 'cv_r2_pool')} | {pm_wide(ab, m, 'r2_logit_test')} "
                      f"| {pm_wide(ab, m, 'rmse_logit_test')} | {pm_wide(ab, m, 'r2_logit_offman')} "
                      f"| {pm_wide(ab, m, 'rmse_logit_offman')} | {pm_wide(ab, m, 'kendall_tau_top_vs_SHAP', 2)} "
                      f"| {pm_wide(ab, m, 'top10_overlap_vs_SHAP', 2)} | {pm_wide(ab, m, 'kendall_tau_top_vs_MC_ST', 2)} "
                      f"| {pm_wide(ab, m, 'top10_overlap_vs_MC_ST', 2)} |")
        print(f"\nThe two screened sets share {ab.loc['shap', 'n_shared_with_other_mean']:.1f} "
              f"± {ab.loc['shap', 'n_shared_with_other_std']:.1f} features.")

    f = RESULTS / "summary_bioresponse_var_ratio.csv"
    if f.exists():
        vr = pd.read_csv(f).set_index("metric")
        ft = pd.read_csv(RESULTS / "summary_bioresponse_feature_types.csv").set_index("type")["n_features_full_data"]
        print("\n### Bioresponse: feature types and the off-manifold Var/D of the PCE\n")
        print("| quantity | value |")
        print("|---|---|")
        for t, label in [("binary", "binary features (2 values)"), ("low_cardinality_3_10", "3–10 distinct values"),
                         ("continuous_gt10", "more than 10 distinct values"), ("constant", "constant")]:
            print(f"| {label}, full data | {int(ft.get(t, 0))} |")
        for m, label, d in [("mean_E_psi1sq_binary", "E[ψ₁²] on the pool, binary features", 3),
                            ("mean_E_psi1sq_low_card", "E[ψ₁²] on the pool, 3–10 values", 3),
                            ("mean_E_psi1sq_continuous", "E[ψ₁²] on the pool, > 10 values", 3),
                            ("share_D_terms_with_binary", "share of D in terms with a binary feature", 3),
                            ("predicted_ratio_from_moments", "predicted Var/D from the pool moments", 3),
                            ("observed_ratio_offmanifold", "observed Var/D off-manifold", 3)]:
            print(f"| {label} | {vr.loc[m, 'mean']:.{d}f} ± {vr.loc[m, 'std']:.{d}f} |")

    print("\n### Runtime\n")
    print("| dataset | PCE grid compute (s) | EBM grid compute (s) | explain stage (s) |")
    print("|---|---|---|---|")
    for k, s in summaries.items():
        print(f"| {DATASETS[k]} | {pm(s, 'grid_compute', 'pce', 'seconds', 0)} "
              f"| {pm(s, 'grid_compute', 'ebm', 'seconds', 0)} | {pm(s, 'runtime', 'explain', 'seconds', 1)} |")


if __name__ == "__main__":
    main()
