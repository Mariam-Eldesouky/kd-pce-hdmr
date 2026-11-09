"""Ablation: how the interaction features are screened (Bioresponse).

    python scripts/ablation_screening.py [--dataset bioresponse] [--seeds 0 1 2 3 4]

The main study chooses the k features that may interact by the teacher's mean |SHAP| on the pool.
Here that is compared with a screening that does not use SHAP: a first-order LASSO PCE (r = 1, same p,
all features) is fitted on the pool (inputs + teacher logits only), and the k features with the largest
first-order Sobol' index S_i are kept. Everything else is the selected configuration of that seed
(p, r, q, k, first_order_all from pce_grid.csv), so the screening is the only difference.

Per seed and screening method it records: CV R^2 on the pool, fidelity on test and off-manifold,
and the rank agreement (Kendall tau, top-k overlap) of the student's S_T with teacher SHAP and MC S_T.
Writes results/<dataset>/seed<k>/ablation_screening.csv and results/summary_<dataset>_ablation_screening.csv.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kdpce.data import load_config, load_dataset  # noqa: E402
from kdpce.explain import rank_agreement, screening_indices  # noqa: E402
from kdpce.metrics import fidelity, off_manifold  # noqa: E402
from kdpce.pce_hdmr import PCEHDMRStudent  # noqa: E402

RESULTS = ROOT / "results"


def lasso_first_order_screening(X_pool, t_pool, p, k, seed, names, max_terms):
    """Indices of the k features with the largest S_i in a first-order LASSO PCE (no SHAP, no labels)."""
    fo = PCEHDMRStudent(p=p, r=1, q=1.0, screened=None, first_order_all=True, seed=seed,
                        max_terms=max_terms, feature_names=names, verbose=False).fit(X_pool, t_pool)
    s1 = fo.sobol()[0].set_index("feature")["S1"].reindex(names).fillna(0)
    nonzero = s1[s1 > 0].sort_values(ascending=False, kind="stable")
    top = list(nonzero.index[:k])                      # fewer than k if LASSO keeps fewer features
    pos = {c: i for i, c in enumerate(names)}
    return sorted(pos[c] for c in top), len(nonzero), fo.cv_r2_


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="bioresponse")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = ap.parse_args()
    name = args.dataset
    cfg = load_config(name)
    X, _, _ = load_dataset(name)
    names = list(X.columns)
    all_rows = []
    for seed in args.seeds:
        out = RESULTS / name / f"seed{seed}"
        splits = dict(np.load(out / "splits.npz"))
        logits = dict(np.load(out / "teacher_logits.npz"))
        pred = np.load(out / "pce_predictions.npz")
        X_pool, X_test = X.iloc[splits["distill"]], X.iloc[splits["test"]]
        t_pool, t_test = logits["distill"], logits["test"]
        X_off, t_off = off_manifold(X_test, seed), pred["teacher_offman"]   # same off-manifold set as the study

        grid = pd.read_csv(out / "pce_grid.csv")
        g = grid[grid.selected & (grid.variant == "lasso")].iloc[0]
        p, r, q, k, foa = int(g.p), int(g.r), float(g.q), int(g.k), bool(g.first_order_all)
        shap_pool = pd.read_csv(out / "shap_pool.csv")
        shap_test = pd.read_csv(out / "shap_test.csv").set_index("feature")["mean_abs_shap"].reindex(names)
        mc_st = pd.read_csv(out / "teacher_mc_sobol.csv").set_index("feature")["ST"].reindex(names)

        t0 = time.perf_counter()
        lasso_idx, n_nonzero, fo_cv = lasso_first_order_screening(X_pool, t_pool, p, k, seed, names,
                                                                  cfg["pce"]["max_terms"])
        lasso_seconds = time.perf_counter() - t0
        screens = {"shap": screening_indices(shap_pool, names, k), "lasso_first_order": lasso_idx}
        common = len(set(screens["shap"]) & set(screens["lasso_first_order"]))
        print(f"seed {seed}: selected p={p} r={r} q={q} k={k} foa={foa}; first-order LASSO keeps {n_nonzero} "
              f"features (CV R^2 {fo_cv:.3f}, {lasso_seconds:.0f} s); screened sets share {common} of {k}")

        for method, idx in screens.items():
            st = PCEHDMRStudent(p=p, r=r, q=q, screened=idx, first_order_all=foa, seed=seed,
                                max_terms=cfg["pce"]["max_terms"], feature_names=names, verbose=False)
            st.fit(X_pool, t_pool)                     # inputs + teacher logits only
            if method == "shap" and not np.isclose(st.cv_r2_, g.cv_r2_pool):
                print(f"  WARNING: SHAP-screened refit CV R^2 {st.cv_r2_:.6f} != stored {g.cv_r2_pool:.6f}")
            row = {"dataset": name, "seed": seed, "screening": method, "p": p, "r": r, "q": q, "k": k,
                   "first_order_all": foa, "n_screened": len(idx), "n_shared_with_other": common,
                   "n_nonzero_first_order_lasso": n_nonzero, "n_terms": st.n_terms, "cv_r2_pool": st.cv_r2_,
                   "D78_screened": names.index("D78") in idx if "D78" in names else np.nan}
            for tag, Xs, ts in [("test", X_test, t_test), ("offman", X_off, t_off)]:
                row.update({f"{m}_{tag}": v for m, v in fidelity(ts, st.predict(Xs)).items()})
            pce_st = st.sobol()[0].set_index("feature")["ST"].reindex(names).fillna(0)
            agr = rank_agreement({"PCE_ST": pce_st, "SHAP": shap_test, "MC_ST": mc_st})
            for ref in ["SHAP", "MC_ST"]:
                a = agr[(agr.method_a == "PCE_ST") & (agr.method_b == ref)].iloc[0]
                for c in ["kendall_tau_all", "kendall_tau_top", "top5_overlap", "top10_overlap"]:
                    row[f"{c}_vs_{ref}"] = a[c]
            all_rows.append(row)
            print(f"  {method:18s} terms={st.n_terms} cv_r2={st.cv_r2_:.3f} test_r2={row['r2_logit_test']:.3f} "
                  f"offman_r2={row['r2_logit_offman']:.3f} tau_top(SHAP)={row['kendall_tau_top_vs_SHAP']:.2f} "
                  f"top10(MC)={row['top10_overlap_vs_MC_ST']:.2f}", flush=True)
        pd.DataFrame([r for r in all_rows if r["seed"] == seed]).to_csv(out / "ablation_screening.csv", index=False)

    # Summary over every seed that has a complete per-seed file on disk (so seeds can be run separately).
    files = sorted((RESULTS / name).glob("seed*/ablation_screening.csv"))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    print(f"summary over seeds {sorted(df.seed.unique().tolist())}")
    metrics = [c for c in df.columns if c not in ("dataset", "seed", "screening", "p", "r", "q", "k",
                                                  "first_order_all")]
    df[metrics] = df[metrics].astype(float)
    summ = df.groupby("screening")[metrics].agg(["mean", "std", "count"])
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    summ = summ.reset_index()
    summ.insert(0, "dataset", name)
    summ.to_csv(RESULTS / f"summary_{name}_ablation_screening.csv", index=False)
    show = ["cv_r2_pool", "r2_logit_test", "rmse_logit_test", "r2_logit_offman", "rmse_logit_offman",
            "kendall_tau_top_vs_SHAP", "top10_overlap_vs_SHAP", "kendall_tau_top_vs_MC_ST", "top10_overlap_vs_MC_ST"]
    print(summ[["screening"] + [f"{c}_mean" for c in show]].T.to_string())


if __name__ == "__main__":
    main()
