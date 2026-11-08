"""Aggregate per-seed CSVs into across-seed summaries (mean, std, number of seeds).

Reads results/<dataset>/seed*/[quick/] and writes, in results/:
  summary_<dataset>.csv           tidy: section, item, metric, mean, std, n_seeds
  summary_<dataset>_pce_grid.csv  PCE grid by configuration (for the trade-off figure)
  summary_<dataset>_ebm_grid.csv  EBM grid by K
  summary_<dataset>_explanations.csv  per method x feature: mean/std of score and rank
(with suffix _quick when aggregating quick-mode results)
"""
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "results"


def seed_dirs(name, quick=False):
    dirs = sorted((RESULTS / name).glob("seed*"), key=lambda p: int(p.name[4:]))
    return [d / "quick" if quick else d for d in dirs]


def _collect(name, filename, quick=False):
    frames = []
    for d in seed_dirs(name, quick):
        f = d / filename
        if f.exists():
            df = pd.read_csv(f)
            if "seed" not in df.columns:
                df.insert(0, "seed", int(d.parent.name[4:] if quick else d.name[4:]))
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _mean_std(df, by, metrics, section):
    g = df.groupby(by)[metrics]
    m, s, n = g.mean(), g.std(), g.count()
    rows = []
    for item in m.index:
        for metric in metrics:
            if n.loc[item, metric] == 0:          # no data for this item/metric (e.g. teacher R^2)
                continue
            rows.append({"section": section, "item": item if not isinstance(item, tuple) else " | ".join(map(str, item)),
                         "metric": metric, "mean": m.loc[item, metric], "std": s.loc[item, metric],
                         "n_seeds": int(n.loc[item, metric])})
    return rows


def aggregate(name, quick=False):
    """Write the summary CSVs for one dataset; return the main summary DataFrame."""
    tag = f"{name}_quick" if quick else name
    rows = []

    tm = _collect(name, "teacher_metrics.csv")                      # teacher lives in seed<k>/, not quick/
    if quick and len(tm):                                           # only seeds that have a quick run
        tm = tm[tm.seed.isin([int(d.parent.name[4:]) for d in seed_dirs(name, quick) if d.exists()])]
    if len(tm):
        rows += _mean_std(tm[tm.subset == "test"], "subset", ["accuracy", "auc"], "teacher")

    fid = _collect(name, "fidelity.csv", quick)
    if len(fid):
        metrics = [c for c in fid.columns if c.endswith(("_test", "_offman")) or c in
                   ("n_terms", "n_parameters", "max_order", "cv_r2_pool", "fit_seconds")]
        metrics = [c for c in metrics if fid[c].notna().any()]
        rows += _mean_std(fid, "model", metrics, "fidelity")
        for model, d in fid[fid.model != "teacher"].groupby("model"):  # which config was selected per seed
            for cfg, cnt in d["config"].value_counts().items():
                rows.append({"section": "selected_config", "item": model, "metric": cfg,
                             "mean": cnt, "std": float("nan"), "n_seeds": len(d)})

    agr = _collect(name, "agreement.csv", quick)
    if len(agr):
        agr["pair"] = agr["method_a"] + " vs " + agr["method_b"]
        rows += _mean_std(agr, "pair", ["kendall_tau_all", "kendall_tau_top", "top5_overlap", "top10_overlap"],
                          "agreement")

    vc = _collect(name, "pce_variance_check.csv", quick)
    if len(vc):
        rows += _mean_std(vc, "data", ["D_analytic", "var_empirical", "ratio_var_over_D"], "pce_variance_check")
    ev = _collect(name, "ebm_variance_check.csv", quick)
    if len(ev):
        ev["item"] = "test"
        rows += _mean_std(ev, "item", ["ratio_sum_term_var_over_var_pred"], "ebm_variance_check")

    # Wall-clock per stage: only seeds where the stage ran in one go (a resumed run measures only the rest).
    rt = _collect(name, "runtime.csv", quick)
    if len(rt):
        rows += _mean_std(rt[~rt["resumed"].astype(bool)], "stage", ["seconds"], "runtime")
    # Compute time summed over the grid rows (fit + CV per configuration); unaffected by resuming.
    for stage, fname, cols in [("pce", "pce_grid.csv", ["fit_seconds"]),
                               ("ebm", "ebm_grid.csv", ["fit_seconds", "cv_seconds"])]:
        g = _collect(name, fname, quick)
        if len(g):
            per_seed = g.groupby("seed")[cols].sum().sum(axis=1).rename("seconds").reset_index()
            per_seed["stage"] = stage
            rows += _mean_std(per_seed, "stage", ["seconds"], "grid_compute")

    summary = pd.DataFrame(rows)
    summary.insert(0, "dataset", name)
    summary.to_csv(RESULTS / f"summary_{tag}.csv", index=False)

    pg = _collect(name, "pce_grid.csv", quick)
    if len(pg):
        pg = pg[(pg.status == "ok") & (pg.get("variant", "lasso") == "lasso")]
        keys = ["p", "r", "q", "k", "first_order_all"]
        cols = ["n_terms", "max_order", "cv_r2_pool", "r2_logit_test", "r2_logit_offman",
                "label_agreement_test", "fit_seconds"]
        out = pg.groupby(keys)[cols].agg(["mean", "std", "count"])
        out.columns = [f"{a}_{b}" for a, b in out.columns]
        out["times_selected"] = pg.groupby(keys)["selected"].sum()
        out.reset_index().to_csv(RESULTS / f"summary_{tag}_pce_grid.csv", index=False)

    eg = _collect(name, "ebm_grid.csv", quick)
    if len(eg):
        cols = ["n_terms", "n_parameters", "cv_r2_pool", "r2_logit_test", "r2_logit_offman",
                "label_agreement_test", "fit_seconds"]
        out = eg.groupby("interactions")[cols].agg(["mean", "std", "count"])
        out.columns = [f"{a}_{b}" for a, b in out.columns]
        out["times_selected"] = eg.groupby("interactions")["selected"].sum()
        out.reset_index().to_csv(RESULTS / f"summary_{tag}_ebm_grid.csv", index=False)

    ex = _collect(name, "explanations.csv", quick)
    if len(ex):
        out = ex.groupby(["method", "feature"])[["score", "rank"]].agg(["mean", "std", "count"])
        out.columns = [f"{a}_{b}" for a, b in out.columns]
        out.reset_index().to_csv(RESULTS / f"summary_{tag}_explanations.csv", index=False)

    return summary
