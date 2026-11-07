"""End-to-end experiment driver.

Examples
    python scripts/run_all.py --dataset magic --seeds 0 --stages teacher
    python scripts/run_all.py --dataset all --quick --seeds 0
    python scripts/run_all.py --dataset all --seeds 0 1 2 3 4

Datasets: magic (moderate, 10 features) and bioresponse (high-dimensional, 1,776 features).
`--dataset all` runs these two. Musk can still be run by name but is not part of the study
(its teacher did not generalise to unseen molecules; see the README).

Stages are added phase by phase. Implemented: teacher, pce.
Each (dataset, seed) writes to results/<dataset>/seed<k>/.
"""
import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kdpce.data import load_config, load_dataset, make_splits, split_summary  # noqa: E402
from kdpce.explain import screening_indices, teacher_shap_global  # noqa: E402
from kdpce.metrics import accuracy_metrics, fidelity, off_manifold  # noqa: E402
from kdpce.pce_hdmr import PCEHDMRStudent  # noqa: E402
from kdpce.teacher import (load_teacher, save_teacher, teacher_logit,  # noqa: E402
                           teacher_metrics, train_teacher)

ALL_STAGES = ["teacher", "pce"]
STUDY_DATASETS = ["magic", "bioresponse"]


def set_global_seeds(seed):
    """numpy's global RNG; xgboost/sklearn/EBM get `random_state=seed` explicitly."""
    np.random.seed(seed)


def fmt(v):
    return f"{v:.4f}"


# ----------------------------------------------------------------------------------------
# Stage: teacher
# ----------------------------------------------------------------------------------------
def stage_teacher(name, cfg, X, y, groups, seed, out, retrain=False):
    """Split, train-and-freeze (or load) the teacher, save logits for every subset."""
    splits = make_splits(X, y, groups, seed)
    np.savez(out / "splits.npz", **splits)
    summ = split_summary(splits, y, groups)
    summ.to_csv(out / "split_summary.csv", index=False)
    print(summ.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    tr = splits["train"]
    t0 = time.perf_counter()
    if (out / "teacher.json").exists() and not retrain:
        model = load_teacher(out)                      # frozen: only loaded, never refitted
        info = json.loads((out / "teacher_info.json").read_text())
        print(f"  loaded frozen teacher (sha256 {info['sha256'][:12]}...)")
    else:
        g_tr = None if groups is None else groups[tr]
        model, info, grid = train_teacher(X.iloc[tr], y[tr], g_tr, cfg["teacher"], seed)
        grid.to_csv(out / "teacher_grid.csv", index=False)
        digest = save_teacher(model, info, out)
        model = load_teacher(out)                      # use the frozen file from here on
        info["sha256"] = digest
        print(f"  trained teacher: depth={info['max_depth']} lr={info['learning_rate']} "
              f"trees={info['n_estimators']} sha256 {digest[:12]}...")
    seconds = time.perf_counter() - t0

    logits = {k: teacher_logit(model, X.iloc[idx]) for k, idx in splits.items()}
    np.savez(out / "teacher_logits.npz", **logits)

    rows = []
    for k, idx in splits.items():
        m = teacher_metrics(model, X.iloc[idx], y[idx])
        rows.append({"dataset": name, "seed": seed, "subset": k, **m,
                     "logit_mean": logits[k].mean(), "logit_std": logits[k].std()})
    met = pd.DataFrame(rows)
    met.to_csv(out / "teacher_metrics.csv", index=False)
    print(met[["subset", "accuracy", "auc", "logit_mean", "logit_std"]]
          .to_string(index=False, float_format=fmt))
    print(f"  teacher stage time: {seconds:.1f} s")


def stage_dir(out, quick):
    """Student outputs: results/<ds>/seed<k>/ (full) or .../seed<k>/quick/ (--quick).
    The frozen teacher always lives in results/<ds>/seed<k>/ and is shared by both modes."""
    d = out / "quick" if quick else out
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_frozen(out):
    """Splits, frozen teacher and its logits saved by the teacher stage."""
    if not (out / "teacher.json").exists():
        raise FileNotFoundError(f"No frozen teacher in {out}; run the teacher stage first.")
    splits = dict(np.load(out / "splits.npz"))
    logits = dict(np.load(out / "teacher_logits.npz"))
    return load_teacher(out), splits, logits


# ----------------------------------------------------------------------------------------
# Stage: PCE student
# ----------------------------------------------------------------------------------------
def pce_grid(cfg_pce):
    """List of dicts (p, r, q, k, first_order_all), skipping configurations that are duplicates.

    Duplicates: q < 1 changes nothing when r = 1 or p = 1 (a single degree k has q-norm k);
    with r = 1, the screening k only matters if first-order terms are restricted to the k features.
    """
    ks = cfg_pce.get("screening_k") or [None]
    foas = cfg_pce.get("first_order_on_all_features") or [True]
    out, seen = [], set()
    for p, r, q, k, foa in itertools.product(cfg_pce["p"], cfg_pce["r"], cfg_pce["q"], ks, foas):
        if r > p:
            continue
        if q < 1.0 and (r == 1 or p == 1):
            continue
        if k is None:
            foa = True
        if r == 1 and foa:
            k = None                                   # all features, no interactions: k irrelevant
        key = (p, r, q, k, foa)
        if key not in seen:
            seen.add(key)
            out.append(dict(zip(["p", "r", "q", "k", "first_order_all"], key)))
    return out


def stage_pce(name, cfg, X, y, seed, out, quick):
    model, splits, logits = load_frozen(out)
    out = stage_dir(out, quick)
    pool, test = splits["distill"], splits["test"]
    X_pool, X_test = X.iloc[pool], X.iloc[test]
    t_pool, t_test = logits["distill"], logits["test"]

    # Off-manifold test set (same marginals, independence) and the teacher's logits there.
    X_off = off_manifold(X_test, seed)
    t_off = teacher_logit(model, X_off)

    # SHAP on the pool (screening only) and on the test set (explanation, Phase 4).
    t0 = time.perf_counter()
    shap_pool = teacher_shap_global(model, X_pool)
    shap_pool.to_csv(out / "shap_pool.csv", index=False)
    shap_test = teacher_shap_global(model, X_test)
    shap_test.to_csv(out / "shap_test.csv", index=False)
    print(f"  SHAP (pool + test): {time.perf_counter() - t0:.1f} s")

    cfg_pce = cfg["pce_quick"] if quick else cfg["pce"]
    names = list(X.columns)
    rows, fitted, seen_sets = [], {}, {}
    for g in pce_grid(cfg_pce):
        screened = None if g["k"] is None else screening_indices(shap_pool, names, g["k"])
        student = PCEHDMRStudent(p=g["p"], r=g["r"], q=g["q"], screened=screened,
                                 first_order_all=g["first_order_all"], seed=seed,
                                 max_terms=cfg_pce["max_terms"], feature_names=names)
        row = {"dataset": name, "seed": seed, **g, "k": g["k"] if g["k"] else "all"}
        cand = student.candidate_terms(X_pool)
        if cand is not None:                          # skip configs whose candidate set we already fitted
            key = frozenset(cand)
            if key in seen_sets:
                print(f"    PCE p={g['p']} r={g['r']} q={g['q']} k={row['k']}: same candidate set "
                      f"as {seen_sets[key]}, skipped")
                continue
            seen_sets[key] = f"p={g['p']} r={g['r']} q={g['q']} k={row['k']}"
        try:
            student.fit(X_pool, t_pool)               # inputs + teacher logits only
        except ValueError as e:                       # too many candidate terms
            print(f"    skipped: {e}")
            rows.append({**row, "status": "skipped_too_many_terms"})
            continue
        row.update(status="ok", n_candidates=student.n_candidates_, n_terms=student.n_terms,
                   max_order=student.max_order, cv_r2_pool=student.cv_r2_,
                   fit_seconds=student.fit_seconds_)
        for tag, Xs, ts in [("test", X_test, t_test), ("offman", X_off, t_off)]:
            row.update({f"{k}_{tag}": v for k, v in fidelity(ts, student.predict(Xs)).items()})
        row.update({f"{k}_vs_y_test": v
                    for k, v in accuracy_metrics(y[test], student.predict(X_test)).items()})
        rows.append(row)
        fitted[len(rows) - 1] = student
        print(f"      active={student.n_terms} order={student.max_order} "
              f"cv_r2={fmt(student.cv_r2_)} test_r2={fmt(row['r2_logit_test'])} "
              f"offman_r2={fmt(row['r2_logit_offman'])} ({student.fit_seconds_:.1f} s)")

    grid = pd.DataFrame(rows)
    ok = grid[grid.status == "ok"]
    best_i = int(ok["cv_r2_pool"].idxmax())            # chosen by CV on the pool, never by test
    grid["selected"] = False
    grid.loc[best_i, "selected"] = True
    best = fitted[best_i]
    g = grid.loc[best_i]
    print(f"  selected PCE (max CV R^2 on pool): p={g.p} r={g.r} q={g.q} k={g.k} "
          f"first_order_all={g.first_order_all} -> {best.n_terms} terms")

    # Hybrid LARS variant of the selected configuration (reported, not used for selection).
    hyb = PCEHDMRStudent(p=int(g.p), r=int(g.r), q=float(g.q), screened=best.screened,
                         first_order_all=bool(g.first_order_all), ols_refit=True, seed=seed,
                         max_terms=cfg_pce["max_terms"], feature_names=names, verbose=False)
    hyb.fit(X_pool, t_pool)
    hrow = {**grid.loc[best_i].to_dict(), "selected": False, "variant": "hybrid_ols_refit",
            "n_terms": hyb.n_terms, "fit_seconds": hyb.fit_seconds_}
    for tag, Xs, ts in [("test", X_test, t_test), ("offman", X_off, t_off)]:
        hrow.update({f"{k}_{tag}": v for k, v in fidelity(ts, hyb.predict(Xs)).items()})
    hrow.update({f"{k}_vs_y_test": v for k, v in accuracy_metrics(y[test], hyb.predict(X_test)).items()})
    grid["variant"] = "lasso"
    grid = pd.concat([grid, pd.DataFrame([hrow])], ignore_index=True)
    grid.to_csv(out / "pce_grid.csv", index=False)

    # Explanations of the selected student.
    per_feature, pairs, D = best.sobol()
    per_feature.to_csv(out / "pce_sobol.csv", index=False)
    pairs.to_csv(out / "pce_sobol_pairs.csv", index=False)
    best.terms_table().to_csv(out / "pce_terms.csv", index=False)
    vc = pd.DataFrame([{"data": "pool", **best.variance_check()},
                       {"data": "test", **best.variance_check(X_test)},
                       {"data": "offmanifold", **best.variance_check(X_off)}])
    vc.to_csv(out / "pce_variance_check.csv", index=False)

    teacher_acc = accuracy_metrics(y[test], t_test)
    print(f"  teacher on test: acc={fmt(teacher_acc['accuracy'])} auc={fmt(teacher_acc['auc'])}")
    print("  selected + hybrid rows:")
    cols = ["variant", "n_terms", "cv_r2_pool", "r2_logit_test", "rmse_logit_test", "label_agreement_test",
            "mean_kl_test", "r2_logit_offman", "accuracy_vs_y_test", "auc_vs_y_test"]
    print(grid[grid.selected | (grid.variant == "hybrid_ols_refit")][cols]
          .to_string(index=False, float_format=fmt))
    print("  variance check (D = sum c^2 vs empirical variance of predictions):")
    print(vc.to_string(index=False, float_format=fmt))
    print("  top features by PCE total Sobol' index:")
    print(per_feature.sort_values("ST", ascending=False).head(8).to_string(index=False, float_format=fmt))
    if len(pairs):
        print("  top pairs S_ij:")
        print(pairs.head(5).to_string(index=False, float_format=fmt))


# ----------------------------------------------------------------------------------------
def run(name, seeds, quick, stages, retrain_teacher):
    cfg = load_config(name)
    X, y, groups = load_dataset(name)
    for seed in seeds:
        print(f"\n===== {name} seed {seed} {'(quick)' if quick else ''} =====")
        set_global_seeds(seed)
        out = ROOT / "results" / name / f"seed{seed}"
        out.mkdir(parents=True, exist_ok=True)
        if "teacher" in stages:
            stage_teacher(name, cfg, X, y, groups, seed, out, retrain=retrain_teacher)
        if "pce" in stages:
            t0 = time.perf_counter()
            stage_pce(name, cfg, X, y, seed, out, quick)
            print(f"  PCE stage time: {time.perf_counter() - t0:.1f} s")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="all", choices=["magic", "bioresponse", "musk", "all"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--quick", action="store_true", help="small grids, fewer MC samples")
    ap.add_argument("--stages", nargs="+", default=ALL_STAGES, choices=ALL_STAGES)
    ap.add_argument("--retrain-teacher", action="store_true",
                    help="overwrite a saved teacher (normally it is only loaded)")
    args = ap.parse_args()
    names = STUDY_DATASETS if args.dataset == "all" else [args.dataset]
    for name in names:
        run(name, args.seeds, args.quick, args.stages, args.retrain_teacher)


if __name__ == "__main__":
    main()
