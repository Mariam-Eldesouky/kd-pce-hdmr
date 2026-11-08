"""End-to-end experiment driver.

Examples
    python scripts/run_all.py --dataset magic --seeds 0 --stages teacher
    python scripts/run_all.py --dataset all --quick --seeds 0
    python scripts/run_all.py --dataset all --seeds 0 1 2 3 4

Datasets: magic (moderate, 10 features) and bioresponse (high-dimensional, 1,776 features).
`--dataset all` runs these two. Musk can still be run by name but is not part of the study
(its teacher did not generalise to unseen molecules; see the README).

Stages: teacher, pce, ebm, explain (explain needs the pce and ebm outputs).
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

from kdpce.aggregate import aggregate  # noqa: E402
from kdpce.data import load_config, load_dataset, make_splits, split_summary  # noqa: E402
from kdpce.ebm_student import ebm_complexity, ebm_variance_shares, fit_ebm_student  # noqa: E402
from kdpce.explain import (agreement_matrix, rank_agreement, screening_indices,  # noqa: E402
                           teacher_mc_sobol, teacher_shap_global)
from kdpce.metrics import accuracy_metrics, fidelity, off_manifold  # noqa: E402
from kdpce.pce_hdmr import PCEHDMRStudent  # noqa: E402
from kdpce.teacher import (load_teacher, save_teacher, teacher_logit,  # noqa: E402
                           teacher_metrics, train_teacher)

ALL_STAGES = ["teacher", "pce", "ebm", "explain"]
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

    def make_student(g, **extra):
        screened = None if g["k"] in (None, "all") else screening_indices(shap_pool, names, int(g["k"]))
        return PCEHDMRStudent(p=int(g["p"]), r=int(g["r"]), q=float(g["q"]), screened=screened,
                              first_order_all=bool(g["first_order_all"]), seed=seed,
                              max_terms=cfg_pce["max_terms"], feature_names=names, **extra)

    # Each configuration's row is saved as soon as it is done (status ok / duplicate / skipped),
    # so an interrupted run resumes where it stopped. Fits are deterministic given the seed.
    partial = out / "pce_grid_partial.csv"
    keycols = ["p", "r", "q", "k", "first_order_all"]
    rows = pd.read_csv(partial).to_dict("records") if partial.exists() else []
    done = {tuple(str(r[c]) for c in keycols) for r in rows}
    seen_sets = {}
    best_student, best_cv = None, -np.inf             # keep only the best student in memory
    for g in pce_grid(cfg_pce):
        row = {"dataset": name, "seed": seed, **g, "k": g["k"] if g["k"] else "all"}
        student = make_student(row)
        cand = student.candidate_terms(X_pool)
        key = frozenset(cand) if cand is not None else None
        label = f"p={g['p']} r={g['r']} q={g['q']} k={row['k']} foa={g['first_order_all']}"
        if key is not None and key in seen_sets:      # same candidate set as a config already seen
            print(f"    PCE {label}: same candidate set as {seen_sets[key]}, skipped")
            if tuple(str(row[c]) for c in keycols) not in done:
                rows.append({**row, "status": "duplicate"})
                pd.DataFrame(rows).to_csv(partial, index=False)
            continue
        if key is not None:
            seen_sets[key] = label
        if tuple(str(row[c]) for c in keycols) in done:
            print(f"    PCE {label}: already done (from {partial.name}), skipped")
            continue
        try:
            student.fit(X_pool, t_pool)               # inputs + teacher logits only
        except ValueError as e:                       # too many candidate terms
            print(f"    skipped: {e}")
            rows.append({**row, "status": "skipped_too_many_terms"})
            pd.DataFrame(rows).to_csv(partial, index=False)
            continue
        row.update(status="ok", n_candidates=student.n_candidates_, n_terms=student.n_terms,
                   max_order=student.max_order, cv_r2_pool=student.cv_r2_,
                   fit_seconds=student.fit_seconds_)
        for tag, Xs, ts in [("test", X_test, t_test), ("offman", X_off, t_off)]:
            row.update({f"{k}_{tag}": v for k, v in fidelity(ts, student.predict(Xs)).items()})
        row.update({f"{k}_vs_y_test": v
                    for k, v in accuracy_metrics(y[test], student.predict(X_test)).items()})
        rows.append(row)
        pd.DataFrame(rows).to_csv(partial, index=False)
        if student.cv_r2_ > best_cv:
            best_student, best_cv = student, student.cv_r2_
        print(f"      active={student.n_terms} order={student.max_order} "
              f"cv_r2={fmt(student.cv_r2_)} test_r2={fmt(row['r2_logit_test'])} "
              f"offman_r2={fmt(row['r2_logit_offman'])} ({student.fit_seconds_:.1f} s)", flush=True)

    grid = pd.DataFrame(rows)
    grid["k"] = grid["k"].astype(str)
    ok = grid[grid.status == "ok"]
    best_i = int(ok["cv_r2_pool"].idxmax())            # chosen by CV on the pool, never by test
    grid["selected"] = False
    grid.loc[best_i, "selected"] = True
    g = grid.loc[best_i]
    if best_student is not None and np.isclose(best_cv, g.cv_r2_pool):
        best = best_student
    else:                                              # best config came from an interrupted run
        best = make_student(g, verbose=False).fit(X_pool, t_pool)
    print(f"  selected PCE (max CV R^2 on pool): p={g.p} r={g.r} q={g.q} k={g.k} "
          f"first_order_all={g.first_order_all} -> {best.n_terms} terms")
    np.savez(out / "pce_predictions.npz", test=best.predict(X_test), offman=best.predict(X_off),
             teacher_test=t_test, teacher_offman=t_off)

    # Hybrid LARS variant of the selected configuration (reported, not used for selection).
    hyb = make_student(g, ols_refit=True, verbose=False)
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
# Stage: EBM student
# ----------------------------------------------------------------------------------------
def stage_ebm(name, cfg, X, y, seed, out, quick):
    from sklearn.model_selection import KFold

    model, splits, logits = load_frozen(out)
    out = stage_dir(out, quick)
    pool, test = splits["distill"], splits["test"]
    X_pool, X_test = X.iloc[pool], X.iloc[test]
    t_pool, t_test = logits["distill"], logits["test"]
    X_off = off_manifold(X_test, seed)                 # same off-manifold set as the PCE stage
    t_off = teacher_logit(model, X_off)

    cfg_ebm = cfg["ebm_quick"] if quick else cfg["ebm"]
    kwargs = cfg_ebm.get("kwargs") or {}
    folds = list(KFold(cfg_ebm["cv_folds"], shuffle=True, random_state=seed).split(X_pool))

    # Each K's row is saved as soon as it is done, so an interrupted run loses at most one K.
    # A rerun skips the Ks already in the partial file (EBM fits are deterministic given the seed).
    partial = out / "ebm_grid_partial.csv"
    rows = pd.read_csv(partial).to_dict("records") if partial.exists() else []
    done = {int(r["interactions"]) for r in rows}
    best_model, best_cv = None, -np.inf               # keep only the best model in memory
    for K in cfg_ebm["interactions"]:
        if K in done:
            print(f"    EBM K={K}: already done (from {partial.name}), skipped")
            continue
        # CV fidelity on the pool (used to choose K; the test set is not used).
        t0 = time.perf_counter()
        sse = 0.0
        for tr, va in folds:
            m, _ = fit_ebm_student(X_pool.iloc[tr], t_pool[tr], K, seed, **kwargs)
            sse += np.sum((m.predict(X_pool.iloc[va]) - t_pool[va]) ** 2)
        cv_r2 = 1 - sse / np.sum((t_pool - t_pool.mean()) ** 2)
        cv_seconds = time.perf_counter() - t0
        # Final fit on the whole pool.
        m, fit_s = fit_ebm_student(X_pool, t_pool, K, seed, **kwargs)
        row = {"dataset": name, "seed": seed, "interactions": K, "cv_r2_pool": cv_r2,
               "fit_seconds": fit_s, "cv_seconds": cv_seconds, **ebm_complexity(m)}
        for tag, Xs, ts in [("test", X_test, t_test), ("offman", X_off, t_off)]:
            row.update({f"{k}_{tag}": v for k, v in fidelity(ts, m.predict(Xs)).items()})
        row.update({f"{k}_vs_y_test": v for k, v in accuracy_metrics(y[test], m.predict(X_test)).items()})
        rows.append(row)
        pd.DataFrame(rows).to_csv(partial, index=False)
        if cv_r2 > best_cv:
            best_model, best_cv = m, cv_r2
        del m
        print(f"    EBM K={K}: terms={row['n_terms']} params={row['n_parameters']} cv_r2={fmt(cv_r2)} "
              f"test_r2={fmt(row['r2_logit_test'])} offman_r2={fmt(row['r2_logit_offman'])} "
              f"(fit {fit_s:.1f} s, CV {cv_seconds:.1f} s)", flush=True)
    grid = pd.DataFrame(rows)
    grid = grid[grid["interactions"].isin(cfg_ebm["interactions"])].reset_index(drop=True)
    best_K = int(grid.loc[grid["cv_r2_pool"].idxmax(), "interactions"])
    grid["selected"] = grid["interactions"] == best_K
    grid.to_csv(out / "ebm_grid.csv", index=False)

    if best_model is not None and best_cv == grid["cv_r2_pool"].max():
        best = best_model
    else:                                              # best K came from an earlier, interrupted run
        best, _ = fit_ebm_student(X_pool, t_pool, best_K, seed, **kwargs)
    np.savez(out / "ebm_predictions.npz", test=best.predict(X_test), offman=best.predict(X_off),
             teacher_test=t_test, teacher_offman=t_off)
    per_feature, per_term, check = ebm_variance_shares(best, X_test, list(X.columns))
    per_feature.to_csv(out / "ebm_importance.csv", index=False)
    per_term.to_csv(out / "ebm_terms.csv", index=False)
    pd.DataFrame([check]).to_csv(out / "ebm_variance_check.csv", index=False)
    print(f"  selected EBM (max CV R^2 on pool): K={best_K}")
    cols = ["interactions", "n_terms", "n_parameters", "cv_r2_pool", "r2_logit_test", "label_agreement_test",
            "mean_kl_test", "r2_logit_offman", "accuracy_vs_y_test", "auc_vs_y_test"]
    print(grid[cols].to_string(index=False, float_format=fmt))
    print(f"  sum of term variances / Var(prediction) on test: {check['ratio_sum_term_var_over_var_pred']:.4f}")
    print("  top features by EBM total variance share (test):")
    print(per_feature.sort_values("total", ascending=False).head(8).to_string(index=False, float_format=fmt))
    print("  top terms:")
    print(per_term.head(5).to_string(index=False, float_format=fmt))


# ----------------------------------------------------------------------------------------
# Stage: explanations (teacher MC Sobol', agreement) and the combined fidelity table
# ----------------------------------------------------------------------------------------
def stage_explain(name, cfg, X, y, seed, out, quick):
    model, splits, logits = load_frozen(out)
    out = stage_dir(out, quick)
    pool, test = splits["distill"], splits["test"]
    X_pool, X_test = X.iloc[pool], X.iloc[test]
    names = list(X.columns)

    # Teacher MC Sobol' on independent resamples of the pool's marginals.
    N = cfg["mc_sobol"]["N_quick"] if quick else cfg["mc_sobol"]["N"]
    t0 = time.perf_counter()
    mc, V = teacher_mc_sobol(model, X_pool, N, seed)
    mc_seconds = time.perf_counter() - t0
    mc.to_csv(out / "teacher_mc_sobol.csv", index=False)
    print(f"  teacher MC Sobol': N={N}, d={len(names)}, {N * (len(names) + 2):,} teacher calls, "
          f"{mc_seconds:.1f} s, Var(logit) under independent inputs = {V:.4f} "
          f"(on test data: {np.var(logits['test']):.4f})")
    print(mc.sort_values("ST", ascending=False).head(8).to_string(index=False, float_format=fmt))
    print(f"  sum of first-order indices: {mc.S1.sum():.4f} (1 - sum = share of variance from interactions)")

    # Collect every explanation as one importance score per feature.
    pce = pd.read_csv(out / "pce_sobol.csv").set_index("feature")
    ebm = pd.read_csv(out / "ebm_importance.csv").set_index("feature")
    shap_test = pd.read_csv(out / "shap_test.csv").set_index("feature")
    mc_i = mc.set_index("feature")
    scores = {"PCE_ST": pce["ST"], "PCE_S1": pce["S1"], "EBM_total": ebm["total"],
              "EBM_first": ebm["first"], "EBM_meanabs": ebm["mean_abs_contrib"],
              "SHAP": shap_test["mean_abs_shap"], "MC_ST": mc_i["ST"], "MC_S1": mc_i["S1"]}
    scores = {k: v.reindex(names) for k, v in scores.items()}
    expl = pd.concat([pd.DataFrame({"dataset": name, "seed": seed, "method": k, "feature": v.index,
                                    "score": v.to_numpy(),
                                    "rank": v.rank(ascending=False, method="min").to_numpy()})
                      for k, v in scores.items()], ignore_index=True)
    expl.to_csv(out / "explanations.csv", index=False)

    agree = rank_agreement(scores)
    agree.insert(0, "seed", seed)
    agree.insert(0, "dataset", name)
    agree.to_csv(out / "agreement.csv", index=False)
    agreement_matrix(agree, "kendall_tau_all").to_csv(out / "agreement_matrix_tau_all.csv")
    agreement_matrix(agree, "kendall_tau_top").to_csv(out / "agreement_matrix_tau_top.csv")
    key = ["PCE_ST", "EBM_total", "SHAP", "MC_ST"]
    sel = agree[agree.method_a.isin(key) & agree.method_b.isin(key)]
    print("  agreement between the main methods:")
    print(sel[["method_a", "method_b", "kendall_tau_all", "kendall_tau_top", "top5_overlap", "top10_overlap"]]
          .to_string(index=False, float_format=fmt))

    # Combined fidelity table: teacher + selected PCE + selected EBM.
    pg = pd.read_csv(out / "pce_grid.csv")
    eg = pd.read_csv(out / "ebm_grid.csv")
    p = pg[pg.selected].iloc[0]
    e = eg[eg.selected].iloc[0]
    tacc = accuracy_metrics(y[test], logits["test"])
    metric_cols = [c for c in pg.columns if c.endswith(("_test", "_offman"))]
    fid = pd.DataFrame([
        {"dataset": name, "seed": seed, "model": "teacher", "config": "xgboost",
         "accuracy_vs_y_test": tacc["accuracy"], "auc_vs_y_test": tacc["auc"]},
        {"dataset": name, "seed": seed, "model": "PCE", "config": f"p={p.p} r={p.r} q={p.q} k={p.k}",
         "n_terms": p.n_terms, "max_order": p.max_order, "cv_r2_pool": p.cv_r2_pool,
         "fit_seconds": p.fit_seconds, **{c: p[c] for c in metric_cols}},
        {"dataset": name, "seed": seed, "model": "EBM", "config": f"K={e.interactions}",
         "n_terms": e.n_terms, "n_parameters": e.n_parameters, "max_order": e.max_order,
         "cv_r2_pool": e.cv_r2_pool, "fit_seconds": e.fit_seconds, **{c: e[c] for c in metric_cols}},
    ])
    fid.to_csv(out / "fidelity.csv", index=False)
    print("  fidelity (selected students):")
    print(fid[["model", "config", "n_terms", "r2_logit_test", "rmse_logit_test", "label_agreement_test",
               "mean_kl_test", "r2_logit_offman", "accuracy_vs_y_test", "auc_vs_y_test"]]
          .to_string(index=False, float_format=fmt))
    return mc_seconds


def record_runtime(out, quick, name, seed, stage, seconds, resumed=False):
    """Append/replace one row of runtime.csv in the stage folder.
    resumed=True: the stage continued from a partial file, so `seconds` is only the remaining work."""
    path = stage_dir(out, quick) / "runtime.csv"
    df = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=["dataset", "seed", "stage", "seconds", "resumed"])
    old = df[df.stage == stage]
    if resumed and len(old) and not bool(old["resumed"].iloc[0]):
        return                                         # keep the complete (non-resumed) measurement
    df = df[df.stage != stage]
    df = pd.concat([df, pd.DataFrame([{"dataset": name, "seed": seed, "stage": stage, "seconds": seconds, "resumed": resumed}])])
    df.to_csv(path, index=False)


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
        for stage, fn in [("pce", stage_pce), ("ebm", stage_ebm), ("explain", stage_explain)]:
            if stage in stages:
                print(f"  --- stage {stage} ---")
                # A stage that resumes from a partial file only times the remaining work; flag it.
                resumed = (stage_dir(out, quick) / f"{stage}_grid_partial.csv").exists()
                t0 = time.perf_counter()
                fn(name, cfg, X, y, seed, out, quick)
                seconds = time.perf_counter() - t0
                record_runtime(out, quick, name, seed, stage, seconds, resumed)
                print(f"  {stage} stage time: {seconds:.1f} s{' (resumed run)' if resumed else ''}")
    summary = aggregate(name, quick)                   # over every seed found on disk
    print(f"\n  aggregated {name}: results/summary_{name}{'_quick' if quick else ''}*.csv "
          f"({summary['n_seeds'].max() if len(summary) else 0} seed(s))")


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
