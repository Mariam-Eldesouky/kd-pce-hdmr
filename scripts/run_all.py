"""End-to-end experiment driver.

Examples
    python scripts/run_all.py --dataset magic --seeds 0 --stages teacher
    python scripts/run_all.py --dataset all --quick --seeds 0
    python scripts/run_all.py --dataset all --seeds 0 1 2 3 4

Stages are added phase by phase. Implemented: teacher.
Each (dataset, seed) writes to results/<dataset>/seed<k>/.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kdpce.data import load_config, load_dataset, make_splits, split_summary  # noqa: E402
from kdpce.teacher import (load_teacher, save_teacher, teacher_logit,  # noqa: E402
                           teacher_metrics, train_teacher)

ALL_STAGES = ["teacher"]


def set_global_seeds(seed):
    """numpy's global RNG; xgboost/sklearn/EBM get `random_state=seed` explicitly."""
    np.random.seed(seed)


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
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"  teacher stage time: {seconds:.1f} s")
    return model, splits, logits


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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="all", choices=["magic", "musk", "bioresponse", "all"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--quick", action="store_true", help="small grids, fewer MC samples")
    ap.add_argument("--stages", nargs="+", default=ALL_STAGES, choices=ALL_STAGES)
    ap.add_argument("--retrain-teacher", action="store_true",
                    help="overwrite a saved teacher (normally it is only loaded)")
    args = ap.parse_args()
    names = ["magic", "musk"] if args.dataset == "all" else [args.dataset]
    for name in names:
        run(name, args.seeds, args.quick, args.stages, args.retrain_teacher)


if __name__ == "__main__":
    main()
