"""Teacher: an XGBoost classifier that is trained once per seed, frozen, and only loaded afterwards.

Procedure (all on the teacher-train subset; the distillation pool and test set are never touched):
  1. Build 5 cross-validation folds of teacher-train: StratifiedKFold, or StratifiedGroupKFold
     when there are groups (Musk), so no molecule is in both the fitting and validation parts.
     (A single 15% validation slice was tried first. On Musk it held only 7 molecules, 2 musk,
     and early stopping kept 1 tree. See explanations/02_teacher.md.)
  2. For each (max_depth, learning_rate) in a small grid, run `xgb.cv` on those folds with early
     stopping on the mean validation AUC. Record the best number of trees and its mean CV AUC.
  3. Pick the configuration with the highest mean CV AUC.
  4. Refit that configuration on the whole teacher-train subset with that number of trees.
  5. Save it as JSON and store the SHA-256 hash of the file. `load_teacher` checks the hash, so
     any change to the frozen model is detected.

Distillation target: the logit, log(p / (1 - p)), from `predict(..., output_margin=True)`.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score

from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

N_FOLDS = 5


def _base_params(seed):
    # Default subsample/colsample = 1.0, so training is deterministic given the seed.
    return dict(objective="binary:logistic", tree_method="hist", random_state=seed, n_jobs=-1)


def cv_folds(y, groups, seed, n_folds=N_FOLDS):
    """List of (fit_idx, val_idx); grouped when groups is not None."""
    if groups is None:
        cv = StratifiedKFold(n_folds, shuffle=True, random_state=seed)
        return list(cv.split(np.zeros(len(y)), y))
    cv = StratifiedGroupKFold(n_folds, shuffle=True, random_state=seed)
    return list(cv.split(np.zeros(len(y)), y, groups))


def train_teacher(X_train, y_train, groups_train, cfg_teacher, seed, log=print):
    """Tune by (grouped) 5-fold CV inside teacher-train, then refit on all of teacher-train.

    Returns (model, info dict, grid DataFrame).
    """
    y_train = np.asarray(y_train)
    folds = cv_folds(y_train, groups_train, seed)
    dtrain = xgb.DMatrix(X_train, label=y_train)

    rows = []
    for depth in cfg_teacher["max_depth"]:
        for lr in cfg_teacher["learning_rate"]:
            params = {"objective": "binary:logistic", "tree_method": "hist", "max_depth": depth,
                      "eta": lr, "seed": seed, "eval_metric": ["logloss", "auc"]}
            # Early stopping uses the LAST metric in eval_metric (auc, maximised).
            hist = xgb.cv(params, dtrain, num_boost_round=cfg_teacher["max_n_estimators"],
                          folds=folds, early_stopping_rounds=cfg_teacher["early_stopping_rounds"],
                          seed=seed, verbose_eval=False)
            best = hist.iloc[-1]                # xgb.cv truncates the history at the best round
            rows.append({"max_depth": depth, "learning_rate": lr, "n_estimators": len(hist),
                         "cv_auc": best["test-auc-mean"], "cv_auc_std": best["test-auc-std"],
                         "cv_logloss": best["test-logloss-mean"]})
            log(f"  teacher grid depth={depth} lr={lr}: trees={len(hist)} "
                f"cv_auc={rows[-1]['cv_auc']:.4f}±{rows[-1]['cv_auc_std']:.4f} "
                f"cv_logloss={rows[-1]['cv_logloss']:.4f}")
    grid = pd.DataFrame(rows)
    best = grid.sort_values(["cv_auc", "cv_logloss"], ascending=[False, True]).iloc[0]

    model = xgb.XGBClassifier(max_depth=int(best.max_depth), learning_rate=float(best.learning_rate),
                              n_estimators=int(best.n_estimators), **_base_params(seed))
    model.fit(X_train, y_train, verbose=False)
    info = {"max_depth": int(best.max_depth), "learning_rate": float(best.learning_rate),
            "n_estimators": int(best.n_estimators), "cv_auc": float(best.cv_auc),
            "cv_auc_std": float(best.cv_auc_std), "n_folds": N_FOLDS,
            "grouped_cv": groups_train is not None, "seed": seed}
    return model, info, grid


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_teacher(model, info, out_dir):
    """Write teacher.json, teacher.sha256 and teacher_info.json; return the hash."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "teacher.json"
    model.save_model(path)
    digest = sha256_of(path)
    (out_dir / "teacher.sha256").write_text(digest + "\n")
    (out_dir / "teacher_info.json").write_text(json.dumps({**info, "sha256": digest}, indent=2))
    return digest


def load_teacher(out_dir):
    """Load the frozen teacher, refusing if the file hash differs from the stored one."""
    out_dir = Path(out_dir)
    path = out_dir / "teacher.json"
    expected = (out_dir / "teacher.sha256").read_text().strip()
    actual = sha256_of(path)
    if actual != expected:
        raise RuntimeError(f"Teacher hash mismatch in {out_dir}: {actual} != {expected}")
    model = xgb.XGBClassifier()
    model.load_model(path)
    return model


def teacher_logit(model, X):
    """The teacher's raw margin (log-odds). This is the students' regression target."""
    return model.predict(X, output_margin=True)


def teacher_metrics(model, X, y):
    """Accuracy (threshold 0.5) and ROC-AUC against true labels."""
    p = model.predict_proba(X)[:, 1]
    return {"accuracy": accuracy_score(y, (p >= 0.5).astype(int)), "auc": roc_auc_score(y, p)}
