"""Dataset loading.

Datasets come from OpenML (downloaded on first use, cached under data/openml_cache).
Each dataset is described by a YAML file in configs/.

`load_dataset(name)` returns
    X       : DataFrame of numeric features only (ID / name columns removed)
    y       : int array of 0/1 labels; which original class is 1 is set in the config
    groups  : array of group labels (Musk: molecule name) or None

`make_splits(X, y, groups, seed)` returns index arrays for the three disjoint subsets
    teacher-train (60%) / distillation pool (20%) / test (20%).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"
CACHE_DIR = REPO_ROOT / "data" / "openml_cache"


def load_config(name):
    """Read configs/<name>.yaml into a dict."""
    with open(CONFIG_DIR / f"{name}.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_dataset(name):
    """Load a dataset by config name ('magic' or 'musk').

    Raises a RuntimeError with a clear message if the download fails, instead of
    silently falling back to something else.
    """
    import openml  # imported here so the rest of the package works without it

    cfg = load_config(name)
    openml.config.set_root_cache_directory(str(CACHE_DIR))
    try:
        ds = openml.datasets.get_dataset(
            cfg["openml_id"],
            download_data=True,
            download_qualities=False,
            download_features_meta_data=False,
        )
        df, target, _, _ = ds.get_data(target=cfg["target"], dataset_format="dataframe")
    except Exception as e:  # network or server error
        raise RuntimeError(
            f"Could not download OpenML dataset {cfg['openml_id']} ({name}): {e}"
        ) from e

    # Groups: kept aside, never used as a feature.
    groups = None
    if cfg.get("group_column"):
        groups = df[cfg["group_column"]].astype(str).to_numpy()
        df = df.drop(columns=[cfg["group_column"]])

    # Drop any other ID / name columns if OpenML returned them.
    df = df.drop(columns=[c for c in cfg.get("drop_columns", []) if c in df.columns])

    # Clean column names (MAGIC's names end with ':').
    df.columns = [str(c).rstrip(":") for c in df.columns]

    # Everything left must be numeric. Fail loudly otherwise.
    non_numeric = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise ValueError(f"{name}: non-numeric feature columns left: {non_numeric}")
    X = df.astype(float).reset_index(drop=True)

    # Binary target: 1 = the configured positive class.
    y = (target.astype(str).to_numpy() == str(cfg["positive_label"])).astype(int)
    if y.sum() == 0 or y.sum() == len(y):
        raise ValueError(f"{name}: positive label {cfg['positive_label']!r} not found")

    return X, y, groups


# ----------------------------------------------------------------------------------------
# Splits
# ----------------------------------------------------------------------------------------
SPLIT_FRACTIONS = {"train": 0.6, "distill": 0.2, "test": 0.2}


def _stratified_split(y, fractions, seed):
    """Row-level stratified split into len(fractions) parts (used when there are no groups)."""
    from sklearn.model_selection import train_test_split

    idx = np.arange(len(y))
    names = list(fractions)
    out, rest, rest_frac = {}, idx, 1.0
    for name in names[:-1]:
        share = fractions[name] / rest_frac          # fraction of what is left
        part, rest = train_test_split(rest, train_size=share, stratify=y[rest], random_state=seed)
        out[name], rest_frac = np.sort(part), rest_frac - fractions[name]
    out[names[-1]] = np.sort(rest)
    return out


def _grouped_split(y, groups, fractions, seed, n_candidates=2000):
    """Group-level split: every group goes entirely to one subset.

    Musk's molecule sizes are very skewed (1 to 1,044 rows), so a single random draw can be far
    from the target row fractions. We therefore draw `n_candidates` random assignments
    (each group goes to subset k with probability fractions[k]) and keep the one with the
    smallest loss:
        loss = sum_k |row share_k - target_k| + sum_k |positive rate_k - overall positive rate|
    Everything is driven by `seed`, so the split is reproducible.
    """
    rng = np.random.default_rng(seed)
    names = list(fractions)
    probs = np.array([fractions[n] for n in names])
    uniq, g_idx = np.unique(groups, return_inverse=True)
    g_rows = np.bincount(g_idx)                       # rows per group
    g_pos = np.bincount(g_idx, weights=y)             # positive rows per group
    n, base_rate = len(y), y.mean()

    best_loss, best_assign = np.inf, None
    for _ in range(n_candidates):
        assign = rng.choice(len(names), size=len(uniq), p=probs)   # subset of each group
        rows = np.bincount(assign, weights=g_rows, minlength=len(names))
        pos = np.bincount(assign, weights=g_pos, minlength=len(names))
        if np.any(pos == 0) or np.any(pos == rows):   # each subset needs both classes
            continue
        loss = np.abs(rows / n - probs).sum() + np.abs(pos / rows - base_rate).sum()
        if loss < best_loss:
            best_loss, best_assign = loss, assign
    if best_assign is None:
        raise RuntimeError("No valid grouped split found; increase n_candidates.")

    row_assign = best_assign[g_idx]
    return {name: np.flatnonzero(row_assign == k) for k, name in enumerate(names)}


def check_splits(splits, n, groups=None):
    """Assert the subsets are disjoint, cover all rows, and (if groups) share no group."""
    parts = list(splits.values())
    all_idx = np.concatenate(parts)
    assert len(all_idx) == len(np.unique(all_idx)), "splits overlap"
    assert len(all_idx) == n, "splits do not cover all rows"
    if groups is not None:
        sets = [set(groups[p]) for p in parts]
        for i in range(len(sets)):
            for j in range(i + 1, len(sets)):
                assert not (sets[i] & sets[j]), "a group appears in two splits"


def make_splits(X, y, groups, seed, fractions=SPLIT_FRACTIONS):
    """Return {'train', 'distill', 'test'} -> sorted row-index arrays (60/20/20).

    Stratified by y when there are no groups; grouped (see `_grouped_split`) otherwise.
    """
    y = np.asarray(y)
    if groups is None:
        splits = _stratified_split(y, fractions, seed)
    else:
        splits = _grouped_split(y, np.asarray(groups), fractions, seed)
    check_splits(splits, len(y), groups)
    return splits


def split_summary(splits, y, groups=None):
    """Small DataFrame with rows, share, positive rate (and n groups) per subset."""
    rows = []
    for name, idx in splits.items():
        r = {"subset": name, "rows": len(idx), "share": len(idx) / len(y),
             "pos_rate": y[idx].mean()}
        if groups is not None:
            r["groups"] = len(np.unique(groups[idx]))
        rows.append(r)
    return pd.DataFrame(rows)
