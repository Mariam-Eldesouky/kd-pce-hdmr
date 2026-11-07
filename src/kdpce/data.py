"""Dataset loading.

Datasets come from OpenML (downloaded on first use, cached under data/openml_cache).
Each dataset is described by a YAML file in configs/.

`load_dataset(name)` returns
    X       : DataFrame of numeric features only (ID / name columns removed)
    y       : int array of 0/1 labels; which original class is 1 is set in the config
    groups  : array of group labels (Musk: molecule name) or None

The split function `make_splits` is added in Phase 2.
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
