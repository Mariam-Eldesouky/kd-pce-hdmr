"""Phase 1 data check: load each dataset and print its basic facts.

Run: python scripts/check_data.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from kdpce.data import load_config, load_dataset  # noqa: E402


def report(name):
    cfg = load_config(name)
    X, y, groups = load_dataset(name)
    print(f"===== {name} (OpenML id {cfg['openml_id']}) =====")
    print(f"shape: {X.shape[0]} rows x {X.shape[1]} features")
    print(f"positive class = {cfg['positive_label']!r}: {y.sum()} ({y.mean():.1%}); "
          f"negative: {len(y) - y.sum()} ({1 - y.mean():.1%})")
    print(f"missing values: {int(X.isna().sum().sum())}")
    integer_cols = int((X % 1 == 0).all().sum())
    print(f"feature types: all float64; integer-valued columns: {integer_cols}/{X.shape[1]}")
    print(f"distinct values per feature: min {X.nunique().min()}, "
          f"median {int(X.nunique().median())}, max {X.nunique().max()}")
    print(f"exact duplicate feature rows: {int(X.duplicated().sum())}")
    if groups is not None:
        g = pd.DataFrame({"g": groups, "y": y}).groupby("g")["y"].agg(["nunique", "mean", "size"])
        print(f"groups ({cfg['group_column']}): {len(g)} distinct; "
              f"groups with mixed labels: {(g['nunique'] > 1).sum()}; "
              f"positive groups: {(g['mean'] == 1).sum()}, negative groups: {(g['mean'] == 0).sum()}")
        print(f"rows per group: min {g['size'].min()}, median {g['size'].median():.0f}, "
              f"max {g['size'].max()}")
        top = g["size"].nlargest(3)
        print(f"3 largest groups: {top.to_dict()} -> {top.sum() / len(y):.1%} of rows")
    print()


if __name__ == "__main__":
    for name in ["magic", "musk"]:
        report(name)
