"""Leakage checks.

1. Splits are disjoint and cover all rows (synthetic data and both real datasets, seeds 0-4).
2. Musk: no molecule appears in two subsets.
3. The split is reproducible for a fixed seed.
4. (Phases 3-4) Student fit functions take no label argument. Skipped until they exist.
"""
import inspect
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from kdpce.data import check_splits, load_dataset, make_splits  # noqa: E402


def _synthetic(n=1000, n_groups=60, seed=0):
    rng = np.random.default_rng(seed)
    groups = rng.integers(0, n_groups, n).astype(str)
    y = (rng.random(n) < 0.3).astype(int)
    return np.zeros((n, 2)), y, groups


@pytest.mark.parametrize("use_groups", [False, True])
@pytest.mark.parametrize("seed", range(5))
def test_synthetic_splits_disjoint(use_groups, seed):
    X, y, g = _synthetic()
    g = g if use_groups else None
    sp = make_splits(X, y, g, seed)
    check_splits(sp, len(y), g)                      # raises on overlap / missing rows / shared group
    assert set(sp) == {"train", "distill", "test"}


def test_split_reproducible():
    X, y, g = _synthetic()
    a, b = make_splits(X, y, g, 3), make_splits(X, y, g, 3)
    for k in a:
        np.testing.assert_array_equal(a[k], b[k])


@pytest.fixture(scope="module")
def musk():
    try:
        return load_dataset("musk")
    except RuntimeError as e:
        pytest.skip(f"Musk not available: {e}")


@pytest.mark.parametrize("seed", range(5))
def test_musk_groups_do_not_cross(musk, seed):
    X, y, groups = musk
    sp = make_splits(X, y, groups, seed)
    mols = {k: set(groups[v]) for k, v in sp.items()}
    assert not (mols["train"] & mols["distill"])
    assert not (mols["train"] & mols["test"])
    assert not (mols["distill"] & mols["test"])
    assert sum(len(v) for v in sp.values()) == len(y)


def test_magic_splits_disjoint():
    try:
        X, y, groups = load_dataset("magic")
    except RuntimeError as e:
        pytest.skip(f"MAGIC not available: {e}")
    for seed in range(5):
        check_splits(make_splits(X, y, groups, seed), len(y))


def test_students_fit_without_labels():
    """Student fit signatures must be fit(X, target): no parameter named like a label."""
    students = []
    try:
        from kdpce.pce_hdmr import PCEHDMRStudent
        students.append(PCEHDMRStudent.fit)
    except ImportError:
        pass
    try:
        from kdpce.ebm_student import fit_ebm_student
        students.append(fit_ebm_student)
    except ImportError:
        pass
    if not students:
        pytest.skip("no students implemented yet")
    for f in students:
        params = set(inspect.signature(f).parameters)
        assert not params & {"y", "y_true", "labels", "label"}, f"{f.__name__} takes labels: {params}"
