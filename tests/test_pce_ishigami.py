"""PCE student on the Ishigami function, a standard sensitivity-analysis benchmark.

f(x) = sin(x1) + a sin^2(x2) + b x3^4 sin(x1),  x ~ U(-pi, pi)^3,  a = 7, b = 0.1.
Known indices: S1 = 0.3139, S2 = 0.4424, S3 = 0, ST1 = 0.5576, ST2 = 0.4424, ST3 = 0.2437.
The student goes through exactly the same code path as on real data (empirical CDF transform,
Legendre basis, LARS-LASSO), so this tests the whole pipeline, not just the formulas.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from kdpce.pce_hdmr import PCEHDMRStudent  # noqa: E402

A, B = 7.0, 0.1
EXPECTED_S1 = np.array([0.3139, 0.4424, 0.0])
EXPECTED_ST = np.array([0.5576, 0.4424, 0.2437])


def ishigami(X):
    return np.sin(X[:, 0]) + A * np.sin(X[:, 1]) ** 2 + B * X[:, 2] ** 4 * np.sin(X[:, 0])


@pytest.fixture(scope="module")
def fitted():
    rng = np.random.default_rng(1)
    X = rng.uniform(-np.pi, np.pi, size=(4000, 3))
    model = PCEHDMRStudent(p=10, r=3, q=1.0, seed=0, verbose=False).fit(X, ishigami(X))
    Xt = rng.uniform(-np.pi, np.pi, size=(4000, 3))
    return model, Xt


def test_ishigami_sobol(fitted):
    model, _ = fitted
    per_feature, _, D = model.sobol()
    np.testing.assert_allclose(per_feature["S1"].to_numpy(), EXPECTED_S1, atol=0.02)
    np.testing.assert_allclose(per_feature["ST"].to_numpy(), EXPECTED_ST, atol=0.02)
    # Analytic variance of Ishigami: a^2/8 + b pi^4/5 + b^2 pi^8/18 + 1/2 = 13.8446
    assert abs(D - 13.8446) / 13.8446 < 0.03


def test_ishigami_accuracy(fitted):
    model, Xt = fitted
    y = ishigami(Xt)
    r2 = 1 - np.mean((model.predict(Xt) - y) ** 2) / y.var()
    assert r2 > 0.99


def test_fit_signature_has_no_labels():
    import inspect
    assert list(inspect.signature(PCEHDMRStudent.fit).parameters) == ["self", "X", "target"]
