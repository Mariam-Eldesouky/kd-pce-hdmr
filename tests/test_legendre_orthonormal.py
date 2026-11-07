"""Orthonormality of the Legendre product basis under Uniform(-1, 1)^2.

Monte Carlo estimate of the Gram matrix G[a, b] = E[Psi_a(Z) Psi_b(Z)] for all multi-indices of
total degree <= 4 in 2 variables (15 basis functions incl. the constant). It should be the identity.
With N = 400,000 samples, the MC standard error of an entry is about sqrt(Var(Psi_a Psi_b)/N);
the products have variance up to ~ 10, so errors stay below ~0.01. Tolerance 0.03.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from kdpce.pce_hdmr import build_terms, design_matrix, legendre_orthonormal  # noqa: E402


def test_univariate_exact_by_quadrature():
    # Gauss-Legendre quadrature with 20 nodes is exact for polynomials of degree <= 39.
    z, w = np.polynomial.legendre.leggauss(20)
    for n in range(8):
        for m in range(8):
            val = 0.5 * np.sum(w * legendre_orthonormal(n, z) * legendre_orthonormal(m, z))
            assert abs(val - (n == m)) < 1e-10


def test_bivariate_gram_monte_carlo():
    rng = np.random.default_rng(0)
    Z = rng.uniform(-1, 1, size=(400_000, 2))
    terms = build_terms(p=4, r=2, q=1.0, first_order_features=[0, 1],
                        interaction_features=[0, 1], max_deg_per_feature=[4, 4])
    Psi = np.column_stack([np.ones(len(Z)), design_matrix(Z, terms, 4)])
    assert Psi.shape[1] == 15                     # C(4 + 2, 2) = 15 terms of total degree <= 4
    G = Psi.T @ Psi / len(Z)
    assert np.max(np.abs(G - np.eye(15))) < 0.03


def test_multi_index_counts():
    # Total degree <= p in d variables, all orders: C(p + d, d) - 1 non-constant terms.
    terms = build_terms(p=3, r=3, q=1.0, first_order_features=range(4),
                        interaction_features=range(4), max_deg_per_feature=[3] * 4)
    assert len(terms) == 35 - 1
    # Interaction order 1 only: d * p terms.
    terms = build_terms(p=3, r=1, q=1.0, first_order_features=range(4),
                        interaction_features=range(4), max_deg_per_feature=[3] * 4)
    assert len(terms) == 12
    # q-norm 0.5 with p = 2 removes the (1, 1) pair: (1^0.5 + 1^0.5)^2 = 4 > 2.
    terms = build_terms(p=2, r=2, q=0.5, first_order_features=range(2),
                        interaction_features=range(2), max_deg_per_feature=[2, 2])
    assert all(len(s) == 1 for s, _ in terms)
    # A binary feature (cap 1) never gets degree 2.
    terms = build_terms(p=3, r=1, q=1.0, first_order_features=[0, 1],
                        interaction_features=[0, 1], max_deg_per_feature=[3, 1])
    assert [t for t in terms if t[0] == (1,)] == [((1,), (1,))]
