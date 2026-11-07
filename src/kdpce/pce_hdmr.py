"""Sparse Polynomial Chaos Expansion organised as an HDMR (Student 1).

Pipeline
    x  --(empirical CDF, per feature)-->  u in (0,1)  -->  z = 2u - 1 in (-1,1)
    z  --(orthonormal Legendre products)-->  design matrix  Psi[:, alpha]
    teacher logit  ~  c_0 + sum_alpha c_alpha Psi_alpha(z)     (sparse fit: LARS-LASSO)
    Sobol' indices  <-  squared coefficients c_alpha^2

A multi-index alpha is stored *sparsely* as (support, degrees):
    support = tuple of feature indices with a non-zero degree, e.g. (2, 7)
    degrees = tuple of the matching degrees,                     e.g. (1, 3)
meaning Psi_alpha(z) = psi_1(z_2) * psi_3(z_7). This avoids d-long vectors (d = 1,776 for Bioresponse).
The HDMR structure is visible directly: |support| = 1 are the f_j terms, |support| = 2 the f_ij terms, ...

Reference: Blatman & Sudret (2011), "Adaptive sparse polynomial chaos expansion based on least
angle regression", J. Comput. Phys. 230, 2345-2367. Sudret (2008) for PCE-based Sobol' indices.
"""
import itertools
import math
import time

import numpy as np
import pandas as pd
from scipy.special import eval_legendre
from sklearn.linear_model import LassoLarsCV
from sklearn.model_selection import KFold


# ----------------------------------------------------------------------------------------
# 1. Input transform
# ----------------------------------------------------------------------------------------
class EmpiricalCDFTransform:
    """Map each feature to z in (-1, 1) through its empirical CDF (fitted on the distillation pool).

    For a feature with sorted distinct values v_1 < ... < v_m, observed n_1, ..., n_m times out of n:
        u(v_k) = (n_1 + ... + n_{k-1} + 0.5 * n_k) / n          ("mid-rank" CDF)
    Ties: all copies of a value get the *same* u, the midpoint of the CDF jump at that value.
    This is the (rank - 0.5)/n rule extended to ties; it never gives exactly 0 or 1.
    Unseen values: linear interpolation between neighbouring v_k; values outside [v_1, v_m] are
    clipped to u(v_1) or u(v_m).
    Finally z = 2u - 1.

    What this does and does not do:
      * It makes each feature's MARGINAL distribution (approximately) Uniform(-1, 1) on the pool,
        which is the measure under which Legendre polynomials are orthonormal.
      * For a discrete feature with m values, z takes only m values, so the margin is a discrete
        approximation of a uniform (for a binary feature, two points). Polynomials of degree >= m
        in such a feature are redundant, so the multi-index set caps the degree at m - 1.
      * It does NOT remove dependence between features. The joint distribution of z is not the
        product of uniforms, so products of polynomials in different features are not exactly
        orthogonal on the data. Consequence: the analytic Sobol' indices describe the student
        under an *independent-uniform* input model, which is not the data distribution
        (see `PCEHDMRStudent.variance_check`).
    """

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        n = X.shape[0]
        self.values_, self.u_ = [], []
        for j in range(X.shape[1]):
            v, counts = np.unique(X[:, j], return_counts=True)
            below = np.concatenate([[0], np.cumsum(counts)[:-1]])  # rows strictly below v_k
            self.values_.append(v)
            self.u_.append((below + 0.5 * counts) / n)
        self.n_unique_ = np.array([len(v) for v in self.values_])
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        Z = np.empty_like(X)
        for j in range(X.shape[1]):
            if self.n_unique_[j] == 1:
                Z[:, j] = 0.0
            else:
                Z[:, j] = 2.0 * np.interp(X[:, j], self.values_[j], self.u_[j]) - 1.0
        return Z


# ----------------------------------------------------------------------------------------
# 2. Orthonormal Legendre basis
# ----------------------------------------------------------------------------------------
def legendre_orthonormal(n, z):
    """psi_n(z) = sqrt(2n + 1) * P_n(z).

    Under z ~ Uniform(-1, 1):  E[psi_n psi_m] = (1/2) * int_{-1}^{1} psi_n psi_m dz = delta_nm.
    (The classical P_n satisfy int P_n^2 dz = 2/(2n+1); the uniform density 1/2 and the factor
    sqrt(2n+1) cancel that, giving unit variance. psi_0 = 1.)
    """
    return np.sqrt(2 * n + 1) * eval_legendre(n, z)


def univariate_table(Z, features, max_degree):
    """{feature j: array (n, max_degree+1) with psi_0..psi_p of z_j} for the given features."""
    return {j: np.column_stack([legendre_orthonormal(k, Z[:, j]) for k in range(max_degree + 1)])
            for j in features}


def design_matrix(Z, terms, max_degree):
    """Columns Psi_alpha(z) = prod_j psi_{alpha_j}(z_j) for every (support, degrees) in `terms`."""
    feats = sorted({j for supp, _ in terms for j in supp})
    tab = univariate_table(Z, feats, max_degree)
    Psi = np.ones((Z.shape[0], len(terms)))
    for col, (supp, degs) in enumerate(terms):
        for j, k in zip(supp, degs):
            Psi[:, col] *= tab[j][:, k]
    return Psi


# ----------------------------------------------------------------------------------------
# 3. Multi-index set (truncation)
# ----------------------------------------------------------------------------------------
def _degree_tuples(s, p, q):
    """All degree tuples of length s with every entry >= 1, total degree <= p and q-norm <= p."""
    out = []
    for degs in itertools.product(range(1, p + 1), repeat=s):
        if sum(degs) > p:
            continue
        if q < 1.0 and sum(k ** q for k in degs) ** (1.0 / q) > p + 1e-9:
            continue
        out.append(degs)
    return out


def count_terms(p, r, q, first_order_features, interaction_features, max_deg_per_feature):
    """Number of terms without building them (used for the size check)."""
    n = 0
    for s in range(1, r + 1):
        feats = first_order_features if s == 1 else interaction_features
        degs = _degree_tuples(s, p, q)
        if s == 1:
            n += sum(sum(1 for (k,) in degs if k <= max_deg_per_feature[j]) for j in feats)
        else:  # upper bound would be C(len, s) * len(degs); count exactly, it is cheap for k <= 50
            for supp in itertools.combinations(feats, s):
                caps = [max_deg_per_feature[j] for j in supp]
                n += sum(all(k <= c for k, c in zip(d, caps)) for d in degs)
    return n


def build_terms(p, r, q, first_order_features, interaction_features, max_deg_per_feature):
    """List of (support, degrees), alpha != 0, sorted by order then degree.

    Built from supports (sets of features of size s <= r) and degree tuples, never by
    enumerating {0..p}^d. First-order terms use `first_order_features`; terms of order >= 2
    only use `interaction_features` (the screened set on high-dimensional data).
    Degree of feature j is capped at max_deg_per_feature[j] (= distinct values - 1).
    """
    terms = []
    for s in range(1, r + 1):
        feats = sorted(first_order_features if s == 1 else interaction_features)
        degs = _degree_tuples(s, p, q)
        for supp in itertools.combinations(feats, s):
            caps = [max_deg_per_feature[j] for j in supp]
            for d in degs:
                if all(k <= c for k, c in zip(d, caps)):
                    terms.append((supp, d))
    return terms


# ----------------------------------------------------------------------------------------
# 4. The student
# ----------------------------------------------------------------------------------------
class PCEHDMRStudent:
    """Sparse PCE/HDMR student.

    Parameters
    p, r, q        : total degree, maximum interaction order, hyperbolic q-norm (q = 1: plain total degree)
    screened       : optional list of feature indices allowed in interaction terms
    first_order_all: if screened is given, whether first-order terms use all features (True) or only
                     the screened ones (False)
    ols_refit      : "hybrid LARS": after LASSO selects the support, refit those terms by ordinary
                     least squares (removes LASSO's shrinkage of the kept coefficients)
    max_terms      : refuse to build a candidate set larger than this
    """

    def __init__(self, p=3, r=2, q=1.0, screened=None, first_order_all=True, ols_refit=False,
                 max_terms=20000, cv_folds=5, seed=0, feature_names=None, verbose=True):
        self.p, self.r, self.q = p, r, q
        self.screened = None if screened is None else list(screened)
        self.first_order_all = first_order_all
        self.ols_refit = ols_refit
        self.max_terms, self.cv_folds, self.seed = max_terms, cv_folds, seed
        self.feature_names = feature_names
        self.verbose = verbose

    # --- fitting ------------------------------------------------------------------------
    def _feature_sets(self, d):
        inter = list(range(d)) if self.screened is None else self.screened
        first = list(range(d)) if (self.screened is None or self.first_order_all) else self.screened
        return first, inter

    def candidate_terms(self, X):
        """The candidate multi-index set this configuration would use on X (no fitting).
        Used by the experiment driver to skip configurations with identical candidate sets."""
        n_unique = np.asarray(pd.DataFrame(np.asarray(X)).nunique())
        first, inter = self._feature_sets(len(n_unique))
        if count_terms(self.p, self.r, self.q, first, inter, np.minimum(n_unique - 1, self.p)) > self.max_terms:
            return None
        return build_terms(self.p, self.r, self.q, first, inter, np.minimum(n_unique - 1, self.p))

    def fit(self, X, target):
        """Fit on (inputs, teacher logits). There is deliberately no label argument."""
        t0 = time.perf_counter()
        X = np.asarray(X, dtype=float)
        target = np.asarray(target, dtype=float)
        d = X.shape[1]
        if self.feature_names is None:
            self.feature_names = [f"x{j}" for j in range(d)]

        self.transform_ = EmpiricalCDFTransform().fit(X)
        Z = self.transform_.transform(X)
        max_deg = np.minimum(self.transform_.n_unique_ - 1, self.p)

        first, inter = self._feature_sets(d)
        self.n_candidates_ = count_terms(self.p, self.r, self.q, first, inter, max_deg)
        if self.verbose:
            print(f"    PCE p={self.p} r={self.r} q={self.q} "
                  f"k={'all' if self.screened is None else len(self.screened)}: "
                  f"{self.n_candidates_} candidate terms")
        if self.n_candidates_ > self.max_terms:
            raise ValueError(f"{self.n_candidates_} candidate terms > max_terms={self.max_terms}. "
                             "Lower p or r, use q < 1, or screen features.")
        candidates = build_terms(self.p, self.r, self.q, first, inter, max_deg)
        Psi = design_matrix(Z, candidates, self.p)

        # LASSO path by LARS; the penalty is chosen by K-fold CV on this (pool) data only.
        # fit_intercept=True: the intercept plays the role of c_0 (Psi_0 = 1), i.e. f0.
        lars = LassoLarsCV(fit_intercept=True, cv=KFold(self.cv_folds, shuffle=True,
                                                        random_state=self.seed), max_iter=1000)
        lars.fit(Psi, target)
        keep = np.flatnonzero(lars.coef_ != 0)
        coef, intercept = lars.coef_[keep], lars.intercept_
        # CV mean squared error at the chosen penalty -> CV R^2 (used to choose p, r, q, k)
        self.cv_mse_ = float(lars.mse_path_.mean(axis=1).min())
        self.cv_r2_ = 1.0 - self.cv_mse_ / target.var()
        self.alpha_ = float(lars.alpha_)

        if self.ols_refit and 0 < len(keep) < len(target) - 1:
            A = np.column_stack([np.ones(len(target)), Psi[:, keep]])
            sol, *_ = np.linalg.lstsq(A, target, rcond=None)
            intercept, coef = sol[0], sol[1:]

        self.terms_ = [candidates[i] for i in keep]
        self.coef_ = np.asarray(coef, dtype=float)
        self.intercept_ = float(intercept)
        self.fit_seconds_ = time.perf_counter() - t0
        self._Psi_train_ = Psi[:, keep]           # kept for the variance / orthogonality check
        return self

    # --- prediction ---------------------------------------------------------------------
    def predict(self, X):
        Z = self.transform_.transform(np.asarray(X, dtype=float))
        if not self.terms_:
            return np.full(Z.shape[0], self.intercept_)
        return self.intercept_ + design_matrix(Z, self.terms_, self.p) @ self.coef_

    # --- structure ----------------------------------------------------------------------
    @property
    def n_terms(self):
        """Number of active (non-zero) terms, excluding the constant."""
        return len(self.terms_)

    @property
    def max_order(self):
        """Largest interaction order among active terms (0 if only the constant)."""
        return max((len(s) for s, _ in self.terms_), default=0)

    def terms_table(self):
        names = self.feature_names
        return pd.DataFrame({
            "support": ["*".join(names[j] for j in s) for s, _ in self.terms_],
            "degrees": [",".join(map(str, dg)) for _, dg in self.terms_],
            "order": [len(s) for s, _ in self.terms_],
            "coef": self.coef_,
            "coef_sq": self.coef_ ** 2,
        }).sort_values("coef_sq", ascending=False, ignore_index=True)

    # --- Sobol' indices -----------------------------------------------------------------
    def sobol(self):
        """Analytic Sobol' indices from the coefficients.

        If the Psi_alpha are orthonormal under the input distribution, then
            Var[f] = sum_{alpha != 0} c_alpha^2 = D,
        and the part of the variance due to exactly the features in a set u is the sum of c_alpha^2
        over the alpha whose support is exactly u. Hence
            S_i  = sum_{supp(alpha) = {i}}     c_alpha^2 / D   (first order)
            S_Ti = sum_{i in supp(alpha)}      c_alpha^2 / D   (total effect)
            S_ij = sum_{supp(alpha) = {i, j}}  c_alpha^2 / D   (second order)
        Returns (per_feature DataFrame [feature, S1, ST], pairs DataFrame [i, j, S2], D).
        """
        d = len(self.feature_names)
        c2 = self.coef_ ** 2
        D = float(c2.sum())
        S1, ST = np.zeros(d), np.zeros(d)
        pairs = {}
        for (supp, _), v in zip(self.terms_, c2):
            for j in supp:
                ST[j] += v
            if len(supp) == 1:
                S1[supp[0]] += v
            elif len(supp) == 2:
                pairs[supp] = pairs.get(supp, 0.0) + v
        if D > 0:
            S1, ST = S1 / D, ST / D
            pairs = {k: v / D for k, v in pairs.items()}
        per_feature = pd.DataFrame({"feature": self.feature_names, "S1": S1, "ST": ST})
        pair_df = pd.DataFrame([{"i": self.feature_names[a], "j": self.feature_names[b], "S2": v}
                                for (a, b), v in sorted(pairs.items(), key=lambda kv: -kv[1])],
                               columns=["i", "j", "S2"])
        return per_feature, pair_df, D

    def variance_check(self, X=None):
        """Compare D = sum c^2 with the empirical variance of the student's predictions.

        Var_data[f] = c^T Cov(Psi) c. This equals D only if Cov(Psi) = identity on the data,
        i.e. if the basis is orthonormal under the data distribution. Dependence between features
        (and discrete margins) break that, so the gap measures how far the analytic Sobol' indices
        (computed for independent uniform inputs) are from a variance decomposition of the data.
        Also returns the largest |off-diagonal| of the correlation matrix of the active columns,
        and the mean of the diagonal of Cov(Psi) (1 for an orthonormal basis).
        """
        Psi = self._Psi_train_ if X is None else design_matrix(
            self.transform_.transform(np.asarray(X, dtype=float)), self.terms_, self.p)
        _, _, D = self.sobol()
        pred_var = float(np.var(Psi @ self.coef_)) if self.n_terms else 0.0
        out = {"D_analytic": D, "var_empirical": pred_var,
               "ratio_var_over_D": pred_var / D if D > 0 else np.nan}
        if self.n_terms >= 2:
            C = np.corrcoef(Psi, rowvar=False)
            off = np.abs(C[~np.eye(len(C), dtype=bool)])
            out["max_abs_corr_between_terms"] = float(np.nanmax(off))
            out["mean_abs_corr_between_terms"] = float(np.nanmean(off))
            out["mean_column_variance"] = float(Psi.var(axis=0).mean())
        return out
