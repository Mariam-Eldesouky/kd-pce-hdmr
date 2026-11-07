"""Explanations of the teacher.

- `teacher_shap_global`: SHAP global importance (also used to screen features for the PCE).
- `mc_sobol` / `teacher_mc_sobol`: Monte Carlo Sobol' indices of the teacher (the reference).
- `rank_agreement`: Kendall tau and top-k overlap between all explanation methods.
"""
import itertools

import numpy as np
import pandas as pd
from scipy.stats import kendalltau


def teacher_shap_global(model, X):
    """Mean |SHAP| per feature, TreeExplainer on the raw (log-odds) output.

    TreeSHAP is exact for tree ensembles. The SHAP values of a row add up to
    teacher_logit(row) - base value, so they explain the same quantity the students imitate.
    Returns a DataFrame [feature, mean_abs_shap] sorted by importance.
    """
    import shap

    explainer = shap.TreeExplainer(model, model_output="raw")
    values = explainer(X).values
    out = pd.DataFrame({"feature": list(X.columns), "mean_abs_shap": np.abs(values).mean(axis=0)})
    return out.sort_values("mean_abs_shap", ascending=False, ignore_index=True)


def screening_indices(shap_table, columns, k):
    """Column indices of the k features with the largest mean |SHAP| (computed on the pool)."""
    top = shap_table["feature"].head(k).tolist()
    pos = {c: i for i, c in enumerate(columns)}
    return sorted(pos[c] for c in top)


# ----------------------------------------------------------------------------------------
# Monte Carlo Sobol' indices (reference explanation of the teacher)
# ----------------------------------------------------------------------------------------
def sample_independent(X_pool, N, rng):
    """N x d matrix whose columns are drawn independently, each from that feature's pool values.

    Each column has the pool's marginal distribution; the columns are independent of each other.
    """
    X_pool = np.asarray(X_pool, dtype=float)
    idx = rng.integers(0, X_pool.shape[0], size=(N, X_pool.shape[1]))
    return X_pool[idx, np.arange(X_pool.shape[1])]


def _sobol_from_runs(fA, fB, fAB):
    """First-order (Saltelli 2010) and total (Jansen 1999) indices from model runs.

    fA, fB : (N,) outputs on A and B;  fAB : (N, d) outputs on A_B^(i), i = 1..d.
        V    = Var(f), estimated from all 2N runs of A and B
        S_i  = mean( f(B) * (f(A_B^(i)) - f(A)) ) / V
        S_Ti = 0.5 * mean( (f(A) - f(A_B^(i)))^2 ) / V
    Why they work (in words): A_B^(i) shares only feature i with B, so f(B) and f(A_B^(i)) co-vary only
    through feature i -> first order. A_B^(i) shares everything except feature i with A, so f(A) and
    f(A_B^(i)) differ only because of feature i and all its interactions -> total effect.
    """
    V = np.var(np.concatenate([fA, fB]))
    S1 = np.mean(fB[:, None] * (fAB - fA[:, None]), axis=0) / V
    ST = 0.5 * np.mean((fA[:, None] - fAB) ** 2, axis=0) / V
    return S1, ST, V


def mc_sobol(f, X_pool, N, seed, n_boot=200, max_batch_bytes=200_000_000):
    """Monte Carlo Sobol' indices of a function f (numpy (n, d) -> (n,)) with independent inputs.

    A, B are N x d with independent columns resampled from X_pool. A_B^(i) is A with column i taken
    from B. Cost: N (d + 2) evaluations of f, done in batches of at most `max_batch_bytes` of input.
    Confidence intervals: 95% percentile bootstrap over the N rows (the model runs are reused, so it
    is cheap).

    Caveat: because columns are resampled independently, A, B and A_B^(i) contain feature combinations
    that never occur in the data ("off-manifold"). The teacher must extrapolate there, so these indices
    describe the teacher under an independent-input model, not on the data distribution
    (Hooker, Mentch & Zhou 2021, "Unrestricted permutation forces extrapolation").
    """
    rng = np.random.default_rng(seed)
    A = sample_independent(X_pool, N, rng)
    B = sample_independent(X_pool, N, rng)
    d = A.shape[1]
    fA, fB = f(A), f(B)
    fAB = np.empty((N, d))
    cols_per_batch = max(1, int(max_batch_bytes // (N * d * 8)))   # 8 bytes per float64
    for start in range(0, d, cols_per_batch):
        cols = range(start, min(d, start + cols_per_batch))
        stacked = np.repeat(A[None], len(cols), axis=0)       # (c, N, d): c copies of A
        for k, i in enumerate(cols):
            stacked[k, :, i] = B[:, i]                        # copy k gets column i from B
        fAB[:, start:start + len(cols)] = f(stacked.reshape(-1, d)).reshape(len(cols), N).T
    S1, ST, V = _sobol_from_runs(fA, fB, fAB)

    boot_rng = np.random.default_rng(seed + 10_000)
    bS1, bST = np.empty((n_boot, d)), np.empty((n_boot, d))
    for b in range(n_boot):
        r = boot_rng.integers(0, N, N)
        bS1[b], bST[b], _ = _sobol_from_runs(fA[r], fB[r], fAB[r])
    table = pd.DataFrame({
        "S1": S1, "S1_lo": np.percentile(bS1, 2.5, axis=0), "S1_hi": np.percentile(bS1, 97.5, axis=0),
        "ST": ST, "ST_lo": np.percentile(bST, 2.5, axis=0), "ST_hi": np.percentile(bST, 97.5, axis=0),
    })
    return table, V


def teacher_mc_sobol(model, X_pool, N, seed, n_boot=200):
    """mc_sobol applied to the teacher's logit. Returns (DataFrame with a 'feature' column, V)."""
    booster = model.get_booster()

    def f(Xnp):
        return booster.inplace_predict(Xnp, predict_type="margin")

    table, V = mc_sobol(f, X_pool, N, seed, n_boot=n_boot)
    table.insert(0, "feature", list(X_pool.columns))
    return table, V


# ----------------------------------------------------------------------------------------
# Agreement between explanations
# ----------------------------------------------------------------------------------------
def rank_agreement(scores, ks=(5, 10), top_union=20):
    """Pairwise agreement between importance vectors.

    scores: {method: pd.Series indexed by feature name (higher = more important)}.
    For every pair of methods:
      kendall_tau_all : Kendall tau-b over all features (tau-b handles ties; in high dimension most
                        features are tied at or near zero for sparse methods, so this mixes in noise)
      kendall_tau_top : Kendall tau-b over the union of both methods' top-`top_union` features
      top{k}_overlap  : |top_k(a) & top_k(b)| / k
    """
    names = list(scores)
    feats = scores[names[0]].index
    rows = []
    for a, b in itertools.combinations(names, 2):
        sa, sb = scores[a].reindex(feats).fillna(0), scores[b].reindex(feats).fillna(0)
        union = sa.nlargest(top_union).index.union(sb.nlargest(top_union).index)
        row = {"method_a": a, "method_b": b,
               "kendall_tau_all": kendalltau(sa, sb).statistic,
               "kendall_tau_top": kendalltau(sa[union], sb[union]).statistic,
               "n_top_union": len(union)}
        for k in ks:
            row[f"top{k}_overlap"] = len(set(sa.nlargest(k).index) & set(sb.nlargest(k).index)) / k
        rows.append(row)
    return pd.DataFrame(rows)


def agreement_matrix(agreement, column="kendall_tau_all"):
    """Symmetric method x method matrix from the tidy agreement table (diagonal = 1)."""
    names = list(dict.fromkeys(list(agreement.method_a) + list(agreement.method_b)))
    M = pd.DataFrame(np.eye(len(names)), index=names, columns=names)
    for r in agreement.itertuples():
        M.loc[r.method_a, r.method_b] = M.loc[r.method_b, r.method_a] = getattr(r, column)
    return M
