"""EBM / GA2M student (Student 2): InterpretML's ExplainableBoostingRegressor on the teacher's logits.

An EBM is a generalised additive model with a few pairwise terms (GA2M):
    g(x) = b0 + sum_j f_j(x_j) + sum_{(i,j) in P} f_ij(x_i, x_j)
Each f is a lookup table over bins, learned by cyclic gradient boosting on one term at a time.
The number of pairs |P| = K is the complexity knob. Pairs are picked by FAST (Lou et al. 2013).

Per-feature scores comparable with Sobol' indices (computed on a given X, here the test set):
    first_j = Var(f_j(x_j)) / Var(g(x))
    total_j = [Var(f_j(x_j)) + sum_{pairs containing j} Var(f_ij(x_i, x_j))] / Var(g(x))
This is a variance-share approximation, NOT an exact Sobol' decomposition. EBM terms are not orthogonal:
  - mains and pairs can be correlated, because a pair can absorb part of a main effect unless the model is
    "purified" (Lengerich et al. 2020);
  - the data inputs are dependent, so even orthogonal-by-construction terms co-vary on the data.
So the shares need not sum to 1. We also save EBM's built-in importance: mean |f(x)| per term.
"""
import time

import numpy as np
import pandas as pd
from interpret.glassbox import ExplainableBoostingRegressor


def fit_ebm_student(X, target, interactions, seed, **ebm_kwargs):
    """Fit an EBM on (inputs, teacher logits). There is deliberately no label argument.

    Returns (model, fit_seconds).
    """
    t0 = time.perf_counter()
    model = ExplainableBoostingRegressor(interactions=interactions, random_state=seed, **ebm_kwargs)
    model.fit(X, np.asarray(target, dtype=float))
    return model, time.perf_counter() - t0


def ebm_complexity(model):
    """Number of terms, number of pairs, total number of learned bin scores, max order."""
    n_params = int(sum(np.asarray(s).size for s in model.term_scores_))
    orders = [len(f) for f in model.term_features_]
    return {"n_terms": len(orders), "n_pairs": sum(o == 2 for o in orders),
            "n_parameters": n_params, "max_order": max(orders)}


def ebm_term_contributions(model, X):
    """(n, n_terms) matrix of per-term contributions; intercept + row sum = prediction."""
    return model.eval_terms(X)


def ebm_variance_shares(model, X, feature_names):
    """Per-feature first/total variance shares, built-in importance, and a per-term table."""
    C = ebm_term_contributions(model, X)
    pred = model.intercept_ + C.sum(axis=1)
    var_pred = np.var(pred)
    term_var = C.var(axis=0)
    term_mean_abs = np.abs(C).mean(axis=0)

    d = len(feature_names)
    first, total, builtin = np.zeros(d), np.zeros(d), np.zeros(d)
    for t, feats in enumerate(model.term_features_):
        for j in feats:
            total[j] += term_var[t]
            builtin[j] += term_mean_abs[t]          # pair importance credited to both features
        if len(feats) == 1:
            first[feats[0]] += term_var[t]

    per_feature = pd.DataFrame({"feature": feature_names, "first": first / var_pred,
                                "total": total / var_pred, "mean_abs_contrib": builtin})
    per_term = pd.DataFrame({"term": model.term_names_,
                             "order": [len(f) for f in model.term_features_],
                             "var_share": term_var / var_pred, "mean_abs": term_mean_abs})
    per_term = per_term.sort_values("var_share", ascending=False, ignore_index=True)
    check = {"var_pred": float(var_pred), "sum_term_var": float(term_var.sum()),
             "ratio_sum_term_var_over_var_pred": float(term_var.sum() / var_pred)}
    return per_feature, per_term, check
