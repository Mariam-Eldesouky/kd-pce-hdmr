"""Explanations of the teacher.

Phase 3: SHAP global importance (also used to screen features for the PCE on high-dim data).
Phase 4 adds: Monte Carlo Sobol' indices of the teacher and rank agreement between methods.
"""
import numpy as np
import pandas as pd


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
