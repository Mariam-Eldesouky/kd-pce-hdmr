"""Fidelity (student vs teacher), accuracy (vs true labels), and the off-manifold test set."""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def bernoulli_kl(p, q, eps=1e-12):
    """KL(Bern(p) || Bern(q)) per row, in nats. p = teacher, q = student."""
    p = np.clip(p, eps, 1 - eps)
    q = np.clip(q, eps, 1 - eps)
    return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))


def fidelity(teacher_logit, student_logit):
    """How well the student reproduces the teacher.

    r2_logit       : 1 - SSE / SST on logits (1 = perfect copy; 0 = no better than a constant)
    rmse_logit     : root mean squared error on logits
    label_agreement: share of rows where student and teacher predict the same class (sign of logit)
    mean_kl        : mean KL(teacher Bernoulli || student Bernoulli), from sigmoid(logit)
    """
    t, s = np.asarray(teacher_logit), np.asarray(student_logit)
    sse = np.sum((t - s) ** 2)
    sst = np.sum((t - t.mean()) ** 2)
    return {"r2_logit": 1 - sse / sst,
            "rmse_logit": float(np.sqrt(np.mean((t - s) ** 2))),
            "label_agreement": float(np.mean((t > 0) == (s > 0))),
            "mean_kl": float(np.mean(bernoulli_kl(sigmoid(t), sigmoid(s))))}


def accuracy_metrics(y_true, logit):
    """Accuracy (threshold 0.5 on sigmoid(logit), i.e. logit > 0) and ROC-AUC vs true labels."""
    y_true = np.asarray(y_true)
    return {"accuracy": accuracy_score(y_true, (np.asarray(logit) > 0).astype(int)),
            "auc": roc_auc_score(y_true, logit)}


def off_manifold(X, seed):
    """Copy of X with every column permuted independently.

    Each column keeps exactly its marginal distribution, but the dependence between columns is
    destroyed. Rows are therefore mostly combinations the teacher never saw ("off-manifold").
    """
    rng = np.random.default_rng(seed)
    Xp = X.copy()
    for c in Xp.columns:
        Xp[c] = rng.permutation(Xp[c].to_numpy())
    return Xp
