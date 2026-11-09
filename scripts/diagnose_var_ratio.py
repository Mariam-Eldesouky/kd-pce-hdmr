"""Why is the PCE's off-manifold Var/D below 1 on Bioresponse?

    python scripts/diagnose_var_ratio.py [--dataset bioresponse] [--seeds 0 1 2 3 4]

The analytic variance D = sum c^2 assumes every basis function has E[psi^2] = 1, i.e. a uniform z.
After the empirical-CDF transform a feature with few distinct values is NOT uniform. For a binary
feature with share pi of zeros, mid-ranks give z = pi - 1 (zeros) and z = pi (ones), so
    E[psi_1^2] = 3 E[z^2] = 3 pi (1 - pi)  <= 0.75.
Off-manifold the columns are independent, so Var(f) = sum_t c_t^2 prod_{j in t} E[psi_{n_j}(z_j)^2]
(+ cross terms between different degrees of the same feature, which are 0 for binary features
because they only get degree 1). This script measures E[psi_n^2] on the pool, per feature type,
and compares the predicted ratio sum_t c_t^2 prod m / D with the observed off-manifold Var/D.

Writes results/<dataset>/seed<k>/var_ratio_diagnosis.csv and results/summary_<dataset>_var_ratio.csv,
and results/summary_<dataset>_feature_types.csv (committed, like the other summaries).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kdpce.data import load_dataset  # noqa: E402
from kdpce.pce_hdmr import EmpiricalCDFTransform, legendre_orthonormal  # noqa: E402

RESULTS = ROOT / "results"


def feature_type(n_unique):
    if n_unique <= 1:
        return "constant"
    if n_unique == 2:
        return "binary"
    if n_unique <= 10:
        return "low_cardinality_3_10"
    return "continuous_gt10"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="bioresponse")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = ap.parse_args()
    name = args.dataset
    X, y, _ = load_dataset(name)
    names = list(X.columns)

    # Feature types on the whole dataset.
    nu_all = X.nunique()
    counts = nu_all.map(feature_type).value_counts().rename_axis("type").reset_index(name="n_features_full_data")
    counts.to_csv(RESULTS / f"summary_{name}_feature_types.csv", index=False)
    print(f"{name}: {X.shape[1]} features; types on the full data:")
    print(counts.to_string(index=False))

    rows = []
    for seed in args.seeds:
        out = RESULTS / name / f"seed{seed}"
        pool = np.load(out / "splits.npz")["distill"]
        Xp = X.iloc[pool].to_numpy(dtype=float)
        tr = EmpiricalCDFTransform().fit(Xp)                    # same transform as the student (fit on pool)
        Z = tr.transform(Xp)
        ftype = pd.Series([feature_type(n) for n in tr.n_unique_], index=names)

        # Second moments E[psi_n(z_j)^2] on the pool, for every feature and degree used by the student.
        terms = pd.read_csv(out / "pce_terms.csv")
        terms["feats"] = terms["support"].str.split("*")
        terms["degs"] = terms["degrees"].astype(str).str.split(",").map(lambda d: [int(v) for v in d])
        col = {c: i for i, c in enumerate(names)}
        cache = {}

        def m2(f, n):
            if (f, n) not in cache:
                cache[(f, n)] = float(np.mean(legendre_orthonormal(n, Z[:, col[f]]) ** 2))
            return cache[(f, n)]

        prod = np.array([np.prod([m2(f, n) for f, n in zip(fs, ds)]) for fs, ds in zip(terms.feats, terms.degs)])
        c2 = terms["coef_sq"].to_numpy()
        D = c2.sum()
        has_bin = np.array([any(ftype[f] == "binary" for f in fs) for fs in terms.feats])
        has_low = np.array([any(ftype[f] == "low_cardinality_3_10" for f in fs) for fs in terms.feats])

        # Average E[psi_1^2] by feature type over all non-constant features (not only the selected ones).
        psi1 = pd.Series([float(np.mean(legendre_orthonormal(1, Z[:, j]) ** 2)) for j in range(len(names))],
                         index=names)
        vc = pd.read_csv(out / "pce_variance_check.csv").set_index("data")
        row = {"dataset": name, "seed": seed,
               "n_binary_pool": int((ftype == "binary").sum()),
               "n_low_card_pool": int((ftype == "low_cardinality_3_10").sum()),
               "n_continuous_pool": int((ftype == "continuous_gt10").sum()),
               "n_constant_pool": int((ftype == "constant").sum()),
               "mean_E_psi1sq_binary": psi1[ftype == "binary"].mean(),
               "mean_E_psi1sq_low_card": psi1[ftype == "low_cardinality_3_10"].mean(),
               "mean_E_psi1sq_continuous": psi1[ftype == "continuous_gt10"].mean(),
               "share_D_terms_with_binary": c2[has_bin].sum() / D,
               "share_D_terms_with_low_card": c2[has_low].sum() / D,
               "share_D_continuous_only": c2[~has_bin & ~has_low].sum() / D,
               "mean_m_selected_terms_binary": np.average(prod[has_bin], weights=c2[has_bin]) if has_bin.any() else np.nan,
               "mean_m_selected_terms_continuous_only": (np.average(prod[~has_bin & ~has_low],
                                                                    weights=c2[~has_bin & ~has_low])
                                                         if (~has_bin & ~has_low).any() else np.nan),
               "predicted_ratio_from_moments": float((c2 * prod).sum() / D),
               "observed_ratio_offmanifold": float(vc.loc["offmanifold", "ratio_var_over_D"]),
               "observed_ratio_pool": float(vc.loc["pool", "ratio_var_over_D"])}
        rows.append(row)
        print(f"seed {seed}: predicted Var/D from pool moments {row['predicted_ratio_from_moments']:.3f}, "
              f"observed off-manifold {row['observed_ratio_offmanifold']:.3f}; "
              f"E[psi1^2] binary {row['mean_E_psi1sq_binary']:.3f}, continuous {row['mean_E_psi1sq_continuous']:.3f}; "
              f"share of D in terms with a binary feature {row['share_D_terms_with_binary']:.3f}")
        pd.DataFrame([row]).to_csv(out / "var_ratio_diagnosis.csv", index=False)

    df = pd.DataFrame(rows)
    summ = df.drop(columns=["dataset", "seed"]).agg(["mean", "std"]).T.reset_index(names="metric")
    summ.insert(0, "dataset", name)
    summ["n_seeds"] = len(df)
    summ.to_csv(RESULTS / f"summary_{name}_var_ratio.csv", index=False)
    print(summ.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
