"""Figures for the report (vector PDF, sized for a LaTeX column: 3.4 in single, 7.0 in double).

All figures read the across-seed summaries written by `kdpce.aggregate` (and, for Figure 4, the per-seed
prediction files). Error bars are the standard deviation over seeds unless stated otherwise.

Colours (validated with the dataviz skill's palette checker, light mode):
    PCE   #2a78d6 (blue)    EBM #eb6834 (orange)    SHAP #1baf7a (aqua)    teacher MC #eda100 (yellow)
Each method keeps its colour in every figure. The four never share one panel (orange vs yellow is too close),
so Figure 2 uses one panel per method. Yellow and aqua have < 3:1 contrast on white, so every series also has
its own marker shape and a legend or direct label. Text is always in neutral ink, never in a series colour.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from .aggregate import RESULTS, seed_dirs  # noqa: E402

FIGURES = RESULTS.parent / "figures"
COLORS = {"PCE": "#2a78d6", "EBM": "#eb6834", "SHAP": "#1baf7a", "MC": "#eda100"}
MARKERS = {"PCE": "o", "EBM": "s", "SHAP": "D", "MC": "^"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
DIVERGING = LinearSegmentedColormap.from_list(   # blue <-> gray midpoint <-> red (palette's diverging pair)
    "div", ["#184f95", "#6da7ec", "#f0efec", "#ec8a89", "#b83232"])
LABELS = {"magic": "MAGIC (d = 10)", "bioresponse": "Bioresponse (d = 1,776)"}
METHOD_LABELS = {"PCE_ST": "PCE $S_T$", "PCE_S1": "PCE $S_1$", "EBM_total": "EBM total",
                 "EBM_first": "EBM first", "EBM_meanabs": "EBM mean|f|", "SHAP": "SHAP",
                 "MC_ST": "Teacher MC $S_T$", "MC_S1": "Teacher MC $S_1$"}


def setup():
    plt.rcParams.update({
        "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.edgecolor": INK2, "axes.labelcolor": INK,
        "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.linewidth": 0.6,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
        "lines.linewidth": 1.5, "lines.markersize": 4, "pdf.fonttype": 42, "savefig.bbox": "tight",
        "figure.dpi": 150,
    })


def _summary(name, kind, quick):
    tag = f"{name}_quick" if quick else name
    f = RESULTS / f"summary_{tag}_{kind}.csv" if kind else RESULTS / f"summary_{tag}.csv"
    return pd.read_csv(f) if f.exists() else None


def _save(fig, stem, quick):
    FIGURES.mkdir(exist_ok=True)
    path = FIGURES / f"{stem}{'_quick' if quick else ''}.pdf"
    fig.savefig(path, metadata={"CreationDate": None})   # no timestamp: rebuilding gives an identical file
    fig.savefig(path.with_suffix(".png"), dpi=200)   # PNG copy for quick viewing; the report uses the PDF
    plt.close(fig)
    return path


def _err(df, col):
    s = df.get(f"{col}_std")
    return None if s is None or s.isna().all() else s.fillna(0).to_numpy()


# ----------------------------------------------------------------------------------------
# Figure 1: fidelity vs complexity (the interpretability / fidelity trade-off)
# ----------------------------------------------------------------------------------------
def fig_tradeoff(datasets, quick=False):
    """Top row: x = active terms. Bottom row: x = learned parameters (PCE coefficients vs EBM bin scores)."""
    fig, axes = plt.subplots(2, len(datasets), figsize=(7.0, 4.6), squeeze=False)
    for j, name in enumerate(datasets):
        pg, eg = _summary(name, "pce_grid", quick), _summary(name, "ebm_grid", quick)
        for i, xkind in enumerate(["terms", "params"]):
            ax = axes[i, j]
            if pg is not None:
                x = pg["n_terms_mean"] + (1 if xkind == "params" else 0)     # + intercept as a parameter
                ax.errorbar(x, pg["r2_logit_test_mean"], xerr=_err(pg, "n_terms"),
                            yerr=_err(pg, "r2_logit_test"), fmt=MARKERS["PCE"], color=COLORS["PCE"],
                            ms=4, elinewidth=0.6, capsize=0, label="PCE/HDMR (each grid config)")
                sel = pg[pg["times_selected"] > 0]
                ax.scatter(sel["n_terms_mean"] + (1 if xkind == "params" else 0), sel["r2_logit_test_mean"],
                           s=40, facecolors="none", edgecolors=INK, linewidths=0.8, zorder=5,
                           label="selected by CV in ≥1 seed")
            if eg is not None:
                xcol = "n_terms" if xkind == "terms" else "n_parameters"
                eg = eg.sort_values("interactions")
                ax.errorbar(eg[f"{xcol}_mean"], eg["r2_logit_test_mean"], yerr=_err(eg, "r2_logit_test"),
                            fmt=MARKERS["EBM"] + "-", color=COLORS["EBM"], ms=4, lw=1.2, elinewidth=0.6,
                            capsize=0, label="EBM (K = number of pairs)")
                # If the EBM points sit almost on top of each other in x (Bioresponse: every K has ~1,800
                # terms), label only the smallest and largest K so the labels do not collide.
                x = eg[f"{xcol}_mean"]
                crowded = x.max() / x.min() < 3
                for pos, (_, r) in enumerate(eg.iterrows()):
                    if crowded and 0 < pos < len(eg) - 1:
                        continue
                    above = pos == len(eg) - 1                   # largest K: label above the point
                    ax.annotate(f"K={int(r.interactions)}", (r[f"{xcol}_mean"], r["r2_logit_test_mean"]),
                                xytext=(3, 5 if above else -8), textcoords="offset points", fontsize=6, color=INK2)
                sel = eg[eg["times_selected"] > 0]
                ax.scatter(sel[f"{xcol}_mean"], sel["r2_logit_test_mean"], s=40, facecolors="none",
                           edgecolors=INK, linewidths=0.8, zorder=5)
            ax.set_xscale("log")
            ax.set_xlabel("active terms" if xkind == "terms" else
                          "learned parameters (PCE coefficients / EBM bin scores)")
            ax.set_ylabel("fidelity: test $R^2$ on teacher logits")
            if i == 0:
                ax.set_title(LABELS.get(name, name))
    axes[0, 0].legend(loc="lower right")
    fig.tight_layout()
    return _save(fig, "fig1_tradeoff", quick)


# ----------------------------------------------------------------------------------------
# Figure 2: feature rankings, one panel per method (small multiples)
# ----------------------------------------------------------------------------------------
FIG2_METHODS = [("PCE_ST", "PCE"), ("EBM_total", "EBM"), ("SHAP", "SHAP"), ("MC_ST", "MC")]


def fig_rankings(datasets, quick=False, top=10):
    """Top features by teacher MC S_T. Each method's scores are divided by their sum over all features,
    so panels share a 0-1 'share of importance' scale. Error bars: std over seeds."""
    fig, axes = plt.subplots(len(datasets), 4, figsize=(7.0, 2.1 * len(datasets)), squeeze=False,
                             sharey="row")
    for i, name in enumerate(datasets):
        ex = _summary(name, "explanations", quick)
        if ex is None:
            continue
        order = (ex[ex.method == "MC_ST"].nlargest(top, "score_mean")["feature"].tolist())[::-1]
        for j, (method, key) in enumerate(FIG2_METHODS):
            ax = axes[i, j]
            d = ex[ex.method == method].set_index("feature")
            total = d["score_mean"].clip(lower=0).sum()
            vals = d["score_mean"].reindex(order) / total
            errs = d["score_std"].reindex(order).fillna(0) / total
            y = np.arange(len(order))
            ax.hlines(y, 0, vals.clip(lower=0), color=GRID, lw=1.0, zorder=1)
            ax.errorbar(vals, y, xerr=errs, fmt=MARKERS[key], color=COLORS[key], ms=4.5, elinewidth=0.6,
                        capsize=0, zorder=3, markeredgecolor=INK if key == "MC" else COLORS[key],
                        markeredgewidth=0.4)
            ax.set_yticks(y, order)
            ax.set_xlim(left=0)
            ax.grid(axis="y", visible=False)
            if i == 0:
                ax.set_title(METHOD_LABELS[method])
            if i == len(datasets) - 1:
                ax.set_xlabel("share of importance")
            if j == 0:
                ax.set_ylabel(LABELS.get(name, name))
    fig.suptitle(f"Top-{top} features by teacher Monte Carlo $S_T$ (mean over seeds)", fontsize=8, y=1.0)
    fig.tight_layout()
    return _save(fig, "fig2_rankings", quick)


# ----------------------------------------------------------------------------------------
# Figure 3: agreement heatmaps (Kendall tau, mean over seeds)
# ----------------------------------------------------------------------------------------
HEAT_METHODS = ["PCE_ST", "PCE_S1", "EBM_total", "EBM_first", "SHAP", "MC_ST", "MC_S1"]


def _tau_matrix(name, quick, col):
    s = _summary(name, "", quick)
    if s is None:
        return None
    a = s[(s.section == "agreement") & (s.metric == col)]
    M = pd.DataFrame(np.eye(len(HEAT_METHODS)), index=HEAT_METHODS, columns=HEAT_METHODS)
    for _, r in a.iterrows():
        m1, m2 = r["item"].split(" vs ")
        if m1 in HEAT_METHODS and m2 in HEAT_METHODS:
            M.loc[m1, m2] = M.loc[m2, m1] = r["mean"]
    return M


def fig_agreement(datasets, quick=False):
    panels = []
    for name in datasets:
        panels.append((name, "kendall_tau_all", f"{LABELS.get(name, name)}\nKendall τ, all features"))
        if name != "magic":   # with d = 10 the top-20 union is every feature, so it equals tau_all
            panels.append((name, "kendall_tau_top", f"{LABELS.get(name, name)}\nKendall τ, top-20 union"))
    fig, axes = plt.subplots(1, len(panels), figsize=(2.45 * len(panels), 2.6), squeeze=False)
    im = None
    for ax, (name, col, title) in zip(axes[0], panels):
        M = _tau_matrix(name, quick, col)
        if M is None:
            continue
        im = ax.imshow(M.to_numpy(), cmap=DIVERGING, vmin=-1, vmax=1)
        for (r, c), v in np.ndenumerate(M.to_numpy()):
            ax.text(c, r, f"{v:.2f}".replace("0.", "."), ha="center", va="center", fontsize=5.5,  # -0.28 -> -.28
                    color="white" if abs(v) > 0.6 else INK)
        labels = [METHOD_LABELS[m] for m in HEAT_METHODS]
        ax.set_xticks(range(len(labels)), labels, rotation=60, ha="right")
        ax.set_yticks(range(len(labels)), labels if ax is axes[0][0] else [])
        ax.grid(False)
        ax.set_title(title)
    if im is not None:
        fig.colorbar(im, ax=axes[0].tolist(), shrink=0.7, label="Kendall τ (mean over seeds)")
    return _save(fig, "fig3_agreement", quick)


# ----------------------------------------------------------------------------------------
# Figure 4: where the students fail (residual vs teacher confidence; test vs off-manifold)
# ----------------------------------------------------------------------------------------
def _pooled_predictions(name, quick, student):
    t, s, to, so = [], [], [], []
    for d in seed_dirs(name, quick):
        f = d / f"{student.lower()}_predictions.npz"
        if f.exists():
            z = np.load(f)
            t.append(z["teacher_test"]); s.append(z["test"])
            to.append(z["teacher_offman"]); so.append(z["offman"])
    if not t:
        return None
    return tuple(np.concatenate(v) for v in (t, s, to, so))


def fig_failures(datasets, quick=False, n_bins=8):
    fig, axes = plt.subplots(2, len(datasets), figsize=(7.0, 4.4), squeeze=False)
    for j, name in enumerate(datasets):
        ax = axes[0, j]
        for student in ["PCE", "EBM"]:
            p = _pooled_predictions(name, quick, student)
            if p is None:
                continue
            t, s, _, _ = p
            conf, res = np.abs(t), np.abs(s - t)
            edges = np.quantile(conf, np.linspace(0, 1, n_bins + 1))
            idx = np.clip(np.searchsorted(edges, conf, side="right") - 1, 0, n_bins - 1)
            mid = [np.median(conf[idx == b]) for b in range(n_bins)]
            mean = [res[idx == b].mean() for b in range(n_bins)]
            sem = [res[idx == b].std() / np.sqrt((idx == b).sum()) for b in range(n_bins)]
            ax.errorbar(mid, mean, yerr=sem, fmt=MARKERS[student] + "-", color=COLORS[student], ms=4,
                        lw=1.2, elinewidth=0.6, capsize=0, label=student)
        ax.set_xlabel("teacher confidence |logit| (test rows, equal-count bins)")
        ax.set_ylabel("mean |student − teacher| (logit)")
        ax.set_title(LABELS.get(name, name))
        if j == 0:
            ax.legend()

        ax = axes[1, j]
        s = _summary(name, "", quick)
        if s is None:
            continue
        f = s[(s.section == "fidelity")]
        x = np.arange(2)
        for k, student in enumerate(["PCE", "EBM"]):
            vals = [f[(f["item"] == student) & (f.metric == m)] for m in ["r2_logit_test", "r2_logit_offman"]]
            means = [v["mean"].iloc[0] if len(v) else np.nan for v in vals]
            stds = [v["std"].fillna(0).iloc[0] if len(v) else 0 for v in vals]
            ax.bar(x + (k - 0.5) * 0.36, means, width=0.34, yerr=stds, color=COLORS[student],
                   edgecolor="white", linewidth=1.0, label=student, error_kw={"elinewidth": 0.6})
            for xi, m in zip(x + (k - 0.5) * 0.36, means):
                ax.annotate(f"{m:.2f}", (xi, max(m, 0)), xytext=(0, 2), textcoords="offset points",
                            ha="center", fontsize=6, color=INK)
        ax.axhline(0, color=INK2, lw=0.6)
        if j == 0:
            ax.legend(loc="upper right")
        ax.set_xticks(x, ["test set", "off-manifold\n(columns permuted)"])
        ax.set_ylabel("fidelity $R^2$ on logits")
        ax.grid(axis="x", visible=False)
    fig.tight_layout()
    return _save(fig, "fig4_failures", quick)


# ----------------------------------------------------------------------------------------
# Figure 5: MAGIC - PCE analytic Sobol' vs teacher Monte Carlo Sobol'
# ----------------------------------------------------------------------------------------
def fig_sobol_check(name="magic", quick=False):
    """Both methods: mean over seeds +- std over seeds (each seed has its own split, teacher and student).
    The per-seed 95% bootstrap CIs of the MC estimates are reported in the text, not drawn here."""
    pce, mc = [], []
    for d in seed_dirs(name, quick):
        if (d / "pce_sobol.csv").exists() and (d / "teacher_mc_sobol.csv").exists():
            pce.append(pd.read_csv(d / "pce_sobol.csv").set_index("feature"))
            mc.append(pd.read_csv(d / "teacher_mc_sobol.csv").set_index("feature"))
    if not pce:
        return None
    P, M = pd.concat(pce), pd.concat(mc)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5), sharey=True)
    order = M.groupby(level=0)["ST"].mean().sort_values().index
    y = np.arange(len(order))
    for ax, idx, title in [(axes[0], "S1", "first-order $S_i$"), (axes[1], "ST", "total effect $S_{T,i}$")]:
        pm, ps = P.groupby(level=0)[idx].mean()[order], P.groupby(level=0)[idx].std().fillna(0)[order]
        mm, ms_ = M.groupby(level=0)[idx].mean()[order], M.groupby(level=0)[idx].std().fillna(0)[order]
        ax.errorbar(mm, y + 0.15, xerr=ms_, fmt=MARKERS["MC"], color=COLORS["MC"], ms=5,
                    markeredgecolor=INK, markeredgewidth=0.4, elinewidth=0.8, capsize=0,
                    label="teacher, Monte Carlo (± std over seeds)")
        ax.errorbar(pm, y - 0.15, xerr=ps, fmt=MARKERS["PCE"], color=COLORS["PCE"], ms=4, elinewidth=0.8,
                    capsize=0, label="PCE student, analytic (± std over seeds)")
        ax.set_yticks(y, order)
        ax.set_title(title)
        ax.set_xlabel("Sobol' index (independent-input model)")
        ax.grid(axis="y", visible=False)
    axes[1].legend(loc="lower right")
    fig.tight_layout()
    return _save(fig, "fig5_sobol_check_magic", quick)


def make_all(datasets=("magic", "bioresponse"), quick=False):
    setup()
    paths = [fig_tradeoff(datasets, quick), fig_rankings(datasets, quick), fig_agreement(datasets, quick),
             fig_failures(datasets, quick), fig_sobol_check("magic", quick)]
    return [p for p in paths if p is not None]
