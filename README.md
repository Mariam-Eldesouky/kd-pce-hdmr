# kd-pce-hdmr

**Distilling a frozen XGBoost teacher into two interpretable students, a sparse Polynomial Chaos Expansion /
HDMR with analytic Sobol' indices and an Explainable Boosting Machine (GA2M), and checking whether their
explanations can be trusted.**

PhD selection assignment, Part B (TÜBİTAK 1001 project, ITU).

## 1. Purpose

A gradient-boosted tree ensemble (the *teacher*) is trained once and frozen. Two transparent *students* are
fitted to imitate the teacher's **logit** (its raw score before the sigmoid), never the true labels:

1. **PCE/HDMR student.** Inputs are mapped to (−1, 1) with an empirical CDF. The basis is orthonormal Legendre
   polynomials, sparse and truncated by total degree *p*, interaction order *r* and a *q*-norm, and the
   coefficients are chosen by LASSO (LARS) with cross-validation. Because the basis is orthonormal, the
   first-order and total Sobol' indices, and the pairwise ones, follow **analytically** from the squared coefficients.
2. **EBM/GA2M student** (`interpret`): boosted shape functions per feature plus *K* pairwise terms.

The study asks three questions:
- **(a) Fidelity:** how faithfully does each student copy the teacher, on the test set and off the data
  manifold, and how should that be measured?
- **(b) Trade-off:** how does fidelity change with interpretability, i.e. the number of terms or parameters?
- **(c) Agreement:** do the students' explanations agree with each other, with TreeSHAP on the teacher, and
  with a Monte Carlo Sobol' reference computed on the teacher itself? Where and why do the students fail?

The study uses two datasets: **MAGIC Gamma Telescope** (10 features) and **Bioresponse** (1,776 features).

## 2. Protocol (rules the code enforces)

- **Students never see labels.** They are fitted only on (distillation-pool inputs, teacher logits).
  `tests/test_no_leakage.py` checks that the student `fit` functions take no label argument.
- **Splits per seed:** stratified 60 / 20 / 20 into teacher-train / distillation pool / test.
  The **test set is never used** for fitting, tuning, screening or choosing a truncation; every choice is made
  by cross-validation inside the distillation pool (or inside teacher-train for the teacher).
- **The teacher is trained once per seed and frozen.** It is saved as `teacher.json` with a SHA-256 hash
  (`teacher.sha256`), and afterwards only loaded; the hash is checked on every load.
- **Seeds 0–4.** Each seed has its own split, teacher, screening and students. Results are reported as
  mean ± std over the 5 seeds.
- PCE, Legendre basis, analytic Sobol' indices and Monte Carlo Sobol' estimators are implemented in numpy here
  (`scipy.special.eval_legendre` for the polynomials). Both estimators are tested on the Ishigami function.

## 3. Setup

Tested with **Python 3.11.9** on Windows 11. All package versions are pinned in `requirements.txt`.

```bash
python -m venv .venv
# Windows
.venv\Scripts\python -m pip install -r requirements.txt
# Linux / macOS
.venv/bin/python -m pip install -r requirements.txt
```

With GNU make: `make install`. The Makefile uses `.venv/Scripts/python` by default; on Linux/macOS call it as
`make PY=.venv/bin/python <target>`.

## 4. Data

Both datasets are downloaded from **OpenML** on first use and cached in `data/openml_cache/` (git-ignored).
The first run therefore needs internet; later runs do not.

| dataset | OpenML ID | rows × features | target (positive class) | role |
|---|---|---|---|---|
| MAGIC Gamma Telescope | 1120 | 19,020 × 10 | `class` (`g`, gamma) | moderate dimension |
| Bioresponse | 4134 | 3,751 × 1,776 | `target` (`1`) | high dimension |

`make data` (or `python scripts/check_data.py`) downloads the datasets and prints a short report.

- OpenML sometimes answers the parquet download with HTTP 503. The `openml` package then prints
  "Failed to download parquet, fallback on ARFF" and uses the ARFF file. This is harmless.
- **Musk (v2)** was the original high-dimensional dataset. It is still in `configs/musk.yaml` and in the data
  check, but it is **not used in the study**. Under a molecule-grouped split the teacher did not generalise to
  unseen molecules: across 5 seeds the test AUC varied widely, and accuracy fell below the always-non-musk
  baseline on 3 of 5 seeds. Bioresponse replaced it.

## 5. How to run

| command | what it does |
|---|---|
| `make test` | 25 unit tests: leakage, Legendre orthonormality, PCE Sobol' and MC Sobol' on Ishigami |
| `make quick` | seed 0, small grids, fewer Monte Carlo samples, both datasets, then `figures/*_quick.*` |
| `make all` | the full study: seeds 0–4, full grids, both datasets, then the figures |
| `make figures` | rebuild the summaries and figures from existing results |

Without make:

```bash
python scripts/run_all.py --dataset all --quick --seeds 0          # = make quick (then make_figures.py --quick)
python scripts/run_all.py --dataset all --seeds 0 1 2 3 4          # = make all
python scripts/make_figures.py                                     # figures + summary CSVs
python scripts/results_table.py                                    # the tables in section 8
```

Useful options of `run_all.py`:
- `--dataset magic|bioresponse|all`
- `--seeds ...`
- `--stages teacher pce ebm explain`
- `--retrain-teacher`: deliberately replaces a frozen teacher.

**On Windows, for the full run** use `scripts\run_full_windows.bat` from the repository root in a Command Prompt.
- It runs one dataset/seed per process, one at a time, with a log per job in `logs\`, then builds the figures.
- **Every stage is resumable.** Teachers are loaded from disk, and the PCE and EBM grids continue from their
  `*_grid_partial.csv`. If the run is interrupted, start the same command again.

**Outputs**
- `results/<dataset>/seed<k>/`: per seed, git-ignored. It holds the splits, frozen teacher, grids, selected
  students' Sobol' tables and terms, SHAP, teacher MC Sobol', agreement tables, fidelity and runtime.
  Quick runs write to `seed<k>/quick/` and never overwrite full results.
- `results/summary_<dataset>*.csv`: the across-seed summaries; these are committed.

## 6. Runtime

Measured on the machine used for the study: AMD Ryzen 5 5500U (6 cores / 12 threads), 7.3 GB RAM, Windows 11.
Jobs ran one at a time; EBM parallelism is limited in the configs (`n_jobs`) to fit in memory.

- `make test`: about 25 s.
- `make quick` (both datasets, seed 0, teachers already trained): **2,323 s (≈ 39 min)**, measured in a fresh run.
  MAGIC took about 8 min (EBM 451 s) and Bioresponse about 30 min (PCE 403 s, EBM 1,413 s). Training the two
  teachers from scratch adds about 4 min.
  Quick results go to `results/<dataset>/seed0/quick/`, `results/summary_*_quick*.csv` and `figures/*_quick.*`
  (all git-ignored).
- `make all`: about **11.5 hours** for 2 datasets × 5 seeds. That is the sum of the per-seed times: MAGIC about 51–56 min
  per seed, and Bioresponse 1 h 20 min – 1 h 33 min per seed (7 h 12 min for all five, from the batch-file log).
  The run was interrupted once by a Windows Update restart and resumed. Grid compute time per seed, from
  `results/summary_*.csv`:

| dataset | PCE grid compute (s) | EBM grid compute (s) | explain stage (s) |
|---|---|---|---|
| MAGIC | 27 ± 3 | 3500 ± 498 | 0.8 ± 0.2 |
| Bioresponse | 1936 ± 46 | 3003 ± 227 | 34.1 ± 2.9 |

Training a teacher (5-fold CV over 6 settings, then a refit) took about 30–35 s per seed for MAGIC and about
3–4 min per seed for Bioresponse.

## 7. Figures

All figures are built by `scripts/make_figures.py` (code in `src/kdpce/plots.py`) from the files in `results/`.
Each is written as PDF and PNG to `figures/`.

| figure | function in `plots.py` | shows | data |
|---|---|---|---|
| `fig1_tradeoff` | `fig_tradeoff` | fidelity (test R² on teacher logits) vs active terms and vs learned parameters, every grid configuration | `summary_*_pce_grid.csv`, `summary_*_ebm_grid.csv` |
| `fig2_rankings` | `fig_rankings` | top-10 features by teacher MC S_T: shares of importance for PCE S_T, EBM total, SHAP, teacher MC S_T | `summary_*_explanations.csv` |
| `fig3_agreement` | `fig_agreement` | Kendall τ between all explanation methods (all features; for Bioresponse also the top-20 union) | `seed*/agreement_matrix_tau_*.csv` |
| `fig4_failures` | `fig_failures` | top: student error vs teacher confidence; bottom: test vs off-manifold fidelity | `seed*/*_predictions.npz`, `summary_*.csv` |
| `fig5_sobol_check_magic` | `fig_sobol_check` | MAGIC: analytic PCE S_i, S_T,i vs teacher Monte Carlo estimates | `seed*/pce_sobol.csv`, `seed*/teacher_mc_sobol.csv` |

## 8. Results (5 seeds)

The tables below are printed by `python scripts/results_table.py` from `results/summary_*.csv`
(mean ± std over seeds). *Off-manifold* is the test set with each column permuted independently, which breaks
the dependence between features.

| dataset | model | selected config (seeds) | active terms | test R² (logit) | label agreement | off-manifold R² | accuracy vs y | AUC vs y |
|---|---|---|---|---|---|---|---|---|
| MAGIC | teacher (XGBoost) | – | – | – | – | – | 0.880 ± 0.005 | 0.934 ± 0.005 |
| MAGIC | PCE student | p=4 r=3 q=1.0 k=all (5/5) | 345.8 ± 16.5 | 0.942 ± 0.008 | 0.952 ± 0.005 | 0.317 ± 0.113 | 0.872 ± 0.004 | 0.930 ± 0.004 |
| MAGIC | EBM student | K=30 (5/5) | 40.0 ± 0.0 | 0.960 ± 0.005 | 0.960 ± 0.004 | 0.737 ± 0.028 | 0.876 ± 0.003 | 0.930 ± 0.006 |
| Bioresponse | teacher (XGBoost) | – | – | – | – | – | 0.799 ± 0.008 | 0.884 ± 0.008 |
| Bioresponse | PCE student | p=3 r=2 q=0.75 k=50 (1/5); p=2 r=2 q=1.0 k=50 (1/5); p=3 r=2 q=1.0 k=50 (1/5); p=4 r=2 q=0.75 k=50 (1/5); p=3 r=2 q=1.0 k=30 (1/5) | 285.8 ± 43.4 | 0.774 ± 0.026 | 0.869 ± 0.011 | 0.178 ± 0.112 | 0.778 ± 0.015 | 0.846 ± 0.011 |
| Bioresponse | EBM student | K=20 (5/5) | 1796.0 ± 0.0 | 0.743 ± 0.038 | 0.863 ± 0.016 | 0.088 ± 0.068 | 0.770 ± 0.016 | 0.842 ± 0.014 |

| dataset | pair | Kendall τ (all features) | Kendall τ (top-20 union) | top-5 overlap | top-10 overlap |
|---|---|---|---|---|---|
| MAGIC | PCE_ST vs MC_ST | 0.88 ± 0.07 | 0.88 ± 0.07 | 1.00 ± 0.00 | 1.00 ± 0.00 |
| MAGIC | EBM_total vs MC_ST | 0.83 ± 0.06 | 0.83 ± 0.06 | 0.88 ± 0.11 | 1.00 ± 0.00 |
| MAGIC | SHAP vs MC_ST | 0.80 ± 0.07 | 0.80 ± 0.07 | 1.00 ± 0.00 | 1.00 ± 0.00 |
| MAGIC | PCE_ST vs SHAP | 0.80 ± 0.07 | 0.80 ± 0.07 | 1.00 ± 0.00 | 1.00 ± 0.00 |
| MAGIC | EBM_total vs SHAP | 0.78 ± 0.07 | 0.78 ± 0.07 | 0.88 ± 0.11 | 1.00 ± 0.00 |
| MAGIC | PCE_ST vs EBM_total | 0.81 ± 0.07 | 0.81 ± 0.07 | 0.88 ± 0.11 | 1.00 ± 0.00 |
| Bioresponse | PCE_ST vs MC_ST | 0.37 ± 0.03 | -0.12 ± 0.12 | 0.28 ± 0.11 | 0.32 ± 0.11 |
| Bioresponse | EBM_total vs MC_ST | 0.35 ± 0.02 | -0.35 ± 0.07 | 0.24 ± 0.09 | 0.12 ± 0.04 |
| Bioresponse | SHAP vs MC_ST | 0.95 ± 0.01 | 0.42 ± 0.11 | 0.76 ± 0.09 | 0.72 ± 0.16 |
| Bioresponse | PCE_ST vs SHAP | 0.37 ± 0.03 | -0.06 ± 0.08 | 0.40 ± 0.20 | 0.38 ± 0.08 |
| Bioresponse | EBM_total vs SHAP | 0.35 ± 0.02 | -0.34 ± 0.07 | 0.24 ± 0.09 | 0.12 ± 0.04 |
| Bioresponse | PCE_ST vs EBM_total | 0.25 ± 0.02 | -0.28 ± 0.08 | 0.28 ± 0.11 | 0.20 ± 0.00 |

How to read these tables:
- **MAGIC has only 10 features**, so a top-10 overlap of 1.00 is trivial, and the top-20 union contains every feature
  (both τ columns are identical).
- **On Bioresponse, a negative τ on the top-20 union means the methods pick different features**, not that they rank
  the same features in reverse order. Each method's top features get low ranks from the other, so τ is negative by
  construction (two methods with disjoint top-20 sets give τ ≈ −0.6).
- **For the PCE on MAGIC, "k=all" means no screening:** all 10 features are used.

## 9. Design choices (short version)

| choice | why |
|---|---|
| Students fit the teacher **logit**, not the probability | it is unbounded and additive in the trees, so it is a regression target without a link function. Fidelity is then R²/RMSE on logits plus label agreement and Bernoulli KL. |
| Mid-rank **empirical CDF** → z ∈ (−1, 1), then **Legendre** | Legendre polynomials are orthonormal under the uniform distribution; mid-ranks map ties to equal values deterministically. |
| Per-feature degree cap = distinct values − 1 | higher degrees of a binary or few-valued feature are exactly collinear with lower ones. |
| **LassoLarsCV** (5-fold, on the pool), intercept = f₀ | sparse, with the regularisation chosen by CV. A hybrid OLS refit is reported, not used: it overfitted on Bioresponse. |
| Truncation (p, r, q, screening k) and EBM K chosen by **CV R² on the pool** | the test set is never used for any choice. |
| Bioresponse: SHAP screening on the pool (top *k* features may interact; optionally all features in first order) | the number of interaction candidates otherwise explodes with 1,776 features. |
| **Off-manifold** fidelity (columns permuted) | Sobol' indices and MC Sobol' assume independent inputs; this measures how the student behaves there. |
| Teacher **Monte Carlo Sobol'** (Saltelli 2010 first order, Jansen 1999 total, bootstrap CIs) | a like-for-like reference for the PCE's analytic indices, computed on the teacher itself. |
| EBM importances = variance shares of term contributions | as in the assignment. The terms are not orthogonal, so the code also reports Σ Var(terms) / Var(prediction). |
| Kendall τ over all features **and** over the union of both methods' top-20 | with 1,776 features, ties at zero dominate τ over all features. |
| XGBoost tuned by 5-fold CV (AUC) with early stopping, then refit on all of teacher-train | a single validation slice was unstable (on Musk it kept one tree). |

## 10. Limitations

- **Several selected models lie at the edge of their grid:** MAGIC PCE p=4, r=3; MAGIC EBM K=30; Bioresponse PCE
  screening k=50 on 4 of 5 seeds. CV fidelity was still rising slowly there; larger grids were not explored for
  compute reasons.
- **The Sobol' indices (analytic and Monte Carlo) are defined for independent inputs, but both datasets have strongly
  dependent features.** The students are fitted on the data manifold and are much less faithful off it (section 8).
- **Five seeds give a rough estimate of the std.**
- All runtimes are from one laptop.

## 11. Repository layout

```
configs/            dataset configs and grids (magic.yaml, bioresponse.yaml; musk.yaml kept, not used)
src/kdpce/
  data.py           OpenML loading, splits (stratified / grouped), split checks
  teacher.py        XGBoost tuning by CV, training, freezing (SHA-256), loading
  pce_hdmr.py       empirical-CDF transform, Legendre basis, multi-indices, PCE/HDMR student, analytic Sobol'
  ebm_student.py    EBM student, complexity, variance shares
  explain.py        TreeSHAP, MC Sobol' (Saltelli/Jansen) with bootstrap CIs, rank agreement
  metrics.py        fidelity metrics, off-manifold set
  aggregate.py      across-seed summaries
  plots.py          report figures
scripts/
  run_all.py        experiment driver (stages teacher, pce, ebm, explain)
  make_figures.py   summaries + figures
  results_table.py  README tables from the summaries
  check_data.py     dataset download + report
  run_full_windows.bat  full run on Windows, one job at a time, resumable
tests/              leakage, Legendre orthonormality, PCE and MC Sobol' on Ishigami
results/            per-seed outputs (git-ignored) and summary_*.csv (committed)
figures/            fig1–fig5 (PDF + PNG)
```

## 12. References

- Sudret, B. (2008). Global sensitivity analysis using polynomial chaos expansions. *Reliability Engineering & System Safety*, 93(7).
- Blatman, G. & Sudret, B. (2011). Adaptive sparse polynomial chaos expansion based on least angle regression. *Journal of Computational Physics*, 230(6).
- Rabitz, H. & Aliş, Ö. F. (1999). General foundations of high-dimensional model representations. *Journal of Mathematical Chemistry*, 25.
- Sobol', I. M. (2001). Global sensitivity indices for nonlinear mathematical models and their Monte Carlo estimates. *Mathematics and Computers in Simulation*, 55.
- Saltelli, A. et al. (2010). Variance based sensitivity analysis of model output. *Computer Physics Communications*, 181(2).
- Jansen, M. J. W. (1999). Analysis of variance designs for model output. *Computer Physics Communications*, 117.
- Lou, Y., Caruana, R., Gehrke, J. & Hooker, G. (2013). Accurate intelligible models with pairwise interactions. *KDD*.
- Lengerich, B., Tan, S., Chang, C.-H., Hooker, G. & Caruana, R. (2020). Purifying interaction effects with the functional ANOVA. *AISTATS*.
- Lundberg, S. M. et al. (2020). From local explanations to global understanding with explainable AI for trees. *Nature Machine Intelligence*, 2.
- Hooker, G., Mentch, L. & Zhou, S. (2021). Unrestricted permutation forces extrapolation. *Statistics and Computing*, 31.
- Hinton, G., Vinyals, O. & Dean, J. (2015). Distilling the knowledge in a neural network. *arXiv:1503.02531*.
- Ishigami, T. & Homma, T. (1990). An importance quantification technique in uncertainty analysis for computer models. *ISUMA '90*.
