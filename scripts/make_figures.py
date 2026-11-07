"""Build all report figures from the aggregated results.

    python scripts/make_figures.py            # full results  -> figures/fig*.pdf
    python scripts/make_figures.py --quick    # quick results -> figures/fig*_quick.pdf
Run scripts/run_all.py first (it writes results/summary_*.csv).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from kdpce.aggregate import aggregate  # noqa: E402
from kdpce.plots import make_all  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--datasets", nargs="+", default=["magic", "bioresponse"])
    a = ap.parse_args()
    for name in a.datasets:
        aggregate(name, a.quick)                     # refresh summaries from whatever seeds exist
    for p in make_all(a.datasets, a.quick):
        print("wrote", p)
