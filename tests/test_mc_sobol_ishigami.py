"""Monte Carlo Sobol' estimator on the Ishigami function (same benchmark as test_pce_ishigami).

The estimator resamples columns from a 'pool', exactly as for the teacher. Here the pool is 50,000
uniform draws on (-pi, pi)^3, so the resampled inputs are (close to) uniform and independent.
With N = 20,000 the Monte Carlo error of these estimators is roughly 0.01-0.02; tolerance 0.04.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from kdpce.explain import mc_sobol  # noqa: E402
from test_pce_ishigami import EXPECTED_S1, EXPECTED_ST, ishigami  # noqa: E402


def test_mc_sobol_ishigami():
    pool = np.random.default_rng(5).uniform(-np.pi, np.pi, size=(50_000, 3))
    table, V = mc_sobol(ishigami, pool, N=20_000, seed=0, n_boot=100)
    np.testing.assert_allclose(table["S1"].to_numpy(), EXPECTED_S1, atol=0.04)
    np.testing.assert_allclose(table["ST"].to_numpy(), EXPECTED_ST, atol=0.04)
    assert abs(V - 13.8446) / 13.8446 < 0.05
    # The bootstrap interval should contain the estimate and be reasonably narrow.
    assert np.all(table["ST_lo"] <= table["ST"]) and np.all(table["ST"] <= table["ST_hi"])
    assert np.all(table["ST_hi"] - table["ST_lo"] < 0.1)
