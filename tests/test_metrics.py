from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from trframenet_revision.metrics import bootstrap_mean_ci, exact_mcnemar, paired_bootstrap_diff


def test_bootstrap_mean():
    mean, lo, hi = bootstrap_mean_ci([1, 1, 0, 0], n_boot=500, seed=1)
    assert mean == 0.5 and lo <= mean <= hi


def test_mcnemar():
    r = exact_mcnemar([1,1,0,0],[1,0,1,0])
    assert r["a_only"] == 1 and r["b_only"] == 1


def test_paired_bootstrap():
    r = paired_bootstrap_diff([1,1,1,0],[1,0,0,0], n_boot=500, seed=1)
    assert r["difference"] == 0.5
