import numpy as np
import pytest

from cordon import did, estimate


def primary(percent, low, high):
    return {"percent": percent, "wild_low_pct": low, "wild_high_pct": high}


PASSING_FAKE = {"percent": 0.5, "low": -2.0, "high": 3.0}


def test_verdict_detected_needs_every_check():
    out = estimate.verdict(primary(-5, -8, -2), 0.04, PASSING_FAKE, [-4.0, -6.0], [-5.0, -4.0])
    assert out["word"] == "detected"


def test_verdict_null_when_interval_covers_zero():
    out = estimate.verdict(primary(-1, -4, 2), 0.04, PASSING_FAKE, [-4.0, -6.0], [-5.0])
    assert out["word"] == "null"


@pytest.mark.parametrize(
    ("randomisation", "fake", "synth_values", "comparison"),
    [
        (0.3, PASSING_FAKE, [-4.0, -6.0], [-5.0]),
        (0.04, {"percent": 9.0, "low": 1.0, "high": 12.0}, [-4.0, -6.0], [-5.0]),
        (0.04, PASSING_FAKE, [3.0, -6.0], [-5.0]),
        (0.04, PASSING_FAKE, [-4.0, -6.0], [-5.0, 2.0]),
    ],
)
def test_verdict_sensitive_when_any_check_fails(randomisation, fake, synth_values, comparison):
    out = estimate.verdict(primary(-5, -8, -2), randomisation, fake, synth_values, comparison)
    assert out["word"] == "sensitive"


def test_merged_clusters_halves_ward_labels(panel_factory):
    p = panel_factory(n_zones=8, n_treated=4, first_week=-4, last_week=3)
    labels = np.array(["core:M:1", "core:M:2", "core:M:3", "core:M:4"] * 2)
    merged = estimate.merged_clusters(
        did.Panel(p.weeks, p.zones, p.groups, labels, p.counts, p.weights)
    )
    assert sorted(set(merged.clusters.tolist())) == ["core:M:1", "core:M:2"]


def test_placebo_blocks_run_on_far_clusters_only(panel_factory):
    p = panel_factory(effect=-0.3, n_zones=24, n_treated=6)
    out = estimate.placebo_blocks(p)
    assert len(out) == 6
    assert max(abs(row["beta"]) for row in out) < 0.1
