import numpy as np
import pytest
import statsmodels.api as sm

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


def test_ppml_matches_a_full_statsmodels_glm(panel_factory):
    p = panel_factory(effect=-0.08, n_zones=24, n_treated=6, first_week=-60, last_week=39)
    beta, se = estimate.fit_ppml(p, ["core"])

    t, n = p.counts.shape
    columns, names = did.post_terms(p, ["core"])
    more, more_names = did.seasonal_terms(p, ["core"])
    terms = np.stack(columns + more, axis=-1).reshape(t * n, len(columns) + len(more))
    used = terms.std(axis=0) > 0
    x = np.hstack(
        [terms[:, used], np.tile(np.eye(n), (t, 1)), np.repeat(np.eye(t)[:, 1:], n, axis=0)]
    )
    groups = np.tile(np.unique(p.clusters, return_inverse=True)[1], t)
    reference = sm.GLM(p.counts.ravel(), x, family=sm.families.Poisson()).fit(
        cov_type="cluster", cov_kwds={"groups": groups}, tol=1e-10
    )
    index = [nm for nm, u in zip(names + more_names, used, strict=True) if u].index("core")
    assert beta == pytest.approx(reference.params[index], abs=1e-7)
    # statsmodels also scales by (N-1)/(N-K) counting the zone dummies. They are nested within
    # clusters, so that term is left out here and is removed from the reference to compare.
    nested_df = np.sqrt((x.shape[0] - 1) / (x.shape[0] - x.shape[1]))
    assert se == pytest.approx(reference.bse[index] / nested_df, rel=0.01)
