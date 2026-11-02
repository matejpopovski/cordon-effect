import numpy as np
import polars as pl
import pytest
import statsmodels.api as sm

from cordon import config, did


def test_absorb_matches_explicit_dummies_with_weights(panel_factory):
    p = panel_factory(effect=-0.08, n_zones=20, n_treated=6, first_week=-20, last_week=19)
    columns, names = did.post_terms(p, ["core"])
    fitted = did.fit(p, columns, names)

    t, n = p.counts.shape
    y = p.log_y.ravel()
    x = columns[0].ravel()
    dummies = np.hstack([np.tile(np.eye(n), (t, 1)), np.repeat(np.eye(t)[:, 1:], n, axis=0)])
    explicit = sm.WLS(y, np.column_stack([x, dummies]), weights=np.tile(p.weights, t)).fit()
    assert fitted.coef("core") == pytest.approx(explicit.params[0], abs=1e-9)


def test_recovers_an_injected_effect(panel_factory):
    p = panel_factory(effect=-0.1)
    fitted = did.fit_post(p, ["core"])
    assert fitted.coef("core") == pytest.approx(-0.1, abs=0.01)


def test_seasonal_terms_remove_a_group_specific_season(panel_factory):
    p = panel_factory(effect=-0.05, seasonal_gap=0.3, last_week=25)
    without = did.fit_post(p, ["core"], seasonal=False).coef("core")
    with_terms = did.fit_post(p, ["core"]).coef("core")
    assert abs(with_terms + 0.05) < 0.01
    assert abs(without + 0.05) > abs(with_terms + 0.05)


def test_trend_term_absorbs_a_treated_group_drift(panel_factory):
    p = panel_factory(effect=-0.05)
    drift = np.outer(p.weeks / 52 * 0.04, (p.groups == "core").astype(float))
    drifting = did.Panel(
        p.weeks, p.zones, p.groups, p.clusters, p.counts * np.exp(drift), p.weights
    )
    plain = did.fit_post(drifting, ["core"]).coef("core")
    trended = did.fit_post(drifting, ["core"], trend_group="core").coef("core")
    assert abs(trended + 0.05) < abs(plain + 0.05)
    assert trended == pytest.approx(-0.05, abs=0.015)


def test_event_study_is_flat_before_and_steps_at_treatment(panel_factory):
    p = panel_factory(effect=-0.1, first_week=-156, last_week=51, noise=0.01, cluster_noise=0.005)
    fitted, labels = did.fit_event_study(p, ["core"])
    pre = [fitted.coef(f"bin[core,{b}]") for b in labels if b * 4 < -52]
    post = [fitted.coef(f"bin[core,{b}]") for b in labels if b >= 0]
    assert max(abs(b) for b in pre) < 0.03
    assert np.mean(post) == pytest.approx(-0.1, abs=0.02)


def test_event_bins_mark_2024_as_the_reference_year():
    bins = did.event_bins(np.array([-53, -52, -1, 0, 3, 4]))
    assert np.isnan(bins[1]) and np.isnan(bins[2])
    assert bins[0] == -14
    assert bins[3] == 0 and bins[4] == 0 and bins[5] == 1


def test_thanksgiving_weeks_are_the_weeks_containing_the_holiday():
    from datetime import date

    weeks = np.array([config.week_index(date(2024, 11, 28)), config.week_index(date(2024, 11, 21))])
    assert did.thanksgiving_weeks(weeks).tolist() == [True, False]


def zone_frame() -> tuple[pl.DataFrame, pl.DataFrame]:
    rows = []
    for week in range(-3, 2):
        rows.append((week, "hv", 1, "entering", "peak", 100))
        rows.append((week, "hv", 1, "within", "peak", 50))
        rows.append((week, "yellow", 1, "within", "peak", 40))
        rows.append((week, "hv", 2, "outside", "peak", 80))
        rows.append((week, "hv", 2, "to_core", "peak", 7))
    frame = pl.DataFrame(
        rows, schema=["week", "service", "zone", "flow", "period", "trips"], orient="row"
    )
    zones = pl.DataFrame({"location_id": [1, 2], "group": ["core", "far"], "cluster": ["a", "b"]})
    return frame, zones


def test_load_panel_counts_the_right_flows_for_each_group():
    frame, zones = zone_frame()
    p = did.load_panel(frame, zones, first_week=-3, last_week=1, min_pre_trips=1)
    assert p.zones.tolist() == [1, 2]
    assert p.counts[0].tolist() == [190.0, 80.0]


def test_load_panel_filters_service_and_comparison_flows():
    frame, zones = zone_frame()
    p = did.load_panel(
        frame,
        zones,
        service="hv",
        first_week=-3,
        last_week=1,
        min_pre_trips=1,
        comparison_flows=("outside", "to_core"),
    )
    assert p.counts[0].tolist() == [150.0, 87.0]


def test_load_panel_drops_units_with_any_empty_week():
    frame, zones = zone_frame()
    sparse = frame.filter(~((pl.col("zone") == 2) & (pl.col("week") == 0)))
    p = did.load_panel(sparse, zones, first_week=-3, last_week=1, min_pre_trips=1)
    assert p.zones.tolist() == [1]
