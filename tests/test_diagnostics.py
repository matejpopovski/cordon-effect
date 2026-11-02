import numpy as np
import pytest

from cordon import config, diagnostics, did


def drifting(panel, slope_per_year: float, group: str = "core") -> did.Panel:
    member = (panel.groups == group).astype(float)
    return did.Panel(
        panel.weeks,
        panel.zones,
        panel.groups,
        panel.clusters,
        panel.counts * np.exp(np.outer(panel.weeks / 52 * slope_per_year, member)),
        panel.weights,
    )


def test_extrapolated_trend_separates_a_step_from_a_long_drift(panel_factory):
    p = drifting(panel_factory(effect=-0.05, first_week=-156, last_week=51), -0.06)
    fitted = diagnostics.fit_extrapolated_trend(p, "core")
    assert fitted.coef("core") == pytest.approx(-0.05 - 0.06 * 0.5, abs=0.015)


def test_extrapolated_trend_ignores_slope_changes_after_the_start(panel_factory):
    base = panel_factory(effect=0.0, first_week=-156, last_week=51)
    member = (base.groups == "core").astype(float)
    after = np.outer(np.where(base.weeks >= 0, -0.2 * base.weeks / 52, 0.0), member)
    bent = did.Panel(
        base.weeks,
        base.zones,
        base.groups,
        base.clusters,
        base.counts * np.exp(after),
        base.weights,
    )
    fitted = diagnostics.fit_extrapolated_trend(bent, "core")
    assert abs(fitted.coef("trend")) < 0.02


def test_post_coefficient_is_the_mean_post_week_effect(panel_factory):
    p = panel_factory(effect=0.0, first_week=-156, last_week=51, noise=0.001, cluster_noise=0.0)
    member = (p.groups == "core").astype(float)
    shape = np.where(p.weeks >= 0, -0.1 - 0.1 * (p.weeks >= 26), 0.0)
    moved = did.Panel(
        p.weeks,
        p.zones,
        p.groups,
        p.clusters,
        p.counts * np.exp(np.outer(shape, member)),
        p.weights,
    )
    fitted = diagnostics.fit_extrapolated_trend(moved, "core")
    assert fitted.coef("core") == pytest.approx(shape[p.weeks >= 0].mean(), abs=0.005)


def test_growth_break_ignores_a_constant_growth_gap(panel_factory):
    p = drifting(panel_factory(effect=0.0, first_week=-156, last_week=51), -0.06)
    assert abs(diagnostics.growth_break(p, "core", 0, 26, 199)["percent"]) < 1.5


def test_growth_break_finds_a_step(panel_factory):
    p = drifting(panel_factory(effect=-0.08, first_week=-156, last_week=51), -0.06)
    row = diagnostics.growth_break(p, "core", 0, 26, 199)
    assert row["percent"] == pytest.approx(did.pct(-0.08), abs=1.5)


def test_yoy_panel_starts_a_full_season_into_the_data(panel_factory):
    p = panel_factory(first_week=-104, last_week=51)
    window = diagnostics.yoy_panel(p, -52, 51)
    assert window.weeks[0] == -52
    expected = p.log_y[p.weeks == 0] - p.log_y[p.weeks == -config.SEASON_LENGTH]
    assert window.log_y[window.weeks == 0] == pytest.approx(expected)
