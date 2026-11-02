"""Pins the claims in the article and README to the committed results.

The page takes numbers from results/ at render time, but sentences such as "the interval
includes zero" are written by hand. If a rerun ever changes one of those facts, these fail.
"""

from __future__ import annotations

import json
import re

import pytest

from cordon import charts, config, report


@pytest.fixture(scope="module")
def facts() -> report.Facts:
    return report.Facts.load()


@pytest.fixture(scope="module")
def page(facts) -> str:
    return report.page(report.body(facts))


def test_the_pre_committed_verdict_is_sensitive_because_of_the_placebo(facts):
    verdict = facts.summary["verdict"]
    assert verdict["word"] == "sensitive"
    assert verdict["checks"]["fake_start_interval_includes_zero"] is False
    assert verdict["checks"]["wild_interval_excludes_zero"] is True


def test_every_fake_start_is_large_and_negative(facts):
    assert all(row["high"] < -5 for row in facts.placebo_dates)


def test_the_zone_was_flat_while_far_zones_grew(facts):
    assert abs(facts.level_change("core", 2024, 2025)) < 1
    assert facts.level_change("far", 2023, 2024) > 5
    assert facts.level_change("far", 2024, 2025) > 5
    assert facts.level_change("core", 2024, 2025, "hv") < 0
    assert facts.level_change("core", 2024, 2025, "yellow") > 0


def test_the_toll_date_break_is_the_smallest_of_the_three_januaries(facts):
    rows = facts.anniversaries
    assert abs(rows["2025"]["percent"]) < abs(rows["2024"]["percent"])
    assert abs(rows["2025"]["percent"]) < abs(rows["2026"]["percent"])
    assert rows["2025"]["wild_low_pct"] <= 0 <= rows["2025"]["wild_high_pct"]


def test_gap_does_not_follow_the_charge(facts):
    trend = facts.trend["real_start"]
    assert trend["surcharge_ring"]["percent"] < 0.8 * trend["core"]["percent"]
    assert trend["mixed"]["percent"] / trend["core"]["percent"] > 0.4
    assert 100 * facts.flows["group_charged_share"]["surcharge_ring"] < 10
    clean = facts.trend["core_vs_ring"]
    assert clean["wild_low_pct"] <= 0 <= clean["wild_high_pct"]


def test_no_gain_just_outside_the_line(facts):
    assert facts.spill["rings"]["surcharge_ring"]["percent"] < 0
    assert facts.spill["rings"]["near"]["percent"] < 0


def test_centre_to_edge_slope_predates_the_toll(facts):
    values = [row["percent"] for row in facts.post_hoc["gradient_fake_2024"]]
    assert values[0] < values[2] < values[3] < values[-1]


def test_full_displacement_still_leaves_a_negative_estimate(facts):
    assert facts.spill["displacement_bound"][-1]["percent"] < -5


def test_published_design_reproduces_and_fails_a_year_earlier(facts):
    design = facts.post_hoc["published_design"]
    assert abs(design["real_2025_vs_2024"]["percent"] - (-5.95)) < 0.5
    assert design["placebo_2024_vs_2023"]["high"] < 0
    assert design["placebo_2024_vs_2023"]["percent"] < design["real_2025_vs_2024"]["percent"]


def test_first_stage_is_sharp(facts):
    assert facts.flows["core_rate_week_before"] < 0.01
    assert facts.flows["core_rate_first_week"] > 0.95
    assert facts.flows["core_rate_lowest_quarter_by_service"] > 0.95
    assert facts.battery_park_rate > 90


def test_charge_amounts_did_not_change_around_the_2026_step():
    amounts = json.loads((config.RESULTS / "fee_amounts.json").read_text())
    for service, fee, surcharge in (("hv", 1.5, 2.75), ("yellow", 0.75, 2.5)):
        for label in (f"{service}_2025-12", f"{service}_2026-02"):
            assert fee in {row["amount"] for row in amounts[label]["cbd_congestion_fee"]}
            assert surcharge in {row["amount"] for row in amounts[label]["congestion_surcharge"]}


def test_page_states_the_verdict_and_the_limits(page):
    assert "<strong>sensitive</strong>" in page
    assert "says nothing about traffic overall" in page
    assert "What this cannot tell you" in page


def test_page_never_claims_a_traffic_or_pollution_effect(page):
    text = re.sub(r"<[^>]+>", " ", page).lower()
    for phrase in (
        "reduced traffic",
        "cut traffic",
        "reduced congestion",
        "cleaner air",
        "reduced pollution",
        "cut congestion",
    ):
        assert phrase not in text


def test_page_has_no_em_dashes(page):
    assert "—" not in page


def test_readme_numbers_match_the_results(facts):
    readme = (config.REPO / "README.md").read_text().replace(charts.MINUS, "-")
    expected = [
        facts.primary["percent"],
        facts.fake["percent"],
        facts.anniversaries["2024"]["percent"],
        facts.anniversaries["2025"]["percent"],
        facts.anniversaries["2026"]["percent"],
        facts.trend["through_2025"]["percent"],
        facts.trend["core_vs_ring"]["percent"],
        facts.synthetic["synthetic_control"]["percent"],
        facts.synthetic["synthetic_did"]["percent"],
        facts.post_hoc["published_design"]["placebo_2024_vs_2023"]["percent"],
    ]
    for value in expected:
        assert f"{value:+.1f}%" in readme, value
