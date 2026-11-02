"""Pushes a synthetic aggregate with known truth through the whole estimation stage."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from cordon import config, did, estimate

TRUE_CORE = -0.10
TRUE_RING = 0.04


def world() -> tuple[pl.DataFrame, pl.DataFrame]:
    rng = np.random.default_rng(3)
    plan = [("core", 12, 0.0, 3), ("mixed", 3, 0.0, 1), ("surcharge_ring", 3, 1.0, 1)]
    plan += [("near", 8, 2.0, 2), ("far", 36, 5.0, 6)]
    zones, rows = [], []
    weeks = np.arange(config.FIRST_WEEK, config.LAST_WEEK + 1)
    week_effect = rng.normal(0, 0.05, len(weeks))
    season = np.sin(2 * np.pi * np.mod(weeks, 52) / 52)
    location = 0
    for group, count, km, clusters in plan:
        for i in range(count):
            location += 1
            cluster = f"{group}:X:{i % clusters + 1}"
            spread = km + (4 * (i % 3) if group == "far" else 0)
            overlap = 1.0 if group == "core" and i < 6 else 0.8 if group == "core" else 0.0
            zones.append((location, group, overlap, spread, cluster))
            base = rng.normal(7.0, 0.4)
            effect = {"core": TRUE_CORE, "surcharge_ring": TRUE_RING}.get(group, 0.0)
            seasonal = 0.15 * season if group == "core" else 0.0
            log_y = (
                base
                + week_effect
                + seasonal
                + effect * (weeks >= 0)
                + rng.normal(0, 0.015, len(weeks))
            )
            trips = np.exp(log_y).round().astype(int)
            # Yellow scales with the zone so both services carry the same true effect.
            flows = ({"core": ("entering", "leaving", "within"), "mixed": ("mixed",)}).get(
                group, ("outside",)
            )
            for flow in flows:
                for w, n in zip(weeks, trips, strict=True):
                    rows.append((int(w), "hv", location, flow, "peak", int(n // len(flows)), 0))
                    rows.append((int(w), "yellow", location, flow, "overnight", int(n // 8), 0))
    frame = pl.DataFrame(
        rows,
        schema=["week", "service", "zone", "flow", "period", "trips", "flex_trips"],
        orient="row",
    )
    table = pl.DataFrame(
        zones, schema=["location_id", "group", "overlap", "km_to_crz", "cluster"], orient="row"
    )
    return frame, table


@pytest.fixture(scope="module")
def data():
    return world()


def test_every_specification_runs_and_primary_recovers_truth(data):
    frame, zones = data
    primary = did.load_panel(frame, zones)
    rows = [r.row() for r in estimate.robustness(frame, zones, primary, draws=299)]
    assert len({r["label"] for r in rows}) == len(rows)
    head = rows[0]
    assert head["beta"] == pytest.approx(TRUE_CORE, abs=0.01)
    assert head["wild_low_pct"] < did.pct(TRUE_CORE) < head["wild_high_pct"]


def test_ring_spillover_is_recovered_and_the_bound_shrinks_the_estimate(data):
    frame, zones = data
    primary = did.load_panel(frame, zones)
    spill = estimate.spillovers(primary, zones, draws=49)
    assert spill["rings"]["surcharge_ring"]["percent"] == pytest.approx(did.pct(TRUE_RING), abs=1.0)
    bound = [row["percent"] for row in spill["displacement_bound"]]
    assert bound[0] < bound[-1] < 0


def test_placebo_dates_are_clean_and_synthetic_agrees(data):
    frame, zones = data
    primary = did.load_panel(frame, zones)
    for row in estimate.placebo_dates(frame, zones, draws=49):
        assert abs(row["percent"]) < 2.0
    synthetic = estimate.synthetic_results(frame, zones, primary)
    assert synthetic["synthetic_control"]["percent"] == pytest.approx(did.pct(TRUE_CORE), abs=2.5)
    assert synthetic["synthetic_did"]["percent"] == pytest.approx(did.pct(TRUE_CORE), abs=2.5)


def test_event_study_shows_the_step(data):
    frame, zones = data
    primary = did.load_panel(frame, zones)
    events = estimate.event_study(primary)
    post = [r["percent"] for r in events["series"]["core"] if r["week"] >= 0]
    assert np.mean(post) == pytest.approx(did.pct(TRUE_CORE), abs=1.5)
    assert events["max_abs_pre_pct"] < 3.0
