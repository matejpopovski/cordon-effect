"""Renders the feature page and README figures from results/.

Every number in the prose comes from a results file through `Facts`, so a rerun cannot leave
the text saying something the estimates no longer say. Claims that depend on the sign or size
of a result are guarded by tests/test_results.py.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from functools import cached_property
from html import escape
from pathlib import Path

import numpy as np

from . import charts, config
from .charts import fmt_count, fmt_pct

REPO_URL = "https://github.com/matejpopovski/cordon-effect"
NARROW = 420
GAP_AXIS = 30


def read(name: str):
    return json.loads((config.RESULTS / name).read_text())


def signed(value: float, digits: int = 1) -> str:
    return fmt_pct(value, digits)


def unsigned(value: float, digits: int = 1) -> str:
    return f"{abs(value):.{digits}f}%"


def interval(low: float, high: float) -> str:
    return f"{signed(low)} to {signed(high)}"


def table(headers: list[str], rows: list[list[str]]) -> str:
    head = "".join(f"<th>{escape(h)}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{escape(str(c))}</td>" for c in row) + "</tr>" for row in rows
    )
    return f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def details(summary: str, content: str) -> str:
    return f"<details><summary>{escape(summary)}</summary>{content}</details>"


def both(render) -> str:
    """Render a chart for desktop and phone. SVG text scales with its viewBox, so one drawing
    wide enough for a laptop leaves 12px labels at about 5px on a phone."""
    return (
        f'<div class="wide-only">{render(760)}</div><div class="narrow-only">{render(NARROW)}</div>'
    )


def figure(title: str, subtitle: str, chart: str, extra: str = "", source: str = "") -> str:
    note = f'<p class="source">{source}</p>' if source else ""
    return (
        f"<figure><figcaption>{escape(title)}<span>{escape(subtitle)}</span></figcaption>"
        f"{chart}{note}{extra}</figure>"
    )


@dataclass
class Facts:
    summary: dict
    robustness: list[dict]
    placebo_dates: list[dict]
    placebo_blocks: list[dict]
    events: dict
    synthetic: dict
    spill: dict
    flows: dict
    post_hoc: dict
    series: dict
    manifest: dict

    @classmethod
    def load(cls) -> Facts:
        return cls(
            summary=read("summary.json"),
            robustness=read("robustness.json"),
            placebo_dates=read("placebo_dates.json"),
            placebo_blocks=read("placebo_blocks.json"),
            events=read("event_study.json"),
            synthetic=read("synthetic.json"),
            spill=read("spillovers.json"),
            flows=read("flows.json"),
            post_hoc=read("post_hoc.json"),
            series=read("group_series.json"),
            manifest=json.loads((config.DATA / "manifest.json").read_text()),
        )

    def spec(self, label: str) -> dict:
        return next(r for r in self.robustness if r["label"] == label)

    @cached_property
    def primary(self) -> dict:
        return self.summary["primary"]

    @cached_property
    def fake(self) -> dict:
        return next(d for d in self.placebo_dates if d["fake_week"] == -52)

    @cached_property
    def trend(self) -> dict:
        return self.post_hoc["extrapolated_trend"]

    @cached_property
    def breaks(self) -> dict:
        return self.post_hoc["break_tests"]

    @cached_property
    def anniversaries(self) -> dict:
        rows = self.breaks["anniversaries"]
        return {r["start"][:4]: r for r in rows}

    @cached_property
    def low_read(self) -> float:
        return min(self.trend_robust_values)

    @cached_property
    def high_read(self) -> float:
        return max(self.trend_robust_values)

    @cached_property
    def trend_robust_values(self) -> list[float]:
        return [
            self.trend["through_2025"]["percent"],
            *[r["percent"] for r in self.breaks["all_vs_far"]],
            self.trend["core_vs_ring"]["percent"],
            self.synthetic["seasonal_difference_2025"]["percent"],
        ]

    def level(self, group: str, service: str, year: int) -> float:
        rows = self.post_hoc["levels"]["rows"]
        return next(r for r in rows if r["group"] == group and r["service"] == service)[f"y{year}"]

    def level_total(self, group: str, year: int) -> float:
        return self.level(group, "hv", year) + self.level(group, "yellow", year)

    def level_change(
        self, group: str, first: int, second: int, service: str | None = None
    ) -> float:
        if service is None:
            return 100 * (self.level_total(group, second) / self.level_total(group, first) - 1)
        return 100 * (self.level(group, service, second) / self.level(group, service, first) - 1)

    def mechanism(self, service: str, flow: str, year: int) -> dict:
        return next(
            r
            for r in self.post_hoc["mechanism"]
            if r["service"] == service and r["flow"] == flow and r["year"] == year
        )

    @cached_property
    def qa(self) -> dict:
        with (config.AGGREGATES / "qa.csv").open() as handle:
            rows = [r for r in csv.DictReader(handle) if r["service"] == "yellow"]
        return {r["month"]: r for r in rows}

    def flex_share(self, month: str) -> float:
        row = self.qa[month]
        return 100 * int(row["flex_rows"]) / int(row["kept"])

    def unpaired_share(self, month: str) -> float:
        row = self.qa[month]
        return 100 * int(row["negative_kept"]) / int(row["kept"])

    @cached_property
    def battery_park_rate(self) -> float:
        with (config.AGGREGATES / "first_stage.csv").open() as handle:
            row = next(r for r in csv.DictReader(handle) if r["zone"] == "12")
        return 100 * float(row["outside_end_rate"])

    def charged(self, touch: str) -> float:
        rows = self.flows["charged_share_post"]
        return 100 * next(r for r in rows if r["touch"] == touch)["rate"]

    @cached_property
    def rows_read(self) -> float:
        return sum(v["raw_rows"] for v in self.manifest.values()) / 1e9

    @cached_property
    def gb_read(self) -> float:
        return sum(v.get("bytes", 0) for v in self.manifest.values()) / 1e9


def trailing_mean(values: list[float], window: int = config.SEASON_LENGTH) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return np.convolve(array, np.ones(window) / window, mode="valid")


def divergence_chart(f: Facts, width: int = 760) -> tuple[str, list[list[str]]]:
    """Trailing year of trips against the same measure at the start of 2023."""
    weeks = np.array(f.series["weeks"])[config.SEASON_LENGTH - 1 :]
    core = trailing_mean(f.series["trips"]["all_core"])
    far = trailing_mean(f.series["trips"]["all_far"])
    base = np.searchsorted(weeks, -105)
    core_pct = 100 * (core / core[base] - 1)
    far_pct = 100 * (far / far[base] - 1)
    keep = weeks >= -105
    chart = charts.series_chart(
        weeks[keep].tolist(),
        {
            "Zones 4 km or more away": (far_pct[keep].tolist(), "orange"),
            "Trips touching the toll zone": (core_pct[keep].tolist(), "blue"),
        },
        "Trips touching the toll zone against far zones",
        "Far zones grew steadily from 2023 while trips touching the zone stayed flat",
        -5,
        float(np.ceil(far_pct[keep].max() / 5) * 5) + 3,
        label_at={
            "Zones 4 km or more away": (-70, 24.0),
            "Trips touching the toll zone": (8, 7.5),
        },
        width=width,
    )
    points = [int(k) for k in (-105, -53, -1, 51, 89)]
    rows = [
        [
            str(charts.date_of_week(k)),
            signed(float(core_pct[weeks == k][0])),
            signed(float(far_pct[weeks == k][0])),
        ]
        for k in points
    ]
    return chart, rows


def flex_chart(width: int = 760) -> tuple[str, list[list[str]]]:
    with (config.AGGREGATES / "qa.csv").open() as handle:
        rows = [r for r in csv.DictReader(handle) if r["service"] == "yellow"]
    from datetime import date

    weeks, flex, table_rows = [], [], []
    for r in rows:
        year, month = (int(p) for p in r["month"].split("-"))
        weeks.append(config.week_index(date(year, month, 15)))
        flex.append(100 * int(r["flex_rows"]) / int(r["kept"]))
        table_rows.append([r["month"], f"{flex[-1]:.1f}%", f"{int(r['kept']):,}"])
    chart = charts.series_chart(
        weeks,
        {"Flex Fare share of yellow records": (flex, "blue")},
        "Flex Fare share of yellow taxi records",
        "Share of yellow rows coded as Flex Fare, by month",
        0,
        35,
        height=270,
        width=width,
        signed_axis=False,
    )
    return chart, table_rows


def smooth(values: list[float], window: int = 13) -> np.ndarray:
    """Centred moving average, with short windows at the ends."""
    array = np.asarray(values, dtype=float)
    weights = np.convolve(np.ones_like(array), np.ones(window), mode="same")
    return np.convolve(array, np.ones(window), mode="same") / weights


def placebo_paths(f: Facts) -> tuple[list[list[float]], int]:
    """Smoothed donor gaps that fit the axis, and how many very small donors did not."""
    smoothed = [smooth(g) for g in f.synthetic["synthetic_control"]["placebo_gaps_pct"]]
    kept = [g.tolist() for g in smoothed if np.abs(g).max() <= GAP_AXIS]
    return kept, len(smoothed) - len(kept)


def gaps_chart(f: Facts, width: int = 760) -> str:
    sc = f.synthetic["synthetic_control"]
    return charts.series_chart(
        f.synthetic["weeks"],
        {"Toll zone": (smooth(sc["gaps_pct"]).tolist(), "blue")},
        "Gap between the zone and its synthetic control",
        "The zone's gap beside the gaps of far-zone clusters, which had no toll",
        -GAP_AXIS,
        GAP_AXIS,
        ghosts=placebo_paths(f)[0],
        label_at={"Toll zone": (56, -24)},
        width=width,
    )


def zone_map(f: Facts) -> str:
    features = json.loads((config.REFERENCE / "zones_map.geojson").read_text())["features"]
    with (config.AGGREGATES / "first_stage.csv").open() as handle:
        first_stage = {int(r["zone"]): r for r in csv.DictReader(handle)}
    with (config.REFERENCE / "zones.csv").open() as handle:
        groups = {int(r["location_id"]): r for r in csv.DictReader(handle)}
    classes = {
        "core": "z-core",
        "mixed": "z-mixed",
        "surcharge_ring": "z-ring",
        "near": "z-near",
        "far": "z-far",
        "airport_or_unknown": "z-air",
    }
    mapping = {k: classes[v["group"]] for k, v in groups.items()}
    tips = {}
    for location, row in groups.items():
        measured = first_stage.get(location)
        share = ""
        if measured and measured["outside_end_rate"]:
            value = 100 * float(measured["outside_end_rate"])
            share = f": {value:.0f}% of trips with an outside end charged"
        tips[location] = f"{row['name']}{share}"
    return charts.map_chart(
        features,
        mapping,
        tips,
        "Taxi zones by exposure group",
        "Zones inside the toll area, partly inside, in the older surcharge ring, near and far",
        labels=[(-73.9625, 40.7675, "60th St")],
    )


def legend(items: list[tuple[str, str]]) -> str:
    spans = "".join(f'<span><i style="background:{c}"></i>{escape(t)}</span>' for t, c in items)
    return f'<div class="legend">{spans}</div>'


def body(f: Facts) -> str:
    p = f.primary
    fake = f.fake
    jan24, jan25, jan26 = (f.anniversaries[y] for y in ("2024", "2025", "2026"))
    h13, h26, h52 = f.breaks["all_vs_far"]
    hv13, hv26, hv52 = f.breaks["uber_lyft_vs_far"]
    core_trend = f.trend["real_start"]["core"]
    ring_trend = f.trend["real_start"]["surcharge_ring"]
    mixed_trend = f.trend["real_start"]["mixed"]
    near_trend = f.trend["real_start"]["near"]
    fake_trend = f.trend["fake_start"]["-52"]
    through = f.trend["through_2025"]
    ring_clean = f.trend["core_vs_ring"]
    sc = f.synthetic["synthetic_control"]
    sdid = f.synthetic["synthetic_did"]
    sd = f.synthetic["seasonal_difference_2025"]
    fake_sc = f.synthetic["fake_start_2024"]
    rings = f.spill["rings"]
    bound = {row["displaced_share"]: row["percent"] for row in f.spill["displacement_bound"]}
    pub = f.post_hoc["published_design"]
    randomisation = f.summary["randomisation"]
    yellow = f.spec("Yellow taxis")
    metered = f.spec("Yellow, metered rows only")
    hv = f.spec("Uber and Lyft")
    linear = f.spec("Core linear pre-trend")
    within_hv = {y: f.mechanism("hv", "Within the core", y) for y in (2024, 2025)}
    far_hv = {y: f.mechanism("hv", "Far to far", y) for y in (2024, 2025)}
    within_y = {y: f.mechanism("yellow", "Within the core", y) for y in (2024, 2025)}
    far_y = {y: f.mechanism("yellow", "Far to far", y) for y in (2023, 2024, 2025)}

    _, divergence_rows = divergence_chart(f)
    _, flex_rows = flex_chart()
    flows = f.flows
    ring_charged = 100 * flows["group_charged_share"]["surcharge_ring"]
    near_charged = 100 * flows["group_charged_share"]["near"]
    mixed_charged = f.charged("touches mixed, not core")
    within_2023 = f.mechanism("hv", "Within the core", 2023)
    far_2023 = f.mechanism("hv", "Far to far", 2023)
    ring_weekly = fmt_count(round(abs(rings["surcharge_ring"]["implied_weekly_change"]), -3))
    near_weekly = fmt_count(round(abs(rings["near"]["implied_weekly_change"]), -3))
    ring_only = signed(f.spec("Ring only, 60th to 96th")["percent"])
    far_only = signed(f.spec("Reference 12 km or more")["percent"])
    peak = signed(f.spec("Peak-toll hours")["percent"])
    overnight = signed(f.spec("Overnight hours")["percent"])
    before_rate = 100 * flows["core_rate_week_before"]
    first_rate = 100 * flows["core_rate_first_week"]
    floor_rate = np.floor(100 * flows["core_rate_lowest_quarter_by_service"])
    pre_gap = np.mean([r["percent"] for r in f.events["series"]["core"] if -104 <= r["week"] < -52])
    post_gap = np.mean([r["percent"] for r in f.events["series"]["core"] if 0 <= r["week"] < 52])
    pub_real = pub["real_2025_vs_2024"]
    pub_fake = pub["placebo_2024_vs_2023"]

    event_rows = [r for r in f.events["series"]["core"] if r["week"] >= config.PRIMARY_PRE_WEEK]
    event = both(
        lambda w: charts.event_study_chart(
            event_rows,
            "Toll zone against far zones, compared with the same weeks of 2024",
            "The zone stood above 2024 levels through 2023 and below them from 2025",
            -25,
            15,
            width=w,
        )
    )

    fake_rows = [
        {
            "label": f"Fake start, {d['start']}",
            "value": d["percent"],
            "low": d["low"],
            "high": d["high"],
        }
        for d in f.placebo_dates
    ]
    fake_rows.append(
        {
            "label": "Real start, 2025-01-05",
            "value": p["percent"],
            "low": p["wild_low_pct"],
            "high": p["wild_high_pct"],
            "emphasis": True,
        }
    )
    fake_chart = both(
        lambda w: charts.interval_chart(
            fake_rows,
            "Estimates at fake and real start dates",
            "Every fake start gives about the same answer as the real one",
            -16,
            2,
            width=w,
        )
    )

    break_rows = [
        {
            "label": r["title"],
            "value": r["percent"],
            "low": r["wild_low_pct"],
            "high": r["wild_high_pct"],
            "emphasis": r["start"].startswith("2025"),
            "section": "Same test each January, 26 weeks either side",
        }
        for r in f.breaks["anniversaries"]
    ]
    for r, width in zip((h13, h26, h52), (13, 26, 52), strict=True):
        break_rows.append(
            {
                "label": f"All trips, {width} weeks either side",
                "value": r["percent"],
                "low": r["wild_low_pct"],
                "high": r["wild_high_pct"],
                "section": "At the real start, by window",
            }
        )
    for r, width in zip((hv13, hv26, hv52), (13, 26, 52), strict=True):
        break_rows.append(
            {
                "label": f"Uber and Lyft, {width} weeks either side",
                "value": r["percent"],
                "low": r["wild_low_pct"],
                "high": r["wild_high_pct"],
                "section": "At the real start, by window",
            }
        )
    break_chart = both(
        lambda w: charts.interval_chart(
            break_rows,
            "Shift in year-on-year growth, toll zone against far zones",
            "A shift at the toll's start is small beside the same test in other years",
            -10,
            6,
            width=w,
        )
    )

    charge_rows = [
        ("Toll zone", f"{f.charged('touches core'):.0f}% of trips charged", core_trend),
        ("Mixed 59th-60th St zones", f"{mixed_charged:.0f}% charged", mixed_trend),
        ("Upper East and West Side", f"{ring_charged:.0f}% charged", ring_trend),
        ("Near zones under 4 km", f"{near_charged:.0f}% charged", near_trend),
    ]
    charge_chart = both(
        lambda w: charts.interval_chart(
            [
                {
                    "label": f"{title}, {share}",
                    "value": row["percent"],
                    "low": row["wild_low_pct"],
                    "high": row["wild_high_pct"],
                    "emphasis": title == "Toll zone",
                }
                for title, share, row in charge_rows
            ],
            "Change against far zones, beyond each group's own pre-toll trend",
            "The gap does not follow the charge",
            -12,
            2,
            label_width=330,
            width=w,
        )
    )

    names = {
        "core": "Toll zone",
        "mixed": "Mixed zones",
        "surcharge_ring": "Upper East and West Side",
        "near": "Under 4 km",
        "far_4_8": "4 to 8 km",
        "far_8_12": "8 to 12 km",
    }
    gradient_rows = []
    for rows, section in (
        (f.post_hoc["gradient_fake_2024"], "Fake start, January 2024: a year before any toll"),
        (f.spill["gradient"], "Real start, January 2025"),
    ):
        for r in rows:
            gradient_rows.append(
                {
                    "label": names[r["group"]],
                    "value": r["percent"],
                    "low": r["low"],
                    "high": r["high"],
                    "emphasis": r["group"] == "core",
                    "section": section,
                }
            )
    gradient_chart = both(
        lambda w: charts.interval_chart(
            gradient_rows,
            "Change against zones 12 km or more away",
            "The slope from centre to edge was there before the toll",
            -18,
            2,
            width=w,
        )
    )

    forest_labels = [
        ("Headline", ["Primary"]),
        ("By flow", ["Entering the zone", "Leaving the zone", "Within the zone"]),
        (
            "By service",
            ["Uber and Lyft", "All services, no Flex rows"],
        ),
        (
            "By time of day and year",
            ["Peak-toll hours", "Overnight hours", "2025 only", "2026 only"],
        ),
        (
            "By comparison group",
            [
                "All outside zones pooled",
                "Near zones only, under 4 km",
                "Ring only, 60th to 96th",
                "Reference 8 km or more",
                "Reference 12 km or more",
                "Naive raw pickups",
            ],
        ),
        (
            "By definition and method",
            [
                "Strict core, 20 zones",
                "Core plus mixed zones",
                "Unweighted",
                "PPML (Poisson)",
                "Pre-period from 2022",
                "Pre-period ends Nov 2024",
                "No seasonal terms",
                "Core linear pre-trend",
                "Holiday weeks dropped",
                "World Cup weeks dropped",
                "Clusters merged pairwise",
            ],
        ),
    ]
    forest_rows = []
    for section, labels in forest_labels:
        for label in labels:
            r = f.spec(label)
            forest_rows.append(
                {
                    "label": label,
                    "value": r["percent"],
                    "low": r["wild_low_pct"] if r["wild_low_pct"] is not None else r["cr_low_pct"],
                    "high": r["wild_high_pct"]
                    if r["wild_high_pct"] is not None
                    else r["cr_high_pct"],
                    "emphasis": label == "Primary",
                    "section": section,
                }
            )
    forest = both(
        lambda w: charts.interval_chart(
            forest_rows,
            "Every pre-specified estimate of the effect on trips touching the zone",
            "Dots are estimates, lines are wild bootstrap 95% intervals",
            -24,
            6,
            width=w,
        )
    )
    forest_table = table(
        ["Specification", "Estimate", "Wild 95% interval", "Zones", "Clusters", "Weeks"],
        [
            [
                r["label"],
                signed(r["percent"]),
                interval(r["wild_low_pct"], r["wild_high_pct"])
                if r["wild_low_pct"] is not None
                else "not bootstrapped",
                r["zones"],
                r["clusters"],
                r["weeks"],
            ]
            for r in f.robustness
        ],
    )

    placebo_strip = both(
        lambda w: charts.strip_chart(
            [b["percent"] for b in f.placebo_blocks],
            p["percent"],
            "Primary estimate against placebo blocks",
            "The primary estimate lies outside the placebo blocks",
            -16,
            8,
            "Toll zone",
            "Each dot is one group of far zones given the same treatment",
            width=w,
        )
    )

    level_rows = [
        [
            title,
            fmt_count(f.level_total(g, 2024) if s is None else f.level(g, s, 2024)),
            fmt_count(f.level_total(g, 2025) if s is None else f.level(g, s, 2025)),
            fmt_count(f.level_total(g, 2026) if s is None else f.level(g, s, 2026)),
            signed(f.level_change(g, 2024, 2025, s)),
        ]
        for title, g, s in (
            ("Toll zone, all recorded trips", "core", None),
            ("Toll zone, Uber and Lyft", "core", "hv"),
            ("Toll zone, yellow taxis", "core", "yellow"),
            ("Zones 4 km or more away, all", "far", None),
            ("Upper East and West Side, all", "surcharge_ring", None),
        )
    ]

    bound_rows = [[f"{int(d * 100)}%", signed(v)] for d, v in sorted(bound.items())]

    verdict_word = f.summary["verdict"]["word"]
    return f"""
<header class="hero">
<p class="kicker">Congestion pricing, in the trip records</p>
<h1>The 13% drop that started before the toll</h1>
<p class="deck">Compare Manhattan's toll zone with the outer city and recorded taxi and Uber trips
look down {unsigned(p["percent"], 0)}. But the same comparison finds a {
        unsigned(fake["percent"], 0)
    }
"drop" a year before tolling began. Around the toll's real start, the trend barely moved.</p>
<p class="byline">An analysis of {f.rows_read:.1f} billion TLC trip records, January 2022 to
September 2026. Recorded for-hire trips only: this says nothing about traffic overall.</p>
</header>

<div class="stats">
<div class="stat"><b>{signed(p["percent"])}</b><span>Pre-specified estimate: trips touching the
zone, against zones 4 km or more away</span></div>
<div class="stat"><b>{signed(fake["percent"])}</b><span>The same estimate with a fake start in
January 2024. With no toll, it should be near zero</span></div>
<div class="stat"><b>{signed(jan25["percent"])}</b><span>Post-hoc: shift in growth at the real
start, 26 weeks either side ({interval(jan25["wild_low_pct"], jan25["wild_high_pct"])})</span></div>
</div>

<p>New York began tolling Manhattan below 60th Street on 5 January 2025. Taxis and app cars do not
pay the $9 driver toll. They pay a flat charge on each trip: $0.75 for a yellow cab, $1.50 for Uber
or Lyft. So what happened to their trips is an open question, and it could go either way. The charge
makes a ride slightly dearer. But if private traffic thins out, for-hire cars may get quicker and
more available, and some former drivers may hail one.</p>

<p>This analysis set out to measure that. The design, including the exact words we would use for
each possible outcome, was committed before any estimate was run. Under those rules the result is
<strong>{verdict_word}</strong>: we cannot report the {unsigned(p["percent"], 0)} as an effect of
the toll, because the test built to catch a false positive caught one.</p>

<p>Our read, which goes beyond what was pre-specified: over the first year the toll moved these
trips very little, somewhere between {signed(f.low_read)} and {signed(f.high_read)} across checks
built to survive the pre-existing trend. An effect of a few percent in either direction cannot be
ruled out. The large gap between the zone and the rest of the city is real, but it was already
opening, and through 2025 it kept opening at much the same pace. What happened in 2026 is
unresolved.</p>

<h2>The gap is real, and it is old</h2>

<p>Before any modelling, here are the plain counts. In the same 38 weeks of each year, recorded
trips touching the zone went from {fmt_count(f.level_total("core", 2024))} a week in 2024 to
{fmt_count(f.level_total("core", 2025))} in 2025, a change of
{signed(f.level_change("core", 2024, 2025))}. Inside that near-flat total, Uber and Lyft fell
{unsigned(f.level_change("core", 2024, 2025, "hv"))} while yellow records rose
{unsigned(f.level_change("core", 2024, 2025, "yellow"))}.</p>

<p>Meanwhile, outer zones kept growing: {signed(f.level_change("far", 2023, 2024))} from 2023 to
2024, before any toll, then {signed(f.level_change("far", 2024, 2025))} from 2024 to 2025. Any
method that compares the zone with those places inherits that growth as if it were a toll
effect.</p>

{
        figure(
            "The outer city was pulling away before the toll",
            "Change in weekly trips, trailing 52-week average, against the year ending January 2023",
            legend(
                [
                    ("Trips touching the toll zone", "var(--blue)"),
                    ("Zones 4 km or more away", "var(--orange)"),
                ]
            )
            + both(lambda w: divergence_chart(f, w)[0]),
            details(
                "Table",
                table(["Week starting", "Touching the zone", "4 km or more away"], divergence_rows),
            ),
        )
    }

{
        details(
            "Weekly trips, same 38 weeks of each year",
            table(["Group", "2024", "2025", "2026", "2024 to 2025"], level_rows),
        )
    }

<h2>Who actually pays</h2>

<p>The toll line is 60th Street, but taxi zones were drawn long before it and stop at 59th. The
FDR Drive and West Side Highway are excluded, while the shoreline zones beside them are not. So we
overlaid the MTA's geofence on the TLC's zone map, then checked the result against the
<code>cbd_congestion_fee</code> column, which records the charge on each trip.</p>

<p>That column is a direct signal of treatment, and it is sharp. In the last week before tolling,
it was positive on {before_rate:.2f}% of trips touching the zone's 37 core zones. In the first week,
it was positive on {first_rate:.1f}%, and it has stayed above {floor_rate:.0f}% for both services in
every quarter since. It also reads the boundary better than geometry alone does: Battery Park is only
9% inside the geofence by area, yet {f.battery_park_rate:.0f}% of its trips to outside zones are
charged.</p>

<p>Seven zones are neither one thing nor the other. Lincoln Square, Lenox Hill, the Upper East Side's
southern zone and Central Park all begin at 59th, so a sliver of each is inside the line. They are
kept apart from both sides of every comparison. The charge is also easy to confuse with an older
one: since 2019, trips touching Manhattan south of 96th Street have paid a $2.50 or $2.75 state
surcharge. That charge did not change, so it is not the treatment. But it means the Upper East and
Upper West Side paid the old charge throughout and do not pay the new one on trips that stay out
of the zone, which makes them the closest available comparison.</p>

{
        figure(
            "What each taxi zone is in this analysis",
            "Outline: the MTA geofence. Hover a zone for its name and charged share.",
            '<div class="mapwrap">'
            + legend(
                [
                    ("Toll zone (37)", "var(--blue)"),
                    ("Mixed (7)", "#86b6ef"),
                    ("Older surcharge only (5)", "var(--orange)"),
                    ("Under 4 km", "#f5b79e"),
                    ("Reference", "var(--grid)"),
                    ("Airports", "var(--ghost)"),
                ]
            )
            + zone_map(f)
            + "</div>",
            source="Sources: MTA Central Business District Geofence (data.ny.gov), TLC taxi zones.",
        )
    }

<h2>The test the headline failed</h2>

<p>The pre-specified estimate compares weekly trips touching the core zones with trips between
zones at least 4 km from the line, adjusting for each zone, each week and each group's seasonal
pattern. It gives {signed(p["percent"])}, with a wild bootstrap interval of
{interval(p["wild_low_pct"], p["wild_high_pct"])}. Its precision is not the problem.</p>

{
        figure(
            "The zone slid against far zones well before 2025",
            "Trips touching the zone against far zones, each 4-week bin against the same weeks of 2024",
            event,
            details(
                "Table",
                table(
                    ["Bin starting", "Estimate", "95% interval"],
                    [
                        [r["start"], signed(r["percent"]), interval(r["low"], r["high"])]
                        for r in event_rows
                    ],
                ),
            ),
            source="Bands are cluster-robust 95% intervals. 2024 is the reference year, so it has no bins.",
        )
    }

<p>Read the chart left to right. In 2023 the zone stood about {
        unsigned(pre_gap, 0)
    } above where 2024
would put it. In 2025 it stood about {unsigned(post_gap, 0)} below. That is one steady slide, not a
level that drops when tolling starts. The large step is at January 2026, a full year after the start,
and we could not explain it. The charge amounts in the records are the same in December 2025 and
February 2026, and we found no rule change.</p>

<p>The placebo makes this concrete. Move the start date to January 2024, drop everything after the
real start, and run the identical regression. It returns {signed(fake["percent"])}, with an interval
of {interval(fake["low"], fake["high"])}. Every other fake start says the same. A method that
reports a {unsigned(fake["percent"], 0)} effect of a toll that did not exist cannot be trusted to
report the effect of one that did.</p>

{
        figure(
            "Fake start dates give nearly the same answer",
            "Same regression with invented start dates, using only data from before the real toll",
            fake_chart,
            source="Lines are wild cluster bootstrap 95% intervals.",
        )
    }

<p>Some standard checks do pass, and it is worth saying why they did not catch this. Against 28
placebo blocks of far zones, the primary estimate is larger than all of them
(p = {randomisation["p"]:.3f}, the smallest this test can give). But every placebo block is drawn
from the outer city, and none of them shares Manhattan's trend. A test can only reject what it
has the power to see. The fake-date test could see the trend. The placebo-block test could not.</p>

<h2>Did anything change at the start?</h2>

<p>From here on, the analysis is post-hoc. None of it was in the original design: it was written after
seeing the failure above, so it deserves less weight. The narrower question is whether growth in the
zone shifted when tolling began, relative to how it was already behaving.</p>

<p>Take each zone's year-on-year change, which removes seasons, and compare the weeks just after
5 January 2025 with the weeks just before. A steady divergence cancels out. A toll that hit the zone
would show up as a step. For all recorded trips the shift is {signed(h13["percent"])} over 13
weeks either side, {signed(h26["percent"])} over 26 and {signed(h52["percent"])} over 52. For Uber
and Lyft alone it is {signed(hv13["percent"])}, {signed(hv26["percent"])} and
{signed(hv52["percent"])}. So Uber and Lyft show a small fall, while yellow records, which were
rising for other reasons, offset it in the total.</p>

{
        figure(
            "Growth barely shifted at the toll's start",
            "Change in year-on-year growth, toll zone against far zones, after against before each date",
            break_chart,
            source="Post-hoc. Lines are wild cluster bootstrap 95% intervals.",
        )
    }

<p>The intervals look tight, and they should not be taken at face value. The same test at January
2024 gives {signed(jan24["percent"])}, and at January 2026 gives {signed(jan26["percent"])}, with
no toll starting at either. Growth gaps of that size arrive without any policy to cause them, so
an effect of a few percent at the real start could hide inside that normal movement.</p>

<p>A second approach fits a straight line to the zone's pre-toll gap, using pre-toll weeks only, and
measures departures from it. Through the end of 2025 the departure is {signed(through["percent"])}
({interval(through["wild_low_pct"], through["wild_high_pct"])}). Over the whole window to September
2026 it is {signed(core_trend["percent"])}, nearly all of it from the unexplained step in 2026.
The method is imperfect: at a fake start in January 2024 it still returns
{signed(fake_trend["percent"])}, so about two points of any figure here may be drift.</p>

<h2>The zones that pay nothing moved the same way</h2>

<p>If the charge were pushing trips down, the gap should follow the charge. It does not. Against far
zones, and beyond their own pre-toll trends, the Upper East and West Side moved
{signed(ring_trend["percent"])} on trips where only {
        ring_charged:.0f}% paid the CRZ charge, close to
the zone's {signed(core_trend["percent"])}. Mixed zones, at {mixed_charged:.0f}% charged, moved
{
        signed(mixed_trend["percent"])
    }, which is well over a quarter of the core's change. Compare the zone
directly with the Upper East and West Side, and the gap is {signed(ring_clean["percent"])}
({interval(ring_clean["wild_low_pct"], ring_clean["wild_high_pct"])}).</p>

{
        figure(
            "The gap does not follow the charge",
            "Departure from each group's pre-toll trend, against zones 4 km or more away",
            charge_chart,
            source="Post-hoc. Lines are wild cluster bootstrap 95% intervals.",
        )
    }

<p>Peak hours say the same thing. The private car toll falls by 75% overnight and the for-hire charge
does not, so anything flowing from less private traffic should show up at peak. The estimates are
{peak} at peak and {overnight} overnight. Both come from the pre-specified comparison, so the levels
carry the same trend problem. What matters is how close they are.</p>

<h2>How much did the comparison zones move?</h2>

<p>Displaced drivers and riders do not vanish. A rider who now gets out at 61st Street or a driver
who now waits in Long Island City changes counts outside the zone, which breaks the assumption that
comparison zones are unaffected. The design treated this as the central threat, and gave zones near
the line their own terms rather than using them as controls.</p>

<p>Displacement into neighbours would look like a gain just outside the line. There is no such gain.
Against far zones, the Upper East and West Side moved {signed(rings["surcharge_ring"]["percent"])},
about {ring_weekly} fewer trips a week than the far zones' path implies, and zones under 4 km moved
{signed(rings["near"]["percent"])}, about {near_weekly} fewer. The estimate for the zone does move
with the choice of comparison, from {ring_only} against the ring to {far_only} against zones 12 km
out.</p>

<p>That slope from centre to edge is not a toll effect either. A fake start in January 2024
produces the same shape, a year before anyone paid anything.</p>

{
        figure(
            "The centre-to-edge slope was there a year earlier",
            "Each ring against zones 12 km or more away, at a fake start and at the real one",
            gradient_chart,
            source="Post-hoc for the fake start. Lines are cluster-robust 95% intervals.",
        )
    }

<p>A gradient cannot rule out displacement spread evenly across the far zones. So we size that
directly. Core trips are {f.spill["core_to_far_ratio"]:.2f} times far-zone trips, so if a share of
any change in the core reappeared evenly in far zones, the estimate would be overstated by this
much:</p>

{table(["Share of the change displaced to far zones", "Implied estimate"], bound_rows)}

<p>Even if every lost trip had reappeared in far zones, an extreme case, the estimate would still be
{signed(bound[1.0])}. Displacement on its own cannot close the gap between the headline and the
checks above. The pre-existing trend can.</p>

<h2>A second method disagrees with itself</h2>

<p>Synthetic control builds a stand-in for the zone from a weighted blend of far-zone clusters.
Here that blend ends up as one cluster in upper Manhattan, and it fits poorly: its pre-period error is
{100 * sc["rmspe_pre"]:.0f}%, about the size of the effect in question. It gives
{signed(sc["percent"])}, and synthetic difference-in-differences gives {signed(sdid["percent"])}.
The zone's average gap is larger than any placebo donor's (p = {sc["att_p"]:.2f}). But so is its
pre-toll misfit, and the ratio of the two is ordinary (p = {sc["ratio_p"]:.2f}). With a fake January
2024 start, synthetic control still returns {signed(fake_sc["synthetic_control"]["percent"])}, while
synthetic difference-in-differences returns {signed(fake_sc["synthetic_did"]["percent"])}. The same
method run on year-on-year changes for 2025, which handles the trend directly, gives
{signed(sd["percent"])}.</p>

{
        figure(
            "A large gap, on a fit that was already poor",
            f"13-week average gap. Blue: toll zone. Grey: far-zone clusters, treated the same way. {placebo_paths(f)[1]} very small clusters run off this axis and are not drawn.",
            both(lambda w: gaps_chart(f, w)),
        )
    }

<h2>The published estimate, one year earlier</h2>

<p>Pandey, Guler and Gayah (<em>Findings</em>, 2026) estimate that Uber and Lyft trips in the zone fell
5.95% over the first 14 weeks of 2025, on weekdays. Our approximation of that design, using all days
because these aggregates do not separate weekdays, gets {signed(pub_real["percent"])}, so the result
reproduces. Running the same comparison a year earlier, 2024 against 2023 with no toll, gives
{signed(pub_fake["percent"])} ({interval(pub_fake["low"], pub_fake["high"])}). This is our
approximation, not their code, but the pattern matches what we found everywhere else.</p>

<h2>The yellow cab trap</h2>

<p>Yellow records need care, and this analysis nearly got them wrong. The usual cleaning step drops
negative-amount rows as reversals. In this data, rows with a negative amount and no matching original
went from {f.unpaired_share("2024-12"):.0f}% of yellow trips in December 2024 to
{f.unpaired_share("2025-05"):.0f}% in May 2025. Most are Flex Fare rows, and their distances and
durations match ordinary Flex trips. They are real trips with odd coding. Filtering on the sign would
have deleted them mostly after the toll, and faked a fall. We now remove only matched pairs.</p>

<p>The bigger problem is that Flex Fare, which became a permanent TLC program in September 2024,
grew from {f.flex_share("2024-12"):.0f}% of yellow records in December 2024 to
{
        f.flex_share(
            "2025-02"
        ):.0f}% by February 2025, and it went everywhere. Yellow trips between far zones rose from {
        fmt_count(far_y[2024]["trips_per_week"])
    } to
{fmt_count(far_y[2025]["trips_per_week"])} a week from 2024 to 2025, and
{
        100
        * far_y[2025][
            "flex_share"
        ]:.0f}% of the 2025 trips were Flex. We could not establish the cause.
That makes far zones useless as a yellow comparison, and the pre-specified yellow estimate of
{signed(yellow["percent"])} is an artifact, not a finding. Metered rows alone give
{signed(metered["percent"])} with an interval of
{interval(metered["wild_low_pct"], metered["wild_high_pct"])}, and the interval includes zero.</p>

{
        figure(
            "Flex Fare surged just as tolling began",
            "Share of yellow taxi records coded as Flex Fare, by month",
            both(lambda w: flex_chart(w)[0]),
            details("Table", table(["Month", "Flex share", "Yellow trips kept"], flex_rows)),
        )
    }

<h2>Every pre-specified estimate</h2>

<p>Nearly all of these agree, which sounds reassuring until you remember the fake dates agree too.
What matters here is which rows move. The row with a linear pre-trend ({signed(linear["percent"])})
and the ring comparison ({
        ring_only
    }) are the ones that deal with the trend, and they sit far from the
rest. The first of those fits one slope across pre and post weeks, so the post-toll weeks leak into
its trend, which is why the post-hoc version uses pre-toll weeks only. Uber and Lyft alone give
{signed(hv["percent"])}. Yellow-only rows are left off the chart because of the Flex problem above,
and appear in the table.</p>

{
        figure(
            "Pre-specified estimates",
            "All against far zones unless stated. The primary failed its fake-start placebo.",
            forest,
            details("All estimates as a table", forest_table),
        )
    }

{
        figure(
            "The toll zone against placebo blocks",
            "The real estimate is outside all 28 placebo blocks. It is the wrong test for this problem.",
            placebo_strip,
        )
    }

<h2>What about fares and speeds?</h2>

<p>The mean base fare on Uber and Lyft trips inside the zone went from
${within_2023["fare_per_trip"]:.2f} in 2023 to ${within_hv[2024]["fare_per_trip"]:.2f} in 2024 and
${
        within_hv[2025][
            "fare_per_trip"
        ]:.2f} in 2025, excluding the $1.50 charge. So fares inside the zone
were already rising before the toll. Far-zone fares went from ${far_2023["fare_per_trip"]:.2f} to
${far_hv[2024]["fare_per_trip"]:.2f} to ${
        far_hv[2025]["fare_per_trip"]:.2f}. Recorded speeds on trips
inside the zone were {within_hv[2024]["mph"]:.1f} mph for Uber and Lyft in 2024 and
{within_hv[2025]["mph"]:.1f} mph in 2025, and {within_y[2024]["mph"]:.1f} and
{
        within_y[2025][
            "mph"
        ]:.1f} mph for yellow cabs. These are plain averages for trips that happened, with
no causal adjustment. They are not a measure of traffic or congestion.</p>

<h2>What this cannot tell you</h2>

<ul>
<li><strong>Anything about total traffic, congestion or pollution.</strong> TLC records cover yellow
taxis and high-volume for-hire cars. Private cars, trucks, buses and everything else are absent.
Nothing here contradicts studies of those.</li>
<li><strong>Where a lost trip went.</strong> The subway, a bicycle, a car or nowhere all look the
same in these records.</li>
<li><strong>What happens along a route.</strong> Records have a pickup and a dropoff zone. A trip through
the zone shows only in the fee column.</li>
<li><strong>What drivers did between trips.</strong> There is no driver identifier, so repositioning
is invisible. The displacement sizing above is a bound, not a measurement.</li>
<li><strong>Why the outer city grew so fast</strong>, or what happened in January 2026.</li>
<li><strong>Whether a citywide or Manhattan-wide shock coincided with the toll.</strong> Offices,
tourism and app pricing all changed in this period. If something lifted the zone just as the toll
pushed it down, a flat line would hide both.</li>
</ul>

<h2>How this was done</h2>

<p>The pipeline scans TLC parquet lazily with Polars, one month at a time, on the streaming engine.
It read {f.gb_read:.0f} GB and {f.rows_read:.2f} billion rows at a peak of 3.65 GB of memory, and
keeps only small aggregates. Inference uses wild cluster restricted bootstrap with Webb weights over
{f.summary["panel"]["clusters"]} spatial clusters, {
        f.summary["panel"]["core_clusters"]
    } of them in the
zone. With so few treated clusters this is known to be imperfect, which is why placebo tests carry
more of the weight here.</p>

<p>The design, its deviations and the post-hoc additions are in
<a href="{REPO_URL}/blob/main/DESIGN.md">DESIGN.md</a>. Code and all results are in the
<a href="{REPO_URL}">repository</a>.</p>

<footer>
<p><strong>Sources.</strong> NYC TLC Trip Record Data and zone files; MTA Central Business
District Geofence (data.ny.gov); mta.info tolling pages; TLC data dictionaries, March 2025; TLC
congestion surcharge page and Flex Fare rules; A. Pandey, I. Guler and V. Gayah, "The Effect of the
New York City Congestion Toll on Trips Served by Transportation Network Companies",
<em>Findings</em>, 2026. Methods: Butts (2023) on spatial spillovers, Roodman et al. (2019) and
MacKinnon and Webb (2017) on wild cluster bootstrap, Arkhangelsky et al. (2021) on synthetic
difference-in-differences.</p>
<p>TLC publishes these records without warranty of accuracy. Weeks run Sunday to Saturday. Percentages
are exp(b) - 1, where b is the regression coefficient on log trips.</p>
</footer>
"""


def page(content: str) -> str:
    assets = charts.ASSETS
    css = (assets / "chart.css").read_text() + (assets / "page.css").read_text()
    script = (assets / "tip.js").read_text()
    description = (
        "What 1.3 billion TLC trip records show about congestion pricing and for-hire trips."
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>The Cordon Effect</title>
<meta name="description" content="{description}">
<style>{css}</style>
</head>
<body>
<main>{content}</main>
<div id="tip" role="status"></div>
<script>{script}</script>
</body>
</html>
"""


def write_figures(f: Facts, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    chart, _ = divergence_chart(f)
    (directory / "divergence.svg").write_text(charts.standalone(chart))
    event_rows = [r for r in f.events["series"]["core"] if r["week"] >= config.PRIMARY_PRE_WEEK]
    event = charts.event_study_chart(
        event_rows,
        "Toll zone against far zones",
        "Each bin against the same weeks of 2024",
        -25,
        15,
    )
    (directory / "event_study.svg").write_text(charts.standalone(event))


def report_stage() -> None:
    facts = Facts.load()
    config.SITE.mkdir(exist_ok=True)
    (config.SITE / "index.html").write_text(page(body(facts)))
    write_figures(facts, config.REPO / "docs" / "figures")
    size = (config.SITE / "index.html").stat().st_size / 1e6
    print(f"site/index.html {size:.2f} MB")
