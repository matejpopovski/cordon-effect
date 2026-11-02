"""Post-hoc diagnostics, added after the pre-specified estimates failed their placebo.

None of this is in the original design; see the deviations log in DESIGN.md. The pre-specified
primary estimate fails because the core was already losing ground to the far zones. These
checks ask the narrower question that remains: did anything change at the toll's start, over
and above that trend?
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import polars as pl

from . import config, did
from .estimate import EXPOSED, by_distance, only_groups
from .wild import WildBootstrap


def report(fitted: did.Fit, name: str, draws: int) -> dict:
    boot = WildBootstrap(fitted, name, draws, config.SEED)
    low, high = boot.interval()
    return {
        "percent": did.pct(fitted.coef(name)),
        "wild_low_pct": did.pct(low),
        "wild_high_pct": did.pct(high),
        "wild_p": boot.p_value(0.0),
        "cr_se": fitted.stderr(name),
    }


def fit_extrapolated_trend(p: did.Panel, group: str, start: int = 0) -> did.Fit:
    """Effect of `start` on one group against the other, with the trend from before it only.

    One dummy per post-week for the group frees the post-period from the trend, so the slope
    comes entirely from pre-period weeks. The columns are reparameterised as an overall Post
    term plus sum-to-zero contrasts, which makes the Post coefficient the mean post-week effect.
    """
    member = (p.groups == group).astype(float)
    post_weeks = p.weeks[p.weeks >= start]
    columns = [np.outer(p.weeks >= start, member)]
    names = [group]
    last = post_weeks[-1]
    for k in post_weeks[:-1]:
        contrast = (p.weeks == k).astype(float) - (p.weeks == last).astype(float)
        columns.append(np.outer(contrast, member))
        names.append(f"contrast[{k}]")
    pre_length = (p.weeks < start).sum()
    centred = np.where(p.weeks < start, (p.weeks - start) / 52.0, 0.0)
    columns.append(np.outer(centred, member))
    names.append("trend")
    season = np.mod(p.weeks, config.SEASON_LENGTH)
    for s in sorted(set(season[:pre_length].tolist()))[1:]:
        columns.append(np.outer(season == s, member))
        names.append(f"season[{s}]")
    return did.fit(p, columns, names)


def trend_table(primary: did.Panel, extended: did.Panel, draws: int) -> dict:
    """Pre-only trend extrapolation for each exposed group, and the same at fake starts."""
    out: dict = {"real_start": {}, "fake_start": {}}
    for group in ("core", "mixed", "surcharge_ring", "near"):
        two_arm = only_groups(primary, [group, "far"])
        fitted = fit_extrapolated_trend(two_arm, group)
        out["real_start"][group] = {
            **report(fitted, group, draws),
            "trend_pct_per_year": did.pct(fitted.coef("trend")),
        }

    pre_only = extended.subset_weeks(extended.weeks < 0)
    two_arm = only_groups(pre_only, ["core", "far"])
    for fake in (-52, -39, -26):
        fitted = fit_extrapolated_trend(two_arm, "core", start=fake)
        out["fake_start"][str(fake)] = {
            "start": str(config.week_start(fake)),
            **report(fitted, "core", draws),
            "trend_pct_per_year": did.pct(fitted.coef("trend")),
        }

    both = only_groups(primary, ["core", "surcharge_ring"])
    out["core_vs_ring"] = report(fit_extrapolated_trend(both, "core"), "core", draws)

    for label, last in (("through_2025", 51), ("first_26_weeks", 25)):
        window = only_groups(primary, ["core", "far"])
        window = window.subset_weeks(window.weeks <= last)
        out[label] = report(fit_extrapolated_trend(window, "core"), "core", draws)
    return out


def yoy_panel(p: did.Panel, first: int, last: int) -> did.Panel:
    """Replace log trips with the 52-week log change, for weeks first to last.

    The panel type takes logs of `counts`, so the change is stored exponentiated. Differencing
    removes zone seasonality, and a constant growth gap between groups lands in the group's
    fixed effect, so a level shift in growth is what the Post term picks up.
    """
    lag = config.SEASON_LENGTH
    index = np.flatnonzero((p.weeks >= first) & (p.weeks <= last))
    change = p.log_y[index] - p.log_y[index - lag]
    return replace(p, weeks=p.weeks[index], counts=np.exp(change))


def growth_break(p: did.Panel, group: str, start: int, horizon: int, draws: int) -> dict:
    """Did year-on-year growth in `group` shift at `start`, relative to the other group?"""
    window = yoy_panel(p, start - horizon, start + horizon - 1)
    columns, names = did.post_terms(window, [group], start=start)
    return {
        "group": group,
        "start": str(config.week_start(start)),
        "horizon_weeks": horizon,
        **report(did.fit(window, columns, names), group, draws),
    }


def break_tests(extended: did.Panel, hv: did.Panel, draws: int) -> dict:
    """Growth breaks at the true start and at the anniversaries around it."""
    core_far = only_groups(extended, ["core", "far"])
    core_ring = only_groups(extended, ["core", "surcharge_ring"])
    hv_far = only_groups(hv, ["core", "far"])
    out = {"all_vs_far": [], "uber_lyft_vs_far": [], "all_vs_ring": [], "anniversaries": []}
    for horizon in (13, 26, 52):
        out["all_vs_far"].append(growth_break(core_far, "core", 0, horizon, draws))
        out["uber_lyft_vs_far"].append(growth_break(hv_far, "core", 0, horizon, draws))
        out["all_vs_ring"].append(growth_break(core_ring, "core", 0, horizon, draws))
    for start, title in (
        (-52, "Jan 2024, placebo"),
        (0, "Jan 2025, toll"),
        (52, "Jan 2026, no known event"),
    ):
        row = growth_break(core_far, "core", start, 26, draws)
        out["anniversaries"].append({**row, "title": title})
    return out


def gradient_before(extended: did.Panel, zones: pl.DataFrame) -> list[dict]:
    """The distance gradient at a fake start of January 2024, a year before the toll."""
    pre = extended.subset_weeks(extended.weeks < 0)
    banded = by_distance(pre, zones)
    groups = [*EXPOSED, "far_4_8", "far_8_12"]
    fitted = did.fit_post(banded, groups, start=-52)
    return [
        {
            "group": g,
            "percent": did.pct(fitted.coef(g)),
            "low": did.pct(fitted.coef(g) - 1.96 * fitted.stderr(g)),
            "high": did.pct(fitted.coef(g) + 1.96 * fitted.stderr(g)),
        }
        for g in groups
    ]


def detrended_event_study(p: did.Panel, width: int = 4) -> list[dict]:
    """Core against far in 4-week bins, around the line fitted to pre-period weeks alone."""
    two_arm = only_groups(p, ["core", "far"])
    core = two_arm.groups == "core"
    w = two_arm.weights
    gap = (two_arm.log_y[:, core] * w[core]).sum(axis=1) / w[core].sum() - (
        two_arm.log_y[:, ~core] * w[~core]
    ).sum(axis=1) / w[~core].sum()

    weeks = two_arm.weeks
    season = np.mod(weeks, config.SEASON_LENGTH)
    pre = weeks < 0
    design = np.column_stack(
        [weeks / 52.0, *[season == s for s in sorted(set(season[pre].tolist()))]]
    ).astype(float)
    coefficients = np.linalg.lstsq(design[pre], gap[pre], rcond=None)[0]
    residual = gap - design @ coefficients
    bins = np.floor_divide(weeks, width)
    rows = []
    for label in sorted(set(bins.tolist())):
        mask = bins == label
        rows.append(
            {
                "week": int(label * width),
                "start": str(config.week_start(int(label * width))),
                "percent": did.pct(float(residual[mask].mean())),
            }
        )
    return rows


def level_table(frame: pl.DataFrame, zones: pl.DataFrame) -> dict:
    """Plain trip counts by group for the same 38 weeks of each year, with no modelling."""
    table = frame.filter(pl.col("flow") != "to_core").join(
        zones.select(pl.col("location_id").cast(frame.schema["zone"]).alias("zone"), "group"),
        on="zone",
    )
    selected = table.filter(
        ((pl.col("group") == "core") & pl.col("flow").is_in(["entering", "leaving", "within"]))
        | (pl.col("group").is_in(["far", "near", "surcharge_ring"]) & (pl.col("flow") == "outside"))
    )

    def window(first: int) -> pl.DataFrame:
        return (
            selected.filter(pl.col("week").is_between(first, first + 37))
            .group_by("group", "service")
            .agg((pl.col("trips").sum() / 38).alias(f"w{first}"))
        )

    joined = window(-104).join(window(-52), on=["group", "service"])
    joined = joined.join(window(0), on=["group", "service"]).join(
        window(52), on=["group", "service"]
    )
    return {
        "weeks": "Same 38 weeks each year, starting the first Sunday of January",
        "rows": joined.sort("group", "service")
        .rename({"w-104": "y2023", "w-52": "y2024", "w0": "y2025", "w52": "y2026"})
        .to_dicts(),
    }


def published_design(frame: pl.DataFrame, zones: pl.DataFrame) -> dict:
    """An all-days approximation of Pandey et al. (2026), then the same design a year earlier.

    Their sample is weekdays, which these aggregates do not separate. Everything else follows
    the paper: Uber and Lyft trips, first 14 weeks against the same weeks a year before,
    Manhattan zones fully outside the charged area with trips into it removed, unweighted, and
    standard errors clustered by zone.
    """
    hv = did.load_panel(frame, zones, service="hv", first_week=config.FIRST_WEEK)
    manhattan = set(zones.filter(pl.col("borough") == "Manhattan")["location_id"].to_list())
    keep = (hv.groups == "core") | (
        np.isin(hv.zones, sorted(manhattan)) & np.isin(hv.groups, ["surcharge_ring", "near", "far"])
    )
    selected = hv.subset_zones(keep)
    selected = replace(
        selected,
        groups=np.where(selected.groups == "core", "core", "far"),
        clusters=selected.zones.astype(str),
    )

    def design(start: int) -> dict:
        weeks = selected.weeks
        window = ((weeks >= start) & (weeks < start + 14)) | (
            (weeks >= start - 52) & (weeks < start - 38)
        )
        fitted = did.fit_post(
            selected.subset_weeks(window), ["core"], seasonal=False, weighted=False, start=start
        )
        beta, se = fitted.coef("core"), fitted.stderr("core")
        return {
            "start": str(config.week_start(start)),
            "percent": did.pct(beta),
            "low": did.pct(beta - 1.96 * se),
            "high": did.pct(beta + 1.96 * se),
        }

    return {"real_2025_vs_2024": design(0), "placebo_2024_vs_2023": design(-52)}


def mechanism_table(flows: pl.DataFrame) -> list[dict]:
    """Fare, distance and speed per trip by flow and year, descriptive only."""
    rows = []
    selections = {
        "Within the core": (pl.col("pu_group") == "core") & (pl.col("do_group") == "core"),
        "Far to far": (pl.col("pu_group") == "far") & (pl.col("do_group") == "far"),
    }
    for service in ("hv", "yellow"):
        subset = flows.filter(pl.col("service") == service)
        for title, where in selections.items():
            for year, first in ((2023, -104), (2024, -52), (2025, 0), (2026, 52)):
                s = subset.filter(where & pl.col("week").is_between(first, first + 37))
                rows.append(
                    {
                        "service": service,
                        "flow": title,
                        "year": year,
                        "fare_per_trip": s["fare"].sum() / max(s["fare_trips"].sum(), 1),
                        "mph": s["miles"].sum() / max(s["seconds"].sum(), 1) * 3600,
                        "miles_per_trip": s["miles"].sum() / max(s["speed_trips"].sum(), 1),
                        "trips_per_week": s["trips"].sum() / 38,
                        "flex_share": s["flex_trips"].sum() / max(s["trips"].sum(), 1),
                    }
                )
    return rows


def run(
    frame: pl.DataFrame, flows: pl.DataFrame, zones: pl.DataFrame, primary: did.Panel, draws: int
) -> dict:
    extended = did.load_panel(frame, zones, first_week=config.FIRST_WEEK)
    hv = did.load_panel(frame, zones, service="hv", first_week=config.FIRST_WEEK)
    yellow_ring = did.load_panel(frame, zones, service="yellow")
    hv_ring = did.load_panel(frame, zones, service="hv")

    def ring_only(p: did.Panel) -> dict:
        two_arm = only_groups(p, ["core", "surcharge_ring"])
        return report(did.fit_post(two_arm, ["core"]), "core", draws)

    ring_placebo = only_groups(
        extended.subset_weeks(extended.weeks < 0), ["core", "surcharge_ring"]
    )
    near_placebo = only_groups(extended.subset_weeks(extended.weeks < 0), ["core", "near"])
    return {
        "note": "Post-hoc. Not part of the pre-specified design.",
        "extrapolated_trend": trend_table(primary, extended, draws),
        "break_tests": break_tests(extended, hv, draws),
        "gradient_fake_2024": gradient_before(extended, zones),
        "detrended_event_study": detrended_event_study(extended),
        "levels": level_table(frame, zones),
        "mechanism": mechanism_table(flows),
        "published_design": published_design(frame, zones),
        "ring_comparisons": {
            "yellow": ring_only(yellow_ring),
            "uber_lyft": ring_only(hv_ring),
            "fake_start_2024_ring": report(
                did.fit_post(ring_placebo, ["core"], start=-52), "core", draws
            ),
            "fake_start_2024_near": report(
                did.fit_post(near_placebo, ["core"], start=-52), "core", draws
            ),
        },
    }
