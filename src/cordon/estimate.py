"""Runs every estimator in DESIGN.md from the committed aggregates and writes results/."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, replace
from datetime import date

import numpy as np
import polars as pl
import statsmodels.api as sm

from . import config, did, panel, synth
from .wild import WildBootstrap, wild_test

EXPOSED = ["core", "mixed", "surcharge_ring", "near"]
WORLD_CUP = (date(2026, 6, 11), date(2026, 7, 19))
NOVEMBER_NOTICE_WEEK = -9


def relabel(p: did.Panel, mapping: dict[str, str]) -> did.Panel:
    return replace(p, groups=np.array([mapping.get(g, g) for g in p.groups]))


def only_groups(p: did.Panel, groups: list[str]) -> did.Panel:
    return p.subset_zones(np.isin(p.groups, groups))


def merged_clusters(p: did.Panel) -> did.Panel:
    """Merge neighbouring Ward labels pairwise. Labels follow dendrogram order."""

    def merge(label: str) -> str:
        head, _, tail = label.rpartition(":")
        return f"{head}:{(int(tail) + 1) // 2}"

    return replace(p, clusters=np.array([merge(c) for c in p.clusters]))


def by_distance(p: did.Panel, zones: pl.DataFrame) -> did.Panel:
    """Split the far group into distance bands for the spillover gradient."""
    km = dict(zip(zones["location_id"].to_list(), zones["km_to_crz"].to_list(), strict=True))
    edges = config.GRADIENT_EDGES_KM

    def band(zone: int, group: str) -> str:
        if group != "far":
            return group
        distance = float(km[int(zone)])
        if distance < edges[1]:
            return "far_4_8"
        return "far_8_12" if distance < edges[2] else "far_12_plus"

    return replace(
        p,
        groups=np.array([band(z, g) for z, g in zip(p.zones, p.groups, strict=True)]),
    )


@dataclass
class Estimate:
    label: str
    family: str
    beta: float
    se: float
    wild_low: float | None
    wild_high: float | None
    wild_p: float | None
    zones: int
    clusters: int
    weeks: int

    def row(self) -> dict:
        return {
            "label": self.label,
            "family": self.family,
            "beta": self.beta,
            "percent": did.pct(self.beta),
            "se": self.se,
            "cr_low_pct": did.pct(self.beta - 1.96 * self.se),
            "cr_high_pct": did.pct(self.beta + 1.96 * self.se),
            "wild_low_pct": None if self.wild_low is None else did.pct(self.wild_low),
            "wild_high_pct": None if self.wild_high is None else did.pct(self.wild_high),
            "wild_p": self.wild_p,
            "zones": self.zones,
            "clusters": self.clusters,
            "weeks": self.weeks,
        }


def estimate(
    label: str,
    family: str,
    p: did.Panel,
    exposed: list[str],
    draws: int,
    name: str = "core",
    **kwargs,
) -> Estimate:
    fitted = did.fit_post(p, exposed, **kwargs)
    wild = wild_test(fitted, name, draws, config.SEED)
    return Estimate(
        label,
        family,
        fitted.coef(name),
        fitted.stderr(name),
        wild.ci_low,
        wild.ci_high,
        wild.p_value,
        len(p.zones),
        fitted.n_clusters,
        len(p.weeks),
    )


def fit_ppml(p: did.Panel, exposed: list[str]) -> tuple[float, float]:
    """Poisson with explicit dummies. Slower, but it is the check on the log-linear form."""
    t, n = p.counts.shape
    y = p.counts.ravel()
    zone_dummies = np.tile(np.eye(n), (t, 1))
    week_dummies = np.repeat(np.eye(t)[:, 1:], n, axis=0)
    columns, names = did.post_terms(p, exposed)
    more, more_names = did.seasonal_terms(p, exposed)
    terms = np.stack(columns + more, axis=-1).reshape(t * n, len(columns) + len(more))
    used = terms.std(axis=0) > 0
    x = np.hstack([terms[:, used], zone_dummies, week_dummies])
    groups = np.tile(np.unique(p.clusters, return_inverse=True)[1], t)
    result = sm.GLM(y, x, family=sm.families.Poisson()).fit(
        cov_type="cluster", cov_kwds={"groups": groups}, maxiter=50, tol=1e-9
    )
    index = [nm for nm, u in zip(names + more_names, used, strict=True) if u].index("core")
    return float(result.params[index]), float(result.bse[index])


def robustness(frame: pl.DataFrame, zones: pl.DataFrame, primary: did.Panel, draws: int) -> list:
    """The pre-specified set. Every entry reports the core coefficient."""
    rows = [estimate("Primary", "primary", primary, EXPOSED, draws)]

    for flow, title in (
        ("entering", "Entering the zone"),
        ("leaving", "Leaving the zone"),
        ("within", "Within the zone"),
    ):
        p = did.load_panel(frame, zones, core_flows=(flow,))
        rows.append(estimate(title, "flow", p, EXPOSED, draws))

    for service, title in (("hv", "Uber and Lyft"), ("yellow", "Yellow taxis")):
        p = did.load_panel(frame, zones, service=service)
        rows.append(estimate(title, "service", p, EXPOSED, draws))

    for period, title in (("peak", "Peak-toll hours"), ("overnight", "Overnight hours")):
        p = did.load_panel(frame, zones, period=period)
        rows.append(estimate(title, "period", p, EXPOSED, draws))

    rows.append(
        estimate("2025 only", "window", primary.subset_weeks(primary.weeks <= 51), EXPOSED, draws)
    )
    rows.append(
        estimate(
            "2026 only",
            "window",
            primary.subset_weeks((primary.weeks < 0) | (primary.weeks >= 52)),
            EXPOSED,
            draws,
        )
    )

    all_outside = {"surcharge_ring": "far", "near": "far"}
    pooled = only_groups(relabel(primary, all_outside), ["core", "far"])
    rows.append(estimate("All outside zones pooled", "comparison", pooled, ["core"], draws))

    ring = only_groups(primary, ["core", "surcharge_ring"])
    rows.append(estimate("Ring only, 60th to 96th", "comparison", ring, ["core"], draws))

    near = only_groups(primary, ["core", "near"])
    rows.append(estimate("Near zones only, under 4 km", "comparison", near, ["core"], draws))

    banded = by_distance(primary, zones)
    rows.append(
        estimate(
            "Reference 8 km or more",
            "comparison",
            only_groups(banded, ["core", "far_8_12", "far_12_plus"]),
            ["core"],
            draws,
        )
    )
    rows.append(
        estimate(
            "Reference 12 km or more",
            "comparison",
            only_groups(banded, ["core", "far_12_plus"]),
            ["core"],
            draws,
        )
    )

    naive = did.load_panel(
        frame, zones, core_flows=("within", "leaving"), comparison_flows=("outside", "to_core")
    )
    rows.append(estimate("Naive raw pickups", "comparison", naive, EXPOSED, draws))

    strict = set(zones.filter(pl.col("overlap") >= 0.999)["location_id"].to_list())
    fully_inside = (primary.groups != "core") | np.isin(primary.zones, sorted(strict))
    rows.append(
        estimate(
            "Strict core, 20 zones",
            "definition",
            primary.subset_zones(fully_inside),
            EXPOSED,
            draws,
        )
    )
    with_mixed = relabel(primary, {"mixed": "core"})
    rows.append(
        estimate(
            "Core plus mixed zones",
            "definition",
            with_mixed,
            ["core", "surcharge_ring", "near"],
            draws,
        )
    )

    rows.append(estimate("Unweighted", "estimator", primary, EXPOSED, draws, weighted=False))
    beta, se = fit_ppml(primary, EXPOSED)
    rows.append(
        Estimate(
            "PPML (Poisson)",
            "estimator",
            beta,
            se,
            None,
            None,
            None,
            len(primary.zones),
            len(set(primary.clusters.tolist())),
            len(primary.weeks),
        )
    )

    extended = did.load_panel(frame, zones, first_week=config.FIRST_WEEK)
    rows.append(estimate("Pre-period from 2022", "window", extended, EXPOSED, draws))

    early = primary.subset_weeks((primary.weeks < NOVEMBER_NOTICE_WEEK) | (primary.weeks >= 0))
    rows.append(estimate("Pre-period ends Nov 2024", "window", early, EXPOSED, draws))

    rows.append(
        estimate("No seasonal terms", "specification", primary, EXPOSED, draws, seasonal=False)
    )
    rows.append(
        estimate(
            "Core linear pre-trend", "specification", primary, EXPOSED, draws, trend_group="core"
        )
    )

    holidays = (
        (np.mod(primary.weeks, config.SEASON_LENGTH) >= 49)
        | (np.mod(primary.weeks, config.SEASON_LENGTH) == 0)
        | did.thanksgiving_weeks(primary.weeks)
    )
    rows.append(
        estimate("Holiday weeks dropped", "window", primary.subset_weeks(~holidays), EXPOSED, draws)
    )
    cup = (primary.weeks >= config.week_index(WORLD_CUP[0])) & (
        primary.weeks <= config.week_index(WORLD_CUP[1])
    )
    rows.append(
        estimate("World Cup weeks dropped", "window", primary.subset_weeks(~cup), EXPOSED, draws)
    )

    coarse = merged_clusters(primary)
    rows.append(estimate("Clusters merged pairwise", "inference", coarse, EXPOSED, draws))
    return rows


def placebo_blocks(p: did.Panel) -> list[dict]:
    """Every far cluster as pseudo-core against the remaining far zones."""
    far = only_groups(p, ["far"])
    out = []
    for cluster in sorted(set(far.clusters.tolist())):
        pseudo = replace(far, groups=np.where(far.clusters == cluster, "core", "far"))
        fitted = did.fit_post(pseudo, ["core"])
        out.append(
            {
                "cluster": cluster,
                "beta": fitted.coef("core"),
                "percent": did.pct(fitted.coef("core")),
                "zones": int((far.clusters == cluster).sum()),
            }
        )
    return out


def placebo_dates(frame: pl.DataFrame, zones: pl.DataFrame, draws: int) -> list[dict]:
    """Fake starts inside the true pre-period, using only data from before the real start."""
    p = did.load_panel(frame, zones, first_week=config.FIRST_WEEK, last_week=-1)
    out = []
    for fake in (-78, -65, -52, -39, -26, -13):
        fitted = did.fit_post(p, EXPOSED, start=fake)
        wild = wild_test(fitted, "core", draws, config.SEED)
        out.append(
            {
                "fake_week": fake,
                "start": str(config.week_start(fake)),
                "percent": did.pct(fitted.coef("core")),
                "low": did.pct(wild.ci_low),
                "high": did.pct(wild.ci_high),
                "wild_p": wild.p_value,
            }
        )
    return out


def event_study(p: did.Panel) -> dict:
    fitted, labels = did.fit_event_study(p, ["core", "surcharge_ring", "near"])
    series = {}
    for group in ("core", "surcharge_ring", "near"):
        rows = []
        for label in labels:
            name = f"bin[{group},{label}]"
            if name in fitted.names:
                b, s = fitted.coef(name), fitted.stderr(name)
                rows.append(
                    {
                        "bin": label,
                        "week": label * 4,
                        "start": str(config.week_start(label * 4)),
                        "percent": did.pct(b),
                        "low": did.pct(b - 1.96 * s),
                        "high": did.pct(b + 1.96 * s),
                    }
                )
        series[group] = rows
    pre_2023 = [r for r in series["core"] if -104 <= r["week"] < -52]
    return {"series": series, "max_abs_pre_pct": max(abs(r["percent"]) for r in pre_2023)}


def cluster_series(p: did.Panel, group: str = "far") -> tuple[np.ndarray, list[str]]:
    selected = only_groups(p, [group])
    names = sorted(set(selected.clusters.tolist()))
    series = np.column_stack(
        [selected.counts[:, selected.clusters == c].sum(axis=1) for c in names]
    )
    return np.log(series), names


def synthetic_results(frame: pl.DataFrame, zones: pl.DataFrame, primary: did.Panel) -> dict:
    treated = np.log(primary.counts[:, primary.groups == "core"].sum(axis=1))
    donors, names = cluster_series(primary)
    pre = primary.weeks < 0

    out: dict = {"weeks": primary.weeks.tolist(), "donor_clusters": len(names)}
    methods = {"synthetic_control": synth.synthetic_control, "synthetic_did": synth.synthetic_did}
    for key, method in methods.items():
        fitted = method(treated, donors, pre)
        placebo = synth.in_space_placebo(method, treated, donors, pre, fitted)
        half_width = 1.96 * placebo.sd
        out[key] = {
            "percent": did.pct(fitted.att),
            "beta": fitted.att,
            "placebo_low_pct": did.pct(fitted.att - half_width),
            "placebo_high_pct": did.pct(fitted.att + half_width),
            "rmspe_pre": fitted.rmspe_pre,
            "rmspe_post": fitted.rmspe_post,
            "ratio_p": placebo.ratio_p,
            "att_p": placebo.att_p,
            "placebo_sd": placebo.sd,
            "gaps_pct": [did.pct(g) for g in fitted.gaps],
            "placebo_gaps_pct": np.round(100 * (np.exp(placebo.gaps) - 1), 2).tolist(),
            "placebo_att_pct": [did.pct(a) for a in placebo.atts],
            "top_weights": sorted(
                [
                    {"cluster": n, "weight": float(w)}
                    for n, w in zip(names, fitted.weights, strict=True)
                    if w > 0.01
                ],
                key=lambda r: -r["weight"],
            )[:8],
        }

    near_donors, _ = cluster_series(primary, "near")
    wider = np.hstack([donors, near_donors])
    out["synthetic_control_with_near"] = {
        "percent": did.pct(synth.synthetic_control(treated, wider, pre).att)
    }

    long = did.load_panel(frame, zones, first_week=config.FIRST_WEEK)
    long_core = np.log(long.counts[:, long.groups == "core"].sum(axis=1))
    long_donors, _ = cluster_series(long)

    lag = config.SEASON_LENGTH
    usable = (long.weeks >= config.PRIMARY_PRE_WEEK) & (long.weeks <= 51)
    index = np.flatnonzero(usable)
    d_core = long_core[index] - long_core[index - lag]
    d_donors = long_donors[index] - long_donors[index - lag]
    d_pre = long.weeks[index] < 0
    differenced = synth.synthetic_control(d_core, d_donors, d_pre)
    out["seasonal_difference_2025"] = {
        "percent": did.pct(differenced.att),
        "rmspe_pre": differenced.rmspe_pre,
    }

    window = long.weeks < 0
    time_placebo = {}
    for key, method in methods.items():
        fake = method(long_core[window], long_donors[window], long.weeks[window] < -52)
        time_placebo[key] = {"percent": did.pct(fake.att), "rmspe_pre": fake.rmspe_pre}
    out["fake_start_2024"] = time_placebo
    return out


def spillovers(primary: did.Panel, zones: pl.DataFrame, draws: int) -> dict:
    """The four sizing exercises from the design."""
    fitted = did.fit_post(primary, EXPOSED)
    base_weeks = (primary.weeks >= -52) & (primary.weeks < 0)
    baseline = {
        g: float(primary.counts[base_weeks][:, primary.groups == g].sum(axis=1).mean())
        for g in ("core", "mixed", "surcharge_ring", "near", "far")
    }
    rings = {}
    for group in ("core", "mixed", "surcharge_ring", "near"):
        booted = WildBootstrap(fitted, group, draws, config.SEED)
        low, high = booted.interval()
        change = baseline[group] * (np.exp(fitted.coef(group)) - 1)
        rings[group] = {
            "percent": did.pct(fitted.coef(group)),
            "wild_low_pct": did.pct(low),
            "wild_high_pct": did.pct(high),
            "wild_p": booted.p_value(0.0),
            "baseline_weekly_trips": baseline[group],
            "implied_weekly_change": float(change),
            "cr_se": fitted.stderr(group),
        }

    banded = by_distance(primary, zones)
    gradient_groups = [*EXPOSED, "far_4_8", "far_8_12"]
    gradient_fit = did.fit_post(banded, gradient_groups)
    gradient = [
        {
            "group": g,
            "percent": did.pct(gradient_fit.coef(g)),
            "low": did.pct(gradient_fit.coef(g) - 1.96 * gradient_fit.stderr(g)),
            "high": did.pct(gradient_fit.coef(g) + 1.96 * gradient_fit.stderr(g)),
        }
        for g in gradient_groups
    ]

    share = baseline["core"] / baseline["far"]
    beta = fitted.coef("core")
    bound = [
        {
            "displaced_share": d,
            "percent": did.pct(beta / (1 + d * share)),
        }
        for d in (0.0, 0.25, 0.5, 1.0)
    ]
    return {
        "rings": rings,
        "gradient": gradient,
        "core_to_far_ratio": share,
        "displacement_bound": bound,
    }


def flow_summary(frames: pl.DataFrame) -> dict:
    """Descriptive tables that need ratios, from the group-level aggregate."""

    def rate_table(df: pl.DataFrame) -> list[dict]:
        return (
            df.group_by("touch")
            .agg(pl.col("trips").sum(), pl.col("charged").sum())
            .with_columns((pl.col("charged") / pl.col("trips")).alias("rate"))
            .sort("touch")
            .to_dicts()
        )

    core = (pl.col("pu_group") == "core") | (pl.col("do_group") == "core")
    mixed = (pl.col("pu_group") == "mixed") | (pl.col("do_group") == "mixed")
    labelled = frames.with_columns(
        pl.when(core)
        .then(pl.lit("touches core"))
        .when(mixed)
        .then(pl.lit("touches mixed, not core"))
        .otherwise(pl.lit("neither"))
        .alias("touch")
    )
    pre_charge = labelled.filter(pl.col("week") < 0)
    post = labelled.filter(pl.col("week") >= 0)

    comparison = (
        post.filter(
            (pl.col("touch") == "neither")
            & pl.col("pu_group").is_in(["far", "near", "surcharge_ring"])
            & pl.col("do_group").is_in(["far", "near", "surcharge_ring"])
        )
        .select(pl.col("charged").sum() / pl.col("trips").sum())
        .item()
    )

    def yearly(where: pl.Expr, service: str) -> dict:
        subset = frames.filter(where & (pl.col("service") == service))

        def mean(window: pl.Expr) -> dict:
            s = subset.filter(window)
            trips = s["trips"].sum()
            return {
                "fare_per_trip": s["fare"].sum() / max(s["fare_trips"].sum(), 1),
                "mph": s["miles"].sum() / max(s["seconds"].sum(), 1) * 3600,
                "trips_per_week": trips / max(s["week"].n_unique(), 1),
            }

        return {
            "2024": mean(pl.col("week").is_between(-52, -1)),
            "2025": mean(pl.col("week").is_between(0, 51)),
        }

    within = (pl.col("pu_group") == "core") & (pl.col("do_group") == "core")
    far = (pl.col("pu_group") == "far") & (pl.col("do_group") == "far")
    return {
        "charged_share_pre": rate_table(pre_charge),
        "charged_share_post": rate_table(post),
        "comparison_charged_share": comparison,
        "mechanism": {
            f"{name}_{service}": yearly(where, service)
            for service in ("hv", "yellow")
            for name, where in (("within_core", within), ("far_to_far", far))
        },
    }


def verdict(
    primary: dict,
    randomisation_p: float,
    fake: dict,
    synth_signs: list[float],
    comparison_percents: list[float],
) -> dict:
    """Apply the wording rules fixed in DESIGN.md before estimation."""
    excludes_zero = primary["wild_low_pct"] > 0 or primary["wild_high_pct"] < 0
    checks = {
        "wild_interval_excludes_zero": bool(excludes_zero),
        "randomisation_p_below_0_10": bool(randomisation_p < 0.10),
        "fake_start_smaller_than_estimate": bool(abs(fake["percent"]) < abs(primary["percent"])),
        "fake_start_interval_includes_zero": bool(fake["low"] <= 0 <= fake["high"]),
        "synthetic_share_sign": bool(
            all(np.sign(s) == np.sign(primary["percent"]) for s in synth_signs)
        ),
        "sign_stable_across_comparisons": bool(
            len({float(np.sign(c)) for c in comparison_percents}) == 1
        ),
    }
    if not excludes_zero:
        word = "null"
    elif all(checks.values()):
        word = "detected"
    else:
        word = "sensitive"
    return {"word": word, "checks": checks}


def estimate_stage(draws: int = config.WILD_DRAWS) -> None:
    started = time.time()
    config.RESULTS.mkdir(exist_ok=True)
    frame = pl.read_parquet(config.AGGREGATES / "zone_week.parquet")
    flows = pl.read_parquet(config.AGGREGATES / "flow_week.parquet")
    zones = panel.zone_groups()
    primary = did.load_panel(frame, zones)

    rows = [r.row() for r in robustness(frame, zones, primary, draws)]
    blocks = placebo_blocks(primary)
    dates = placebo_dates(frame, zones, draws)
    events = event_study(primary)
    extended_events = event_study(did.load_panel(frame, zones, first_week=config.FIRST_WEEK))
    synthetic = synthetic_results(frame, zones, primary)
    spill = spillovers(primary, zones, draws)
    flow = flow_summary(flows)

    head = rows[0]
    betas = np.array([b["beta"] for b in blocks])
    randomisation_p = float((1 + (np.abs(betas) >= abs(head["beta"])).sum()) / (len(betas) + 1))
    fake = next(d for d in dates if d["fake_week"] == -52)
    comparison = [r["percent"] for r in rows if r["family"] == "comparison"]
    outcome = verdict(
        head,
        randomisation_p,
        fake,
        [synthetic["synthetic_control"]["percent"], synthetic["synthetic_did"]["percent"]],
        comparison,
    )

    sizes = {g: int((primary.groups == g).sum()) for g in sorted(set(primary.groups.tolist()))}
    group_weeks = {
        g: primary.counts[:, primary.groups == g].sum(axis=1).tolist()
        for g in ("core", "mixed", "surcharge_ring", "near", "far")
    }
    summary = {
        "primary": head,
        "verdict": outcome,
        "randomisation": {
            "p": randomisation_p,
            "blocks": len(betas),
            "abs_quantile_95_pct": did.pct(float(np.quantile(np.abs(betas), 0.95))),
            "spread_pct": [did.pct(float(betas.min())), did.pct(float(betas.max()))],
        },
        "panel": {
            "zones": sizes,
            "first_week": str(config.week_start(int(primary.weeks[0]))),
            "last_week": str(config.week_start(int(primary.weeks[-1]))),
            "weeks": len(primary.weeks),
            "pre_weeks": int((primary.weeks < 0).sum()),
            "post_weeks": int((primary.weeks >= 0).sum()),
            "clusters": len(set(primary.clusters.tolist())),
            "core_clusters": len(set(primary.clusters[primary.groups == "core"].tolist())),
        },
        "draws": draws,
        "event_study_max_abs_pre_pct": events["max_abs_pre_pct"],
        "seconds": round(time.time() - started, 1),
    }
    outputs = {
        "summary.json": summary,
        "robustness.json": rows,
        "placebo_blocks.json": blocks,
        "placebo_dates.json": dates,
        "event_study.json": events,
        "event_study_2022.json": extended_events,
        "synthetic.json": synthetic,
        "spillovers.json": spill,
        "flows.json": flow,
        "group_series.json": {"weeks": primary.weeks.tolist(), "trips": group_weeks},
    }
    for name, payload in outputs.items():
        (config.RESULTS / name).write_text(json.dumps(payload, indent=1, default=float) + "\n")
    print(json.dumps(summary, indent=1, default=float))
