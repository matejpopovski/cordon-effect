"""Difference-in-differences on a balanced zone-by-week panel.

Weights are constant within a zone, so weighted two-way demeaning has a closed form. That lets
every specification run in well under a second even with hundreds of seasonal terms, which is
what makes bootstrapping and placebo loops over dozens of blocks cheap.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date, timedelta

import numpy as np
import polars as pl
import statsmodels.api as sm

from . import config


@dataclass(frozen=True)
class Panel:
    """Balanced panel. `counts` has shape (weeks, zones)."""

    weeks: np.ndarray
    zones: np.ndarray
    groups: np.ndarray
    clusters: np.ndarray
    counts: np.ndarray
    weights: np.ndarray

    @property
    def log_y(self) -> np.ndarray:
        return np.log(self.counts)

    def subset_zones(self, mask: np.ndarray) -> Panel:
        return replace(
            self,
            zones=self.zones[mask],
            groups=self.groups[mask],
            clusters=self.clusters[mask],
            counts=self.counts[:, mask],
            weights=self.weights[mask],
        )

    def subset_weeks(self, mask: np.ndarray) -> Panel:
        return replace(self, weeks=self.weeks[mask], counts=self.counts[mask, :])


@dataclass(frozen=True)
class Fit:
    names: list[str]
    beta: np.ndarray
    se: np.ndarray
    nobs: int
    n_clusters: int
    x: np.ndarray
    y: np.ndarray
    sqrt_w: np.ndarray
    cluster_ids: np.ndarray
    shape: tuple[int, int]
    weights: np.ndarray

    def coef(self, name: str) -> float:
        return float(self.beta[self.names.index(name)])

    def stderr(self, name: str) -> float:
        return float(self.se[self.names.index(name)])


def pct(log_points: float) -> float:
    return 100 * (float(np.exp(log_points)) - 1)


def load_panel(
    frame: pl.DataFrame,
    zones: pl.DataFrame,
    service: str = "all",
    core_flows: tuple[str, ...] = ("entering", "leaving", "within"),
    period: str | None = None,
    first_week: int = config.PRIMARY_PRE_WEEK,
    last_week: int = config.LAST_WEEK,
    pre_end: int = -1,
    comparison_flows: tuple[str, ...] = ("outside",),
    min_pre_trips: float = config.MIN_PRE_WEEKLY_TRIPS,
) -> Panel:
    """Build the unit-by-week matrix: core units count core_flows, comparison units count
    comparison_flows, and each mixed zone counts trips touching it but no core zone."""
    table = frame.filter(pl.col("week").is_between(first_week, last_week))
    if service != "all":
        table = table.filter(pl.col("service") == service)
    if period is not None:
        table = table.filter(pl.col("period") == period)

    units = zones.filter(pl.col("group").is_in(["core", "mixed", "surcharge_ring", "near", "far"]))
    flows_by_group = {
        "core": core_flows,
        "mixed": ("mixed",),
        "surcharge_ring": comparison_flows,
        "near": comparison_flows,
        "far": comparison_flows,
    }
    wanted = pl.concat(
        [
            pl.DataFrame({"group": [group] * len(flows), "flow": list(flows)})
            for group, flows in flows_by_group.items()
        ]
    )
    selected = (
        table.join(units.select("location_id", "group"), left_on="zone", right_on="location_id")
        .join(wanted, on=["group", "flow"])
        .group_by("week", "zone")
        .agg(pl.col("trips").sum())
    )
    weeks = np.arange(first_week, last_week + 1)
    wide = (
        selected.pivot(on="zone", index="week", values="trips")
        .join(
            pl.DataFrame({"week": weeks}).cast({"week": selected.schema["week"]}),
            on="week",
            how="right",
        )
        .sort("week")
        .fill_null(0)
    )
    ids = sorted(int(c) for c in wide.columns if c != "week")
    counts = wide.select([str(i) for i in ids]).to_numpy().astype(float)

    pre = weeks <= pre_end
    pre_mean = counts[pre].mean(axis=0)
    keep = (counts > 0).all(axis=0) & (pre_mean >= min_pre_trips)
    meta = units.select("location_id", "group", "cluster").to_dicts()
    lookup = {row["location_id"]: row for row in meta}
    zone_ids = np.array(ids)[keep]
    return Panel(
        weeks=weeks,
        zones=zone_ids,
        groups=np.array([lookup[i]["group"] for i in zone_ids]),
        clusters=np.array([lookup[i]["cluster"] for i in zone_ids]),
        counts=counts[:, keep],
        weights=pre_mean[keep],
    )


def thanksgiving_weeks(weeks: np.ndarray) -> np.ndarray:
    """True for weeks containing Thanksgiving, the one holiday that drifts across season bins."""

    def thanksgiving(year: int) -> date:
        first = date(year, 11, 1)
        return first + timedelta(days=(3 - first.weekday()) % 7 + 21)

    marks = {config.week_index(thanksgiving(y)) for y in range(2021, 2028)}
    return np.isin(weeks, sorted(marks))


def absorb(matrix: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted two-way demeaning of a (weeks, zones, k) array, weights fixed within zone."""
    w = weights / weights.sum()
    zone_mean = matrix.mean(axis=0, keepdims=True)
    week_mean = np.einsum("tnk,n->tk", matrix, w)[:, None, :]
    grand = np.einsum("nk,n->k", zone_mean[0], w)[None, None, :]
    return matrix - zone_mean - week_mean + grand


def seasonal_terms(panel: Panel, groups: list[str]) -> tuple[list[np.ndarray], list[str]]:
    """Group by season-of-year dummies plus group by Thanksgiving for each listed group."""
    season = np.mod(panel.weeks, config.SEASON_LENGTH)
    columns, names = [], []
    for group in groups:
        member = (panel.groups == group).astype(float)
        for s in sorted(set(season))[1:]:
            columns.append(np.outer(season == s, member))
            names.append(f"season[{group},{s}]")
        columns.append(np.outer(thanksgiving_weeks(panel.weeks), member))
        names.append(f"thanksgiving[{group}]")
    return columns, names


def post_terms(
    panel: Panel, groups: list[str], start: int = 0, end: int | None = None
) -> tuple[list[np.ndarray], list[str]]:
    active = panel.weeks >= start
    if end is not None:
        active &= panel.weeks <= end
    return [np.outer(active, (panel.groups == g).astype(float)) for g in groups], list(groups)


def fit(panel: Panel, columns: list[np.ndarray], names: list[str], weighted: bool = True) -> Fit:
    """Partial out zone and week effects, then fit WLS with cluster-robust covariance."""
    t, n = panel.counts.shape
    weights = panel.weights if weighted else np.ones(n)
    stack = np.stack(columns, axis=-1)
    x = absorb(stack, weights).reshape(t * n, len(columns))
    y = absorb(panel.log_y[..., None], weights).ravel()

    w = np.tile(weights, t)
    cluster_ids = np.tile(np.unique(panel.clusters, return_inverse=True)[1], t)

    keep = x.std(axis=0) > 1e-12
    result = sm.WLS(y, x[:, keep], weights=w).fit(
        cov_type="cluster", cov_kwds={"groups": cluster_ids}
    )
    beta = np.full(len(names), np.nan)
    se = np.full(len(names), np.nan)
    beta[keep] = result.params
    se[keep] = result.bse
    return Fit(
        names=[nm for nm, k in zip(names, keep, strict=True) if k],
        beta=beta[keep],
        se=se[keep],
        nobs=t * n,
        n_clusters=len(set(cluster_ids.tolist())),
        x=x[:, keep],
        y=y,
        sqrt_w=np.sqrt(w),
        cluster_ids=cluster_ids,
        shape=(t, n),
        weights=weights,
    )


def fit_post(
    panel: Panel,
    exposed: list[str],
    seasonal: bool = True,
    weighted: bool = True,
    trend_group: str | None = None,
    start: int = 0,
) -> Fit:
    """Post by group terms against the groups not listed, the usual primary-style regression."""
    columns, names = post_terms(panel, exposed, start=start)
    if seasonal:
        more, more_names = seasonal_terms(panel, exposed)
        columns += more
        names += more_names
    if trend_group is not None:
        columns.append(np.outer(panel.weeks / 52.0, (panel.groups == trend_group).astype(float)))
        names.append(f"trend[{trend_group}]")
    return fit(panel, columns, names, weighted=weighted)


def event_bins(weeks: np.ndarray, width: int = 4) -> np.ndarray:
    """Bin index of each week; the 2024 reference year (weeks -52 to -1) maps to NaN."""
    bins = np.floor_divide(weeks, width).astype(float)
    bins[(weeks >= -52) & (weeks <= -1)] = np.nan
    return bins


def fit_event_study(panel: Panel, groups: list[str], width: int = 4) -> tuple[Fit, list[int]]:
    """Each bin relative to the same weeks of 2024, with seasons identified from 2024 alone."""
    bins = event_bins(panel.weeks, width)
    labels = sorted({int(b) for b in bins if not np.isnan(b)})
    columns, names = [], []
    for group in groups:
        member = (panel.groups == group).astype(float)
        for label in labels:
            columns.append(np.outer(bins == label, member))
            names.append(f"bin[{group},{label}]")
    more, more_names = seasonal_terms(panel, groups)
    # Thanksgiving falls in the reference year once, so its term would be collinear there.
    keep = [i for i, nm in enumerate(more_names) if not nm.startswith("thanksgiving")]
    return fit(
        panel, columns + [more[i] for i in keep], names + [more_names[i] for i in keep]
    ), labels
