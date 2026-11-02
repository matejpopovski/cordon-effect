"""Collapses interim origin-destination tables into the small aggregates that are committed.

Trip counts are assigned to one unit each through the flow rules in DESIGN.md. The rules live
in `flow_expressions` so tests can check them on a handful of hand-built trips.
"""

from __future__ import annotations

import json

import polars as pl

from . import config

COUNTS = (
    "trips",
    "charged",
    "surcharged",
    "surcharge_known",
    "speed_trips",
    "miles",
    "seconds",
    "fare",
    "fare_trips",
    "flex_trips",
)


def zone_groups() -> pl.DataFrame:
    return pl.read_csv(config.REFERENCE / "zones.csv", infer_schema_length=None)


def with_groups(lf: pl.LazyFrame, zones: pl.DataFrame) -> pl.LazyFrame:
    lookup = zones.lazy().select(pl.col("location_id").cast(pl.Int16), "group")
    return (
        lf.join(lookup.rename({"location_id": "pu", "group": "pu_group"}), on="pu", how="left")
        .join(lookup.rename({"location_id": "do", "group": "do_group"}), on="do", how="left")
        .with_columns(
            pl.col("pu_group").fill_null("airport_or_unknown"),
            pl.col("do_group").fill_null("airport_or_unknown"),
        )
    )


def flow_expressions() -> tuple[pl.Expr, pl.Expr]:
    """Return (flow, zone): the trip's class and the single unit it is assigned to."""
    pu_core = pl.col("pu_group") == "core"
    do_core = pl.col("do_group") == "core"
    pu_mixed = pl.col("pu_group") == "mixed"
    do_mixed = pl.col("do_group") == "mixed"
    flow = (
        pl.when(pu_core & do_core)
        .then(pl.lit("within"))
        .when(pu_core)
        .then(pl.lit("leaving"))
        .when(do_core)
        .then(pl.lit("entering"))
        .when(pu_mixed | do_mixed)
        .then(pl.lit("mixed"))
        .otherwise(pl.lit("outside"))
    )
    zone = (
        pl.when(pu_core)
        .then(pl.col("pu"))
        .when(do_core)
        .then(pl.col("do"))
        .when(pu_mixed)
        .then(pl.col("pu"))
        .when(do_mixed)
        .then(pl.col("do"))
        .otherwise(pl.col("pu"))
    )
    return flow.alias("flow"), zone.alias("zone")


def zone_week_table(od: pl.LazyFrame) -> pl.LazyFrame:
    """Units by week. `to_core` repeats entering trips at their pickup zone for the naive check."""
    flow, zone = flow_expressions()
    keys = ["week", "service", "zone", "flow", "period"]
    sums = pl.col("trips", "charged", "flex_trips").sum()
    assigned = od.with_columns(flow, zone).group_by(keys).agg(sums)
    naive = (
        od.with_columns(flow)
        .filter(pl.col("flow") == "entering")
        .with_columns(pl.lit("to_core").alias("flow"), pl.col("pu").alias("zone"))
        .group_by(keys)
        .agg(sums)
    )
    return pl.concat([assigned, naive]).sort(keys)


def flow_week_table(od: pl.LazyFrame) -> pl.LazyFrame:
    """Group-to-group totals carrying the fields that need ratios: speed, fare, charge rates."""
    keys = ["week", "service", "pu_group", "do_group", "period"]
    return od.group_by(keys).agg(pl.col(*COUNTS).sum()).sort(keys)


def first_stage_table(od: pl.LazyFrame, zones: pl.DataFrame) -> pl.LazyFrame:
    """Charged and surcharged rates by zone in the post-period, against the zone's overlap."""
    post = od.filter(pl.col("week") >= 0)
    endpoints = pl.concat(
        [
            post.select(
                pl.col("pu").alias("zone"),
                "trips",
                "charged",
                "surcharged",
                "surcharge_known",
                "do_group",
                pl.col("pu").eq(pl.col("do")).alias("same"),
            ),
            post.filter(pl.col("pu") != pl.col("do")).select(
                pl.col("do").alias("zone"),
                "trips",
                "charged",
                "surcharged",
                "surcharge_known",
                pl.col("pu_group").alias("do_group"),
                pl.lit(False).alias("same"),
            ),
        ]
    )
    outward = pl.col("do_group").is_in(["surcharge_ring", "near", "far"])
    return (
        endpoints.group_by("zone")
        .agg(
            pl.col("trips").sum().alias("trips"),
            pl.col("charged").sum().alias("charged"),
            pl.col("trips").filter(pl.col("same")).sum().alias("same_zone_trips"),
            pl.col("charged").filter(pl.col("same")).sum().alias("same_zone_charged"),
            pl.col("trips").filter(outward).sum().alias("outside_end_trips"),
            pl.col("charged").filter(outward).sum().alias("outside_end_charged"),
            pl.col("surcharged").sum().alias("surcharged"),
            pl.col("surcharge_known").sum().alias("surcharge_known"),
        )
        .join(
            zones.lazy().select(
                pl.col("location_id").cast(pl.Int16).alias("zone"), "name", "group", "overlap"
            ),
            on="zone",
        )
        .with_columns(
            (pl.col("charged") / pl.col("trips")).round(4).alias("charged_rate"),
            (pl.col("outside_end_charged") / pl.col("outside_end_trips"))
            .round(4)
            .alias("outside_end_rate"),
            (pl.col("same_zone_charged") / pl.col("same_zone_trips"))
            .round(4)
            .alias("same_zone_rate"),
            (pl.col("surcharged") / pl.col("surcharge_known")).round(4).alias("surcharge_rate"),
        )
        .sort("zone")
    )


def panel_stage() -> None:
    zones = zone_groups()
    config.AGGREGATES.mkdir(parents=True, exist_ok=True)
    od = with_groups(
        pl.scan_parquet(config.INTERIM / "od_*.parquet").filter(
            pl.col("week").is_between(config.FIRST_WEEK, config.LAST_WEEK)
        ),
        zones,
    )
    outputs = {
        "zone_week.parquet": zone_week_table(od),
        "flow_week.parquet": flow_week_table(od),
    }
    for name, frame in outputs.items():
        frame.sink_parquet(
            config.AGGREGATES / name, compression="zstd", compression_level=19, engine="streaming"
        )
    first_stage_table(od, zones).sink_csv(config.AGGREGATES / "first_stage.csv", engine="streaming")

    qa = [json.loads(line) for line in (config.INTERIM / "qa.jsonl").read_text().splitlines()]
    pl.DataFrame(qa).unique(["service", "month"], keep="last").sort("month", "service").write_csv(
        config.AGGREGATES / "qa.csv"
    )
    for name in (*outputs, "first_stage.csv", "qa.csv"):
        print(name, f"{(config.AGGREGATES / name).stat().st_size / 1e6:.2f} MB")
