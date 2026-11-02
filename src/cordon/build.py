"""Turns one month of raw trip parquet into one small origin-destination aggregate.

Everything here is lazy and runs on the streaming engine. The output grain (week, service,
pickup zone, dropoff zone, toll period) has a few hundred thousand groups a month, so memory
is bounded by that group count rather than by the 20 million rows in a high-volume file.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl

from . import config

# A yellow reversal is a negative row sharing all of these with its positive original.
TRIP_KEY = ("vendor", "pickup", "dropoff", "pu", "do")


def normalise_yellow(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Project the stable yellow columns. Schemas drift across years, so nothing is inferred."""
    names = lf.collect_schema().names()
    fee = pl.col("cbd_congestion_fee") if "cbd_congestion_fee" in names else pl.lit(0.0)
    return lf.select(
        pl.lit("yellow").alias("service"),
        pl.col("VendorID").cast(pl.Int16).alias("vendor"),
        pl.col("tpep_pickup_datetime").alias("pickup"),
        pl.col("tpep_dropoff_datetime").alias("dropoff"),
        pl.col("PULocationID").cast(pl.Int16).alias("pu"),
        pl.col("DOLocationID").cast(pl.Int16).alias("do"),
        pl.col("trip_distance").cast(pl.Float64).alias("miles"),
        (pl.col("tpep_dropoff_datetime") - pl.col("tpep_pickup_datetime"))
        .dt.total_seconds()
        .cast(pl.Float64)
        .alias("seconds"),
        pl.col("fare_amount").cast(pl.Float64).alias("fare"),
        pl.col("total_amount").cast(pl.Float64).alias("total"),
        fee.cast(pl.Float64).alias("fee"),
        pl.col("congestion_surcharge").cast(pl.Float64).alias("surcharge"),
        (pl.col("payment_type").fill_null(0) == 0).alias("flex"),
    )


def normalise_hv(lf: pl.LazyFrame) -> pl.LazyFrame:
    names = lf.collect_schema().names()
    fee = pl.col("cbd_congestion_fee") if "cbd_congestion_fee" in names else pl.lit(0.0)
    return lf.select(
        pl.lit("hv").alias("service"),
        pl.lit(0).cast(pl.Int16).alias("vendor"),
        pl.col("pickup_datetime").alias("pickup"),
        pl.col("dropoff_datetime").alias("dropoff"),
        pl.col("PULocationID").cast(pl.Int16).alias("pu"),
        pl.col("DOLocationID").cast(pl.Int16).alias("do"),
        pl.col("trip_miles").cast(pl.Float64).alias("miles"),
        pl.col("trip_time").cast(pl.Float64).alias("seconds"),
        pl.col("base_passenger_fare").cast(pl.Float64).alias("fare"),
        pl.col("base_passenger_fare").cast(pl.Float64).alias("total"),
        fee.cast(pl.Float64).alias("fee"),
        pl.col("congestion_surcharge").cast(pl.Float64).alias("surcharge"),
        pl.lit(False).alias("flex"),
    )


def drop_reversal_twins(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Keep one row per trip key.

    A negative amount does not mean a reversal. From January 2025 a large and growing number
    of Flex Fare rows carry small negative amounts with no positive twin, and their distances
    and durations match ordinary Flex trips, so they are real trips. Filtering on the sign
    would delete them mostly after the toll started and fake a fall in yellow trips.
    Only a matched pair is a reversal, so matching on the key removes exactly those.
    """
    return lf.unique(subset=list(TRIP_KEY), keep="any")


def add_flags(lf: pl.LazyFrame, year: int, month: int) -> pl.LazyFrame:
    in_month = (pl.col("pickup").dt.year() == year) & (pl.col("pickup").dt.month() == month)
    bad_distance = pl.col("miles").fill_null(0) <= 0
    return lf.with_columns(
        in_month.alias("in_month"), bad_distance.alias("bad_distance")
    ).with_columns((pl.col("in_month") & ~pl.col("bad_distance")).alias("keep"))


def qa_counts(flagged: pl.LazyFrame) -> pl.LazyFrame:
    return flagged.select(
        pl.len().alias("unique_rows"),
        (~pl.col("in_month")).sum().alias("out_of_month"),
        (pl.col("in_month") & pl.col("bad_distance")).sum().alias("bad_distance"),
        pl.col("keep").sum().alias("kept"),
        (pl.col("keep") & pl.col("flex")).sum().alias("flex_rows"),
        (pl.col("keep") & (pl.col("fare") < 0)).sum().alias("negative_kept"),
    )


def toll_period() -> pl.Expr:
    """Peak under the MTA schedule: 5am-9pm weekdays, 9am-9pm weekends."""
    hour = pl.col("pickup").dt.hour()
    weekend = pl.col("pickup").dt.weekday() >= 6
    peak = pl.when(weekend).then((hour >= 9) & (hour < 21)).otherwise((hour >= 5) & (hour < 21))
    return pl.when(peak).then(pl.lit("peak")).otherwise(pl.lit("overnight")).alias("period")


def aggregate(flagged: pl.LazyFrame) -> pl.LazyFrame:
    anchor = pl.lit(config.POLICY_START, dtype=pl.Date)
    speed_ok = (
        (pl.col("seconds") > 60)
        & (pl.col("miles") / (pl.col("seconds") / 3600) >= 0.5)
        & (pl.col("miles") / (pl.col("seconds") / 3600) <= 80)
    )
    return (
        flagged.filter(pl.col("keep"))
        .with_columns(
            ((pl.col("pickup").dt.date() - anchor).dt.total_days() // 7)
            .cast(pl.Int16)
            .alias("week"),
            toll_period(),
            speed_ok.alias("speed_ok"),
        )
        .group_by("week", "service", "pu", "do", "period")
        .agg(
            pl.len().cast(pl.Int32).alias("trips"),
            (pl.col("fee").abs() > 0).sum().cast(pl.Int32).alias("charged"),
            (pl.col("surcharge").abs() > 0).sum().cast(pl.Int32).alias("surcharged"),
            pl.col("surcharge").is_not_null().sum().cast(pl.Int32).alias("surcharge_known"),
            pl.col("speed_ok").sum().cast(pl.Int32).alias("speed_trips"),
            pl.col("miles").filter(pl.col("speed_ok")).sum().cast(pl.Float32).alias("miles"),
            pl.col("seconds").filter(pl.col("speed_ok")).sum().cast(pl.Float32).alias("seconds"),
            pl.col("fare")
            .filter(pl.col("fare").is_not_null())
            .sum()
            .cast(pl.Float32)
            .alias("fare"),
            (pl.col("fare") >= 0).sum().cast(pl.Int32).alias("fare_trips"),
            pl.col("flex").sum().cast(pl.Int32).alias("flex_trips"),
        )
    )


def process_file(path: Path, service: str, year: int, month: int) -> tuple[pl.DataFrame, dict]:
    """Aggregate one parquet file, returning the table and its QA counters."""
    lf = pl.scan_parquet(path)
    normalised = normalise_yellow(lf) if service == "yellow" else normalise_hv(lf)
    deduped = drop_reversal_twins(normalised) if service == "yellow" else normalised
    flagged = add_flags(deduped, year, month)
    table, qa, raw = pl.collect_all(
        [aggregate(flagged), qa_counts(flagged), normalised.select(pl.len().alias("raw_rows"))],
        engine="streaming",
    )
    counters = qa.row(0, named=True)
    raw_rows = raw.item()
    return table, {
        "raw_rows": raw_rows,
        "reversal_twins": raw_rows - counters.pop("unique_rows"),
        **counters,
    }


def month_label(year: int, month: int) -> str:
    return date(year, month, 1).strftime("%Y-%m")
