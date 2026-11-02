"""The download-and-build stage: one month at a time, keeping raw parquet only while needed."""

from __future__ import annotations

import json
import time

import polars as pl

from . import build, config, download, geography


def zones_stage() -> None:
    """Rebuild the zone classification and map from the geofence and the TLC shapefile."""
    zones_zip, lookup = download.fetch_reference(config.RAW / "reference")
    geofence = config.REFERENCE / "crz_geofence.geojson"
    zones = geography.build_zones(geofence, zones_zip)
    geography.write_zone_table(zones, config.REFERENCE / "zones.csv")
    geography.write_simplified_map(geofence, zones_zip, config.REFERENCE / "zones_map.geojson")
    print(f"{len(zones)} zones classified; lookup at {lookup.name}")


def build_stage(first: tuple[int, int], last: tuple[int, int], keep_raw: bool = False) -> None:
    """Download, aggregate and discard each month. A finished month is skipped on rerun."""
    config.INTERIM.mkdir(parents=True, exist_ok=True)
    manifest = config.DATA / "manifest.json"
    qa_path = config.INTERIM / "qa.jsonl"
    for year, month in config.months(first, last):
        label = build.month_label(year, month)
        for service in download.SERVICES:
            out = config.INTERIM / f"od_{service}_{label}.parquet"
            if out.exists():
                continue
            started = time.time()
            raw = config.RAW / f"{service}_{label}.parquet"
            if raw.exists():
                entry = {"url": download.trip_url(service, year, month), "reused_local": True}
            else:
                entry = download.fetch(download.trip_url(service, year, month), raw)
            table, qa = build.process_file(raw, service, year, month)
            partial = out.with_suffix(".tmp")
            table.write_parquet(partial, compression="zstd", compression_level=9)
            partial.replace(out)
            download.update_manifest(manifest, f"{service}_{label}", {**entry, **qa})
            with qa_path.open("a") as handle:
                handle.write(json.dumps({"service": service, "month": label, **qa}) + "\n")
            if not keep_raw:
                raw.unlink()
            print(
                f"{service} {label}: {qa['kept']:,} kept of {qa['raw_rows']:,}, "
                f"{table.height:,} groups, {time.time() - started:.0f}s",
                flush=True,
            )
            del table


def fee_check_stage() -> None:
    """Record charge amounts either side of January 2026, reading only those two columns.

    The event study has a step in January 2026. This is the check that the charge itself did
    not change then. Remote parquet is scanned by byte range, so little is downloaded.
    """
    columns = ("cbd_congestion_fee", "congestion_surcharge")
    out = {}
    for service in download.SERVICES:
        for year, month in ((2025, 12), (2026, 2)):
            lazy = pl.scan_parquet(download.trip_url(service, year, month))
            entry = {}
            for column in columns:
                counts = (
                    lazy.group_by(column)
                    .agg(pl.len().alias("rows"))
                    .sort("rows", descending=True)
                    .head(3)
                    .collect(engine="streaming")
                )
                entry[column] = [
                    {"amount": row[column], "rows": row["rows"]} for row in counts.to_dicts()
                ]
            out[f"{service}_{year}-{month:02d}"] = entry
    (config.RESULTS / "fee_amounts.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))
