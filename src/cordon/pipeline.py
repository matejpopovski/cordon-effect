"""The download-and-build stage: one month at a time, keeping raw parquet only while needed."""

from __future__ import annotations

import json
import time

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
