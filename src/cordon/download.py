"""Fetches TLC source files, recording what was fetched so a rerun can detect revisions."""

from __future__ import annotations

import hashlib
import json
import time
import urllib.request
from pathlib import Path

from . import config

SERVICES = {"yellow": "yellow_tripdata", "hv": "fhvhv_tripdata"}
USER_AGENT = "cordon-effect-pipeline (research; github.com/matejpopovski/cordon-effect)"


def trip_url(service: str, year: int, month: int) -> str:
    return f"{config.TRIP_BASE}/{SERVICES[service]}_{year}-{month:02d}.parquet"


def fetch(url: str, dest: Path, attempts: int = 4) -> dict:
    """Download url to dest atomically, retrying because long runs meet transient 5xx errors."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, attempts + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            digest = hashlib.sha256()
            size = 0
            with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as f:
                while chunk := response.read(1 << 20):
                    f.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
                modified = response.headers.get("Last-Modified", "")
                expected = response.headers.get("Content-Length")
            if expected is not None and int(expected) != size:
                raise OSError(f"short read for {url}: {size} of {expected}")
            partial.replace(dest)
            return {
                "url": url,
                "bytes": size,
                "sha256": digest.hexdigest(),
                "last_modified": modified,
            }
        except OSError:
            partial.unlink(missing_ok=True)
            if attempt == attempts:
                raise
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def fetch_reference(directory: Path) -> tuple[Path, Path]:
    """Fetch the taxi zone shapefile and the zone lookup. The geofence is committed."""
    zones = directory / "taxi_zones.zip"
    if not zones.exists():
        fetch(config.ZONES_ZIP_URL, zones)
    lookup = directory / "taxi_zone_lookup.csv"
    if not lookup.exists():
        fetch(config.ZONE_LOOKUP_URL, lookup)
    return zones, lookup


def update_manifest(path: Path, key: str, entry: dict) -> None:
    manifest = json.loads(path.read_text()) if path.exists() else {}
    manifest[key] = entry
    path.write_text(json.dumps(dict(sorted(manifest.items())), indent=1) + "\n")
