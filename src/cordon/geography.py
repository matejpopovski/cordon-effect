"""Maps TLC taxi zones onto the tolled area.

Taxi zones were drawn long before the toll and do not follow it: their northern edge sits at
59th Street while the toll line is 60th, and shoreline zones include the excluded FDR and West
Side Highway corridors. So the mapping comes from overlaying the MTA geofence on the zone
polygons, and is then checked against who was actually charged (see the first_stage table).
"""

from __future__ import annotations

import csv
import json
import warnings
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import shapefile
import shapely
from pyproj import Transformer
from scipy.cluster.hierarchy import fcluster, linkage
from shapely.geometry import shape
from shapely.ops import unary_union

from . import config

SQFT_PER_SQKM = 10_763_910.4
FT_PER_KM = 3_280.84


@dataclass(frozen=True)
class Zone:
    location_id: int
    name: str
    borough: str
    group: str
    overlap: float
    km_to_crz: float
    centroid_x: float
    centroid_y: float
    cluster: str


def load_geofence(path: Path):
    """Return the geofence as one geometry in the taxi zones' CRS (EPSG:2263, feet)."""
    features = json.loads(path.read_text())["features"]
    transformer = Transformer.from_crs(4326, 2263, always_xy=True)

    def to_state_plane(coords: np.ndarray) -> np.ndarray:
        x, y = transformer.transform(coords[:, 0], coords[:, 1])
        return np.column_stack([x, y])

    polygons = [shapely.transform(shape(f["geometry"]), to_state_plane).buffer(0) for f in features]
    return unary_union(polygons)


def load_zone_shapes(zip_path: Path) -> dict[int, tuple[str, str, object]]:
    """Return {location_id: (name, borough, geometry)}, dissolving multi-record zones."""
    parts: dict[int, list] = {}
    meta: dict[int, tuple[str, str]] = {}
    with zipfile.ZipFile(zip_path) as archive:
        prefix = next(n for n in archive.namelist() if n.endswith(".shp"))[:-4]
        reader = shapefile.Reader(
            shp=archive.open(prefix + ".shp"),
            shx=archive.open(prefix + ".shx"),
            dbf=archive.open(prefix + ".dbf"),
        )
        for record in reader.iterShapeRecords():
            attrs = record.record.as_dict()
            location = int(attrs["LocationID"])
            parts.setdefault(location, []).append(shape(record.shape.__geo_interface__).buffer(0))
            meta[location] = (attrs["zone"], attrs["borough"])
    return {k: (*meta[k], unary_union(v)) for k, v in parts.items()}


def assign_group(location: int, overlap: float, km: float) -> str:
    """The exposure group of a zone. Order matters: airports never act as comparison units."""
    if location in config.AIRPORTS or location in config.UNKNOWN_ZONES:
        return "airport_or_unknown"
    if overlap >= config.CORE_OVERLAP:
        return "core"
    if overlap > config.MIN_OVERLAP:
        return "mixed"
    if location in config.SURCHARGE_RING:
        return "surcharge_ring"
    return "near" if km < config.NEAR_RING_KM else "far"


def ward_clusters(xy: np.ndarray, zones_per_cluster: int = 6) -> np.ndarray:
    """Deterministic spatial clusters from centroids, about six zones each.

    A group held in a single cluster would have its treatment dummy absorb that cluster's
    scores and report a variance of zero, so small groups get one cluster per zone.
    """
    n = len(xy)
    if n <= zones_per_cluster + 1:
        return np.arange(1, n + 1)
    k = max(1, round(n / zones_per_cluster))
    return fcluster(linkage(xy, method="ward"), t=k, criterion="maxclust")


def build_zones(geofence_path: Path, zones_zip: Path) -> list[Zone]:
    crz = load_geofence(geofence_path)
    shapes = load_zone_shapes(zones_zip)
    rows = []
    for location, (name, borough, geom) in sorted(shapes.items()):
        overlap = geom.intersection(crz).area / geom.area
        km = geom.distance(crz) / FT_PER_KM
        rows.append((location, name, borough, overlap, km, geom.centroid.x, geom.centroid.y))

    grouped: dict[tuple[str, str], list[int]] = {}
    groups = {}
    for i, (location, _, borough, overlap, km, _, _) in enumerate(rows):
        groups[i] = assign_group(location, overlap, km)
        grouped.setdefault((groups[i], borough), []).append(i)

    # Clusters nest inside exposure groups so that treatment never varies within a cluster.
    labels: dict[int, str] = {}
    for (group, borough), members in sorted(grouped.items()):
        xy = np.array([[rows[i][5], rows[i][6]] for i in members])
        for i, c in zip(members, ward_clusters(xy), strict=True):
            labels[i] = f"{group}:{borough}:{c}"

    return [
        Zone(r[0], r[1], r[2], groups[i], round(r[3], 4), round(r[4], 3), r[5], r[6], labels[i])
        for i, r in enumerate(rows)
    ]


def write_zone_table(zones: list[Zone], path: Path) -> None:
    fields = ["location_id", "name", "borough", "group", "overlap", "km_to_crz", "cluster"]
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        for z in zones:
            writer.writerow([getattr(z, f) for f in fields])
        # Zones 264 and 265 exist in trip records but have no polygon.
        for location, name in ((264, "Unknown"), (265, "Outside of NYC")):
            writer.writerow([location, name, "Unknown", "airport_or_unknown", 0.0, "", "none"])


def write_simplified_map(geofence_path: Path, zones_zip: Path, path: Path) -> None:
    """Write a small lon/lat GeoJSON of zones and the toll polygon for the report's map."""
    crz = load_geofence(geofence_path)
    back = Transformer.from_crs(2263, 4326, always_xy=True)

    def to_lonlat(coords: np.ndarray) -> np.ndarray:
        x, y = back.transform(coords[:, 0], coords[:, 1])
        return np.round(np.column_stack([x, y]), 4)

    features = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for location, (_, _, geom) in sorted(load_zone_shapes(zones_zip).items()):
            small = shapely.transform(geom.simplify(180), to_lonlat)
            features.append(
                {
                    "type": "Feature",
                    "properties": {"location_id": location},
                    "geometry": shapely.geometry.mapping(small),
                }
            )
        toll = shapely.transform(crz.simplify(60), to_lonlat)
    features.append(
        {
            "type": "Feature",
            "properties": {"location_id": 0},
            "geometry": shapely.geometry.mapping(toll),
        }
    )
    path.write_text(json.dumps({"type": "FeatureCollection", "features": features}))
