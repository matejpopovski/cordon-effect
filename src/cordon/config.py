"""Every date, threshold and path that defines the design, in one place.

The values here are the design in DESIGN.md. Changing one changes the estimand, so anything
that varies across robustness checks is passed explicitly rather than edited here.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
REFERENCE = DATA / "reference"
AGGREGATES = DATA / "aggregates"
RESULTS = REPO / "results"
SITE = REPO / "site"

TRIP_BASE = "https://d37ci6vzurychx.cloudfront.net/trip-data"
ZONES_ZIP_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip"
ZONE_LOOKUP_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv"
GEOFENCE_URL = "https://data.ny.gov/resource/srxy-5nxn.geojson"

# Tolling began at midnight on Sunday 5 January 2025. Weeks run Sunday to Saturday and are
# numbered from that Sunday, so week 0 is the first full tolled week and no week straddles it.
POLICY_START = date(2025, 1, 5)
PANEL_START = date(2022, 1, 2)
PRIMARY_PRE_START = date(2023, 1, 8)
FIRST_MONTH = (2022, 1)
LAST_MONTH = (2026, 9)
# The last published month ends on a Wednesday, so its final partial week is dropped.
LAST_WEEK = 89
SEASON_LENGTH = 52

# A zone counts as inside the Congestion Relief Zone when this share of its area lies inside
# the MTA geofence. Zones below it but above MIN_OVERLAP are "mixed": some of their trips are
# charged and some are not, so they belong to neither arm.
CORE_OVERLAP = 0.75
MIN_OVERLAP = 0.005

# Outside the charged zone but inside the older $2.50/$2.75 surcharge area (south of 96th).
# Verified against congestion_surcharge incidence in the trip records.
SURCHARGE_RING = (236, 238, 239, 262, 263)
AIRPORTS = (1, 132, 138)
UNKNOWN_ZONES = (264, 265)

# Distance from the geofence, in km, that separates the exposed near ring from the reference.
NEAR_RING_KM = 4.0
GRADIENT_EDGES_KM = (4.0, 8.0, 12.0)

# Comparison units need a usable log outcome in every week.
MIN_PRE_WEEKLY_TRIPS = 25

WILD_DRAWS = 9_999
SEED = 20250105


def week_index(day: date) -> int:
    return (day - POLICY_START).days // 7


def week_start(k: int) -> date:
    return POLICY_START + timedelta(days=7 * k)


FIRST_WEEK = week_index(PANEL_START)
PRIMARY_PRE_WEEK = week_index(PRIMARY_PRE_START)


def months(first: tuple[int, int] = FIRST_MONTH, last: tuple[int, int] = LAST_MONTH):
    """Yield (year, month) pairs from first to last inclusive."""
    y, m = first
    while (y, m) <= last:
        yield y, m
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
