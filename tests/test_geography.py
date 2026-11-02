import numpy as np
import polars as pl

from cordon import config, geography


def test_assign_group_rules():
    assert geography.assign_group(161, 1.0, 0.0) == "core"
    assert geography.assign_group(88, 0.79, 0.0) == "core"
    assert geography.assign_group(142, 0.14, 0.0) == "mixed"
    assert geography.assign_group(236, 0.0, 1.3) == "surcharge_ring"
    assert geography.assign_group(145, 0.0, 0.8) == "near"
    assert geography.assign_group(2, 0.0, 12.0) == "far"


def test_airports_are_never_units_whatever_their_geometry():
    assert geography.assign_group(132, 0.0, 15.0) == "airport_or_unknown"
    assert geography.assign_group(1, 0.0, 11.0) == "airport_or_unknown"


def test_small_groups_get_one_cluster_per_zone():
    xy = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]])
    assert len(set(geography.ward_clusters(xy).tolist())) == 5


def test_large_groups_are_clustered_spatially():
    rng = np.random.default_rng(1)
    centres = np.array([[0, 0], [100, 0], [0, 100]], dtype=float)
    xy = np.vstack([c + rng.normal(0, 1, (6, 2)) for c in centres])
    labels = geography.ward_clusters(xy)
    assert len(set(labels.tolist())) == 3
    assert len(set(labels[:6].tolist())) == 1


def test_committed_zone_table_matches_the_design():
    zones = pl.read_csv(config.REFERENCE / "zones.csv", infer_schema_length=None)
    counts = dict(zones.group_by("group").len().iter_rows())
    assert counts["core"] == 37
    assert counts["mixed"] == 7
    assert counts["surcharge_ring"] == 5
    assert zones.filter(pl.col("group") == "mixed")["location_id"].sort().to_list() == [
        12,
        43,
        140,
        141,
        142,
        143,
        237,
    ]
    assert zones.filter(pl.col("overlap") >= 0.999).height == 20


def test_treatment_never_varies_within_a_cluster():
    zones = pl.read_csv(config.REFERENCE / "zones.csv", infer_schema_length=None)
    per_cluster = zones.group_by("cluster").agg(pl.col("group").n_unique())
    assert per_cluster["group"].max() == 1
