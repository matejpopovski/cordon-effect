import polars as pl

from cordon import panel

ZONES = pl.DataFrame(
    {
        "location_id": [161, 230, 142, 236, 7, 2, 132],
        "group": ["core", "core", "mixed", "surcharge_ring", "near", "far", "airport_or_unknown"],
    }
)


def trips(pairs: list[tuple[int, int]]) -> pl.LazyFrame:
    n = len(pairs)
    return pl.LazyFrame(
        {
            "week": [0] * n,
            "service": ["hv"] * n,
            "period": ["peak"] * n,
            "pu": [p[0] for p in pairs],
            "do": [p[1] for p in pairs],
            "trips": [1] * n,
            "charged": [1] * n,
            "flex_trips": [0] * n,
        },
        schema_overrides={"pu": pl.Int16, "do": pl.Int16},
    )


def assigned(pairs: list[tuple[int, int]]) -> list[tuple[str, int]]:
    od = panel.with_groups(trips(pairs), ZONES)
    flow, zone = panel.flow_expressions()
    return list(od.with_columns(flow, zone).select("flow", "zone").collect().iter_rows())


def test_flow_rules_from_the_design_table():
    assert assigned([(161, 230)]) == [("within", 161)]
    assert assigned([(161, 2)]) == [("leaving", 161)]
    assert assigned([(2, 230)]) == [("entering", 230)]
    assert assigned([(142, 2)]) == [("mixed", 142)]
    assert assigned([(2, 142)]) == [("mixed", 142)]
    assert assigned([(2, 7)]) == [("outside", 2)]


def test_core_end_beats_mixed_end():
    assert assigned([(142, 230)]) == [("entering", 230)]
    assert assigned([(230, 142)]) == [("leaving", 230)]


def test_airport_trips_to_core_are_treated_and_airport_pickups_stay_outside():
    assert assigned([(132, 161)]) == [("entering", 161)]
    assert assigned([(132, 2)]) == [("outside", 132)]


def test_unmapped_zones_fall_through_as_unknown():
    assert assigned([(265, 161)]) == [("entering", 161)]


def test_each_trip_counts_once_outside_the_naive_helper_flow():
    pairs = [(161, 230), (161, 2), (2, 230), (142, 2), (2, 7), (236, 7)]
    table = panel.zone_week_table(panel.with_groups(trips(pairs), ZONES)).collect()
    assert table.filter(pl.col("flow") != "to_core")["trips"].sum() == len(pairs)
    naive = table.filter(pl.col("flow") == "to_core")
    assert naive["zone"].to_list() == [2]
