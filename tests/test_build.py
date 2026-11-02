from datetime import datetime

import polars as pl

from cordon import build


def yellow_frame(with_fee: bool) -> pl.DataFrame:
    frame = pl.DataFrame(
        {
            "tpep_pickup_datetime": [
                datetime(2025, 1, 4, 23, 50),
                datetime(2025, 1, 5, 0, 10),
                datetime(2025, 1, 5, 12, 0),
                datetime(2025, 1, 5, 12, 0),
                datetime(2024, 12, 31, 12, 0),
                datetime(2025, 1, 6, 12, 0),
            ],
            "tpep_dropoff_datetime": [
                datetime(2025, 1, 5, 0, 5),
                datetime(2025, 1, 5, 0, 30),
                datetime(2025, 1, 5, 12, 20),
                datetime(2025, 1, 5, 12, 20),
                datetime(2024, 12, 31, 12, 30),
                datetime(2025, 1, 6, 12, 10),
            ],
            "VendorID": [2, 2, 2, 2, 2, 2],
            "PULocationID": [161, 161, 161, 161, 161, 161],
            "DOLocationID": [230, 230, 230, 230, 230, 230],
            "trip_distance": [1.0, 2.0, 1.5, 1.5, 1.0, 0.0],
            "fare_amount": [10.0, 12.0, 11.0, -11.0, 9.0, 8.0],
            "total_amount": [15.0, 18.0, 16.0, -16.0, 13.0, 12.0],
            "congestion_surcharge": [2.5, 2.5, 2.5, -2.5, 2.5, None],
            "payment_type": [1, 1, 0, 1, 1, 1],
        }
    )
    if with_fee:
        frame = frame.with_columns(
            pl.Series("cbd_congestion_fee", [0.0, 0.75, 0.75, -0.75, 0.0, 0.0])
        )
    return frame


def run(tmp_path, frame: pl.DataFrame, service: str, year: int = 2025, month: int = 1):
    path = tmp_path / "month.parquet"
    frame.write_parquet(path)
    return build.process_file(path, service, year, month)


def test_reversal_twins_zero_distance_and_stray_months_are_dropped_and_counted(tmp_path):
    table, qa = run(tmp_path, yellow_frame(True), "yellow")
    assert qa == {
        "raw_rows": 6,
        "reversal_twins": 1,
        "out_of_month": 1,
        "bad_distance": 1,
        "kept": 3,
        "flex_rows": 1,
        "negative_kept": 0,
    }
    assert table["trips"].sum() == 3


def test_unpaired_negative_flex_rows_are_real_trips_and_count_once(tmp_path):
    frame = pl.DataFrame(
        {
            "VendorID": [2],
            "tpep_pickup_datetime": [datetime(2025, 1, 6, 9, 0)],
            "tpep_dropoff_datetime": [datetime(2025, 1, 6, 9, 17)],
            "PULocationID": [79],
            "DOLocationID": [161],
            "trip_distance": [2.5],
            "fare_amount": [-4.75],
            "total_amount": [-4.75],
            "congestion_surcharge": [None],
            "payment_type": [0],
            "cbd_congestion_fee": [-0.75],
        },
        schema_overrides={"congestion_surcharge": pl.Float64},
    )
    table, qa = run(tmp_path, frame, "yellow")
    assert qa["kept"] == 1
    assert qa["negative_kept"] == 1
    assert qa["reversal_twins"] == 0
    row = table.row(0, named=True)
    assert (row["trips"], row["charged"], row["flex_trips"]) == (1, 1, 1)
    assert row["fare_trips"] == 0


def test_two_different_trips_with_the_same_times_are_not_merged(tmp_path):
    frame = (
        yellow_frame(True)
        .head(2)
        .with_columns(
            pl.col("tpep_pickup_datetime").first(),
            pl.col("tpep_dropoff_datetime").first(),
            pl.Series("DOLocationID", [230, 231]),
        )
    )
    _, qa = run(tmp_path, frame, "yellow", 2025, 1)
    assert qa["reversal_twins"] == 0


def test_week_boundary_uses_floor_division_on_the_policy_sunday(tmp_path):
    table, _ = run(tmp_path, yellow_frame(True), "yellow")
    trips = dict(table.group_by("week").agg(pl.col("trips").sum()).iter_rows())
    assert trips == {-1: 1, 0: 2}


def test_fee_counts_only_positive_charges(tmp_path):
    table, _ = run(tmp_path, yellow_frame(True), "yellow")
    assert table.filter(pl.col("week") == 0)["charged"].sum() == 2
    assert table.filter(pl.col("week") == -1)["charged"].sum() == 0


def test_files_from_before_the_fee_column_still_build(tmp_path):
    table, _ = run(tmp_path, yellow_frame(False), "yellow")
    assert table["charged"].sum() == 0
    assert table["trips"].sum() == 3


def test_integer_width_drift_is_normalised(tmp_path):
    frame = yellow_frame(True).with_columns(pl.col("PULocationID").cast(pl.Int64))
    table, _ = run(tmp_path, frame, "yellow")
    assert table.schema["pu"] == pl.Int16


def test_toll_period_follows_the_mta_schedule():
    frame = pl.DataFrame(
        {
            "pickup": [
                datetime(2025, 1, 6, 5, 0),
                datetime(2025, 1, 6, 4, 59),
                datetime(2025, 1, 6, 20, 59),
                datetime(2025, 1, 6, 21, 0),
                datetime(2025, 1, 5, 8, 59),
                datetime(2025, 1, 5, 9, 0),
                datetime(2025, 1, 4, 20, 59),
            ]
        }
    )
    assert frame.select(build.toll_period())["period"].to_list() == [
        "peak",
        "overnight",
        "peak",
        "overnight",
        "overnight",
        "peak",
        "peak",
    ]


def test_hv_columns_map_onto_the_common_shape(tmp_path):
    frame = pl.DataFrame(
        {
            "pickup_datetime": [datetime(2025, 1, 6, 12, 0)],
            "dropoff_datetime": [datetime(2025, 1, 6, 12, 15)],
            "PULocationID": [79],
            "DOLocationID": [161],
            "trip_miles": [3.0],
            "trip_time": [900],
            "base_passenger_fare": [20.0],
            "congestion_surcharge": [2.75],
            "cbd_congestion_fee": [1.5],
        }
    )
    table, qa = run(tmp_path, frame, "hv")
    assert qa["kept"] == 1
    row = table.row(0, named=True)
    assert (row["service"], row["charged"], row["surcharged"]) == ("hv", 1, 1)
    assert abs(row["miles"] / row["seconds"] * 3600 - 12.0) < 1e-5
