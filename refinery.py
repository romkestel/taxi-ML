import polars as pl
import numpy as np
import glob


def data_refinery(input_file, output_path):
    df = pl.read_parquet(input_file)

    df = df.with_columns(
        [
            (
                (
                    pl.col("tpep_dropoff_datetime") - pl.col("tpep_pickup_datetime")
                ).dt.total_seconds()
                / 60.0
            ).alias("duration_mins")
        ]
    )
    df = df.filter(
        (pl.col("fare_amount") > 0.0)
        & (pl.col("trip_distance") > 0.0)
        & (pl.col("total_amount") > 0.0)
        & (pl.col("passenger_count") > 0.0)
        & (pl.col("passenger_count") < 9.0)
        & (pl.col("duration_mins") > 0.5)
        & (pl.col("duration_mins") < 180.0)
    )

    df = df.with_columns(
        [
            (
                (pl.col("tip_amount") / (pl.col("total_amount") - pl.col("tip_amount")))
                * 100.0
            ).alias("tip_percentage")
        ]
    )

    df = df.filter(
        (pl.col("tip_percentage") >= 0.0) & (pl.col("tip_percentage") <= 100.0)
    )

    df = df.with_columns(
        [
            pl.col("tpep_pickup_datetime").dt.hour().alias("pickup_hour"),
            pl.col("tpep_pickup_datetime").dt.weekday().alias("pickup_day"),
            (pl.col("tpep_pickup_datetime").dt.weekday() > 5.0).alias(
                "is_weekend_pickup"
            ),
            pl.col("tpep_dropoff_datetime").dt.hour().alias("dropoff_hour"),
            pl.col("tpep_dropoff_datetime").dt.weekday().alias("dropoff_day"),
            (pl.col("tpep_dropoff_datetime").dt.weekday() > 5.0).alias(
                "is_weekend_dropoff"
            ),
        ]
    )

    df = df.with_columns(
        [
            (np.sin(2.0 * np.pi * pl.col("pickup_hour") / 24.0)).alias(
                "pickup_hour_sin"
            ),
            (np.cos(2 * np.pi * pl.col("pickup_hour") / 24.0)).alias("pickup_hour_cos"),
            (np.sin(2.0 * np.pi * pl.col("dropoff_hour") / 24.0)).alias(
                "dropoff_hour_sin"
            ),
            (np.cos(2 * np.pi * pl.col("dropoff_hour") / 24.0)).alias(
                "dropoff_hour_cos"
            ),
        ]
    )

    df = df.with_columns(
        [
            pl.col("pulocationid").cast(pl.Int64),
            pl.col("dolocationid").cast(pl.Int64),
            pl.col("vendorid").cast(pl.Int64),
            pl.col("ratecodeid").cast(pl.Int64),
            ((pl.col("trip_distance") / pl.col("duration_mins")) * 60.0).alias(
                "avg_speed_mph"
            ),
        ]
    )

    money_cols = [
        "fare_amount",
        "extra",
        "mta_tax",
        "tolls_amount",
        "improvement_surcharge",
        "congestion_surcharge",
    ]
    df = df.with_columns([pl.col(c).fill_null(0.0) for c in money_cols])

    df = df.filter(pl.col("avg_speed_mph").is_finite())

    df = df.drop(["tpep_pickup_datetime", "tpep_dropoff_datetime", "tip_amount"])

    df.write_parquet(output_path)

    print(f"✅ Refinery Complete! Final features: {df.columns}")
    print(f"Final Row Count: {len(df)}")


if __name__ == "__main__":
    INPUT_PATTERN = "./clean_data/clean_2019-*.parquet"
    OUTPUT = "./data/refined_whole_source.parquet"
    if glob.glob(INPUT_PATTERN):
        data_refinery(INPUT_PATTERN, OUTPUT)
    else:
        print(f"COULD NOT FIND ANY FILES MATCHING {INPUT_PATTERN}")
