import polars as pl
import geopandas as gpd


# 1. Setting sources
def pipeline_cleaner(datasource: str):
    db_uri = f"sqlite://./data/2019/{datasource}.sqlite"
    zones_path = "./data/taxi_zones/taxi_zones.dbf"
    query = "SELECT * FROM tripdata"
    # 2. Lazy-Extract the data from the tables

    raw_trips = pl.read_database_uri(
        query=query, uri=db_uri, engine="connectorx"
    ).lazy()
    zones_lookup = pl.from_pandas(
        gpd.read_file(zones_path).drop(columns=["geometry"], errors="ignore")
    ).lazy()

    # 3. Transformation Pipeline

    clean_df = (
        raw_trips
        .with_columns(
            pl.col("pulocationid").cast(pl.Int32)
        )
        .filter(
            (pl.col("fare_amount") > 0)
            & (pl.col("trip_distance") > 0)
            & (pl.col("passenger_count") > 0)
        )
        .join(zones_lookup, left_on="pulocationid", right_on="LocationID", how="left")
        .collect()
    )

    clean_df.write_parquet(
        f"clean_{datasource}.parquet", compression="zstd", compression_level=3
    )
    print(f"Pipeline Complete! Processed {len(clean_df)} clean rows.")
