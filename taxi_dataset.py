import torch
from torch.utils.data import Dataset
import polars as pl


class TaxiDataset(Dataset):
    def __init__(self, file_path):
        df = pl.read_parquet(file_path)

        self.cat_features = torch.tensor(
            df.select(
                [
                    "pulocationid",
                    "dolocationid",
                    "vendorid",
                    "ratecodeid",
                    "pickup_day",
                    "dropoff_day",
                ]
            ).to_numpy(),
            dtype=torch.long,
        )

        self.cont_features = torch.tensor(
            df.select(
                [
                    (pl.col("trip_distance") / 50.0).clip(0, 1),
                    (pl.col("duration_mins") / 180.0).clip(0, 1),
                    (pl.col("avg_speed_mph") / 100.0).clip(0, 1),
                    "pickup_hour_sin",
                    "pickup_hour_cos",
                    "dropoff_hour_sin",
                    "dropoff_hour_cos",
                    (pl.col("fare_amount") / 200.0).clip(0, 1),
                    (pl.col("congestion_surcharge") / 10.0).clip(0, 1),
                ]
            ).to_numpy(),
            dtype=torch.float32,
        )

        self.target = torch.tensor(
            (df.select("tip_percentage") / 100.0).to_numpy(), dtype=torch.float32
        )

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        return self.cat_features[idx], self.cont_features[idx], self.target[idx]
