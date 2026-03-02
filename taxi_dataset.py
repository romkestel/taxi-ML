import torch
from torch.utils.data import Dataset
import polars as pl


class TaxiDataset(Dataset):
    def __init__(self, file_path):
        df = pl.read_parquet(file_path)

        self.features = torch.tensor(
            df.select(
                [
                    (pl.col("trip_distance") / 50.0).clip(0, 1),
                    (pl.col("duration_mins") / 120.0).clip(0, 1),
                    (pl.col("pickup_hour") / 23.0),
                    (pl.col("day_of_week") / 6.0),
                ]
            ).to_numpy(),
            dtype=torch.float32,
        )

        self.target = torch.tensor(
            (df.select("tip_percentage") / 100.0).to_numpy(), dtype=torch.float32
        )

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.target[idx]
