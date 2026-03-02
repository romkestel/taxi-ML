import torch
import torch.nn as nn
from typing import cast


class WideAndDeepModel(nn.Module):
    def __init__(self, cat_dims, num_cont):
        super(WideAndDeepModel, self).__init__()

        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(num_embeddings=dim, embedding_dim=min(50, (dim + 1) // 2))
                for dim in cat_dims
            ]
        )

        embedding_sum = sum(
            cast(nn.Embedding, emb).embedding_dim for emb in self.embeddings
        )

        self.deep_layers = nn.Sequential(
            nn.Linear(embedding_sum + num_cont, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, cat_data, cont_data):

        embedded = [emb(cat_data[:, i]) for i, emb in enumerate(self.embeddings)]
        x_cat = torch.cat(embedded, dim=1)

        x = torch.cat([x_cat, cont_data], dim=1)

        return self.deep_layers(x).squeeze()
