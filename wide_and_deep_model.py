from torch import embedding
import torch
import torch.nn as nn


class WideAndDeepModel(nn.Module):
    def __init__(self, cat_dims, num_cont):
        super(WideAndDeepModel, self).__init__()

        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(num_embeddings=dim, embedding_dim=min(50, (dim + 1) // 2))
                for dim in cat_dims
            ]
        )

        embedding_sum = sum([emb.embedding_dim for emb in self.embeddings])

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
