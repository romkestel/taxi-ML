import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from wide_and_deep_model import WideAndDeepModel
from taxi_dataset import TaxiDataset

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

train_ds = TaxiDataset("./Dataset/train_compressed.parquet")
train_loader = DataLoader(train_ds, batch_size=4096, shuffle=True, num_workers=4)


cat_dims = (train_ds.cat_features.max(dim=0)[0] + 1).tolist()
print(f"Dynamic Embedding Size: {cat_dims}")
model = WideAndDeepModel(cat_dims=cat_dims, num_cont=9).to(device)


criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(7):
    model.train()
    epoch_loss = 0

    for cat, cont, target in train_loader:
        cat, cont, target = cat.to(device), cont.to(device), target.squeeze().to(device)

        preds = model(cat, cont)
        loss = criterion(preds, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    print(f"Epoch {epoch + 1} | Loss: {epoch_loss / len(train_loader):.6f}")
torch.save(model.state_dict(), "wide_deep_taxi.pth")
