from torch.utils.data import DataLoader
import tip_predictor as tp
import taxi_dataset as td
import torch
import torch.optim as optim
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = tp.TipPredictor().to(device)
dataset = td.TaxiDataset("./data/train_tips.parquet")
loader = DataLoader(dataset, batch_size=4096, shuffle=True, num_workers=8)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


for epoch in range(5):
    total_loss = 0
    for batch_features, batch_targets in loader:
        x, y = batch_features.to(device), batch_targets.to(device)

        predictions = model(x)
        loss = criterion(predictions, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1} | Loss: {total_loss / len(loader):.4f}")

torch.save(model.state_dict(), "taxi_model.pth")
