import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from unet_model import UNet
from dataset import PolygonDataset
import wandb

wandb.init(project="ayna-polygon-colorizer")

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Hyperparameters
EPOCHS = 300
LR = 1e-3
BATCH_SIZE = 16

model = UNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()

train_dataset = PolygonDataset("dataset/training")
val_dataset = PolygonDataset("dataset/validation")
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

def train_one_epoch():
    model.train()
    total_loss = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def validate():
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            total_loss += loss.item()
    return total_loss / len(val_loader)

best_loss = float('inf')

for epoch in range(EPOCHS):
    train_loss = train_one_epoch()
    val_loss = validate()
    print(f"Epoch {epoch+1} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    wandb.log({"epoch": epoch+1, "train_loss": train_loss, "val_loss": val_loss})

    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(model.state_dict(), "best_unet.pth")
        artifact = wandb.Artifact("model", type="model")
        artifact.add_file("best_unet.pth")
        wandb.log_artifact(artifact)

