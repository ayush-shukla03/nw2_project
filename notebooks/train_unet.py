import torch
import torch.nn as nn
import numpy as np
import os
import glob
import rasterio
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
from sklearn.model_selection import train_test_split

PATCHES_DIR = '/mnt/nw2data/nw2_project/data/processed/patches'
MODEL_DIR = '/mnt/nw2data/nw2_project/models'
os.makedirs(MODEL_DIR, exist_ok=True)

# Hyperparameters
NUM_CLASSES = 5
BATCH_SIZE = 4
EPOCHS = 30
LR = 1e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {DEVICE}')

class BrahmaputraDataset(Dataset):
    def __init__(self, patch_paths):
        self.patch_paths = patch_paths

    def __len__(self):
        return len(self.patch_paths)

    def __getitem__(self, idx):
        patch_path = self.patch_paths[idx]
        label_path = patch_path.replace('/images/', '/labels/').replace('.tif', '_label.tif')

        # Load image
        with rasterio.open(patch_path) as src:
            image = src.read().astype(np.float32)  # (4, 256, 256)

        # Load label
        with rasterio.open(label_path) as src:
            label = src.read(1).astype(np.int64)  # (256, 256)

        # Handle nodata
        label[label == 255] = 0

        # Normalize image to 0-1
        image = np.clip(image, 0, 1)

        return torch.tensor(image), torch.tensor(label)

# Get all patches that have labels
all_patches = sorted(glob.glob(f'{PATCHES_DIR}/images/*.tif'))
all_patches = [p for p in all_patches
               if os.path.exists(p.replace('/images/', '/labels/').replace('.tif', '_label.tif'))
               and 'test_patch' not in p]

print(f'Found {len(all_patches)} labelled patches')

# Train/val split
train_patches, val_patches = train_test_split(all_patches, test_size=0.2, random_state=42)
print(f'Train: {len(train_patches)} | Val: {len(val_patches)}')

train_dataset = BrahmaputraDataset(train_patches)
val_dataset = BrahmaputraDataset(val_patches)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# U-Net model
model = smp.Unet(
    encoder_name='resnet34',
    encoder_weights='imagenet',
    in_channels=4,          # B02, B03, B04, B08
    classes=NUM_CLASSES,
).to(DEVICE)

# Loss and optimizer
criterion = nn.CrossEntropyLoss(ignore_index=255)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

# Training loop
best_val_loss = float('inf')

for epoch in range(EPOCHS):
    # Train
    model.train()
    train_loss = 0
    for images, labels in train_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)

    # Validate
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.numel()

    val_loss /= len(val_loader)
    val_acc = correct / total * 100
    scheduler.step(val_loss)

    print(f'Epoch {epoch+1:3d}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.1f}%')

    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), f'{MODEL_DIR}/unet_best.pth')
        print(f'  ✓ Best model saved')

print(f'\nTraining complete. Best val loss: {best_val_loss:.4f}')
print(f'Model saved to {MODEL_DIR}/unet_best.pth')