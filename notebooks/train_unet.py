import torch
import torch.nn as nn
import numpy as np
import os
import glob
import rasterio
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
from sklearn.model_selection import train_test_split
import random

PATCHES_DIR = '/mnt/nw2data/nw2_project/data/processed/patches'
MODEL_DIR = '/mnt/nw2data/nw2_project/models'
os.makedirs(MODEL_DIR, exist_ok=True)

# Hyperparameters
NUM_CLASSES = 5
BATCH_SIZE = 4
EPOCHS = 50
LR = 1e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {DEVICE}')

class BrahmaputraDataset(Dataset):
    def __init__(self, patch_paths, augment=False):
        self.patch_paths = patch_paths
        self.augment = augment

    def __len__(self):
        return len(self.patch_paths)

    def augment_data(self, image, label):
        # Random horizontal flip
        if random.random() > 0.5:
            image = np.flip(image, axis=2).copy()
            label = np.flip(label, axis=1).copy()

        # Random vertical flip
        if random.random() > 0.5:
            image = np.flip(image, axis=1).copy()
            label = np.flip(label, axis=0).copy()

        # Random 90 degree rotation
        k = random.randint(0, 3)
        image = np.rot90(image, k, axes=(1, 2)).copy()
        label = np.rot90(label, k, axes=(0, 1)).copy()

        # Random brightness adjustment
        if random.random() > 0.5:
            factor = random.uniform(0.8, 1.2)
            image = np.clip(image * factor, 0, 1)

        # Random channel noise
        if random.random() > 0.5:
            noise = np.random.normal(0, 0.01, image.shape).astype(np.float32)
            image = np.clip(image + noise, 0, 1)

        return image, label

    def __getitem__(self, idx):
        patch_path = self.patch_paths[idx]
        label_path = patch_path.replace('/images/', '/labels/').replace('.tif', '_label.tif')

        with rasterio.open(patch_path) as src:
            image = src.read().astype(np.float32)

        with rasterio.open(label_path) as src:
            label = src.read(1).astype(np.int64)

        label[label == 255] = 0
        image = np.clip(image, 0, 1)

        if self.augment:
            image, label = self.augment_data(image, label)

        return torch.tensor(image.copy()), torch.tensor(label.copy())

# Get all patches that have labels
all_patches = sorted(glob.glob(f'{PATCHES_DIR}/images/*.tif'))
all_patches = [p for p in all_patches
               if os.path.exists(p.replace('/images/', '/labels/').replace('.tif', '_label.tif'))
               and 'test_patch' not in p]

print(f'Found {len(all_patches)} labelled patches')

# Train/val split
train_patches, val_patches = train_test_split(all_patches, test_size=0.2, random_state=42)
print(f'Train: {len(train_patches)} | Val: {len(val_patches)}')

train_dataset = BrahmaputraDataset(train_patches, augment=True)
val_dataset   = BrahmaputraDataset(val_patches,   augment=False)

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
class_weights = torch.tensor([1.0, 2.0, 1.0, 1.0, 3.0]).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=255)
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