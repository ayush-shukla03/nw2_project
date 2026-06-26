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
EPOCHS_PHASE1 = 80
EPOCHS_PHASE2 = 20
LR_PHASE1 = 1e-4
LR_PHASE2 = 1e-5
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {DEVICE}')

# Exact manual patches (converted from _label.tif to image names)
MANUAL_IMAGE_NAMES = [
    "T46RBP_y1242_x9383.tif", "T46RBP_y1389_x279.tif", "T46RBP_y2017_x880.tif",
    "T46RCP_y0_x0.tif", "T46RCP_y0_x2959.tif", "T46RCP_y0_x7781.tif",
    "T46RCP_y1034_x2749.tif", "T46RCP_y335_x7241.tif", "T46RCP_y52_x2399.tif",
    "T46RCQ_y10087_x1278.tif", "T46RCQ_y9716_x2127.tif",
    "T46RDQ_y10724_x612.tif", "T46RDQ_y6778_x3115.tif"
]

class BrahmaputraDataset(Dataset):
    def __init__(self, patch_paths, augment=False):
        self.patch_paths = patch_paths
        self.augment = augment

    def __len__(self):
        return len(self.patch_paths)

    def augment_data(self, image, label):
        if random.random() > 0.5:
            image = np.flip(image, axis=2).copy()
            label = np.flip(label, axis=1).copy()
        if random.random() > 0.5:
            image = np.flip(image, axis=1).copy()
            label = np.flip(label, axis=0).copy()
        k = random.randint(0, 3)
        image = np.rot90(image, k, axes=(1, 2)).copy()
        label = np.rot90(label, k, axes=(0, 1)).copy()
        if random.random() > 0.5:
            factor = random.uniform(0.8, 1.2)
            image = np.clip(image * factor, 0, 1)
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

# ─── Data Separation ─────────────────────────────────────────────────────────
all_patches = sorted(glob.glob(f'{PATCHES_DIR}/images/*.tif'))
manual_patches = []
auto_patches = []

for p in all_patches:
    label_p = p.replace('/images/', '/labels/').replace('.tif', '_label.tif')
    if not os.path.exists(label_p) or 'test_patch' in p:
        continue
    if os.path.basename(p) in MANUAL_IMAGE_NAMES:
        manual_patches.append(p)
    else:
        auto_patches.append(p)

print(f'Found {len(manual_patches)} Manual patches and {len(auto_patches)} Auto patches.')

# Phase 1 Splits (Auto Labels)
auto_train, auto_val = train_test_split(auto_patches, test_size=0.2, random_state=42)
# Phase 2 Splits (Manual Labels) - Reserving 3 for pure manual validation
manual_train, manual_val = train_test_split(manual_patches, test_size=3, random_state=42)

# Oversample the manual training data so an epoch has enough batches
manual_train_oversampled = manual_train * 10 

train_loader_p1 = DataLoader(BrahmaputraDataset(auto_train, augment=True), batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader_p1   = DataLoader(BrahmaputraDataset(auto_val, augment=False), batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

train_loader_p2 = DataLoader(BrahmaputraDataset(manual_train_oversampled, augment=True), batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader_p2   = DataLoader(BrahmaputraDataset(manual_val, augment=False), batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# ─── Model Setup ─────────────────────────────────────────────────────────────
model = smp.Unet(
    encoder_name='resnet34',
    encoder_weights='imagenet',
    in_channels=4,
    classes=NUM_CLASSES,
).to(DEVICE)

class_weights = torch.tensor([1.0, 2.0, 1.0, 1.0, 3.0]).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=255)

# ─── Phase 1: Train on Auto Labels ───────────────────────────────────────────
print("\n" + "="*50)
print(f" PHASE 1: Pre-training on Auto Labels ({EPOCHS_PHASE1} Epochs)")
print("="*50)

optimizer = torch.optim.Adam(model.parameters(), lr=LR_PHASE1)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
best_val_loss_p1 = float('inf')

for epoch in range(EPOCHS_PHASE1):
    model.train()
    train_loss = 0
    for images, labels in train_loader_p1:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader_p1)

    model.eval()
    val_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for images, labels in val_loader_p1:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.numel()
            
    val_loss /= len(val_loader_p1)
    val_acc = correct / total * 100
    scheduler.step(val_loss)

    print(f'P1 Epoch {epoch+1:3d}/{EPOCHS_PHASE1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.1f}%')

    if val_loss < best_val_loss_p1:
        best_val_loss_p1 = val_loss
        torch.save(model.state_dict(), f'{MODEL_DIR}/unet_phase1_best.pth')

# ─── Phase 2: Fine-Tune on Manual Labels ─────────────────────────────────────
print("\n" + "="*50)
print(f" PHASE 2: Fine-tuning on Manual Labels ({EPOCHS_PHASE2} Epochs)")
print("="*50)

# Load the best weights from Phase 1
model.load_state_dict(torch.load(f'{MODEL_DIR}/unet_phase1_best.pth'))

# Freeze the ResNet encoder to prevent catastrophic forgetting
for param in model.encoder.parameters():
    param.requires_grad = False

# New optimizer that only updates the decoder with a much smaller learning rate
optimizer_p2 = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR_PHASE2)
best_val_loss_p2 = float('inf')

for epoch in range(EPOCHS_PHASE2):
    model.train()
    train_loss = 0
    for images, labels in train_loader_p2:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer_p2.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer_p2.step()
        train_loss += loss.item()
    train_loss /= len(train_loader_p2)

    model.eval()
    val_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for images, labels in val_loader_p2:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.numel()
            
    val_loss /= len(val_loader_p2)
    val_acc = correct / total * 100

    print(f'P2 Epoch {epoch+1:2d}/{EPOCHS_PHASE2} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.1f}%')

    if val_loss < best_val_loss_p2:
        best_val_loss_p2 = val_loss
        torch.save(model.state_dict(), f'{MODEL_DIR}/unet_final_best.pth')

print(f'\nTraining complete.')
print(f'Final production model saved to {MODEL_DIR}/unet_final_best.pth')