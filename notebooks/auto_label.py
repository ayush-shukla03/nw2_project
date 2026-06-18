import rasterio
import numpy as np
import os
import glob

SENTINEL_DIR = '/mnt/nw2data/nw2_project/data/raw/sentinel2'
PATCHES_DIR = '/mnt/nw2data/nw2_project/data/processed/patches'
LABELS_DIR = f'{PATCHES_DIR}/labels'
os.makedirs(LABELS_DIR, exist_ok=True)

def auto_label_patch(patch_path):
    with rasterio.open(patch_path) as src:
        b02 = src.read(1).astype(float)  # Blue
        b03 = src.read(2).astype(float)  # Green
        b04 = src.read(3).astype(float)  # Red
        b08 = src.read(4).astype(float)  # NIR
        profile = src.profile


    # Indices
    ndwi = (b03 - b08) / (b03 + b08 + 1e-10)
    ndvi = (b08 - b04) / (b08 + b04 + 1e-10)
    brightness = (b02 + b03 + b04) / 3

    # Classification rules
    # Start with unclassified
    label = np.zeros(b02.shape, dtype=np.uint8)

    # Class 3 — Vegetated island (high NDVI)
    label[ndvi > 0.2] = 3

    # Class 4 — Mudflat (low brightness, low NDVI, low NDWI)
    label[(brightness < 0.05) & (ndvi < 0.05) & (ndwi < -0.2)] = 4
    # Class 1 — Active sandbar (high brightness, low NDVI, low NDWI)
    label[(brightness > 0.15) & (ndvi < 0.1)] = 1

    # Class 2 — Sediment-laden shallow (moderate NDWI, high turbidity)
    label[(ndwi > -0.15) & (ndwi < 0.0) & (ndvi < 0.1)] = 2

    # Class 0 — Open water (high NDWI or very low NIR)
    label[ndwi > 0.0] = 0

    # Nodata mask
    label[(b02 == 0) & (b03 == 0)] = 255

    # Save label raster
    patch_name = os.path.basename(patch_path).replace('.tif', '_label.tif')
    label_path = os.path.join(LABELS_DIR, patch_name)

    profile.update({
        'count': 1,
        'dtype': 'uint8',
        'driver': 'GTiff',
        'compress': 'lzw'
    })

    with rasterio.open(label_path, 'w', **profile) as dst:
        dst.write(label, 1)

    # Class distribution
    classes = {0:'water', 1:'sandbar', 2:'shallow', 3:'vegetation', 4:'mudflat'}
    dist = {classes[i]: (label==i).sum()/label.size*100 for i in range(5)}
    return dist

# Process all patches
patches = sorted(glob.glob(f'{PATCHES_DIR}/images/*.tif'))
print(f'Auto-labelling {len(patches)} patches...\n')

for path in patches:
    name = os.path.basename(path)
    dist = auto_label_patch(path)
    dist_str = ' | '.join([f'{k}:{v:.0f}%' for k,v in dist.items()])
    print(f'{name:40} → {dist_str}')

print(f'\n✓ Done. Labels saved to {LABELS_DIR}')
du = os.popen(f'du -sh {LABELS_DIR}').read().strip()
print(f'Disk usage: {du}')