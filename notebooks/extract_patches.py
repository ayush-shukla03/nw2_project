import rasterio
import numpy as np
import os
import glob
import random
from rasterio.windows import Window

SENTINEL_DIR = "/mnt/nw2data/nw2_project/data/raw/sentinel2"
PATCHES_DIR = "/mnt/nw2data/nw2_project/data/processed/patches"
os.makedirs(f"{PATCHES_DIR}/images", exist_ok=True)
os.makedirs(f"{PATCHES_DIR}/labels", exist_ok=True)

PATCH_SIZE = 256
PATCHES_PER_TILE = 15  # only 15 patches per tile × 8 tiles = 120 total

def extract_patches_from_tile(tile_dir, tile_name, n_patches=PATCHES_PER_TILE):
    print(f"\nExtracting patches from {tile_name}...")

    # Load water mask to find interesting areas
    water_mask_path = f"/mnt/nw2data/nw2_project/data/processed/classified/{tile_name}_water_mask.tif"
    if not os.path.exists(water_mask_path):
        print(f"  No water mask found, skipping")
        return 0

    # Load bands
    bands = {}
    for band in ["B02", "B03", "B04", "B08"]:
        pattern = os.path.join(
            tile_dir,
            "*.SAFE/GRANULE/*/IMG_DATA/R10m/*_{}_*.jp2".format(band)
        )
        files = glob.glob(pattern)
        if not files:
            print(f"  Band {band} not found, skipping")
            return 0
        with rasterio.open(files[0]) as src:
            bands[band] = src.read(1).astype(np.float32) / 10000.0
            profile = src.profile

    with rasterio.open(water_mask_path) as src:
        water_mask = src.read(1)

    height, width = bands["B02"].shape

    # Find patches near the river (where water mask has detections)
    # Sample random locations but bias toward water pixels
    water_rows, water_cols = np.where(water_mask > 0)

    if len(water_rows) == 0:
        print(f"  No water pixels found, skipping")
        return 0

    count = 0
    attempts = 0
    used_positions = set()

    while count < n_patches and attempts < n_patches * 10:
        attempts += 1

        # 70% of patches near water, 30% random (to capture non-water classes)
        if random.random() < 0.7 and len(water_rows) > 0:
            idx = random.randint(0, len(water_rows) - 1)
            cy = water_rows[idx]
            cx = water_cols[idx]
            y = max(0, cy - PATCH_SIZE // 2)
            x = max(0, cx - PATCH_SIZE // 2)
        else:
            y = random.randint(0, height - PATCH_SIZE)
            x = random.randint(0, width - PATCH_SIZE)

        # Ensure patch fits
        if y + PATCH_SIZE > height or x + PATCH_SIZE > width:
            continue

        # Avoid duplicate positions
        pos_key = (y // 128, x // 128)
        if pos_key in used_positions:
            continue
        used_positions.add(pos_key)

        # Stack bands
        patch = np.stack([
            bands["B02"][y:y+PATCH_SIZE, x:x+PATCH_SIZE],
            bands["B03"][y:y+PATCH_SIZE, x:x+PATCH_SIZE],
            bands["B04"][y:y+PATCH_SIZE, x:x+PATCH_SIZE],
            bands["B08"][y:y+PATCH_SIZE, x:x+PATCH_SIZE],
        ])

        # Skip mostly zero patches
        if (patch == 0).sum() / patch.size > 0.3:
            continue

        # Save patch
        patch_name = f"{tile_name}_y{y}_x{x}.tif"
        patch_path = os.path.join(PATCHES_DIR, "images", patch_name)

        patch_profile = profile.copy()
        patch_profile.update({
            'count': 4,
            'height': PATCH_SIZE,
            'width': PATCH_SIZE,
            'dtype': 'float32',
            'driver': 'GTiff',
            'compress': 'lzw'
        })

        with rasterio.open(patch_path, 'w', **patch_profile) as dst:
            dst.write(patch)

        count += 1

    print(f"  ✓ Extracted {count} patches from {tile_name}")
    return count

# Process pre-monsoon tiles only
tiles = sorted([
    d for d in os.listdir(SENTINEL_DIR)
    if os.path.isdir(os.path.join(SENTINEL_DIR, d))
    and not d.startswith('monsoon')
    and not d.startswith('postmonsoon')
])

total = 0
for tile_name in tiles:
    tile_dir = os.path.join(SENTINEL_DIR, tile_name)
    total += extract_patches_from_tile(tile_dir, tile_name)

print(f"\n✓ Total patches extracted: {total}")
print(f"Saved to: {PATCHES_DIR}/images/")
du = os.popen(f"du -sh {PATCHES_DIR}").read()
print(f"Disk usage: {du}")