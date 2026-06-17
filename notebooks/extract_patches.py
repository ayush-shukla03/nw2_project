import rasterio
import numpy as np
import os
import glob
from rasterio.windows import Window

SENTINEL_DIR = "/mnt/nw2data/nw2_project/data/raw/sentinel2"
PATCHES_DIR = "/mnt/nw2data/nw2_project/data/processed/patches"
os.makedirs(f"{PATCHES_DIR}/images", exist_ok=True)
os.makedirs(f"{PATCHES_DIR}/labels", exist_ok=True)

PATCH_SIZE = 256
STRIDE = 128  # 50% overlap between patches

def extract_patches_from_tile(tile_dir, tile_name):
    print(f"\nExtracting patches from {tile_name}...")

    # Load all 4 bands
    bands = {}
    for band in ["B02", "B03", "B04", "B08"]:
        pattern = os.path.join(
            tile_dir,
            "*.SAFE/GRANULE/*/IMG_DATA/R10m/*_{}_*.jp2".format(band)
        )
        files = glob.glob(pattern)
        if not files:
            print(f"  Band {band} not found, skipping tile")
            return 0
        with rasterio.open(files[0]) as src:
            bands[band] = src.read(1).astype(np.float32) / 10000.0
            profile = src.profile

    height, width = bands["B02"].shape
    count = 0

    for y in range(0, height - PATCH_SIZE, STRIDE):
        for x in range(0, width - PATCH_SIZE, STRIDE):
            # Stack bands into 4-channel patch
            patch = np.stack([
                bands["B02"][y:y+PATCH_SIZE, x:x+PATCH_SIZE],
                bands["B03"][y:y+PATCH_SIZE, x:x+PATCH_SIZE],
                bands["B04"][y:y+PATCH_SIZE, x:x+PATCH_SIZE],
                bands["B08"][y:y+PATCH_SIZE, x:x+PATCH_SIZE],
            ])

            # Skip patches that are mostly zero (nodata)
            if (patch == 0).sum() / patch.size > 0.5:
                continue

            # Save patch as GeoTIFF
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

# Process pre-monsoon tiles only for now
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
print("\nNext step: open patches in QGIS and create label files")