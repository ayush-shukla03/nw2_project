import rasterio
import numpy as np
import os
from rasterio.enums import Resampling
import glob
import matplotlib
matplotlib.use('Agg')

SENTINEL_DIR = "/mnt/nw2data/nw2_project/data/raw/sentinel2"
OUTPUT_DIR = "/mnt/nw2data/nw2_project/data/processed/classified"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_band(tile_path, band_name):
    pattern = os.path.join(tile_path, "*.SAFE/GRANULE/*/IMG_DATA/R10m/*_{}_*.jp2".format(band_name))
    files = glob.glob(pattern)
    if not files:
        return None, None
    with rasterio.open(files[0]) as src:
        return src.read(1).astype(float), src.profile

def get_cloud_mask(tile_path, profile):
    """Load cloud probability mask and resample to 10m"""
    pattern = os.path.join(tile_path, "*.SAFE/GRANULE/*/QI_DATA/MSK_CLDPRB_20m.jp2")
    files = glob.glob(pattern)
    if not files:
        print("  No cloud mask found, skipping cloud filtering")
        return None

    with rasterio.open(files[0]) as src:
        # Resample from 20m to 10m to match band resolution
        target_height = profile['height']
        target_width  = profile['width']
        cloud_prob = src.read(
            1,
            out_shape=(1, target_height, target_width),
            resampling=Resampling.nearest
        ).astype(float)

    print(f"  Cloud mask loaded — mean cloud prob: {cloud_prob.mean():.1f}%")
    return cloud_prob

def compute_water_indices(tile_dir, tile_name):
    print(f"\nProcessing {tile_name}...")

    green, profile = get_band(tile_dir, "B03")
    nir, _         = get_band(tile_dir, "B08")
    blue, _        = get_band(tile_dir, "B02")
    red, _         = get_band(tile_dir, "B04")

    if green is None or nir is None:
        print(f"  Skipping {tile_name} — missing bands")
        return None

    # Normalize to reflectance
    green = green / 10000.0
    nir   = nir   / 10000.0
    blue  = blue  / 10000.0
    red   = red   / 10000.0

    # NDWI
    ndwi = np.where(
        (green + nir) > 0,
        (green - nir) / (green + nir),
        0
    )

    # AWEI
    awei = 4 * (green - nir) - (0.25 * nir + 2.75 * red)

    # Water masks with Brahmaputra-tuned thresholds
    ndwi_mask = ndwi > -0.05
    awei_mask = awei > -0.5
    high_ndwi = ndwi < 0.4   # exclude clouds/vegetation

    ensemble = (ndwi_mask | awei_mask) & high_ndwi

    # Apply cloud mask — exclude pixels with >30% cloud probability
    cloud_prob = get_cloud_mask(tile_dir, profile)
    if cloud_prob is not None:
        clear_mask = cloud_prob < 30
        ensemble = ensemble & clear_mask
        cloud_pct = (cloud_prob >= 30).sum() / cloud_prob.size * 100
        print(f"  Cloudy pixels masked: {cloud_pct:.1f}%")

    # Save outputs
    profile.update(dtype=rasterio.float32, count=1, compress='lzw', driver='GTiff')
    ndwi_path = os.path.join(OUTPUT_DIR, f"{tile_name}_NDWI.tif")
    with rasterio.open(ndwi_path, 'w', **profile) as dst:
        dst.write(ndwi.astype(np.float32), 1)

    profile.update(dtype=rasterio.uint8, driver='GTiff')
    mask_path = os.path.join(OUTPUT_DIR, f"{tile_name}_water_mask.tif")
    with rasterio.open(mask_path, 'w', **profile) as dst:
        dst.write(ensemble.astype(np.uint8), 1)

    water_pct = ensemble.sum() / ensemble.size * 100
    print(f"  ✓ Water pixels: {water_pct:.1f}% of tile")
    print(f"  Saved: {mask_path}")

    return ensemble, profile

# Process all tiles
tiles = sorted([
    d for d in os.listdir(SENTINEL_DIR)
    if os.path.isdir(os.path.join(SENTINEL_DIR, d))
])

results = {}
for tile_name in tiles:
    tile_dir = os.path.join(SENTINEL_DIR, tile_name)
    result = compute_water_indices(tile_dir, tile_name)
    if result:
        results[tile_name] = result

print(f"\n✓ Water extraction complete for {len(results)} tiles")
print(f"Output saved to: {OUTPUT_DIR}")