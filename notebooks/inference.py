import torch
import numpy as np
import rasterio
import glob
import os
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from rasterio.windows import Window

MODEL_PATH = '/mnt/nw2data/nw2_project/models/unet_best.pth'
OUTPUT_DIR = '/mnt/nw2data/nw2_project/outputs/maps'
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
PATCH_SIZE = 256
NUM_CLASSES = 5

# Load model
model = smp.Unet(
    encoder_name='resnet34',
    encoder_weights=None,
    in_channels=4,
    classes=NUM_CLASSES,
).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
print('Model loaded')

def predict_tile(tile_name):
    SENTINEL_DIR = '/mnt/nw2data/nw2_project/data/raw/sentinel2'
    tile_dir = os.path.join(SENTINEL_DIR, tile_name)

    # Load bands
    bands = {}
    for band in ['B02', 'B03', 'B04', 'B08']:
        pattern = os.path.join(tile_dir,
            '*.SAFE/GRANULE/*/IMG_DATA/R10m/*_{}_*.jp2'.format(band))
        files = glob.glob(pattern)
        if not files:
            print(f'Band {band} not found')
            return
        with rasterio.open(files[0]) as src:
            bands[band] = src.read(1).astype(np.float32)
            profile = src.profile
            height, width = src.height, src.width

    print(f'Predicting {tile_name} ({height}x{width})...')

    # Full tile prediction using sliding window
    prediction = np.zeros((height, width), dtype=np.float32)
    count = np.zeros((height, width), dtype=np.float32)

    stride = PATCH_SIZE // 2  # 50% overlap for smoother results

    with torch.no_grad():
        for y in range(0, height - PATCH_SIZE + 1, stride):
            for x in range(0, width - PATCH_SIZE + 1, stride):
                patch = np.stack([
                    bands['B02'][y:y+PATCH_SIZE, x:x+PATCH_SIZE],
                    bands['B03'][y:y+PATCH_SIZE, x:x+PATCH_SIZE],
                    bands['B04'][y:y+PATCH_SIZE, x:x+PATCH_SIZE],
                    bands['B08'][y:y+PATCH_SIZE, x:x+PATCH_SIZE],
                ]) / 10000.0  # normalize

                patch = np.clip(patch, 0, 1)
                tensor = torch.tensor(patch).unsqueeze(0).to(DEVICE)
                output = model(tensor)
                pred = output.argmax(dim=1).squeeze().cpu().numpy()

                prediction[y:y+PATCH_SIZE, x:x+PATCH_SIZE] += pred
                count[y:y+PATCH_SIZE, x:x+PATCH_SIZE] += 1

    # Average overlapping predictions
    count[count == 0] = 1
    prediction = (prediction / count).astype(np.uint8)

    # Save prediction raster
    out_profile = profile.copy()
    out_profile.update({'count': 1, 'dtype': 'uint8', 'driver': 'GTiff', 'compress': 'lzw'})
    out_path = os.path.join(OUTPUT_DIR, f'{tile_name}_segmentation.tif')
    with rasterio.open(out_path, 'w', **out_profile) as dst:
        dst.write(prediction, 1)

    # Save visualisation
    cmap = mcolors.ListedColormap(['blue', 'yellow', 'cyan', 'red', 'brown'])
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    plt.figure(figsize=(10, 10))
    plt.imshow(prediction, cmap=cmap, norm=norm)
    labels = ['Water(0)', 'Sandbar(1)', 'Shallow(2)', 'Vegetation(3)', 'Mudflat(4)']
    colors = ['blue', 'yellow', 'cyan', 'red', 'brown']
    patches = [plt.Rectangle((0,0),1,1, color=c) for c in colors]
    plt.legend(patches, labels, loc='lower right', fontsize=10)
    plt.title(f'{tile_name} — U-Net segmentation')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'{tile_name}_segmentation.png'), dpi=100)
    plt.close()

    print(f'✓ Saved to {out_path}')

    # Class distribution
    classes = {0:'water', 1:'sandbar', 2:'shallow', 3:'vegetation', 4:'mudflat'}
    for i, name in classes.items():
        pct = (prediction == i).sum() / prediction.size * 100
        print(f'  {name}: {pct:.1f}%')

# Run on all the tiles
for tile in ['T46RBP', 'T46RBQ', 'T46RCP', 'T46RCQ', 'T46RCR', 'T46RDP', 'T46RDQ', 'T46RDR']:
    predict_tile(tile)