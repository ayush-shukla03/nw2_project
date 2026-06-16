#!/bin/bash

SENTINEL_DIR="/mnt/nw2data/nw2_project/data/raw/sentinel2"
TILES="T46RBP T46RBQ T46RBR T46RCP T46RCQ T46RCR T46RDP T46RDQ"

for TILE in $TILES; do
    echo "============================================"
    echo "Processing $TILE..."
    echo "============================================"

    # Find the L1C .SAFE folder
    L1C=$(ls -d $SENTINEL_DIR/$TILE/*.SAFE 2>/dev/null | grep MSIL1C)

    if [ -z "$L1C" ]; then
        echo "No L1C .SAFE found for $TILE, skipping..."
        continue
    fi

    # Run Sen2Cor
    L2A_Process "$L1C" --resolution=10

    # Delete 20m and 60m bands
    find $SENTINEL_DIR/$TILE/ -type d -name "R20m" -exec rm -rf {} + 2>/dev/null
    find $SENTINEL_DIR/$TILE/ -type d -name "R60m" -exec rm -rf {} + 2>/dev/null

    # Delete original L1C
    rm -rf "$L1C"

    echo "✓ Done with $TILE"
    df -h /mnt/nw2data
    echo ""
done

echo "All tiles processed."
