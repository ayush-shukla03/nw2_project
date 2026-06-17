#!/bin/bash

SENTINEL_DIR="/mnt/nw2data/nw2_project/data/raw/sentinel2/monsoon"
TILES="monsoon_T46RFQ monsoon_T46RFP monsoon_T46REQ monsoon_T46RER monsoon_T46RFR monsoon_T46REP monsoon_T46RGR monsoon_T46RDP"

for TILE in $TILES; do
    echo "============================================"
    echo "Checking $TILE..."
    echo "============================================"

    L2A=$(ls -d $SENTINEL_DIR/$TILE/*.SAFE 2>/dev/null | grep MSIL2A)
    if [ -n "$L2A" ]; then
        echo "✓ $TILE already processed, skipping..."
        continue
    fi

    L1C=$(ls -d $SENTINEL_DIR/$TILE/*.SAFE 2>/dev/null | grep MSIL1C)
    if [ -z "$L1C" ]; then
        echo "No L1C found for $TILE, skipping..."
        continue
    fi

    echo "Processing $TILE..."
    L2A_Process "$L1C" --resolution=10

    find $SENTINEL_DIR/$TILE/ -type d -name "R20m" -exec rm -rf {} + 2>/dev/null
    find $SENTINEL_DIR/$TILE/ -type d -name "R60m" -exec rm -rf {} + 2>/dev/null
    rm -rf "$L1C"

    echo "✓ Done with $TILE"
    df -h /mnt/nw2data
    echo ""
done

echo "All monsoon tiles processed."