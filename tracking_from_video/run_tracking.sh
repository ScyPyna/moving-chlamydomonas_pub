#!/usr/bin/env bash
source /home/user/Scrivania/Videos/.trajs/bin/activate
set -e

echo "======================================="
echo " Microalgae Tracking Pipeline"
echo "======================================="

#indicare il video che vuoi analizzare
VIDEO_ID=432

# directory dello script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# cartella principale del repository
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

# dove stanno i video 
VIDEO_DIR="/home/user/Scrivania/Videos/Novembre2024"

# dove salvare i risultati
OUT_DIR="$ROOT_DIR"

CONFIG="$SCRIPT_DIR/configs/config_tracking.json"

echo ""
echo "Video folder : $VIDEO_DIR"
echo "Output folder: $OUT_DIR"
echo ""

python3 "$SCRIPT_DIR/src/tracking_algae.py" \
    --input "$VIDEO_DIR" \
    --glob "tAlgae${VIDEO_ID}.avi" \
    --config "$CONFIG" \
    --out "$OUT_DIR"

echo ""
echo "Tracking completed."
