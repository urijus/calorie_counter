#!/usr/bin/env bash
set -euo pipefail

# Read optional configuration file
CONFIG_FILE="${CONFIG_FILE:-.train.env}"
[[ -f "$CONFIG_FILE" ]] && source "$CONFIG_FILE"

# Defaults (can be overridden by env or CLI)
MODEL="${MODEL:-yolov8m-seg.pt}" # s, m, l
EPOCHS="${EPOCHS:-30}"
IMG_SIZE="${IMG_SIZE:-640}"
BATCH="${BATCH:-2}"
DATASET_YAML_PATH="${DATASET_YAML_PATH:-config/foodseg103.yaml}"
LR0="${LR0:-0.001}"
WORKERS="${WORKERS:-4}"
OPTIMIZER="${OPTIMIZER:-SGD}"
PROJECT_ROOT="${PROJECT_ROOT:-runs/foodseg}"

# CLI flags override everything else
#     Usage examples:
#         ./train.sh -m yolov8s-seg.pt -e 50
usage() {
  echo "Usage: $0 [-m model] [-e epochs] [-i imgsz] 
  [-b batch] [-y yaml] [-l lr] [-w workers] [-o optimizer]"
  exit 1
}

while getopts ":m:e:i:b:y:l:w:o:" flag; do
  case "${flag}" in
    m) MODEL="$OPTARG" ;;
    e) EPOCHS="$OPTARG" ;;
    i) IMG_SIZE="$OPTARG" ;;
    b) BATCH="$OPTARG" ;;
    y) DATASET_YAML_PATH="$OPTARG" ;;
    l) LR0="$OPTARG" ;;
    w) WORKERS="$OPTARG" ;;
    o) OPTIMIZER="$OPTARG" ;;
    *) usage ;;
  esac
done

# Derive folder names from MODEL
# yolov8m-seg.pt  ->  yolov8m
MODEL_TAG="$(basename "$MODEL" | cut -d'-' -f1)"
NAME="${NAME:-${MODEL_TAG}_food}"                   
MODEL_OUT_DIR="models/${MODEL_TAG}"

mkdir -p "$MODEL_OUT_DIR"

# Show final hyper-params
cat <<EOF
🔧  Training configuration
    MODEL : $MODEL
    EPOCHS     : $EPOCHS
    IMG_SIZE   : $IMG_SIZE
    BATCH      : $BATCH
    LR0        : $LR0
    WORKERS    : $WORKERS
    DATASET    : $DATASET_YAML_PATH
    OPTIMIZER  : $OPTIMIZER
    OUTPUT DIR : $MODEL_OUT_DIR
EOF

# Launch training
yolo segment train \
     model="$MODEL" \
     data="$DATASET_YAML_PATH" \
     imgsz="$IMG_SIZE" \
     epochs="$EPOCHS" \
     batch="$BATCH" \
     workers="$WORKERS" \
     optimizer="$OPTIMIZER" \
     lr0="$LR0" \
     project="$PROJECT_ROOT" \
     name="$NAME"

# Copy best weight and TorchScript export (optional)
BEST_PT="$PROJECT_ROOT/$NAME/weights/best.pt"
if [[ -f "$BEST_PT" ]]; then
  cp "$BEST_PT" "$MODEL_OUT_DIR/"
else
  echo "No best.pt found at $BEST_PT. Training may have failed." >&2
  exit 1
fi

# Export to TorchScript format
EXPORT_DIR=$(mktemp -d)
yolo export model="$BEST_PT" format=torchscript project="$EXPORT_DIR" name=temp
TS_FILE="$EXPORT_DIR/temp/best.torchscript"

if [[ -f "$TS_FILE" ]]; then
  mv "$TS_FILE" "$MODEL_OUT_DIR/"
  rm -rf "$EXPORT_DIR"
else
  echo "TorchScript export not found. Export step failed." >&2
  exit 1
fi

echo "Weights saved to $MODEL_OUT_DIR/"