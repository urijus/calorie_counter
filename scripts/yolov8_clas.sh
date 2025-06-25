#!/usr/bin/env bash
set -euo pipefail

# Auto-download Food-101 if not present  
DATASET_DIR="${FOOD101_PATH:-data/Food101}"

if [[ ! -d "$DATASET_DIR/train" ]]; then
  echo "Downloading Food-101"
  mkdir -p "$(dirname "$DATASET_DIR")"
  wget -qO- https://data.vision.ee.ethz.ch/cvl/food-101.tar.gz | \
        tar -xz -C "$(dirname "$DATASET_DIR")"
  mv "$(dirname "$DATASET_DIR")/food-101" "$DATASET_DIR"
  rm -r "$(dirname "$DATASET_DIR")/food-101"
fi

echo "Creating train/ and val/ dirs"
mkdir -p "$DATASET_DIR/train" "$DATASET_DIR/val"

# The meta/train.txt and meta/test.txt list image paths without extension
while read -r rel; do
  cls=$(dirname "$rel")                        
  mkdir -p "$DATASET_DIR/train/$cls"
  mv "$DATASET_DIR/images/$rel.jpg" "$DATASET_DIR/train/$rel.jpg"
done < "$DATASET_DIR/meta/train.txt"

while read -r rel; do
  cls=$(dirname "$rel")
  mkdir -p "$DATASET_DIR/val/$cls"
  mv "$DATASET_DIR/images/$rel.jpg" "$DATASET_DIR/val/$rel.jpg"
done < "$DATASET_DIR/meta/test.txt"

# Point Ultralytics to the folder that now HAS train/ val/
DATA_YAML="$DATASET_DIR"

# Read optional configuration file
CONFIG_FILE="${CONFIG_FILE:-.train.env}"
[[ -f "$CONFIG_FILE" ]] && source "$CONFIG_FILE"


MODEL="${MODEL_CLAS:-yolov8m-cls.pt}"       
EPOCHS="${EPOCHS_CLAS:-25}"
PATIENCE="${PATIENCE:-5}"
IMG_SIZE="${IMG_SIZE_CLAS:-224}" # Square crops required for this model               
BATCH="${BATCH_CLAS:--1}"
LR0="${LR0:-0.001}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0005}"
OPTIMIZER="${OPTIMIZER:-AdamW}"
AMP="${AMP:-True}"
DEVICE="${DEVICE:-0}"
WORKERS="${WORKERS_CLAS:-4}"
PROJECT_ROOT="${PROJECT_ROOT:-runs/food101}"
RESUME="${RESUME:-False}"

# CLI flags override everything else
usage() {
  echo "Usage: $0 [-m model] [-e epochs] [-i imgsz] [-b batch] [-l lr0] [-w workers] [-o optimizer] [-r resume]"
  exit 1
}

while getopts ":m:e:i:b:l:w:o:r" flag; do
  case "${flag}" in
    m) MODEL="$OPTARG" ;;
    e) EPOCHS="$OPTARG" ;;
    i) IMG_SIZE="$OPTARG" ;;
    b) BATCH="$OPTARG" ;;
    l) LR0="$OPTARG" ;;
    w) WORKERS="$OPTARG" ;;
    o) OPTIMIZER="$OPTARG" ;;
    r) RESUME="$OPTARG" ;;
    *) usage ;;
  esac
done

# Derive folder names from MODEL
MODEL_TAG="$(basename "$MODEL" | cut -d'-' -f1)"
NAME="${NAME:-${MODEL_TAG}_food101}"        
MODEL_OUT_DIR="models/${MODEL_TAG}"
mkdir -p "$MODEL_OUT_DIR"

cat <<EOF
🔧  Classification training
    MODEL     : $MODEL
    EPOCHS    : $EPOCHS   (patience $PATIENCE)
    IMG_SIZE  : $IMG_SIZE
    BATCH     : $BATCH
    OPTIMIZER : $OPTIMIZER   lr0=$LR0
      • LR0       : $LR0
      • W-decay   : $WEIGHT_DECAY
    DATA      : $DATA_YAML
    OUTPUT    : $MODEL_OUT_DIR
    RESUME    : $RESUME
EOF

# # Launch training
# yolo classify train \
#      model="$MODEL" \
#      data="$DATA_YAML" \
#      imgsz="$IMG_SIZE" \
#      epochs="$EPOCHS" \
#      patience="$PATIENCE" \
#      batch="$BATCH" \
#      workers="$WORKERS" \
#      optimizer="$OPTIMIZER" \
#      lr0="$LR0" \
#      weight_decay="$WEIGHT_DECAY" \
#      amp="$AMP" \
#      device="$DEVICE" \
#      project="$PROJECT_ROOT" \
#      name="$NAME" \
#      resume="$RESUME"

# # Copy best weight and TorchScript export
# BEST_PT="$PROJECT_ROOT/$NAME/weights/best.pt"
# if [[ -f "$BEST_PT" ]]; then
#   cp "$BEST_PT" "$MODEL_OUT_DIR/"
# else
#   echo "No best.pt found at $BEST_PT – training may have failed." >&2
#   exit 1
# fi

# # Export to TorchScript format
# EXPORT_DIR=$(mktemp -d)
# yolo export model="$BEST_PT" format=torchscript project="$EXPORT_DIR" name=temp
# TS_FILE="$EXPORT_DIR/temp/best.torchscript"

# if [[ -f "$TS_FILE" ]]; then
#   mv "$TS_FILE" "$MODEL_OUT_DIR/"
#   rm -rf "$EXPORT_DIR"
# fi

# echo "Weights saved to $MODEL_OUT_DIR/"
