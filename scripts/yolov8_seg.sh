#!/usr/bin/env bash
set -euo pipefail

# Read optional configuration file
CONFIG_FILE="${CONFIG_FILE:-.train.env}"
[[ -f "$CONFIG_FILE" ]] && source "$CONFIG_FILE"

MODEL="${MODEL_SEG:-yolov8m-seg.pt}" 
EPOCHS="${EPOCHS_SEG:-150}"
PATIENCE="${PATIENCE:-10}"
IMG_SIZE="${IMG_SIZE_SEG:-640}"
BATCH="${BATCH_SEG:--1}" #-1 is auto batch
LR0="${LR0:-0.001}"
LRF="${LRF:-0.1}" # final_lr = lr0 * lrf
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0005}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-3}"
OPTIMIZER="${OPTIMIZER:-AdamW}"
AMP="${AMP:-True}"
DEVICE="${DEVICE:-0}"
WORKERS="${WORKERS_SEG:-4}"
DATASET_YAML_PATH="${DATASET_YAML_PATH:-config/foodseg103.yaml}"
PROJECT_ROOT="${PROJECT_ROOT:-runs/foodseg}"
RESUME="${RESUME:-False}" 

# CLI flags override everything else
usage() {
  echo "Usage: $0 [-m model] [-e epochs] [-i imgsz] [-b batch] [-y yaml] \
[-l lr0] [-w workers] [-o optimizer] [-r resume]"
  exit 1
}

while getopts ":m:e:i:b:y:l:w:o:r" flag; do
  case "${flag}" in
    m) MODEL="$OPTARG" ;;
    e) EPOCHS="$OPTARG" ;;
    i) IMG_SIZE="$OPTARG" ;;
    b) BATCH="$OPTARG" ;;
    y) DATASET_YAML_PATH="$OPTARG" ;;
    l) LR0="$OPTARG" ;;
    w) WORKERS="$OPTARG" ;;
    o) OPTIMIZER="$OPTARG" ;;
    r) RESUME="$OPTARG" ;;
    *) usage ;;
  esac
done

# Derive folder names from MODEL
# yolov8m-seg.pt  ->  yolov8m
MODEL_TAG="$(basename "$MODEL" | cut -d'-' -f1)"
NAME="${NAME:-${MODEL_TAG}_foodseg}"                   
MODEL_OUT_DIR="models/${MODEL_TAG}"
mkdir -p "$MODEL_OUT_DIR"

# Show final hyper-params
cat <<EOF
🔧  Training configuration
    MODEL         : $MODEL
    EPOCHS        : $EPOCHS  (patience $PATIENCE)
    IMG_SIZE      : $IMG_SIZE
    BATCH         : $BATCH   (auto if -1)
    OPTIMIZER     : $OPTIMIZER
      • LR0       : $LR0
      • LRF       : $LRF
      • W-decay   : $WEIGHT_DECAY
    WARMUP_EPOCHS : $WARMUP_EPOCHS
    AMP           : $AMP
    DEVICE        : $DEVICE
    WORKERS       : $WORKERS
    DATASET       : $DATASET_YAML_PATH
    OUTPUT DIR    : $MODEL_OUT_DIR
    RESUME        : $RESUME
EOF

# Launch training
yolo segment train \
    model="$MODEL" \
    data="$DATASET_YAML_PATH" \
    imgsz="$IMG_SIZE" \
    epochs="$EPOCHS" \
    patience="$PATIENCE" \
    batch="$BATCH" \
    workers="$WORKERS" \
    optimizer="$OPTIMIZER" \
    lr0="$LR0" \
    lrf="$LRF" \
    weight_decay="$WEIGHT_DECAY" \
    warmup_epochs="$WARMUP_EPOCHS" \
    amp="$AMP" \
    device="$DEVICE" \
    project="$PROJECT_ROOT" \
    name="$NAME" \
    resume="$RESUME"           

# Copy best weight and TorchScript export
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