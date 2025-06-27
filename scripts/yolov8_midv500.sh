#!/usr/bin/env bash
set -euo pipefail

# Auto-download MIDV500 if not present  
DATASET_DIR="${MIDV500_PATH:-data/MIDV500}"
RAW_DIR="$DATASET_DIR/_raw" 
BG_DIR="backgrounds"      

echo "Building YOLO dataset in $DATASET_DIR."
if [[ -d "$BG_DIR" && "$(ls -A "$BG_DIR")" ]]; then
  python ./scripts/prep_midv500_yolov8.py "$RAW_DIR" "$DATASET_DIR" --synth "$BG_DIR"
else
  python ./scripts/prep_midv500_yolov8.py "$RAW_DIR" "$DATASET_DIR"
fi

echo "Removing temporary raw data."
rm -rf "$RAW_DIR"

CONFIG_FILE="${CONFIG_FILE:-.train.env}"
[[ -f "$CONFIG_FILE" ]] && source "$CONFIG_FILE"


MODEL="${MODEL_CLAS:-yolov8m-cls.pt}"       
EPOCHS="${EPOCHS_CLAS:-50}"
IMG_SIZE="${IMG_SIZE_CLAS:-640}" # Square crops required for this model               
BATCH="${BATCH_CLAS:-16}"
LR0="${LR0:-0.001}"
MOSAIC="${MOSAIC:1.0}"
HSV_V="${HSV_V:0.4}"
HSV_H="${HSV_H:0.05}"
HSV_S="${HSV_S:0.6}"
DEGREES="${DEGREES:-5}"
SCALE="${SCALE:-0.25}" 
SHEAR="${SHEAR:-2}" 
TRANSLATE="${TRANSLATE:-0.12}" 


WEIGHT_DECAY="${WEIGHT_DECAY:-0.0005}"
OPTIMIZER="${OPTIMIZER:-AdamW}"
AMP="${AMP:-True}"
DEVICE="${DEVICE:-0}"
WORKERS="${WORKERS_CLAS:-4}"
PROJECT_ROOT="${PROJECT_ROOT:-runs/food101}"
RESUME="${RESUME:-False}"

echo "Training YOLOv8-seg model on MIDV-500."

ultralytics train segment \
  model="$MODEL" \
  data="$DATASET_DIR/data.yaml" \
  epochs="$EPOCHS" \
  imgsz="$IMG_SIZE" \
  batch="$BATCH" \
  lr0="$LR0" \
  optimizer="$OPTIMIZER" \
  mosaic="$MOSAIC" \
  hsv_h="$HSV_H" \
  hsv_s="$HSV_S" \
  hsv_v="$HSV_v" \
  degrees="$DEGREES" \
  translate="$TRANSLATE" \
  scale="$SCALE" \
  shear="$SHEAR" \
  project=card_seg \
  runs=train

echo "🎉  Done!  Best weights in runs/segment/train*/weights/best.pt"














# Download MIDV-500
echo "Downloading MIDV-500"
mkdir -p "$(dirname "$DATASET_DIR")"
gdown --id 1oZ0qZI9kqsKMNKn2Hks1faLDOp0uIm3K -O "$DATASET_DIR"/midv.tar.gz       
tar -xzf "$DATASET_DIR"/midv.tar.gz -C raw

# preprocess into one YOLOv8-seg dataset
python prep_midv500_yolov8.py raw dataset_yolo


# fine-tune YOLOv8-seg
ultralytics train segment model=yolov8m-seg.pt \
  data=dataset_yolo/data.yaml \
  epochs=80 imgsz=640 batch=16 \
  lr0=1e-3 optimizer=AdamW \
  mosaic=1.0 hsv_h=0.05 hsv_s=0.6 hsv_v=0.4 \
  degrees=5 translate=0.12 scale=0.25 shear=2 \
  project=card_seg runs=train

echo "Training finished. Best weights in runs/segment/train*/weights/best.pt"
