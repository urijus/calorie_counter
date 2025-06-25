"""
This script provides the necessary methods to segment an image into food items.
"""
import os
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from dotenv import load_dotenv, find_dotenv
from ultralytics import YOLO

from src.detector.utils import show_masks_on_image


load_dotenv()

SEG_CONFIDENCE = float(os.getenv('SEG_CONFIDENCE_THRESHOLD'))
CLS_CONFIDENCE = float(os.getenv('CLAS_CONFIDENCE_THRESHOLD'))
DEBUG_DIR = os.getenv('DEBUG_DIR')

def _solidify_mask(mask: np.ndarray,
                  kernel: int = 8,
                  close_iters: int = 2,
                  dilate_iters: int = 1) -> np.ndarray:
    """
    Expand + close + hole-fill a binary mask.

    Args:
        mask (np.ndarray) Bool mask from YOLOv8.
        kernel (int) Size of the round structuring element (px).
        close_iters (int) How many MORPH_CLOSE passes (to fill cracks and small holes).
        dilate_iters (int) Optional dilation before closing (grows mask a little).

    Returns:
        solidified_mask (np.ndarray) Bool mask (same H×W) as a solid blob.
    """
    m = (mask > 0).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel))

    if dilate_iters:
        m = cv2.dilate(m, k, iterations=dilate_iters)

    if close_iters:
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=close_iters)

    # fill internal holes (so we have a dense blob in the end)
    flood = cv2.bitwise_not(m)
    cv2.floodFill(flood, None, (0, 0), 255)
    holes = cv2.bitwise_not(flood)
    m_filled = cv2.bitwise_or(m, holes)
    return m_filled.astype(bool)

def _run_classifier(cls_model, crop: np.ndarray) -> Tuple[str, float]:
    """Return (label, confidence) predicted by the classifier model."""
    pred = cls_model.predict(crop, imgsz=224, verbose=False)[0]
    idx  = int(pred.probs.top1) # class index
    conf = float(pred.probs.top1conf)
    return pred.names[idx], conf

def detect_and_segment(image, 
                       seg_model, 
                       clas_model,
                       debug_dir: Optional[Path] = Path(DEBUG_DIR),
                       seg_confidence: float=SEG_CONFIDENCE,
                       cls_confidence: float=CLS_CONFIDENCE):
    """
    Args:
        image (np.ndarray, PIL, str): RGB or BGR image array or image path.
        model (ultralytics.YOLO): Loaded YOLOv8 segmentation model.
        confidence (float): Min-confidence to keep a detection.

    Returns:
        masks  (list[np.ndarray]): List of binary masks, one per kept object.
        labels (list[str])       : List of class names for each mask.
        boxes  (list[list[float]]): List of [x1, y1, x2, y2] per mask.
    """
    if debug_dir:
        debug_dir.mkdir(parents=True, exist_ok=True)

    result = seg_model.predict(source=image, save=False, verbose=False)[0]
    output = []

    if result.masks is None:
        return output
    
    H, W = image.shape[:2]
    class_remap: dict[str, str] = {}

    for mask, _ , conf, cls_id in zip(result.masks.data, 
                                       result.boxes.xyxy, 
                                       result.boxes.conf, 
                                       result.boxes.cls.int()):
        # keep confident masks
        if conf < seg_confidence:
            continue

        # solidify mask
        raw_mask = mask.cpu().numpy() # [H, W] binary mask
        solid = _solidify_mask(raw_mask)

        mask_uint8 = solid.astype(np.uint8) * 255
        if mask_uint8.shape != image.shape[:2]:
            mask_uint8 = cv2.resize(mask_uint8, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        masked_img = cv2.bitwise_and(image, image, mask=mask_uint8)

        x, y, w, h = cv2.boundingRect(mask_uint8)
        crop = masked_img[y:y+h, x:x+w]

        seg_label = result.names[int(cls_id)] 

        if seg_label in class_remap:
            final_label = class_remap[seg_label]
        else:
            cls_label, cls_prob = _run_classifier(clas_model, crop)
            final_label = cls_label if cls_prob >= cls_confidence else seg_label
            class_remap[seg_label] = final_label

        print(seg_label, "changed to", final_label)
        output.append((solid, final_label, [x, y, x + w, y + h])) # x1, y1 x2, y2  

    if debug_dir:
        show_masks_on_image(str(debug_dir / f"mask_overlay_yolo.png"), image, output)

    return output


if __name__ == "__main__":
    seg_model = YOLO("./models/foodseg/best.pt")
    clas_model = YOLO("./models/food101/best.pt")
    image = cv2.imread("./data/chicken_test.png")

    items = detect_and_segment(image, seg_model, clas_model, )
    print(f"Detected {len(items)} items:")

    for mask, label, box in items:
        print(f"Label: {label}, Box: {box}")



### NOTE ###
"""
Why does YOLOv8 segmentation head sometimes fail in classifying objects?
1) Food items share colour and shape, we'd need many more examples to learn
the differences.

2) During training we should increase the resolution if VRam allows.

3) The segmentation head is dominated by shape, not fine class detail.

4) My proposed solution is to use the YOLOv8's classification head and fine tune
it with Food101 dataset.
"""