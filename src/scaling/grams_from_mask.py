import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict
from functools import reduce

DENSITY_G_CM2 = {"rice": 1.0, "chicken duck": 1.5, "shellfish": 4, "banana": 0.2}

def grams_from_mask(mask: np.ndarray,
                    label: str,
                    pixels_per_cm: float,
                    density_map: Dict[str, float] = DENSITY_G_CM2) -> float:
    """
    Args
        mask (np.ndarray) Binary mask (H×W).
        label (str) Class name for the mask.
        pixels_per_cm (float) Scale factor.
        density_map (dict) Mapping {label: grams_per_cm2}.

    Returns
        grams (float) Estimated weight of the object represented by mask.
    """
    if label not in density_map:
        raise KeyError(f"No density value for label '{label}'.")
    
    if pixels_per_cm is None:
        raise KeyError(f"No scale found.")

    area_pixels = mask.sum() # total pixels
    area_cm2 = area_pixels / (pixels_per_cm ** 2)  # pixels / cm²
    grams = area_cm2 * density_map[label]
    return grams

def grams_from_items(items: List[Tuple[np.ndarray, str, list]],
                     pixels_per_cm: float,
                     density_map: Dict[str, float] = DENSITY_G_CM2
                     ) -> Dict[str, float]:
    """
    Args
        items (list) The list returned by `detect_and_segment` helper (mask, label, box).
        pixels_per_cm (float) Scale factor.
        density_map (dict) Mapping {label: grams_per_cm2}.

    Returns
        weights (dict) {label: total_grams}.
    """
    per_label = defaultdict(list)
    for mask, label, _ in items:
        per_label[label].append(mask)

    # union masks per unique label and compute grams once
    weights = {}
    for label, mask_list in per_label.items():
        union_mask = reduce(np.logical_or, mask_list).astype(np.float32)
        grams = grams_from_mask(union_mask, label, pixels_per_cm, density_map)
        weights[label] = grams

    return weights

if __name__ == "__main__":
    import cv2
    from ultralytics import YOLO
    from src.detector.yolov8_detector import detect_and_segment
    from src.scaling.credit_card_scaler import px_per_cm_from_card

    model = YOLO("./models/foodseg/best.pt")
    image_path = "./data/banana.jpg" 
    image = cv2.imread(image_path)

    items = detect_and_segment(image, model)
    print(items)
    scale = px_per_cm_from_card(image)
    print(scale)
    weights = grams_from_items(items, scale, DENSITY_G_CM2)
    print(weights)

    from src.detector.utils import show_masks_on_image
    show_masks_on_image("output.png", image, items)

    

