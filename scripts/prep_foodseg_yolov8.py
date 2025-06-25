"""
This script prepares the FoodSeg103 dataset for later use in YOLO-v8 segmentation fine-tuning.
"""
import io
import json
import os
from typing import Dict, List
from pathlib import Path

import cv2
from datasets import load_dataset
import numpy as np
from PIL import Image
from skimage import measure
import tqdm
from dotenv import load_dotenv

from config.data_categories import DATA_CATEGORIES

load_dotenv()

# Constants
ROOT = Path(os.getenv("FOODSEG_PATH", "data/FoodSeg103"))
PNG_COMPRESSION = [
    cv2.IMWRITE_PNG_COMPRESSION,
    int(os.getenv("PNG_COMPRESSION", 3)),
]
SHARD_SIZE = int(os.getenv("SHARD_SIZE", 500))
DATASET_YAML_NAME = os.getenv("DATASET_YAML_PATH", "config/foodseg103.yaml")
MAX_POLY_POINTS = int(os.getenv("MAX_POLY_POINTS", 1000))

# Helpers
def save_mask(label_field, out_path: Path) -> None:
    """
    Save a segmentation mask to disk in PNG format.
    Parameters:
        label_field: The mask to save. Can be a PIL.Image.Image object or a dict
                     containing a "png" key with PNG bytes.
        out_path (Path): The output file path where the mask will be saved.
    Raises:
        ValueError: If label_field is not a supported format.
    """
    if isinstance(label_field, Image.Image):
        label_field.save(out_path, "PNG")
    elif isinstance(label_field, dict) and "png" in label_field:
        Image.open(io.BytesIO(label_field["png"])).save(out_path, "PNG")
    else:
        raise ValueError("Unknown label format")

def explode_index_mask(mask_file: Path, split: str) -> None:
    """
    Converts an index mask PNG file into YOLO polygon label format and saves it as a .txt file.
    Parameters:
        mask_file (Path): Path to the PNG mask file.
        split (str): Dataset split name (e.g., "train", "val", "test").
    Output:
        Writes a .txt label file with the same stem as the mask in the appropriate labels directory.
    """
    idx = np.array(Image.open(mask_file))
    h, w = idx.shape
    inst_ids = np.unique(idx)[1:] # skip background (0)
    lines = []

    for inst in inst_ids:
        cls_id = int(inst) - 1 # yolo needs 0-based class
        # binary mask for this instance
        binary = idx == inst
        # find outer contour
        contours = measure.find_contours(binary, level=0.5)
        if not contours:
            print(f"No contours found for instance {inst} in {mask_file}")
            continue
        contour = max(contours, key=len)
        # down-sample very long polygons
        if len(contour) > MAX_POLY_POINTS:
            step = len(contour) // MAX_POLY_POINTS
            contour = contour[::step]

        # (y,x) ➜ (x,y) and normalise 0-1
        poly_norm = []
        for y, x in contour:
            poly_norm += [x / w, y / h]
        lines.append(" ".join([str(cls_id)] +
                              [f"{p:.6f}" for p in poly_norm]) + "\n")

    # write label file
    label_path = ROOT / f"labels/{split}/{mask_file.stem}.txt"
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text("".join(lines))

def get_class_names(
    ds_dict,
    categories: Dict[int, str] | None = None,
    expected_classes: int = 103,
) -> List[str]:
    """
    Retrieve the list of class names for the dataset.
    Parameters:
        ds_dict: The loaded HuggingFace dataset dictionary.
        categories (Dict[int, str] | None): Optional mapping from class index to class name.
        expected_classes (int): The expected number of classes (default: 103).
    Returns:
        List[str]: A list of class names in order.
    """
    try:
        meta_path = Path(ds_dict.cache_files[0]["filename"]).with_suffix(".json")
        with meta_path.open() as f:
            names = json.load(f)["features"]["label"]["names"]
        if len(names) == expected_classes:
            return names
        print(f"Metadata has {len(names)} classes, expected {expected_classes}")
    except Exception as e:
        print(f"Could not read names from metadata: {e}")

    if categories:
        try:
            return [categories[i + 1] for i in range(expected_classes)]
        except Exception as e:
            print(f"DATA_CATEGORIES fallback failed: {e}")

    print("Falling back to generic food_1 … food_103 names")
    return [f"food_{i}" for i in range(1, expected_classes + 1)]


if __name__ == "__main__":
    ds_dict = load_dataset("EduardoPacheco/FoodSeg103")
    print("Splits found:", list(ds_dict.keys()))

    # create folder skeleton
    for split in ds_dict.keys():
        (ROOT / f"images/{split}").mkdir(parents=True, exist_ok=True)
        (ROOT / f"masks/{split}").mkdir(parents=True, exist_ok=True)

    # export images + index masks (sharded)
    for split, ds in ds_dict.items():
        print(f"Saving {split} images & masks in shards …")
        num_shards = (len(ds) + SHARD_SIZE - 1) // SHARD_SIZE

        for shard_id in range(num_shards):
            sub = ds.shard(num_shards, shard_id)
            for row in tqdm.tqdm(sub, total=len(sub)):
                stem = f"{row['id']:05d}"
                img_path = ROOT / f"images/{split}/{stem}.jpg"
                mask_path = ROOT / f"masks/{split}/{stem}.png"
                if not img_path.exists():
                    row["image"].save(img_path, "JPEG", quality=90)
                if not mask_path.exists():
                    save_mask(row["label"], mask_path)

    # convert each index mask ➜ polygons + label TXT
    for split in ds_dict.keys():
        print(f"Exploding masks → polygons for {split} …")
        for mask_file in tqdm.tqdm((ROOT / f"masks/{split}").glob("*.png")):
            explode_index_mask(mask_file, split)

    # build YAML
    names = get_class_names(ds_dict, DATA_CATEGORIES, expected_classes=103)
    yaml_lines = [
        f"path: {ROOT}",
        "train: images/train",
        "val: images/validation" if "validation" in ds_dict else "val: images/val",
    ]
    if "test" in ds_dict:
        yaml_lines.append("test: images/test")

    yaml_lines += ["", "nc: 103", "names:"]
    yaml_lines += [f"  - {n}" for n in names]

    Path(DATASET_YAML_NAME).write_text("\n".join(yaml_lines))
    print(f"Preparation finished. YAML → {DATASET_YAML_NAME}")