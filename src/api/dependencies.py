import os
from functools import lru_cache
from dotenv import load_dotenv
from ultralytics import YOLO

from src.nutrition.usda_client import USDAClient


load_dotenv()

SEGMENTATION_MODEL_PATH=os.getenv("SEGMENTATION_MODEL_PATH", "models/foodseg/best.pt")
CLASSIFICATION_MODEL_PATH=os.getenv("CLASSIFICATION_MODEL_PATH", "models/food101/best.pt")

@lru_cache
def get_usda_client() -> USDAClient:
    return USDAClient(os.getenv("USDA_API_KEY"))

@lru_cache
def get_segment_model() -> YOLO:
    "Return a single, cached YOLOv8 segmentation model instance."
    return YOLO(SEGMENTATION_MODEL_PATH)

@lru_cache
def get_classification_model() -> YOLO:
    "Return a single, cached YOLOv8 classification model instance."
    return YOLO(CLASSIFICATION_MODEL_PATH)
