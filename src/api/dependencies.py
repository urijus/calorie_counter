import os
from functools import lru_cache
from dotenv import load_dotenv
from ultralytics import YOLO

from src.nutrition.usda_client import USDAClient


load_dotenv()

@lru_cache
def get_usda_client() -> USDAClient:
    return USDAClient(os.getenv("USDA_API_KEY"))



MODEL_PATH = "models/foodseg/best.pt" 

@lru_cache
def get_detector() -> YOLO:
    """Return a single, cached YOLOv8 model instance."""
    return YOLO(MODEL_PATH)
