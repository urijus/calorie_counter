"""
This script defines the routes for the Nutritional Facts API.
"""
from typing import List, Dict
import cv2, numpy as np
from fastapi import APIRouter, File, UploadFile, HTTPException, Depends
from pydantic import BaseModel

from src.api.dependencies import get_usda_client, get_detector          
from src.nutrition.usda_client import USDAClient
from src.detector.yolov8_detector import detect_and_segment
from src.scaling.credit_card_scaler import px_per_cm_from_card
from src.scaling.grams_from_mask import grams_from_items
from src.scaling.utils import mask_to_png_b64
from src.scaling.densities import DENSITY_TABLE             

router = APIRouter()

class MaskOut(BaseModel):
    label: str
    grams: float
    boxes: List[List[float]]   
    masks: List[str]                


class PredictOut(BaseModel):
    items: List[MaskOut]
    totals: Dict[str, float]   # kcal, protein, fat, carb


@router.post("/predict", response_model=PredictOut)
async def predict(
    file: UploadFile = File(...),
    usda: USDAClient = Depends(get_usda_client), 
    model = Depends(get_detector)  
):
    if file.content_type not in ("image/jpeg", "image/png"):
        raise HTTPException(400, "Upload a JPEG or PNG")

    img_bytes = await file.read()
    bgr = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(415, "Could not decode image")

    scale  = px_per_cm_from_card(bgr) or 1.0
    items  = detect_and_segment(bgr, model) # can have more than one same label
    grams_per_label = grams_from_items(items, scale, DENSITY_TABLE) # unique keys
    by_label: dict[str, dict] = {} # label -> dict(mask, box, grams)

    for mask, label, box in items:
        if label not in by_label:
            by_label[label] = {
                "grams": grams_per_label[label],  
                "masks": [mask],                   
                "boxes": [box]
            }
        else:
            by_label[label]["masks"].append(mask)
            by_label[label]["boxes"].append(box)

    outputs: list[MaskOut] = []
    totals  = {"kcal": 0.0, "protein": 0.0, "fat": 0.0, "carb": 0.0}

    for label, info in by_label.items():
        g = info["grams"]                     
        facts100 = usda.get_nutritional_facts(label) # per 100g
        scaled   = {k: v * g / 100.0 for k, v in facts100.items()}

        for k in totals:
            totals[k] += scaled.get(k, 0.0)

        outputs.append(
            MaskOut(
                label=label,
                grams=round(g, 2),
                boxes=[[round(v, 1) for v in box] for box in info["boxes"]],
                masks=[mask_to_png_b64(mask) for mask in info["masks"]],
                nutrition=scaled              
            )
        )

    totals = {nutrient: round(quantity, 2) for nutrient, quantity in totals.items()}
    return PredictOut(items=outputs, totals=totals)