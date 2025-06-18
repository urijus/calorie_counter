"""
This script defines the routes for the Nutritional Facts API.
"""
from pathlib import Path

import base64
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel

from detector.yolov8_detector import detect_and_segment
from scaling.credit_card_scaler import px_per_cm_from_card
from scaling.grams_from_mask import grams_from_mask
from scaling.utils import mask_to_png_b64
from nutrition.usda_client import macro_totals


app = FastAPI(title="Calorie Counter API")

class MaskOut(BaseModel):
    label: str
    grams: float
    box:   list[float]  # [x1,y1,x2,y2]
    mask:  str # base64 PNG

class PredictOut(BaseModel):
    items: list[MaskOut]
    totals: dict # kcal, protein, fat, carb

@app.post("/predict", response_model=PredictOut)
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ("image/jpeg", "image/png"):
        raise HTTPException(400, "Upload a JPEG or PNG.")

    img_bytes = await file.read()
    bgr = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

    if bgr is None:
        raise HTTPException(415, "Could not decode image")

    scale = px_per_cm_from_card(bgr) or 1.0
    items = detect_and_segment(bgr)

    outputs, macros_in = [], []
    for mask, label, box in items:
        grams = grams_from_mask(mask, label, scale)
        outputs.append(MaskOut(label=label, grams=grams, box=box.tolist(),
                               mask=mask_to_png_b64(mask)))
        macros_in.append((label, grams))

    totals = macro_totals(macros_in)
    return PredictOut(items=outputs, totals=totals)
