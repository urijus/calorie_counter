from pathlib import Path
import cv2
from ultralytics import YOLO

from src.detector.yolov8_detector import detect_and_segment
from src.scaling.credit_card_scaler import px_per_cm_from_card
from src.scaling.grams_from_mask import grams_from_items

from src.scaling.densities import DENSITY_TABLE

if __name__=="__main__":
    dbg = Path("dbg"); dbg.mkdir(exist_ok=True)

    model = YOLO("./models/foodseg/best.pt")
    img   = cv2.imread("./data/banana.jpg")

    proc = cv2.resize(img, (640, 480), interpolation=cv2.INTER_AREA)

    # ---- detect & visualise ---------------------------------------
    items = detect_and_segment(proc, model, debug_dir=dbg)
    scale = px_per_cm_from_card(proc, debug_dir=dbg)
    print("px/cm :", scale)

    # ---- grams ----------------------------------------------------
    w = grams_from_items(items, scale, DENSITY_TABLE)
    print(w)