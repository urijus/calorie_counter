"""
This module provides functions to estimate the pixel density of a credit card in an image.
"""
from __future__ import annotations

import os
import ast
from typing import Optional, Tuple

import cv2
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

CARD_WIDTH_CM   = float(os.getenv("CARD_WIDTH_CM", 8.56))
CARD_HEIGHT_CM  = float(os.getenv("CARD_HEIGHT_CM", 5.398))
CARD_ASPECT     = CARD_WIDTH_CM / CARD_HEIGHT_CM # ≈ 1.586

# BRIGHT_L        = int(os.getenv('BRIGHT_L', 120))
# DARK_L        = int(os.getenv('BRIGHT_L', 80))
LOW_CHROMA_THR = 18        # empirical: ≈10 % of max (179)
DARK_L_MIN     = 50        # keep very dark neutrals
BRIGHT_L_MIN   = 120       # keep bright neutrals
CLOSE_K        = (7, 7)    # gentler close than 11×11×2

ASP_TOL         = float(os.getenv('ASP_TOL', 0.30))         
MIN_AREA_FRAC   = float(os.getenv('MIN_AREA_FRAC', 0.005))
MAX_AREA_FRAC   = float(os.getenv('MAX_AREA_FRAC', 0.30))        
FRAME_MARGIN    = int(os.getenv('FRAME_MARGIN', 3))  
POLY_EPS_FRAC   = float(os.getenv('POLY_EPS_FRAC', 0.02))

DEBUG_DIR       = Path(os.getenv("DEBUG_DIR", './debug'))


# def _lab_masks_dual(bgr: np.ndarray) -> list[np.ndarray]:
#     """
#     Return two candidate masks:
#         index 0 → low-chroma (white / grey cards)
#         index 1 → high-chroma (coloured cards)
#     Both are morph-closed and restricted to bright pixels (L > 120).
#     """
#     lab   = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
#     a, b_ = lab[..., 1].astype(np.float32), lab[..., 2].astype(np.float32)
#     chroma = cv2.magnitude(a, b_).astype(np.uint8)

#     # Otsu finds a threshold splitting low vs high chroma
#     _, thr = cv2.threshold(chroma, 0, 255,
#                            cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

#     L = lab[..., 0]
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))

#     # low-chroma mask
#     low = thr.copy()
#     low[L < BRIGHT_L] = 0
#     low = cv2.morphologyEx(low, cv2.MORPH_CLOSE, kernel, iterations=2)

#     low_dark = thr.copy()
#     low_dark[(L < DARK_L) | (L > BRIGHT_L)] = 0
#     low_dark = cv2.morphologyEx(low_dark, cv2.MORPH_CLOSE, kernel, iterations=2)

#     # high-chroma mask (invert)
#     high = cv2.bitwise_not(thr)
#     high[L < 120] = 0
#     high = cv2.morphologyEx(high, cv2.MORPH_CLOSE, kernel, iterations=2)

#     return [low, low_dark, high]

def _lab_masks_dual(bgr: np.ndarray) -> list[np.ndarray]:
    """
    Returns two masks:
        0 → low-chroma neutrals   (dark OR bright, but texture-flat)
        1 → high-chroma colours   (any L)
    A fixed chroma threshold (=distance in a-b plane) is used instead of Otsu,
    so vivid outliers (an apple, a tomato…) no longer dictate the split.
    """
    lab     = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    a       = lab[..., 1].astype(np.int16) - 128   # centre a,b at (0,0)
    b       = lab[..., 2].astype(np.int16) - 128
    chroma  = cv2.magnitude(a.astype(np.float32), b.astype(np.float32)).astype(np.uint8)
    L       = lab[..., 0]

    # --- low-chroma mask ----------------------------------------------------
    low = (chroma < LOW_CHROMA_THR).astype(np.uint8) * 255

    # accept BOTH bright and dark neutrals
    neutral = ((L > DARK_L_MIN) & (L < BRIGHT_L_MIN)) | (L >= BRIGHT_L_MIN)
    low[~neutral] = 0

    # eliminate fabric wrinkles (texture) with a mild close
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, CLOSE_K)
    low    = cv2.morphologyEx(low, cv2.MORPH_CLOSE, kernel, iterations=1)

    # --- high-chroma mask ---------------------------------------------------
    high = cv2.bitwise_not(low)
    high[chroma < LOW_CHROMA_THR] = 0           # keep only genuinely colourful
    high[L < DARK_L_MIN]          = 0           # ignore dark noise
    high  = cv2.morphologyEx(high, cv2.MORPH_CLOSE, kernel, iterations=1)

    return [low, high]

def _contours(img_bin: np.ndarray):
    return cv2.findContours(img_bin, cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)[0]

def _touches_border(x: int, y: int, w: int, h: int,
                    img_w: int, img_h: int,
                    margin: int = FRAME_MARGIN) -> bool:
    """True if bounding-box lies on (or beyond) image frame."""
    return (
        x <= margin
        or y <= margin
        or x + w >= img_w - margin
        or y + h >= img_h - margin
    )

def _card_candidate(contours: list[np.ndarray],img_w: int,img_h: int,
                    aspect: float = CARD_ASPECT,asp_tol: float = ASP_TOL,
                    ) -> Optional[np.ndarray]:
    """Return the single contour that best matches credit-card heuristics."""
    img_area = img_w * img_h
    best_cnt: Optional[np.ndarray] = None
    best_area = 0.0

    for cnt in contours:
        # reject outer frame contours
        x, y, w, h = cv2.boundingRect(cnt)
        if _touches_border(x, y, w, h, img_w, img_h):
            continue

        # rotated rectangle metrics
        _, (rw, rh), _ = cv2.minAreaRect(cnt)
        if rw == 0 or rh == 0:
            continue

        rect_area = rw * rh
        if not (MIN_AREA_FRAC * img_area < rect_area < MAX_AREA_FRAC * img_area):
            continue
        
        asp = max(rw, rh) / min(rw, rh)
        if not (1 - asp_tol) * aspect < asp < (1 + asp_tol) * aspect:
            continue

        # peri   = cv2.arcLength(cnt, True)
        # approx = cv2.approxPolyDP(cnt, POLY_EPS_FRAC * peri, True)
        # if len(approx) != 4:
        #     continue

        # keep the largest surviving rectangle
        if rect_area > best_area:        
            best_area = rect_area
            best_cnt  = cnt

    return best_cnt

def _edge_preprocess(bgr: np.ndarray) -> np.ndarray:
    """Return thickened edge map later for contour search."""
    gray   = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray   = cv2.createCLAHE(4.0, (8, 8)).apply(gray)
    gray   = cv2.bilateralFilter(gray, 7, 75, 75)

    edges  = cv2.Canny(gray, 20, 120)
    edges  = cv2.dilate(edges, None, 1)

    # thicken/close gaps so card interior can be measured
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    return cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

def px_per_cm_from_card(image: np.ndarray,
                        debug_dir: Optional[Path] = None,
                        target_hw: Tuple[int, int] = (480, 640)) -> Optional[float]:
    """
    Detect card and return pixels per centimetre (long edge).
    Returns None if nothing plausible is found.
    """
    # YOLO usually resizes, need to take this into account
    try:
        raw = ast.literal_eval(os.getenv('HEIGHT_WIDTH_YOLO_TUPLE', ''))
        if isinstance(raw, (tuple, list)) and len(raw) == 2:
            targ_h, targ_w = map(int, raw)
        else:
            targ_h, targ_w = target_hw
    except (ValueError, SyntaxError):
        targ_h, targ_w = target_hw

    img_h, img_w   = image.shape[:2]

    if (img_h, img_w) != (targ_h, targ_w):
        image_rs = cv2.resize(image, (targ_w, targ_h),
                              interpolation=cv2.INTER_AREA)
    else:
        image_rs = image 

    H, W = image_rs.shape[:2]

    if debug_dir:
        debug_dir.mkdir(parents=True, exist_ok=True)

    for i, mask in enumerate(_lab_masks_dual(image_rs)):
        if debug_dir:
            cv2.imwrite(str(debug_dir / f"card_mask_{i}.png"), mask)
        card = _card_candidate(_contours(mask), W, H)
        if card is not None:
            if debug_dir:
                dbg = image_rs.copy()
                cv2.drawContours(dbg, [card], -1, (0, 255, 0), 2)
                cv2.imwrite(str(debug_dir / f"card_contour_mask_{i}.png"), dbg)
            _, (rw, rh), _ = cv2.minAreaRect(card)
            return max(rw, rh) / CARD_WIDTH_CM
        
    edges = _edge_preprocess(image_rs)
    if debug_dir:
        cv2.imwrite(str(debug_dir / "card_edge_mask.png"), edges)
    card  = _card_candidate(_contours(edges), W, H)
    if card is not None:
        if debug_dir:
            dbg = image_rs.copy()
            cv2.drawContours(dbg, [card], -1, (255, 0, 0), 2)
            cv2.imwrite(str(debug_dir / "card_contour_edge.png"), dbg)
        _, (rw, rh), _ = cv2.minAreaRect(card)
        return max(rw, rh) / CARD_WIDTH_CM

    print("No credit-card contour found.")
    return None

if __name__ == "__main__":
    img   = cv2.imread("./data/test4.jpg")
    ppcm  = px_per_cm_from_card(img, DEBUG_DIR)
    print("pixels / cm:", ppcm)
