"""
This module provides functions to estimate the pixel density of a credit card in an image.
"""
import os
import cv2
import numpy as np
from dotenv import load_dotenv


load_dotenv()

# Size of an ID-1 card (ISO/IEC 7810)
CARD_WIDTH_CM  = float(os.getenv("CARD_WIDTH_CM", 8.56))
CARD_HEIGHT_CM = float(os.getenv("CARD_HEIGHT_CM", 5.398)) 
ASPECT = CARD_WIDTH_CM / CARD_HEIGHT_CM # ≈ 1.586

def _largest_card_like_shape(contours: list[np.ndarray], aspect:float=ASPECT, tolerance:float=0.3) -> tuple[float, list]:
    """
    Find the contour that:
      • approximates to 4 vertices
      • has aspect ratio ≈ CARD
      • has the largest area
    Returns (width_px, pts)  or (0, None)
    """
    best_w = 0
    best_pts = None
    for cnt in contours:
        # Polygonal approximation
        peri = cv2.arcLength(cnt, True) # perimeter
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True) # estimate the vertices 
        if len(approx) != 4:
            continue  # not a rectangle
        # bounding rect (rotated)
        rect = cv2.minAreaRect(cnt) # find rectangle (center, (w,h), angle)
        w, h = sorted(rect[1])        
        if w == 0 or h == 0:
            continue
        asp = h / w                
        if not (1-tolerance)*aspect < asp < (1+tolerance)*aspect:
            continue

        area = w * h
        if area > best_w * h: # pick largest by width
            best_w = h                 
            best_pts = approx.reshape(-1, 2)

    return best_w, best_pts

def px_per_cm_from_card(image:np.ndarray) -> float | None:
    """
    This function estimates the pixels per centimeter from a credit card in the image.
    """
    gray   = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    gray  = clahe.apply(gray)
    gray  = cv2.bilateralFilter(gray, d=7, sigmaColor=75, sigmaSpace=75)
    edges = cv2.Canny(gray, 20, 120)
    edges = cv2.dilate(edges, None, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    
    long_px, _ = _largest_card_like_shape(contours)
    print(f"Found card-like shape with long edge: {long_px} px")

    if long_px == 0:
        return None

    return CARD_WIDTH_CM / long_px


if __name__ == "__main__":
    img_path = "./data/creditcard.png"
    image = cv2.imread(img_path)

    if image is None:
        print("Could not read the image.")
    else:
        px_per_cm = px_per_cm_from_card(image)
        if px_per_cm is not None:
            print(f"Pixels per centimeter: {px_per_cm:.2f}")
        else:
            print("No credit card-like shape found in the image.")

