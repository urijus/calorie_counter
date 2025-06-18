import cv2, base64
import numpy as np
from io import BytesIO

def mask_to_png_b64(mask_bool: np.ndarray) -> str:
    """bool H×W → base64 PNG string (no header)."""
    # 0/255 uint8 for PNG
    png_buf = cv2.imencode(".png", mask_bool.astype(np.uint8) * 255,
                           [cv2.IMWRITE_PNG_COMPRESSION, 9])[1]
    return base64.b64encode(png_buf).decode("ascii")

def base64_png_to_mask(b64: str) -> np.ndarray:
    png = base64.b64decode(b64)
    img = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_GRAYSCALE)
    return img > 0