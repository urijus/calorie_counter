import random

import cv2
import numpy as np


def show_masks_on_image(save_path, image, items):
    """
    Overlays masks and boxes on the image and displays it.
    
    Args:
        image (np.ndarray): Original image (BGR format).
        items (list): List of (mask, label, box) tuples.
    """
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    mask_canvas = image.copy()
    H, W = image.shape[:2]

    for mask, label, box in items:
        if mask.shape != (H, W):
            mask = cv2.resize(mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)

        mask_bool = mask.astype(bool)

        # Generate a random color
        color = [random.randint(0, 255) for _ in range(3)]

        # Blend mask onto image
        for c in range(3):  # for each channel
            mask_canvas[:, :, c][mask_bool] = (
                0.5 * mask_canvas[:, :, c][mask_bool] + 0.5 * color[c]
            ).astype(np.uint8)

        # Draw bounding box
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(mask_canvas, (x1, y1), (x2, y2), color, 2)
        cv2.putText(mask_canvas, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imwrite(save_path, mask_canvas)
    print(f"Saved image with masks to {save_path}")