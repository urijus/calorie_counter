import cv2
from ultralytics import YOLO



CONFIDENCE_THRESHOLD = 0.3

def detect_and_segment(image, model, confidence=CONFIDENCE_THRESHOLD):
    """
    Args:
        image (np.ndarray, PIL, str): RGB or BGR image array or image path.
        model (ultralytics.YOLO): Loaded YOLOv8 segmentation model.
        confidence (float): Min-confidence to keep a detection.

    Returns:
        masks  (list[np.ndarray]): List of binary masks, one per kept object.
        labels (list[str])       : List of class names for each mask.
        boxes  (list[list[float]]): List of [x1, y1, x2, y2] per mask.
    """
    result = model.predict(source=image, save=False, verbose=False)[0]
    output = []

    if result.masks is not None:
        masks   = result.masks.data # Shape [N, H, W]
        boxes   = result.boxes.xyxy # Shape [N, 4]
        confs   = result.boxes.conf # [N]
        cls_ids = result.boxes.cls.int() # [N]

        for mask, box, conf, cls_id in zip(masks, boxes, confs, cls_ids):
            if conf < confidence:
                continue

            mask_np = mask.cpu().numpy() # [H, W] binary mask
            box_list = box.cpu().numpy().tolist() # [x1, y1, x2, y2]
            label_str = result.names[int(cls_id)]

            output.append((mask_np, label_str, box_list))

    return output


if __name__ == "__main__":
    model = YOLO("./models/foodseg/best.pt")
    image_path = "./data/chicken_test.png" 
    image = cv2.imread(image_path)

    items = detect_and_segment(image, model)
    print(f"Detected {len(items)} items:")

    for mask, label, box in items:
        print(f"Label: {label}, Box: {box}")