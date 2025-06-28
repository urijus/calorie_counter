# Calorie Counter 

This project is a simple but effective calorie counter built using computer vision techniques. Given a photo of a dish (or raw food) with a **reference object** (like a pale-colored credit card), the system estimates the **mass in grams of each segmented food item**. It does so by processing the image, segmenting food items, and calculating area-based mass estimates via masks.

---

## Techniques Used

This project incorporates several key computer vision and image processing concepts:

1. **Segmentation & Classification** using YOLOv8
2. **Morphological operations**: `dilate`, `MORPH_CLOSE`, `floodfill`
3. **LAB color space analysis**: understanding lightness and chroma (color distance from grey)
4. **OTSU thresholding** for binarization
5. **Gaussian Blur** to reduce noise
6. **Edge detection** with CLAHE, bilateral filtering, and Canny

---

## Running the API

Start the FastAPI server:

```bash
python -m uvicorn src.api.fastapi_app:app --reload
```

Make a prediction with curl:
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@./data/chicken_test.png" \
     -o response.json
```

To fine-tune the segmentation model:
```bash
chmod +x ./scripts/fine_tune_yolov8.sh
./scripts/fine_tune_yolov8.sh
```

Ensure proper permissions for labels:

```bash
chmod -R u+rw data/FoodSeg103/labels
```

To resume training:

```python
resume = True
model = "path/to/last.pt"
```

## Project Structure 

## Dataset
The project uses the FoodSeg103 dataset for training and validation.

## Tip for Better Accuracy
Use a pale-colored credit card next to the food to act as a size reference in the image.

## Author Notes
This project was developed as a practical application of advanced image analysis techniques. It reflects a hands-on understanding of:

- Color models (RGB vs LAB)
- Classical and learned segmentation
- Image preprocessing pipelines