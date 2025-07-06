# Calorie Counter 

## Content table
1. [Project Description](#project-description)
2. [Project Structure](#project-structure)
3. [CV Techniques](#cv-techniques)
4. [Run the program](#run-the-program)
5. [Parameters](#parameters)
6. [Review version 1.00](#review-version-1.00)


## [Project Description](#project-description)
**Calorie counter** is a simple project built with the idea to explore common computer vision techniques like segmentation, classification and edge detection.
Given a photo of a dish (or raw food) with a **reference object** (like a credit card), the API estimates the **mass in grams of each segmented food item**. It does so by processing the image, segmenting food items, and calculating area-based mass estimates via masks.
Then it computes the total macro-nutrients for that food item.


## [Project Structure](#project-structure)
The project is divided in 4 main directories:
1) **Models**
Stores the models that the APi will use for prediction.

2) **Data**
Stores the required datasets.

3) **Scripts**
Contains important scripts that control: 
- Download and preprocessing of FoodSeg and Food101 datasets.
- Training of Yolov8 segmentation model on those datasets.

4) **Src**
With 5 modules it performs the following:
- *API*: Controls the API system, from the routes to its dependencies.
- *Detector*: Provides the methods to predict the food items from an image using the segmenation
and classification models.
- *Nutrition*: Controls the communication with the USDA for the nutritonal facts.
- *Scaling*: Scripts to extract the scale of the items using a reference object and estimating
its weight from the segmentation masks.


## [CV Techniques](#cv-techniques)
This project incorporates several key computer vision and image processing concepts:

1. **Segmentation & Classification** using YOLOv8.
2. **Morphological operations**: `dilate`, `MORPH_CLOSE`, `floodfill`
3. **LAB color space analysis**: understanding lightness and chroma.
4. **OTSU thresholding** for binarization.
5. **Gaussian Blur** to reduce noise.
6. **Edge detection** with CLAHE, bilateral filtering, and Canny.

## [Run the program](#run-the-program)
In future version the idea is to dockerize the API for an easier implementation. This version
requires of a manual integration.

1) First we need to fine tune the models with the food datasets.
```bash
chmod +x ./scripts/yolov8_food101.sh
chmod +x ./scripts/yolov8_foodseg.sh
./scripts/yolov8_food101.sh
./scripts/yolov8_foodseg.sh
```

To resume training use the follwing flags:
```python
resume = True
model = "path/to/last.pt"
```

2) Start the FastAPI server:
```bash
python -m uvicorn src.api.fastapi_app:app --reload
```

3) Make a prediction with curl:
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@./data/image_test.png"
```

## [Parameters](#parameters)
The system can be fine tuned in the **.env** file. The most important parameters are:
1) **USE_CLAS_MODEL** Use teh classification model or not. At the moment the dataset is not the correct so it throws error for the different label. It is recommneded to keep as *False*. This problem will be solved in the future.
2) **USDA_API_KEY** Needs to be obtained individually at *https://fdc.nal.usda.gov/api-guide/*
3) The fine tuning of the models can be controled using flags when running the bash command. This are the commands:
-m (model)
-e (epochs)
-i (imgsz)
-b (batch)
-l (lr0)
-w (workers)
-o (optimizer)
-r (resume)

## [Review version 1.00](#review-version-1.00)
There are many considerations to take into account for this first version of the code.
1) The accuracy of the models is not great:
- Ideally we'd need to use a better dataset or fine tune the models for longer to increase performance.
2) Detecting the credit card
- Detecting the reference object is still a challenge depending on the background and the credit card's color.
For best accuracy, teh image use a pale-colored credit card next to the food to act as a size reference in the image. The background should be dark so the contrast is high and the OTSU approach works as expected.
- For best performance, next version will try to use segmentation models to detect the credit card. This approach is much more invariant to color and shapes, and so it is more robust.
3) The UI will be implemented in future versions.