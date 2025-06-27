# calorie_counter
This project aims to create a simpel calorie counter using CV techniques


chmod +x ./scripts/fine_tune_yolov8.sh
chmod -R u+rw data/FoodSeg103/labels


1) morpholocial transformation (dilute, MORPH_CLOSE, floodfill)
2) segmentation + classification w/ yolovo8 
3) understanding LAB colorspace (Lightness, Red/Green axis and Blue/Yellow axis)
    - chroma (how far from grey (a=b=0) a color is) vs lightness (brightness (light/dark))
4) OTSU thresholding (find threshold value that best separates two peaks of pixels of the grayscale image)
5) Gaussian Blur to remove noise
6) edge detection techniques (clahe, bilateral filter and canny)

to resume training, resume = True and model point to where we have last.pt