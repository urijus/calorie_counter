from datasets import load_dataset

ds = load_dataset("EduardoPacheco/FoodSeg103", split="train", cache_dir="data")
print(ds)   