import os
import random
import shutil
from ultralytics import YOLO

#split
'''

folders = [
    "dataset/train/images",
    "dataset/train/labels",
    "dataset/valid/images",
    "dataset/valid/labels",
    "dataset/test/images",
    "dataset/test/labels"
]

for folder in folders:
    os.makedirs(folder, exist_ok=True)

images = [f for f in os.listdir("data") if f.endswith(".jpeg")]

random.shuffle(images)

train_end = int(0.7 * len(images))
valid_end = int(0.9 * len(images))

train = images[:train_end]
valid = images[train_end:valid_end]
test = images[valid_end:]

def copy_files(files, subset):
    for img in files:
        label = img.replace(".jpeg", ".txt")

        shutil.copy(
            os.path.join("data", img),
            os.path.join(f"dataset/{subset}/images", img)
        )

        shutil.copy(
            os.path.join("data", label),
            os.path.join(f"dataset/{subset}/labels", label)
        )

copy_files(train, "train")
copy_files(valid, "valid")
copy_files(test, "test")

print("Dataset split completed.")
'''



# Training and pred

model = YOLO("yolov8n.pt")

results = model.train(
    data="cropweed.yaml",
    epochs=30,
    imgsz=512,
    batch=8 
)

metrics = model.val()
print(metrics)

model.predict(
    source="dataset/test/images",
    save=True,
    conf=0.25
)