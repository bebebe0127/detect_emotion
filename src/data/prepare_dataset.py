import os
import shutil
import sys
from pathlib import Path

from sklearn.model_selection import train_test_split

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from utils.seed import DEFAULT_SEED, set_seed

RAW_DIR = "src/data/raw/test"
PROCESSED_DIR = "data/processed"

def prepare():
    set_seed(DEFAULT_SEED)
    images = []
    labels = []

    for label in os.listdir(RAW_DIR):
        class_dir = os.path.join(RAW_DIR, label)
        if not os.path.isdir(class_dir):
            continue

        for img in os.listdir(class_dir):
            if not img.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            images.append(os.path.join(class_dir, img))
            labels.append(label)

    train_imgs, temp_imgs, train_lbls, temp_lbls = train_test_split(
        images, labels, test_size=0.3, stratify=labels, random_state=DEFAULT_SEED
    )

    val_imgs, test_imgs, val_lbls, test_lbls = train_test_split(
        temp_imgs, temp_lbls, test_size=0.5, stratify=temp_lbls, random_state=DEFAULT_SEED
    )

    splits = {
        "train": (train_imgs, train_lbls),
        "val": (val_imgs, val_lbls),
        "test": (test_imgs, test_lbls),
    }

    for split, (imgs, lbls) in splits.items():
        for img_path, label in zip(imgs, lbls):
            target_dir = os.path.join(PROCESSED_DIR, split, label)
            os.makedirs(target_dir, exist_ok=True)
            shutil.copy(img_path, target_dir)

    print("Dataset prepared successfully")

if __name__ == "__main__":
    prepare()
