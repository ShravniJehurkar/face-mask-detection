"""
Dataset Setup Helper
====================
Downloads the face mask dataset from Kaggle and organises it
into the expected  dataset/with_mask  and  dataset/without_mask  folders.

Requirements:
    pip install kaggle
    Place your kaggle.json API token at ~/.kaggle/kaggle.json

Usage:
    python utils/download_dataset.py
"""

import os
import zipfile
import shutil


KAGGLE_DATASET = "omkargurav/face-mask-dataset"
DEST           = "dataset"


def download():
    try:
        import kaggle  # noqa: F401
    except ImportError:
        print("[ERROR] kaggle not installed. Run: pip install kaggle")
        return

    print(f"[INFO] Downloading dataset: {KAGGLE_DATASET}")
    os.makedirs(DEST, exist_ok=True)
    os.system(f"kaggle datasets download -d {KAGGLE_DATASET} -p /tmp/mask_data --unzip")

    # Rearrange into with_mask / without_mask
    src_with    = "/tmp/mask_data/data/with_mask"
    src_without = "/tmp/mask_data/data/without_mask"

    for src, tgt_name in [(src_with, "with_mask"), (src_without, "without_mask")]:
        tgt = os.path.join(DEST, tgt_name)
        if os.path.isdir(src):
            shutil.copytree(src, tgt, dirs_exist_ok=True)
            print(f"[INFO] Copied {src} → {tgt}")
        else:
            print(f"[WARN] Source not found: {src}")

    print("[INFO] Dataset ready.")


if __name__ == "__main__":
    download()
