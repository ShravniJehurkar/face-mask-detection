"""
Face Mask Detection - Static Image Inference
=============================================
Run mask detection on a single image file and save the annotated result.

Usage:
    python detect_mask_image.py --image path/to/photo.jpg
    python detect_mask_image.py --image photo.jpg --output result.jpg

Author: [Your Name]
"""

import argparse
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

FACE_CASCADE_PATH = "haarcascade/haarcascade_frontalface_default.xml"
MODEL_PATH        = "model/mask_detector.h5"
IMG_SIZE          = (224, 224)
COLORS = {
    "Mask":    (0, 200, 60),
    "No Mask": (0, 40, 220),
}


def detect_on_image(image_path: str, output_path: str = "output.jpg"):
    face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
    model        = load_model(MODEL_PATH)

    frame = cv2.imread(image_path)
    if frame is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))

    print(f"[INFO] {len(faces)} face(s) detected in {image_path}")

    for (x, y, w, h) in faces:
        face_roi = frame[y:y + h, x:x + w]
        face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
        face_res = cv2.resize(face_rgb, IMG_SIZE)
        blob     = np.expand_dims(
            preprocess_input(face_res.astype("float32")), axis=0
        )
        preds    = model.predict(blob, verbose=0)[0]
        idx      = int(np.argmax(preds))
        label    = "Mask" if idx == 0 else "No Mask"
        conf     = float(preds[idx])
        color    = COLORS[label]

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        text = f"{label} ({conf * 100:.1f}%)"
        cv2.putText(
            frame, text, (x, y - 10),
            cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 2, cv2.LINE_AA,
        )
        print(f"  Face at ({x},{y}): {label} — {conf * 100:.1f}% confidence")

    cv2.imwrite(output_path, frame)
    print(f"[INFO] Annotated image saved → {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Static Image Mask Detector")
    parser.add_argument("--image",  required=True, help="Path to input image")
    parser.add_argument("--output", default="output.jpg", help="Output image path")
    args = parser.parse_args()
    detect_on_image(args.image, args.output)
