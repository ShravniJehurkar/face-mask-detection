"""
Face Mask Detection - Real-Time Webcam Inference
=================================================
Detects faces via Haar Cascade and classifies each face as
"Mask" or "No Mask" using a fine-tuned MobileNetV2 model.

Usage:
    python detect_mask_video.py
    python detect_mask_video.py --source path/to/video.mp4
    python detect_mask_video.py --source 0              # webcam index

Author: [Your Name]
"""

import argparse
import time
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# ─── CONFIG ───────────────────────────────────────────────────────────────────
FACE_CASCADE_PATH = "haarcascade/haarcascade_frontalface_default.xml"
MODEL_PATH        = "model/mask_detector.h5"
CONFIDENCE_THRESH = 0.6   # Minimum confidence to display a prediction
IMG_SIZE          = (224, 224)
# ──────────────────────────────────────────────────────────────────────────────

COLORS = {
    "Mask":    (0, 200, 60),    # Green
    "No Mask": (0, 40, 220),    # Red (BGR)
}


def load_resources():
    """Load face cascade and Keras model."""
    face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
    if face_cascade.empty():
        raise FileNotFoundError(
            f"Haar cascade not found at: {FACE_CASCADE_PATH}"
        )
    model = load_model(MODEL_PATH)
    print("[INFO] Model and cascade loaded successfully.")
    return face_cascade, model


def preprocess_face(face_roi: np.ndarray) -> np.ndarray:
    """Resize, preprocess, and expand dims for model inference."""
    face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
    face = cv2.resize(face, IMG_SIZE)
    face = preprocess_input(face.astype("float32"))
    return np.expand_dims(face, axis=0)


def draw_overlay(frame, x, y, w, h, label, confidence, color):
    """Draw bounding box, label, and confidence bar."""
    # Bounding box
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    # Label background pill
    text = f"{label}: {confidence * 100:.1f}%"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.6, 1)
    cv2.rectangle(frame, (x, y - th - 10), (x + tw + 8, y), color, -1)
    cv2.putText(
        frame, text,
        (x + 4, y - 5),
        cv2.FONT_HERSHEY_DUPLEX, 0.6,
        (255, 255, 255), 1, cv2.LINE_AA,
    )

    # Confidence bar below box
    bar_w = int(w * confidence)
    cv2.rectangle(frame, (x, y + h + 4), (x + w, y + h + 12), (50, 50, 50), -1)
    cv2.rectangle(frame, (x, y + h + 4), (x + bar_w, y + h + 12), color, -1)


def run_detection(source, face_cascade, model):
    """Main detection loop."""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {source}")

    fps_time = time.time()
    frame_count = 0
    fps = 0.0

    print("[INFO] Starting detection — press 'q' to quit, 's' to screenshot.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60),
        )

        mask_count    = 0
        no_mask_count = 0

        for (x, y, w, h) in faces:
            # Add padding around detected face
            pad = 10
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(frame.shape[1], x + w + pad)
            y2 = min(frame.shape[0], y + h + pad)

            face_roi = frame[y1:y2, x1:x2]
            if face_roi.size == 0:
                continue

            blob     = preprocess_face(face_roi)
            preds    = model.predict(blob, verbose=0)[0]
            class_id = int(np.argmax(preds))
            label    = "Mask" if class_id == 0 else "No Mask"
            conf     = float(preds[class_id])

            if conf < CONFIDENCE_THRESH:
                label = "Uncertain"
                color = (180, 180, 180)
            else:
                color = COLORS[label]
                if label == "Mask":
                    mask_count += 1
                else:
                    no_mask_count += 1

            draw_overlay(frame, x1, y1, x2 - x1, y2 - y1, label, conf, color)

        # ── HUD ─────────────────────────────────────────────────────────────
        elapsed = time.time() - fps_time
        if elapsed >= 1.0:
            fps = frame_count / elapsed
            frame_count = 0
            fps_time = time.time()

        hud_lines = [
            f"FPS: {fps:.1f}",
            f"Masked: {mask_count}",
            f"No Mask: {no_mask_count}",
        ]
        for i, line in enumerate(hud_lines):
            cv2.putText(
                frame, line,
                (10, 24 + i * 22),
                cv2.FONT_HERSHEY_DUPLEX, 0.6,
                (255, 255, 255), 1, cv2.LINE_AA,
            )

        cv2.imshow("Face Mask Detector  [q=quit  s=screenshot]", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            print("[INFO] Quit signal received.")
            break
        elif key == ord("s"):
            filename = f"screenshot_{int(time.time())}.png"
            cv2.imwrite(filename, frame)
            print(f"[INFO] Screenshot saved → {filename}")

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Real-Time Face Mask Detector")
    parser.add_argument(
        "--source", default=0,
        help="Video source: 0 for webcam, or path to video file",
    )
    args = parser.parse_args()

    # If source is a digit string, convert to int for webcam index
    source = args.source
    if isinstance(source, str) and source.isdigit():
        source = int(source)

    face_cascade, model = load_resources()
    run_detection(source, face_cascade, model)


if __name__ == "__main__":
    main()
