# Real-Time Face Mask Detection

A deep learning–powered computer vision system that detects **whether a person is wearing a face mask** in real time using a webcam or video file. Built with **MobileNetV2 transfer learning** + **OpenCV Haar Cascade** face detection.

---

## Project Highlights

| Feature | Detail |
|---|---|
| Architecture | MobileNetV2 (transfer learning, ImageNet weights) |
| Face Detector | OpenCV Haar Cascade |
| Classes | `with_mask` · `without_mask` |
| Input modes | Webcam · Video file · Static image |
| Extras | FPS counter · Confidence bar · Screenshot (press `s`) |

---

## Project Structure

```
face-mask-detection/
├── haarcascade/
│   └── haarcascade_frontalface_default.xml
├── model/
│   └── mask_detector.h5          # saved after training
├── logs/
│   ├── training_log.csv
│   ├── training_plot.png
│   └── confusion_matrix.png
├── utils/
│   └── download_dataset.py       # optional Kaggle downloader
├── dataset/                      # populate before training
│   ├── with_mask/
│   └── without_mask/
├── train_mask_detector.py        # ← train the model
├── detect_mask_video.py          # ← real-time webcam/video
├── detect_mask_image.py          # ← static image inference
└── requirements.txt
```

---

## Quick Start

### 1 — Clone & enter the repo

```bash
git clone https://github.com/<your-username>/face-mask-detection.git
cd face-mask-detection
```

### 2 — Create a virtual environment

```bash
python -m venv venv

# Linux / macOS
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3 — Install dependencies

```bash
pip install -r requirements.txt
```

---

## Dataset

Download any face-mask image dataset and organise it like this:

```
dataset/
├── with_mask/      # images of people wearing masks
└── without_mask/   # images of people without masks
```

**Recommended dataset (Kaggle):**
[Face Mask Dataset by omkargurav](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset)
https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml

Automated download (requires Kaggle API token at `~/.kaggle/kaggle.json`):

```bash
pip install kaggle
python utils/download_dataset.py
```

---

## Training

```bash
python train_mask_detector.py
```

**Optional arguments:**

| Argument | Default | Description |
|---|---|---|
| `--dataset` | `dataset` | Path to image dataset folder |
| `--model` | `model/mask_detector.h5` | Where to save the trained model |
| `--epochs` | `20` | Number of training epochs |
| `--lr` | `0.0001` | Initial learning rate |

Training output saved to `logs/`:
- `training_log.csv` — epoch-by-epoch metrics
- `training_plot.png` — accuracy & loss curves
- `confusion_matrix.png` — classification results

---

## Real-Time Detection (Webcam)

```bash
python detect_mask_video.py
```

**Video file:**

```bash
python detect_mask_video.py --source path/to/video.mp4
```

**Controls inside the window:**

| Key | Action |
|---|---|
| `q` | Quit |
| `s` | Save screenshot |

---

## Static Image Detection

```bash
python detect_mask_image.py --image photo.jpg
python detect_mask_image.py --image photo.jpg --output result.jpg
```

---

## Model Architecture

```
Input (224×224×3)
    │
MobileNetV2 (frozen ImageNet weights)
    │
AveragePooling2D (7×7)
    │
Flatten
    │
Dense(128, relu)
    │
Dropout(0.5)
    │
Dense(2, softmax)  →  [with_mask, without_mask]
```

**Why MobileNetV2?**
- Lightweight enough for real-time inference on CPU
- Pre-trained on 1.2M ImageNet images — excellent feature extractor
- Depthwise separable convolutions → low parameter count

---

## Training Configuration

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Loss | Categorical Cross-Entropy |
| Learning Rate | 1e-4 (with ReduceLROnPlateau) |
| Augmentation | Rotation, zoom, shift, flip |
| Early Stopping | Patience = 5 |
| Val Split | 20% |

---

## Evaluation

After training, the script automatically prints:

- **Classification Report** (precision, recall, F1-score)
- **Confusion Matrix** (saved to `logs/confusion_matrix.png`)

---

## Requirements

```
tensorflow==2.13.0
opencv-python==4.8.1.78
numpy==1.24.3
matplotlib==3.7.2
scikit-learn==1.3.0
seaborn==0.12.2
imutils==0.5.4
Pillow==10.0.0
```

> Python 3.8 – 3.11 recommended. TensorFlow 2.13 does **not** support Python 3.12.

---

## Troubleshooting

| Issue | Fix |
|---|---|
| `No module named tensorflow` | Run `pip install -r requirements.txt` inside your venv |
| Webcam not opening | Try `--source 1` or check camera permissions |
| Model file not found | Train first: `python train_mask_detector.py` |
| Low accuracy | Ensure dataset is balanced; increase epochs |

---

## License

MIT License — free to use, modify, and distribute.

---

## Acknowledgements

- [Kaggle Face Mask Dataset](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset)
- [MobileNetV2 paper](https://arxiv.org/abs/1801.04381) — Sandler et al., 2018
- [OpenCV](https://opencv.org/) — open source computer vision library
