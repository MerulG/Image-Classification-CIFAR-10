# Image-Classification-CIFAR-10

Computer vision image classification using PyTorch and the CIFAR-10 dataset.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Project Structure

- `src/preprocessing.py` — data loading, normalization, and augmentation pipeline for CIFAR-10
- `src/model.py` — `CIFAR10CNN` architecture (3 conv blocks, adaptive pool, classifier head)
- `src/train.py` — training loop with SGD + CosineAnnealingLR, checkpointing, and curve plots
- `src/evaluate.py` — loads best checkpoint, computes accuracy/F1/classification report, saves confusion matrix
- `src/api.py` — FastAPI inference service with `GET /` health check and `POST /predict` image upload endpoint

## Usage

```bash
# Train
cd src && python train.py

# Evaluate (requires a trained checkpoint at models/best_model.pth)
cd src && python evaluate.py

# Serve locally (requires a trained checkpoint at src/models/best_model.pth)
uvicorn src.api:app --reload --host 0.0.0.0 --port 8000

# Serve via Docker
docker build -f docker/Dockerfile -t cifar10-api .
docker run -p 8000:8000 cifar10-api
```

## Best Model Performance

Trained for 30 epochs with SGD + CosineAnnealingLR on CIFAR-10 (50k train / 10k test).

| Metric | Value |
|---|---|
| Test accuracy | 85.17% |
| Weighted F1 | 0.8507 |

Per-class results:

| Class | Precision | Recall | F1 |
|---|---|---|---|
| plane | 0.87 | 0.85 | 0.86 |
| car | 0.94 | 0.95 | 0.94 |
| bird | 0.84 | 0.74 | 0.79 |
| cat | 0.74 | 0.67 | 0.70 |
| deer | 0.85 | 0.84 | 0.84 |
| dog | 0.74 | 0.83 | 0.78 |
| frog | 0.85 | 0.91 | 0.88 |
| horse | 0.89 | 0.87 | 0.88 |
| ship | 0.89 | 0.94 | 0.91 |
| truck | 0.92 | 0.92 | 0.92 |

Training config: `lr=0.1`, `momentum=0.9`, `weight_decay=1e-4`, `batch_size=64`.

## Stack

- **PyTorch** + **torchvision** — model training and CIFAR-10 data loading
- **MLflow** — experiment tracking
- **JupyterLab** — interactive development
- **Flask** / **FastAPI** — model serving
