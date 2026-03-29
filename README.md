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

## Usage

```bash
# Train
cd src && python train.py

# Evaluate (requires a trained checkpoint at models/best_model.pth)
cd src && python evaluate.py
```

## Stack

- **PyTorch** + **torchvision** — model training and CIFAR-10 data loading
- **MLflow** — experiment tracking
- **JupyterLab** — interactive development
- **Flask** / **FastAPI** — model serving
