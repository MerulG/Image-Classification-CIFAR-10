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

## Stack

- **PyTorch** + **torchvision** — model training and CIFAR-10 data loading
- **MLflow** — experiment tracking
- **JupyterLab** — interactive development
- **Flask** / **FastAPI** — model serving
