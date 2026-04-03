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
- `src/mlflow_config.py` — centralised MLflow config (experiment name, tracking URI, `setup_mlflow()`)
- `src/train.py` — training loop with SGD + CosineAnnealingLR, checkpointing, curve plots, and MLflow run logging
- `src/evaluate.py` — loads best checkpoint, computes accuracy/F1/classification report, saves confusion matrix, and logs metrics/artifacts to MLflow
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

## Model Architecture

`CIFAR10CNN` — a 3-block convolutional network with batch normalisation and dropout.

| Layer | Output channels | Notes |
|---|---|---|
| Block 1 | 32 | Conv→BN→ReLU×2, MaxPool, Dropout2d(0.25) |
| Block 2 | 64 | Conv→BN→ReLU×2, MaxPool, Dropout2d(0.25) |
| Block 3 | 128 | Conv→BN→ReLU×2, MaxPool, Dropout2d(0.25) |
| AdaptiveAvgPool | 128×4×4 | Fixed spatial size before classifier |
| Classifier | 10 | Linear(2048→512)→BN→ReLU→Dropout(0.5)→Linear(512→10) |

## Results

Trained for 30 epochs with SGD + CosineAnnealingLR on CIFAR-10 (50k train / 10k test, 1000 samples per class).

| Metric | Value |
|---|---|
| Test accuracy | 85.17% |
| Weighted F1 | 0.8507 |
| Macro F1 | 0.8444 |

Per-class results:

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| plane | 0.87 | 0.85 | 0.86 | 1000 |
| car | 0.93 | 0.94 | 0.94 | 1000 |
| bird | 0.83 | 0.73 | 0.78 | 1000 |
| cat | 0.75 | 0.64 | 0.69 | 1000 |
| deer | 0.81 | 0.86 | 0.83 | 1000 |
| dog | 0.73 | 0.81 | 0.77 | 1000 |
| frog | 0.85 | 0.91 | 0.88 | 1000 |
| horse | 0.90 | 0.87 | 0.88 | 1000 |
| ship | 0.89 | 0.93 | 0.91 | 1000 |
| truck | 0.90 | 0.93 | 0.91 | 1000 |

`cat` and `dog` are the hardest classes (F1 0.69 and 0.77), which is expected given their visual similarity. `car` is the strongest class at F1 0.94.

Training hyperparameters:

| Param | Value |
|---|---|
| epochs | 30 |
| batch_size | 64 |
| learning_rate | 0.1 |
| momentum | 0.9 |
| weight_decay | 1e-4 |
| optimizer | SGD |
| scheduler | CosineAnnealingLR |

## Experiment Tracking

Runs are tracked with MLflow under the `cifar10-cnn` experiment. Tracking data is stored locally in `mlruns/` at the project root.

```bash
# Train (automatically creates and logs a run)
cd src && python train.py

# View all runs
mlflow ui
# open http://127.0.0.1:5000
```

Each run logs:

- **Params**: all training hyperparameters and architecture name
- **Metrics (per epoch)**: `train_loss`, `train_accuracy`, `val_loss`, `val_accuracy`
- **Metrics (final)**: `test_accuracy`, `test_f1_weighted`
- **Artifacts**: `best_model.pth`, `confusion_matrix.png`, `classification_report.txt`

## Stack

- **PyTorch** + **torchvision** — model training and CIFAR-10 data loading
- **MLflow** — experiment tracking
- **JupyterLab** — interactive development
- **Flask** / **FastAPI** — model serving
