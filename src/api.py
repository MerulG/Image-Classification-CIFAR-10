import logging
from contextlib import asynccontextmanager
from pathlib import Path

import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

from model import CIFAR10CNN
from preprocessing import get_normalization_stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

CHECKPOINT_PATH = Path("models/best_model.pth")

model: CIFAR10CNN
device: torch.device
transform: transforms.Compose


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global model, device, transform

    if not CHECKPOINT_PATH.exists():
        raise RuntimeError(
            f"Model checkpoint not found at '{CHECKPOINT_PATH}'. "
            "Train the model first by running src/train.py."
        )

    device = get_device()
    logger.info("Using device: %s", device)

    mean, std = get_normalization_stats()
    logger.info("Normalization stats — mean: %s  std: %s", mean, std)
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model = CIFAR10CNN()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    logger.info("CIFAR10CNN loaded from %s", CHECKPOINT_PATH)

    yield


app = FastAPI(title="CIFAR-10 Classifier", lifespan=lifespan)


@app.get("/")
def health():
    return {"status": "ok", "model": "CIFAR10CNN", "classes": CLASSES}


@app.post("/predict")
async def predict(file: UploadFile):
    try:
        image = Image.open(file.file)
        image.load()
    except (UnidentifiedImageError, Exception) as exc:
        raise HTTPException(status_code=400, detail=f"Could not open image: {exc}") from exc

    tensor = transform(image).unsqueeze(0).to(device)  # (1, 3, 32, 32)

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).squeeze(0)

    top_idx = probs.argmax().item()
    confidence = round(probs[top_idx].item(), 2)
    probabilities = {cls: round(probs[i].item(), 4) for i, cls in enumerate(CLASSES)}

    return {
        "predicted_class": CLASSES[top_idx],
        "confidence": confidence,
        "probabilities": probabilities,
    }


# Run with: uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
