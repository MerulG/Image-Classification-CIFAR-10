import mlflow
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, classification_report, confusion_matrix

from preprocessing import get_dataloaders
from model import CIFAR10CNN
from train import TrainConfig  # noqa: F401 — needed for unpickling checkpoints

CHECKPOINT_PATH = "models/best_model.pth"
CONFUSION_MATRIX_PATH = "models/confusion_matrix.png"

CLASS_NAMES = ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def get_predictions(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_preds, all_labels = [], []

    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels.numpy())

    return np.concatenate(all_preds), np.concatenate(all_labels)


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    save_path: str,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_title("Confusion Matrix", fontsize=14)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def evaluate() -> None:
    device = get_device()
    print(f"Using device: {device}")

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']} "
          f"(val_loss={checkpoint['val_loss']:.4f}, val_acc={checkpoint['val_acc']:.4f})")

    model = CIFAR10CNN().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    _, test_loader = get_dataloaders(batch_size=128)

    preds, labels = get_predictions(model, test_loader, device)

    overall_acc = (preds == labels).mean()
    print(f"\nOverall test accuracy: {overall_acc:.4f} ({overall_acc * 100:.2f}%)")

    print("\nPer-class accuracy:")
    for i, name in enumerate(CLASS_NAMES):
        mask = labels == i
        class_acc = (preds[mask] == labels[mask]).mean()
        print(f"  {name:<8}: {class_acc:.4f} ({class_acc * 100:.2f}%)")

    weighted_f1 = f1_score(labels, preds, average="weighted")
    print(f"\nWeighted F1 score: {weighted_f1:.4f}")

    report = classification_report(labels, preds, target_names=CLASS_NAMES)
    print("\nClassification report:")
    print(report)

    cm = confusion_matrix(labels, preds)
    plot_confusion_matrix(cm, CLASS_NAMES, CONFUSION_MATRIX_PATH)

    mlflow.log_metrics({
        "test_accuracy": float(overall_acc),
        "test_f1_weighted": float(weighted_f1),
    })

    mlflow.log_artifact(CONFUSION_MATRIX_PATH)

    report_path = "models/classification_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    mlflow.log_artifact(report_path)


if __name__ == "__main__":
    evaluate()
