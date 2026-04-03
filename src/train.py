import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
from tqdm import tqdm

from preprocessing import get_dataloaders
from model import CIFAR10CNN
from mlflow_config import setup_mlflow


@dataclass
class TrainConfig:
    num_epochs: int = 30
    learning_rate: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 1e-4
    batch_size: int = 64
    checkpoint_dir: str = "models/"


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="  train", leave=False):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="    val", leave=False):
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


def plot_curves(
    train_losses: list[float],
    val_losses: list[float],
    train_accs: list[float],
    val_accs: list[float],
    save_path: str,
) -> None:
    epochs = range(1, len(train_losses) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(epochs, train_losses, label="Train")
    ax1.plot(epochs, val_losses, label="Val")
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs, train_accs, label="Train")
    ax2.plot(epochs, val_accs, label="Val")
    ax2.set_title("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Training curves saved to {save_path}")


def train(config: TrainConfig) -> None:
    Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"Using device: {device}")

    train_loader, val_loader = get_dataloaders(batch_size=config.batch_size)

    model = CIFAR10CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config.learning_rate,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.num_epochs
    )

    setup_mlflow()

    with mlflow.start_run() as run:
        mlflow.log_params({
            "epochs": config.num_epochs,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "optimizer": "SGD",
            "scheduler": "CosineAnnealingLR",
            "architecture": "CIFAR10CNN",
            "num_classes": 10,
        })

        train_losses, val_losses = [], []
        train_accs, val_accs = [], []
        best_val_loss = float("inf")

        for epoch in range(1, config.num_epochs + 1):
            current_lr = scheduler.get_last_lr()[0] if epoch > 1 else config.learning_rate

            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            scheduler.step()

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)

            print(
                f"Epoch [{epoch:02d}/{config.num_epochs}]  "
                f"lr={current_lr:.2e}  "
                f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
                f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}"
            )

            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "train_accuracy": train_acc,
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                },
                step=epoch,
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "config": config,
                }
                best_path = os.path.join(config.checkpoint_dir, "best_model.pth")
                torch.save(checkpoint, best_path)
                print(f"  -> Best model saved (val_loss={val_loss:.4f})")

        final_path = os.path.join(config.checkpoint_dir, "final_model.pth")
        torch.save(
            {
                "epoch": config.num_epochs,
                "model_state_dict": model.state_dict(),
                "val_loss": val_losses[-1],
                "val_acc": val_accs[-1],
                "config": config,
            },
            final_path,
        )
        print(f"Final model saved to {final_path}")

        curves_path = os.path.join(config.checkpoint_dir, "training_curves.png")
        plot_curves(train_losses, val_losses, train_accs, val_accs, curves_path)

        mlflow.log_artifact("models/best_model.pth")
        print(f"MLflow run ID: {run.info.run_id}")


if __name__ == "__main__":
    train(TrainConfig())
