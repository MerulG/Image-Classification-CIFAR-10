import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

DATA_DIR = "data/"

CIFAR10_CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)


def compute_mean_std(dataset: torch.utils.data.Dataset) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute per-channel mean and std from a dataset using a cumulative sum approach.

    Mean/std must be computed from training data only — using test data would
    constitute data leakage, as normalisation parameters would then encode
    information about the test distribution and invalidate evaluation metrics.
    """
    loader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=2)

    n_pixels = 0
    channel_sum = torch.zeros(3)
    channel_sum_sq = torch.zeros(3)

    #Ignore labels
    for images, _ in loader:
        # images: (B, C, H, W)
        b, c, h, w = images.shape
        n_pixels += b * h * w
        #1D tensor with 3 values — one per channel
        channel_sum += images.sum(dim=(0, 2, 3))
        channel_sum_sq += (images ** 2).sum(dim=(0, 2, 3))

    mean = channel_sum / n_pixels
    # Var(X) = E[X^2] - E[X]^2
    std = torch.sqrt(channel_sum_sq / n_pixels - mean ** 2)

    print(f"Computed mean: {mean}")
    print(f"Computed std:  {std}")
    return mean, std


def get_dataloaders(batch_size: int = 64) -> tuple[DataLoader, DataLoader]:
    """
    Build train and test DataLoaders for CIFAR-10.

    Steps:
      1. Download the raw dataset with ToTensor() only (no normalisation yet).
      2. Compute mean/std from the training set only.
      3. Rebuild datasets with the full normalisation transforms applied.
      4. Return loaders.
    """
    # Step 1 — download raw data (ToTensor maps [0,255] uint8 → [0,1] float32,
    # which is required before we can compute statistics numerically)
    raw_train = torchvision.datasets.CIFAR10(
        root=DATA_DIR, train=True, download=True, transform=transforms.ToTensor()
    )

    # Step 2 — compute statistics from training set only
    mean, std = compute_mean_std(raw_train)

    # Step 3 — define transforms using the computed statistics

    # Training transforms:
    # - RandomHorizontalFlip: cheap, label-preserving augmentation that halves
    #   effective sample repetition and improves generalisation
    # - RandomCrop(32, padding=4): shifts the image slightly, encouraging
    #   translation invariance without distorting the content
    # - ToTensor: convert PIL image to float32 tensor in [0, 1]
    # - Normalize: zero-centre and scale each channel so that inputs have
    #   approximately unit variance, stabilising gradient flow
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean.tolist(), std.tolist()),
    ])

    # Test transforms:
    # Augmentation is NOT applied at test time because we want a deterministic,
    # unbiased estimate of generalisation performance. Augmenting test images
    # would introduce randomness into evaluation and make results non-reproducible.
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean.tolist(), std.tolist()),
    ])

    # Step 4 — rebuild datasets with correct transforms
    train_dataset = torchvision.datasets.CIFAR10(
        root=DATA_DIR, train=True, download=False, transform=train_transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=DATA_DIR, train=False, download=False, transform=test_transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )

    print(f"Train batches: {len(train_loader)} | Test batches: {len(test_loader)}")
    return train_loader, test_loader


def show_sample_images(train_loader: DataLoader, classes: tuple[str, ...]) -> None:
    """Display a 4x4 grid of sample images from the first training batch."""
    images, labels = next(iter(train_loader))
    images = images[:16]
    labels = labels[:16]

    # Undo normalisation for display: roughly clamp back to [0, 1]
    imgs = images - images.min()
    imgs = imgs / imgs.max()

    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    fig.suptitle("CIFAR-10 Sample Images", fontsize=14)

    for i, ax in enumerate(axes.flat):
        # (C, H, W) → (H, W, C)
        ax.imshow(np.transpose(imgs[i].numpy(), (1, 2, 0)))
        ax.set_title(classes[labels[i].item()], fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    train_loader, test_loader = get_dataloaders(batch_size=64)
    show_sample_images(train_loader, CIFAR10_CLASSES)
