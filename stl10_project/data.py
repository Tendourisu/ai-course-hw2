from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


STL10_MEAN = (0.4467, 0.4398, 0.4066)
STL10_STD = (0.2241, 0.2215, 0.2239)


@dataclass(frozen=True)
class DataLoaders:
    train: DataLoader
    valid: DataLoader
    test: DataLoader
    class_names: list[str]
    train_size: int
    valid_size: int
    test_size: int


def build_transforms(image_size: int, split: str, augmentation: str) -> transforms.Compose:
    """Build transforms for train/valid/test splits."""
    normalize = transforms.Normalize(mean=STL10_MEAN, std=STL10_STD)
    if split == "train" and augmentation == "basic":
        return transforms.Compose(
            [
                transforms.RandomCrop(image_size, padding=8),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )

    return transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ]
    )


def stratified_split(targets: list[int], valid_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    """Split indices class by class so valid keeps the original class balance."""
    if not 0.0 < valid_ratio < 0.5:
        raise ValueError("valid_ratio must be in (0, 0.5).")

    rng = np.random.default_rng(seed)
    targets_array = np.asarray(targets)
    train_indices: list[int] = []
    valid_indices: list[int] = []
    for class_id in sorted(np.unique(targets_array)):
        class_indices = np.flatnonzero(targets_array == class_id)
        rng.shuffle(class_indices)
        valid_count = max(1, int(round(len(class_indices) * valid_ratio)))
        valid_indices.extend(class_indices[:valid_count].tolist())
        train_indices.extend(class_indices[valid_count:].tolist())
    rng.shuffle(train_indices)
    rng.shuffle(valid_indices)
    return train_indices, valid_indices


def create_dataloaders(
    data_root: str | Path,
    image_size: int = 96,
    batch_size: int = 64,
    num_workers: int = 4,
    valid_ratio: float = 0.15,
    augmentation: str = "basic",
    seed: int = 42,
) -> DataLoaders:
    """Create ImageFolder dataloaders for STL-10."""
    root = Path(data_root)
    train_dir = root / "train"
    test_dir = root / "test"
    if not train_dir.exists() or not test_dir.exists():
        raise FileNotFoundError(f"Expected train and test folders under {root}")

    split_dataset = datasets.ImageFolder(train_dir)
    train_indices, valid_indices = stratified_split(split_dataset.targets, valid_ratio, seed)

    train_dataset = datasets.ImageFolder(train_dir, transform=build_transforms(image_size, "train", augmentation))
    valid_dataset = datasets.ImageFolder(train_dir, transform=build_transforms(image_size, "valid", augmentation))
    test_dataset = datasets.ImageFolder(test_dir, transform=build_transforms(image_size, "test", augmentation))

    if train_dataset.classes != test_dataset.classes:
        raise ValueError("Train and test class folders do not match.")

    generator = torch.Generator()
    generator.manual_seed(seed)
    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        Subset(train_dataset, train_indices),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=generator,
    )
    valid_loader = DataLoader(
        Subset(valid_dataset, valid_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return DataLoaders(
        train=train_loader,
        valid=valid_loader,
        test=test_loader,
        class_names=train_dataset.classes,
        train_size=len(train_indices),
        valid_size=len(valid_indices),
        test_size=len(test_dataset),
    )
