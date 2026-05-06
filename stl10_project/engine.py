from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from .utils import AverageMeter


@dataclass(frozen=True)
class EpochMetrics:
    loss: float
    accuracy: float


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute top-1 accuracy as a percentage."""
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item() * 100.0


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> EpochMetrics:
    """Run one training epoch."""
    model.train()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    for images, targets in tqdm(loader, desc="train", leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        loss_meter.update(loss.item(), batch_size)
        acc_meter.update(accuracy_from_logits(logits.detach(), targets), batch_size)

    return EpochMetrics(loss=loss_meter.avg, accuracy=acc_meter.avg)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    desc: str = "eval",
) -> EpochMetrics:
    """Evaluate loss and accuracy."""
    model.eval()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    for images, targets in tqdm(loader, desc=desc, leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, targets)

        batch_size = images.size(0)
        loss_meter.update(loss.item(), batch_size)
        acc_meter.update(accuracy_from_logits(logits, targets), batch_size)

    return EpochMetrics(loss=loss_meter.avg, accuracy=acc_meter.avg)


@torch.no_grad()
def predict(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collect targets, predictions and probabilities from a dataloader."""
    model.eval()
    all_targets: list[np.ndarray] = []
    all_preds: list[np.ndarray] = []
    all_probs: list[np.ndarray] = []

    for images, targets in tqdm(loader, desc="predict", leave=False):
        images = images.to(device, non_blocking=True)
        logits = model(images)
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)
        all_targets.append(targets.numpy())
        all_preds.append(preds.cpu().numpy())
        all_probs.append(probs.cpu().numpy())

    return np.concatenate(all_targets), np.concatenate(all_preds), np.concatenate(all_probs)
