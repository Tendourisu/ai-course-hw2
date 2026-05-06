from __future__ import annotations

import csv
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set common random seeds for repeatable experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_name: str = "auto") -> torch.device:
    """Resolve the training device from CLI input."""
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but PyTorch cannot see any CUDA device.")
    device = torch.device(device_name)
    if device.type == "cuda" and device.index is not None and device.index >= torch.cuda.device_count():
        raise RuntimeError(f"CUDA device {device.index} was requested, but only {torch.cuda.device_count()} devices are visible.")
    return device


def create_run_dir(output_dir: str | Path, run_name: str | None) -> Path:
    """Create an output directory for one experiment."""
    base = Path(output_dir)
    base.mkdir(parents=True, exist_ok=True)
    name = run_name or datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = base / name
    suffix = 1
    while run_dir.exists():
        run_dir = base / f"{name}_{suffix}"
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def save_json(payload: dict[str, Any], path: str | Path) -> None:
    """Write JSON with stable formatting."""
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)


def load_json(path: str | Path) -> dict[str, Any]:
    """Read a JSON object from disk."""
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def save_csv(rows: Iterable[dict[str, Any]], path: str | Path) -> None:
    """Save a sequence of dictionaries as CSV."""
    rows = list(rows)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class AverageMeter:
    """Track the sample-weighted average of a scalar."""

    def __init__(self) -> None:
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int) -> None:
        self.total += value * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.total / max(1, self.count)
