from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .data import create_dataloaders
from .engine import evaluate, predict, train_one_epoch
from .gradcam import generate_gradcam_grid
from .metrics import plot_learning_curves, save_classification_outputs
from .models import build_model
from .utils import create_run_dir, get_device, save_csv, save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CNN models on STL-10.")
    parser.add_argument("--data-root", type=str, default="STL10")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--model", type=str, default="resnet18", choices=["simple_cnn", "resnet18"])
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--valid-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")

    parser.add_argument("--augmentation", type=str, default="basic", choices=["none", "basic"])
    parser.add_argument("--activation", type=str, default="relu", choices=["relu", "tanh", "sigmoid"])
    parser.add_argument("--pooling", type=str, default="max", choices=["max", "avg"])
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--no-batchnorm", action="store_true")

    parser.add_argument("--optimizer", type=str, default="adamw", choices=["sgd", "adamw"])
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--gradcam-samples", type=int, default=10)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.epochs <= 0:
        raise ValueError("epochs must be positive.")
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive.")
    if args.dropout < 0.0 or args.dropout >= 1.0:
        raise ValueError("dropout must be in [0, 1).")


def build_optimizer(args: argparse.Namespace, model: nn.Module) -> torch.optim.Optimizer:
    """Create optimizer from CLI args."""
    if args.optimizer == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=True,
        )
    if args.optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    raise ValueError("Unsupported optimizer.")


def checkpoint_payload(
    args: argparse.Namespace,
    model: nn.Module,
    class_names: list[str],
    epoch: int,
    valid_acc: float,
) -> dict[str, Any]:
    """Build a checkpoint dictionary."""
    return {
        "args": vars(args),
        "class_names": class_names,
        "epoch": epoch,
        "model_state": model.state_dict(),
        "valid_acc": valid_acc,
    }


def main() -> None:
    args = parse_args()
    validate_args(args)
    set_seed(args.seed)

    device = get_device(args.device)
    run_dir = create_run_dir(args.output_dir, args.run_name)
    save_json(vars(args), run_dir / "config.json")

    dataloaders = create_dataloaders(
        data_root=args.data_root,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        valid_ratio=args.valid_ratio,
        augmentation=args.augmentation,
        seed=args.seed,
    )
    model = build_model(
        model_name=args.model,
        num_classes=len(dataloaders.class_names),
        activation=args.activation,
        pooling=args.pooling,
        dropout=args.dropout,
        use_batchnorm=not args.no_batchnorm,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(args, model)

    best_valid_acc = -1.0
    best_epoch = 0
    history: list[dict[str, Any]] = []
    start_time = time.time()

    print(f"Run directory: {run_dir}")
    print(f"Device: {device}")
    print(f"Classes: {', '.join(dataloaders.class_names)}")
    print(f"Split sizes: train={dataloaders.train_size}, valid={dataloaders.valid_size}, test={dataloaders.test_size}")

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            loader=dataloaders.train,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )
        valid_metrics = evaluate(model, dataloaders.valid, criterion, device, desc="valid")

        current_lr = optimizer.param_groups[0]["lr"]
        row = {
            "epoch": epoch,
            "lr": current_lr,
            "train_loss": train_metrics.loss,
            "train_acc": train_metrics.accuracy,
            "valid_loss": valid_metrics.loss,
            "valid_acc": valid_metrics.accuracy,
        }
        history.append(row)
        save_csv(history, run_dir / "history.csv")
        save_json({"history": history}, run_dir / "history.json")
        plot_learning_curves(history, run_dir / "learning_curves.png")

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"lr={current_lr:.6f} | "
            f"train loss={train_metrics.loss:.4f} acc={train_metrics.accuracy:.2f}% | "
            f"valid loss={valid_metrics.loss:.4f} acc={valid_metrics.accuracy:.2f}%"
        )

        if valid_metrics.accuracy > best_valid_acc:
            best_valid_acc = valid_metrics.accuracy
            best_epoch = epoch
            torch.save(
                checkpoint_payload(args, model, dataloaders.class_names, epoch, valid_metrics.accuracy),
                run_dir / "best.pt",
            )

    checkpoint = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    test_metrics = evaluate(model, dataloaders.test, criterion, device, desc="test")
    y_true, y_pred, _ = predict(model, dataloaders.test, device)
    report = save_classification_outputs(y_true, y_pred, dataloaders.class_names, run_dir, prefix="test")

    if args.gradcam_samples > 0:
        generate_gradcam_grid(
            model=model,
            model_name=args.model,
            data_root=Path(args.data_root),
            class_names=dataloaders.class_names,
            output_dir=run_dir / "gradcam",
            device=device,
            image_size=args.image_size,
            max_samples=args.gradcam_samples,
        )

    elapsed = time.time() - start_time
    summary = {
        "best_epoch": best_epoch,
        "best_valid_acc": best_valid_acc,
        "test_acc": test_metrics.accuracy,
        "test_loss": test_metrics.loss,
        "test_macro_f1": report["macro avg"]["f1-score"],
        "test_weighted_f1": report["weighted avg"]["f1-score"],
        "elapsed_sec": elapsed,
        "run_dir": str(run_dir),
    }
    save_json(summary, run_dir / "summary.json")
    print(f"Training complete. Summary saved to {run_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
