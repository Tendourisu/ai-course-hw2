from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

from .data import STL10_MEAN, STL10_STD
from .models import build_model, get_gradcam_target_layer
from .utils import get_device


class GradCAM:
    """Minimal Grad-CAM implementation for CNN classifiers."""

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self.forward_handle = target_layer.register_forward_hook(self._save_activation)
        self.backward_handle = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, _module: torch.nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        self.activations = output.detach()

    def _save_gradient(
        self,
        _module: torch.nn.Module,
        _grad_input: tuple[torch.Tensor, ...],
        grad_output: tuple[torch.Tensor, ...],
    ) -> None:
        self.gradients = grad_output[0].detach()

    def close(self) -> None:
        """Remove PyTorch hooks."""
        self.forward_handle.remove()
        self.backward_handle.remove()

    def __call__(self, image: torch.Tensor, target_class: int | None = None) -> tuple[np.ndarray, int, float]:
        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        logits = self.model(image)
        probs = torch.softmax(logits, dim=1)
        class_id = int(probs.argmax(dim=1).item()) if target_class is None else target_class
        score = logits[:, class_id].sum()
        score.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations or gradients.")

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=image.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = cam - cam.min()
        max_value = cam.max()
        if max_value > 0:
            cam = cam / max_value
        return cam, class_id, float(probs[0, class_id].item())


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """Convert a normalized CHW tensor into a displayable HWC array."""
    mean = torch.tensor(STL10_MEAN).view(3, 1, 1)
    std = torch.tensor(STL10_STD).view(3, 1, 1)
    image = tensor.cpu() * std + mean
    image = image.clamp(0.0, 1.0)
    return image.permute(1, 2, 0).numpy()


def save_overlay(
    image_tensor: torch.Tensor,
    cam: np.ndarray,
    title: str,
    output_path: str | Path,
) -> None:
    """Save original image, heatmap, and overlay side by side."""
    image = denormalize(image_tensor)
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    axes[0].imshow(image)
    axes[0].set_title("Original")
    axes[1].imshow(cam, cmap="jet")
    axes[1].set_title("Grad-CAM")
    axes[2].imshow(image)
    axes[2].imshow(cam, cmap="jet", alpha=0.45)
    axes[2].set_title(title)
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def generate_gradcam_grid(
    model: torch.nn.Module,
    model_name: str,
    data_root: Path,
    class_names: list[str],
    output_dir: str | Path,
    device: torch.device,
    image_size: int = 96,
    max_samples: int = 8,
) -> None:
    """Generate Grad-CAM images from the test set."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=STL10_MEAN, std=STL10_STD),
        ]
    )
    dataset = datasets.ImageFolder(data_root / "test", transform=transform)
    target_layer = get_gradcam_target_layer(model, model_name)
    cam_runner = GradCAM(model.to(device), target_layer)
    sample_indices = first_sample_per_class(dataset.targets, max_samples)

    for output_index, dataset_index in enumerate(sample_indices):
        image, target = dataset[dataset_index]
        image_batch = image.unsqueeze(0).to(device)
        cam, pred, prob = cam_runner(image_batch)
        title = f"pred={class_names[pred]} ({prob:.2f}) / true={class_names[target]}"
        save_overlay(image, cam, title, output / f"gradcam_{output_index:02d}_{class_names[target]}.png")

    cam_runner.close()


def first_sample_per_class(targets: list[int], max_samples: int) -> list[int]:
    """Select the first sample of each class until max_samples is reached."""
    selected: list[int] = []
    seen: set[int] = set()
    for index, target in enumerate(targets):
        if target in seen:
            continue
        selected.append(index)
        seen.add(target)
        if len(selected) >= max_samples:
            break
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Grad-CAM visualizations.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-root", type=str, default="STL10")
    parser.add_argument("--model", type=str, default="resnet18", choices=["simple_cnn", "resnet18"])
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--max-samples", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    class_names = checkpoint["class_names"]
    train_args = checkpoint.get("args", {})
    model = build_model(
        model_name=args.model,
        num_classes=len(class_names),
        activation=train_args.get("activation", "relu"),
        pooling=train_args.get("pooling", "max"),
        dropout=train_args.get("dropout", 0.0),
        use_batchnorm=not train_args.get("no_batchnorm", False),
    )
    model.load_state_dict(checkpoint["model_state"])
    generate_gradcam_grid(
        model=model,
        model_name=args.model,
        data_root=Path(args.data_root),
        class_names=class_names,
        output_dir=args.output_dir,
        device=device,
        image_size=args.image_size,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
