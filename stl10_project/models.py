from __future__ import annotations

import torch
from torch import nn
from torchvision.models import resnet18


class ConvBlock(nn.Module):
    """A compact convolution block used by SimpleCNN."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: str = "relu",
        pooling: str = "max",
        use_batchnorm: bool = True,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=not use_batchnorm),
        ]
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(build_activation(activation))
        layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=not use_batchnorm))
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(build_activation(activation))
        if pooling == "max":
            layers.append(nn.MaxPool2d(kernel_size=2))
        elif pooling == "avg":
            layers.append(nn.AvgPool2d(kernel_size=2))
        else:
            raise ValueError("pooling must be 'max' or 'avg'.")
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SimpleCNN(nn.Module):
    """A small CNN baseline with four convolution stages."""

    def __init__(
        self,
        num_classes: int,
        activation: str = "relu",
        pooling: str = "max",
        dropout: float = 0.0,
        use_batchnorm: bool = True,
    ) -> None:
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3, 32, activation, pooling, use_batchnorm),
            ConvBlock(32, 64, activation, pooling, use_batchnorm),
            ConvBlock(64, 128, activation, pooling, use_batchnorm),
            ConvBlock(128, 256, activation, pooling, use_batchnorm),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(256, num_classes),
        )
        self.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def build_activation(name: str) -> nn.Module:
    """Create an activation module by name."""
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "tanh":
        return nn.Tanh()
    if name == "sigmoid":
        return nn.Sigmoid()
    raise ValueError("activation must be one of: relu, tanh, sigmoid.")


def init_weights(module: nn.Module) -> None:
    """Initialize trainable layers with common CNN defaults."""
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.01)
        nn.init.zeros_(module.bias)


def build_resnet18_stl10(num_classes: int, dropout: float = 0.0) -> nn.Module:
    """Build a ResNet-18 variant adapted to 96x96 images."""
    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        nn.Linear(in_features, num_classes),
    )
    return model


def build_model(
    model_name: str,
    num_classes: int,
    activation: str = "relu",
    pooling: str = "max",
    dropout: float = 0.0,
    use_batchnorm: bool = True,
) -> nn.Module:
    """Factory for supported model architectures."""
    if model_name == "simple_cnn":
        return SimpleCNN(
            num_classes=num_classes,
            activation=activation,
            pooling=pooling,
            dropout=dropout,
            use_batchnorm=use_batchnorm,
        )
    if model_name == "resnet18":
        return build_resnet18_stl10(num_classes=num_classes, dropout=dropout)
    raise ValueError("model must be 'simple_cnn' or 'resnet18'.")


def get_gradcam_target_layer(model: nn.Module, model_name: str) -> nn.Module:
    """Return the last convolution layer for Grad-CAM."""
    if model_name == "simple_cnn":
        conv_layers = [module for module in model.modules() if isinstance(module, nn.Conv2d)]
        return conv_layers[-1]
    if model_name == "resnet18":
        return model.layer4[-1].conv2
    raise ValueError("Unsupported model for Grad-CAM.")
