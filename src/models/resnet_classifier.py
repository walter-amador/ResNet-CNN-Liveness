"""
ResNet-based binary classifier for face liveness detection.

Supports ResNet-18, ResNet-34, and ResNet-50.
The final fully-connected layer is replaced with:
    Dropout(0.5) → Linear(in_features, num_classes)

Differential learning rates are used: the backbone runs at lr/10 and the
classification head runs at the full lr, enabling gentle fine-tuning of
pretrained features while aggressively training the new head.

Usage:
    model = ResNetClassifier("resnet18", num_classes=2, pretrained=True)
    optimizer = model.get_optimizer(lr=1e-4, weight_decay=1e-4)
    logits = model(images)  # shape: (B, num_classes)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


_SUPPORTED = {"resnet18", "resnet34", "resnet50"}


class ResNetClassifier(nn.Module):
    """Pretrained ResNet backbone with a custom binary classification head.

    Args:
        model_name  : "resnet18" | "resnet34" | "resnet50"
        num_classes : number of output classes (2 for live/spoof)
        pretrained  : load ImageNet pretrained weights
        dropout     : dropout probability before the final linear layer
    """

    def __init__(
        self,
        model_name: str = "resnet18",
        num_classes: int = 2,
        pretrained: bool = True,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()

        if model_name not in _SUPPORTED:
            raise ValueError(
                f"Unsupported model '{model_name}'. Choose from: {_SUPPORTED}"
            )

        self.model_name = model_name
        weights = "IMAGENET1K_V1" if pretrained else None
        backbone = getattr(models, model_name)(weights=weights)

        # Store backbone layers (everything except the final FC)
        self.features = nn.Sequential(*list(backbone.children())[:-1])  # output: (B, C, 1, 1)
        in_features = backbone.fc.in_features

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Returns raw logits (no softmax)."""
        x = self.features(x)          # (B, C, 1, 1)
        x = x.flatten(1)              # (B, C)
        return self.classifier(x)     # (B, num_classes)

    def get_optimizer(
        self,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
    ) -> torch.optim.AdamW:
        """Return AdamW with differential learning rates.

        Backbone parameters: lr / 10  (gentle fine-tuning)
        Head parameters:     lr       (full learning rate)
        """
        backbone_params = list(self.features.parameters())
        head_params     = list(self.classifier.parameters())

        return torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": lr / 10},
                {"params": head_params,     "lr": lr},
            ],
            weight_decay=weight_decay,
        )

    def freeze_backbone(self) -> None:
        """Freeze all backbone parameters (stage-1 fine-tuning)."""
        for param in self.features.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreeze all backbone parameters (stage-2 fine-tuning)."""
        for param in self.features.parameters():
            param.requires_grad = True

    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Factory function used by train.py
# ---------------------------------------------------------------------------

def build_resnet(config) -> ResNetClassifier:
    """Instantiate a ResNetClassifier from a Config object."""
    return ResNetClassifier(
        model_name=config.model_name,
        num_classes=config.num_classes,
        pretrained=config.pretrained,
    )
