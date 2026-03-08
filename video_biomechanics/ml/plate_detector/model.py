"""Plate corner detection model.

A lightweight CNN that predicts 5 keypoints (corners) of home plate.
Uses a ResNet18 backbone with a keypoint regression head.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Tuple


class PlateKeypointModel(nn.Module):
    """Lightweight model for detecting plate corners.

    Architecture:
    - ResNet18 backbone (pretrained on ImageNet)
    - Global average pooling
    - FC layers for keypoint regression
    - Output: 10 values (x, y for 5 corners), normalized to [0, 1]
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()

        # Backbone - ResNet18 is small and fast
        backbone = models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)

        # Remove the final FC layer
        self.features = nn.Sequential(*list(backbone.children())[:-1])

        # Feature dimension from ResNet18
        feat_dim = 512

        # Keypoint regression head
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 10),  # 5 corners * 2 (x, y)
            nn.Sigmoid()  # Normalize to [0, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input image tensor (B, 3, H, W)

        Returns:
            Keypoints tensor (B, 10) with values in [0, 1]
            Reshape to (B, 5, 2) for corner coordinates
        """
        features = self.features(x)
        keypoints = self.head(features)
        return keypoints

    def predict_corners(self, x: torch.Tensor, img_size: Tuple[int, int]) -> torch.Tensor:
        """Predict corners in pixel coordinates.

        Args:
            x: Input image tensor (B, 3, H, W)
            img_size: Original image size (H, W)

        Returns:
            Corner coordinates (B, 5, 2) in pixel space
        """
        normalized = self.forward(x)  # (B, 10)
        corners = normalized.view(-1, 5, 2)  # (B, 5, 2)

        # Scale to image size
        h, w = img_size
        corners[:, :, 0] *= w  # x coordinates
        corners[:, :, 1] *= h  # y coordinates

        return corners


class PlateKeypointModelMobileNet(nn.Module):
    """Even lighter model using MobileNetV3.

    Good for real-time inference on CPU.
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()

        # MobileNetV3-Small backbone
        backbone = models.mobilenet_v3_small(
            weights='IMAGENET1K_V1' if pretrained else None
        )

        # Remove classifier
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Feature dimension from MobileNetV3-Small
        feat_dim = 576

        # Keypoint regression head
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 10),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        pooled = self.pool(features)
        keypoints = self.head(pooled)
        return keypoints


def create_model(model_type: str = "resnet18", pretrained: bool = True) -> nn.Module:
    """Factory function to create models.

    Args:
        model_type: "resnet18" or "mobilenet"
        pretrained: Whether to use pretrained backbone

    Returns:
        Model instance
    """
    if model_type == "resnet18":
        return PlateKeypointModel(pretrained=pretrained)
    elif model_type == "mobilenet":
        return PlateKeypointModelMobileNet(pretrained=pretrained)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# Loss function for keypoint regression
class KeypointLoss(nn.Module):
    """Combined loss for keypoint regression.

    Uses MSE for coordinate regression with optional
    wing loss for better convergence near the target.
    """

    def __init__(self, use_wing_loss: bool = True, wing_w: float = 10.0, wing_eps: float = 2.0):
        super().__init__()
        self.use_wing_loss = use_wing_loss
        self.w = wing_w
        self.eps = wing_eps
        self.c = self.w - self.w * np.log(1 + self.w / self.eps) if use_wing_loss else 0

    def wing_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Wing loss - better for keypoint localization than MSE."""
        diff = torch.abs(pred - target)
        loss = torch.where(
            diff < self.w,
            self.w * torch.log(1 + diff / self.eps),
            diff - self.c
        )
        return loss.mean()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.use_wing_loss:
            return self.wing_loss(pred, target)
        else:
            return nn.functional.mse_loss(pred, target)


# Import numpy for wing loss calculation
import numpy as np


if __name__ == "__main__":
    # Test model
    model = create_model("resnet18")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")

    corners = model.predict_corners(x, (720, 1280))
    print(f"Corners shape: {corners.shape}")
    print(f"Sample corners:\n{corners[0]}")
