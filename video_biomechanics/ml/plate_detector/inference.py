"""Inference module for plate corner detection."""

import cv2
import numpy as np
import torch
from pathlib import Path
from typing import Optional, Tuple, List
import albumentations as A
from albumentations.pytorch import ToTensorV2

from model import create_model


class PlateDetector:
    """Neural network-based plate corner detector.

    Usage:
        detector = PlateDetector("checkpoints/best_model.pth")
        corners = detector.detect(frame)  # Returns 5x2 array or None
    """

    def __init__(
        self,
        model_path: Path,
        device: Optional[str] = None,
        confidence_threshold: float = 0.5
    ):
        """
        Args:
            model_path: Path to trained model checkpoint
            device: Device to run inference on (auto-detect if None)
            confidence_threshold: Minimum confidence for valid detection
        """
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold

        # Setup device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # Load model
        self._load_model()

        # Preprocessing transform
        self.transform = A.Compose([
            A.Resize(self.img_size[0], self.img_size[1]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

    def _load_model(self):
        """Load model from checkpoint."""
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)

        self.model_type = checkpoint.get('model_type', 'resnet18')
        self.img_size = checkpoint.get('img_size', (224, 224))

        self.model = create_model(self.model_type, pretrained=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()

        print(f"Loaded {self.model_type} model from {self.model_path}")
        print(f"  Pixel error: {checkpoint.get('pixel_error', 'N/A'):.2f}px")

    def preprocess(self, frame: np.ndarray) -> torch.Tensor:
        """Preprocess frame for model input."""
        # Convert BGR to RGB
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            rgb = frame

        # Apply transforms
        transformed = self.transform(image=rgb)
        tensor = transformed['image'].unsqueeze(0)

        return tensor.to(self.device)

    @torch.no_grad()
    def detect(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Detect plate corners in a frame.

        Args:
            frame: BGR image (H, W, 3)

        Returns:
            corners: (5, 2) array of corner coordinates in pixel space,
                    or None if detection failed
        """
        orig_h, orig_w = frame.shape[:2]

        # Preprocess
        tensor = self.preprocess(frame)

        # Inference
        output = self.model(tensor)  # (1, 10)

        # Reshape to corners
        normalized = output.view(5, 2).cpu().numpy()

        # Scale to original image size
        corners = np.zeros_like(normalized)
        corners[:, 0] = normalized[:, 0] * orig_w
        corners[:, 1] = normalized[:, 1] * orig_h

        # Basic validation: corners should form a reasonable pentagon
        if not self._validate_corners(corners, orig_h, orig_w):
            return None

        return corners.astype(np.float32)

    def _validate_corners(
        self,
        corners: np.ndarray,
        img_h: int,
        img_w: int
    ) -> bool:
        """Validate that detected corners form a reasonable plate shape."""
        # Check all corners are within image bounds (with margin)
        margin = 0.05
        if (corners[:, 0].min() < -img_w * margin or
            corners[:, 0].max() > img_w * (1 + margin) or
            corners[:, 1].min() < -img_h * margin or
            corners[:, 1].max() > img_h * (1 + margin)):
            return False

        # Check corners are in lower portion of image (plate should be at bottom)
        centroid_y = corners[:, 1].mean()
        if centroid_y < img_h * 0.5:
            return False

        # Check polygon area is reasonable
        # Shoelace formula for polygon area
        n = len(corners)
        area = 0.5 * abs(sum(
            corners[i, 0] * corners[(i+1) % n, 1] -
            corners[(i+1) % n, 0] * corners[i, 1]
            for i in range(n)
        ))

        min_area = img_h * img_w * 0.001  # At least 0.1% of image
        max_area = img_h * img_w * 0.15   # At most 15% of image
        if area < min_area or area > max_area:
            return False

        return True

    def detect_with_confidence(
        self,
        frame: np.ndarray,
        n_augments: int = 5
    ) -> Tuple[Optional[np.ndarray], float]:
        """Detect with confidence estimation using test-time augmentation.

        Runs multiple augmented versions and checks consistency.

        Args:
            frame: BGR image
            n_augments: Number of augmented versions to test

        Returns:
            corners: (5, 2) array or None
            confidence: 0-1 score based on detection consistency
        """
        orig_h, orig_w = frame.shape[:2]

        # Run detection on original
        base_corners = self.detect(frame)
        if base_corners is None:
            return None, 0.0

        # Test-time augmentations
        all_corners = [base_corners]

        aug_transforms = [
            A.HorizontalFlip(p=1.0),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
            A.GaussianBlur(blur_limit=3, p=1.0),
        ]

        for aug in aug_transforms[:n_augments-1]:
            augmented = aug(image=frame)['image']
            corners = self.detect(augmented)

            if corners is not None:
                # Flip x coordinates back if horizontal flip was applied
                if isinstance(aug, A.HorizontalFlip):
                    corners[:, 0] = orig_w - corners[:, 0]
                all_corners.append(corners)

        if len(all_corners) < 2:
            return base_corners, 0.5

        # Compute consistency (lower variance = higher confidence)
        all_corners = np.array(all_corners)
        variance = all_corners.var(axis=0).mean()

        # Normalize variance to confidence score
        # Lower variance = higher confidence
        max_acceptable_variance = 100  # pixels squared
        confidence = max(0, 1 - variance / max_acceptable_variance)

        # Return mean of all detections
        mean_corners = all_corners.mean(axis=0)

        return mean_corners.astype(np.float32), confidence


def load_detector(model_path: Optional[Path] = None) -> Optional[PlateDetector]:
    """Load the plate detector if a trained model exists.

    Args:
        model_path: Path to model checkpoint, or None to use default

    Returns:
        PlateDetector instance, or None if no model found
    """
    if model_path is None:
        # Look for default model location
        script_dir = Path(__file__).parent
        model_path = script_dir / "checkpoints" / "best_model.pth"

    if not model_path.exists():
        return None

    return PlateDetector(model_path)


if __name__ == "__main__":
    import sys

    # Test inference
    script_dir = Path(__file__).parent
    model_path = script_dir / "checkpoints" / "best_model.pth"

    if not model_path.exists():
        print(f"No model found at {model_path}")
        print("Train a model first with: python train.py")
        sys.exit(1)

    detector = PlateDetector(model_path)

    # Test on a video frame
    data_dir = script_dir / "../../training_data"
    video_path = data_dir / "session_001" / "primary.mp4"

    if video_path.exists():
        cap = cv2.VideoCapture(str(video_path))
        ret, frame = cap.read()
        cap.release()

        if ret:
            corners = detector.detect(frame)
            if corners is not None:
                print(f"Detected corners:\n{corners}")

                # Visualize
                vis = frame.copy()
                for i, (x, y) in enumerate(corners):
                    cv2.circle(vis, (int(x), int(y)), 8, (0, 255, 0), -1)
                    cv2.putText(vis, str(i), (int(x)+10, int(y)),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                cv2.imwrite(str(script_dir / "test_detection.jpg"), vis)
                print("Saved visualization to test_detection.jpg")
            else:
                print("Detection failed")
