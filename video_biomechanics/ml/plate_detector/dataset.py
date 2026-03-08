"""Dataset for plate corner detection training."""

import cv2
import json
import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Union
import albumentations as A
from albumentations.pytorch import ToTensorV2

from annotations import AnnotationStore, PlateAnnotation


class SyntheticImageDataset(Dataset):
    """Dataset for synthetic images with corner annotations."""

    def __init__(
        self,
        annotations_file: Path,
        images_dir: Path,
        img_size: Tuple[int, int] = (224, 224),
        augment: bool = True
    ):
        """
        Args:
            annotations_file: Path to synthetic annotations JSON
            images_dir: Directory containing synthetic images
            img_size: Target image size (H, W)
            augment: Whether to apply augmentations
        """
        self.images_dir = images_dir
        self.img_size = img_size

        # Load annotations
        with open(annotations_file) as f:
            data = json.load(f)

        self.annotations = data['annotations']
        print(f"Loaded {len(self.annotations)} synthetic image annotations")

        # Create transform
        self.transform = self._create_transform(augment)

    def _create_transform(self, augment: bool) -> A.Compose:
        """Create augmentation pipeline."""
        if augment:
            return A.Compose([
                A.Resize(self.img_size[0], self.img_size[1]),
                A.HorizontalFlip(p=0.5),
                # Lighter augmentation since images already have synthetic transforms
                A.Affine(
                    translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                    scale=(0.9, 1.1),
                    rotate=(-10, 10),
                    p=0.5
                ),
                A.OneOf([
                    A.GaussianBlur(blur_limit=5),
                    A.MotionBlur(blur_limit=5),
                ], p=0.3),
                A.OneOf([
                    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3),
                    A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=30, val_shift_limit=30),
                ], p=0.5),
                A.GaussNoise(std_range=(0.01, 0.05), p=0.2),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
        else:
            return A.Compose([
                A.Resize(self.img_size[0], self.img_size[1]),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        ann = self.annotations[idx]
        image_path = self.images_dir / ann['image_path']

        # Load image
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise RuntimeError(f"Failed to load image: {image_path}")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        orig_h, orig_w = frame.shape[:2]

        # Prepare keypoints
        keypoints = [(float(x), float(y)) for x, y in ann['corners'][:5]]

        # Apply transforms
        transformed = self.transform(image=frame, keypoints=keypoints)
        image = transformed['image']
        new_keypoints = transformed['keypoints']

        # Normalize keypoints to [0, 1]
        target_h, target_w = self.img_size
        normalized = np.zeros(10, dtype=np.float32)

        for i, (x, y) in enumerate(new_keypoints[:5]):
            x = max(0, min(x, target_w))
            y = max(0, min(y, target_h))
            normalized[i * 2] = x / target_w
            normalized[i * 2 + 1] = y / target_h

        keypoints_tensor = torch.tensor(normalized, dtype=torch.float32)
        return image, keypoints_tensor


class PlateDataset(Dataset):
    """Dataset for plate corner detection.

    Loads video frames and their corner annotations.
    Applies augmentations for training.
    """

    def __init__(
        self,
        annotations_file: Path,
        data_dir: Path,
        img_size: Tuple[int, int] = (224, 224),
        augment: bool = True,
        cache_frames: bool = False
    ):
        """
        Args:
            annotations_file: Path to annotations JSON
            data_dir: Base directory containing video files
            img_size: Target image size (H, W)
            augment: Whether to apply augmentations
            cache_frames: Whether to cache frames in memory
        """
        self.data_dir = data_dir
        self.img_size = img_size
        self.augment = augment

        # Load annotations
        self.store = AnnotationStore.load(annotations_file)
        self.annotations = self.store.all_annotations()

        if len(self.annotations) == 0:
            raise ValueError(f"No annotations found in {annotations_file}")

        print(f"Loaded {len(self.annotations)} annotations")

        # Cache for video captures
        self._video_caps: Dict[str, cv2.VideoCapture] = {}

        # Frame cache
        self.cache_frames = cache_frames
        self._frame_cache: Dict[Tuple[str, int], np.ndarray] = {}

        # Augmentation pipeline
        self.transform = self._create_transform(augment)

    def _create_transform(self, augment: bool) -> A.Compose:
        """Create augmentation pipeline with aggressive transforms for better generalization."""
        if augment:
            return A.Compose([
                A.Resize(self.img_size[0], self.img_size[1]),
                A.HorizontalFlip(p=0.5),
                # Aggressive spatial augmentation for position/scale generalization
                A.Affine(
                    translate_percent={"x": (-0.3, 0.3), "y": (-0.3, 0.3)},
                    scale=(0.4, 2.0),
                    rotate=(-45, 45),
                    p=0.9
                ),
                A.OneOf([
                    A.GaussianBlur(blur_limit=5),
                    A.MotionBlur(blur_limit=5),
                ], p=0.3),
                A.OneOf([
                    A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4),
                    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=40, val_shift_limit=40),
                ], p=0.5),
                A.GaussNoise(std_range=(0.01, 0.05), p=0.2),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
        else:
            return A.Compose([
                A.Resize(self.img_size[0], self.img_size[1]),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

    def _get_frame(self, video_path: str, frame_idx: int) -> np.ndarray:
        """Load a frame from video."""
        cache_key = (video_path, frame_idx)

        if self.cache_frames and cache_key in self._frame_cache:
            return self._frame_cache[cache_key].copy()

        full_path = self.data_dir / video_path

        if video_path not in self._video_caps:
            self._video_caps[video_path] = cv2.VideoCapture(str(full_path))

        cap = self._video_caps[video_path]
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if not ret:
            raise RuntimeError(f"Failed to read frame {frame_idx} from {video_path}")

        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if self.cache_frames:
            self._frame_cache[cache_key] = frame.copy()

        return frame

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a training sample.

        Returns:
            image: Tensor of shape (3, H, W)
            keypoints: Tensor of shape (10,) with normalized coordinates
        """
        ann = self.annotations[idx]

        # Load frame
        frame = self._get_frame(ann.video_path, ann.frame_idx)
        orig_h, orig_w = frame.shape[:2]

        # Prepare keypoints for augmentation (must have exactly 5)
        keypoints = [(float(x), float(y)) for x, y in ann.corners[:5]]

        # Apply transforms
        transformed = self.transform(image=frame, keypoints=keypoints)
        image = transformed['image']
        new_keypoints = transformed['keypoints']

        # Normalize keypoints to [0, 1] and ensure exactly 10 values
        target_h, target_w = self.img_size
        normalized = np.zeros(10, dtype=np.float32)

        for i, (x, y) in enumerate(new_keypoints[:5]):
            # Clamp to valid range
            x = max(0, min(x, target_w))
            y = max(0, min(y, target_h))
            normalized[i * 2] = x / target_w
            normalized[i * 2 + 1] = y / target_h

        keypoints_tensor = torch.tensor(normalized, dtype=torch.float32)

        return image, keypoints_tensor

    def __del__(self):
        """Clean up video captures."""
        for cap in self._video_caps.values():
            cap.release()


def create_dataloaders(
    annotations_file: Path,
    data_dir: Path,
    batch_size: int = 16,
    img_size: Tuple[int, int] = (224, 224),
    val_split: float = 0.2,
    num_workers: int = 0,
    synthetic_annotations_file: Optional[Path] = None,
    synthetic_images_dir: Optional[Path] = None
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Create train and validation dataloaders.

    Args:
        annotations_file: Path to video annotations JSON
        data_dir: Base directory containing videos
        batch_size: Batch size
        img_size: Target image size
        val_split: Fraction of data for validation
        num_workers: Number of data loading workers
        synthetic_annotations_file: Optional path to synthetic annotations JSON
        synthetic_images_dir: Optional path to synthetic images directory

    Returns:
        train_loader, val_loader
    """
    # Load video dataset for validation (no augmentation)
    video_dataset = PlateDataset(
        annotations_file, data_dir, img_size, augment=False, cache_frames=True
    )

    # Use video dataset for validation
    n_total = len(video_dataset)
    n_val = max(1, int(n_total * val_split))
    val_indices = torch.randperm(n_total)[:n_val].tolist()

    val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)
    val_loader = torch.utils.data.DataLoader(
        video_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True
    )

    # Create training dataset(s)
    train_datasets = []

    # Video dataset with augmentation
    video_train_dataset = PlateDataset(
        annotations_file, data_dir, img_size, augment=True, cache_frames=True
    )
    train_indices = [i for i in range(n_total) if i not in val_indices]
    train_datasets.append(torch.utils.data.Subset(video_train_dataset, train_indices))

    # Add synthetic dataset if provided
    if synthetic_annotations_file and synthetic_images_dir:
        if synthetic_annotations_file.exists() and synthetic_images_dir.exists():
            synthetic_dataset = SyntheticImageDataset(
                synthetic_annotations_file, synthetic_images_dir, img_size, augment=True
            )
            train_datasets.append(synthetic_dataset)
            print(f"Added {len(synthetic_dataset)} synthetic samples to training")

    # Combine training datasets
    if len(train_datasets) > 1:
        combined_train = ConcatDataset(train_datasets)
    else:
        combined_train = train_datasets[0]

    print(f"Total training samples: {len(combined_train)}")
    print(f"Validation samples: {n_val}")

    train_loader = torch.utils.data.DataLoader(
        combined_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader


if __name__ == "__main__":
    # Test dataset
    import sys

    script_dir = Path(__file__).parent
    annotations_file = script_dir / "plate_annotations.json"
    data_dir = script_dir / "../../training_data"

    if not annotations_file.exists():
        print(f"No annotations file found at {annotations_file}")
        print("Run annotate.py first to create annotations")
        sys.exit(1)

    dataset = PlateDataset(annotations_file, data_dir.resolve(), augment=True)
    print(f"Dataset size: {len(dataset)}")

    # Get a sample
    image, keypoints = dataset[0]
    print(f"Image shape: {image.shape}")
    print(f"Keypoints shape: {keypoints.shape}")
    print(f"Keypoints: {keypoints}")
