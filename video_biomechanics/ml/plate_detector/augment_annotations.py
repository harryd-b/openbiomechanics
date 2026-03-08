"""Generate augmented training data by transforming existing annotations.

Creates synthetic samples by extracting frames, applying affine transforms,
and saving as image files with corresponding annotations.
"""

import cv2
import numpy as np
from pathlib import Path
import json
from typing import List, Tuple, Dict
import random
import shutil

from annotations import AnnotationStore, PlateAnnotation


def get_affine_transform_matrix(
    img_size: Tuple[int, int],
    scale: float,
    rotation_deg: float,
    translate: Tuple[float, float]
) -> np.ndarray:
    """Build 2x3 affine transformation matrix.

    Args:
        img_size: (height, width) of image
        scale: Scale factor (1.0 = no change)
        rotation_deg: Rotation in degrees (counter-clockwise)
        translate: (tx, ty) translation as fraction of image size

    Returns:
        2x3 affine transformation matrix
    """
    h, w = img_size
    cx, cy = w / 2, h / 2

    angle_rad = np.radians(rotation_deg)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

    tx = translate[0] * w
    ty = translate[1] * h

    # Build full transformation: translate to center, scale, rotate, translate back + offset
    # M = T_back * R * S * T_center
    # where T_center translates center to origin, S scales, R rotates, T_back translates back + offset

    # Combined matrix:
    # [cos*s  -sin*s  cx*(1-cos*s) + cy*sin*s + tx]
    # [sin*s   cos*s  cy*(1-cos*s) - cx*sin*s + ty]

    M = np.array([
        [cos_a * scale, -sin_a * scale, cx * (1 - cos_a * scale) + cy * sin_a * scale + tx],
        [sin_a * scale,  cos_a * scale, cy * (1 - cos_a * scale) - cx * sin_a * scale + ty]
    ], dtype=np.float32)

    return M


def transform_corners(corners: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Apply affine transform matrix to corner coordinates.

    Args:
        corners: (N, 2) array of corner coordinates
        M: 2x3 affine transformation matrix

    Returns:
        Transformed corners (N, 2)
    """
    # Convert to homogeneous coordinates
    ones = np.ones((corners.shape[0], 1), dtype=np.float32)
    corners_h = np.hstack([corners, ones])  # (N, 3)

    # Apply transform
    transformed = corners_h @ M.T  # (N, 2)

    return transformed


def is_valid_corners(corners: np.ndarray, img_size: Tuple[int, int], margin: float = 0.05) -> bool:
    """Check if corners are within valid image bounds."""
    h, w = img_size

    # Check bounds with margin
    if (corners[:, 0].min() < -w * margin or
        corners[:, 0].max() > w * (1 + margin) or
        corners[:, 1].min() < -h * margin or
        corners[:, 1].max() > h * (1 + margin)):
        return False

    # Check area (not too small or too large)
    n = len(corners)
    area = 0.5 * abs(sum(
        corners[i, 0] * corners[(i+1) % n, 1] -
        corners[(i+1) % n, 0] * corners[i, 1]
        for i in range(n)
    ))

    min_area = h * w * 0.001   # 0.1% of image
    max_area = h * w * 0.20    # 20% of image

    return min_area <= area <= max_area


def generate_synthetic_images(
    store: AnnotationStore,
    data_dir: Path,
    output_dir: Path,
    num_augments_per_video: int = 20,
    scale_range: Tuple[float, float] = (0.5, 1.8),
    rotation_range: Tuple[float, float] = (-30, 30),
    translate_range: Tuple[float, float] = (-0.25, 0.25),
    balance_video_types: bool = True
) -> Dict[str, List[Tuple[int, int]]]:
    """Generate synthetic training images with transformed corners.

    Args:
        store: Original annotation store
        data_dir: Base data directory containing videos
        output_dir: Directory to save synthetic images
        num_augments_per_video: Number of augmented versions per unique video
        scale_range: (min_scale, max_scale)
        rotation_range: (min_deg, max_deg)
        translate_range: (min_frac, max_frac) as fraction of image size
        balance_video_types: If True, generate more samples for underrepresented types

    Returns:
        Dictionary mapping image filename to corner coordinates
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get one annotation per unique video
    unique_annotations = {}
    for ann in store.all_annotations():
        if ann.video_path not in unique_annotations:
            unique_annotations[ann.video_path] = ann

    # Analyze distribution for balancing
    primary_videos = [p for p in unique_annotations.keys() if 'primary' in p]
    secondary_videos = [p for p in unique_annotations.keys() if 'secondary' in p]

    print(f"Found {len(primary_videos)} primary videos, {len(secondary_videos)} secondary videos")

    # Calculate augments per video type to balance
    if balance_video_types and len(primary_videos) > 0 and len(secondary_videos) > 0:
        # We want equal total samples from each type
        target_total = num_augments_per_video * max(len(primary_videos), len(secondary_videos))
        primary_augments = target_total // len(primary_videos)
        secondary_augments = target_total // len(secondary_videos)
        print(f"Balancing: {primary_augments} augments per primary, {secondary_augments} per secondary")
    else:
        primary_augments = secondary_augments = num_augments_per_video

    annotations_map = {}
    total_generated = 0

    for video_path, ann in unique_annotations.items():
        is_primary = 'primary' in video_path
        target_augments = primary_augments if is_primary else secondary_augments

        # Load video and extract frame
        full_path = data_dir / video_path
        if not full_path.exists():
            print(f"Warning: Video not found: {full_path}")
            continue

        cap = cv2.VideoCapture(str(full_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, ann.frame_idx)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            print(f"Warning: Could not read frame {ann.frame_idx} from {video_path}")
            continue

        h, w = frame.shape[:2]
        img_size = (h, w)
        corners = np.array(ann.corners, dtype=np.float32)

        # Generate base name for this video
        video_name = video_path.replace('\\', '_').replace('/', '_').replace('.mp4', '')

        # Save original frame
        orig_filename = f"{video_name}_orig.jpg"
        cv2.imwrite(str(output_dir / orig_filename), frame)
        annotations_map[orig_filename] = [(int(x), int(y)) for x, y in corners]

        # Generate augmented versions
        aug_count = 0
        attempts = 0
        max_attempts = target_augments * 5

        while aug_count < target_augments and attempts < max_attempts:
            attempts += 1

            # Random transform parameters
            scale = random.uniform(*scale_range)
            rotation = random.uniform(*rotation_range)
            tx = random.uniform(*translate_range)
            ty = random.uniform(*translate_range)

            # Get transformation matrix
            M = get_affine_transform_matrix(img_size, scale, rotation, (tx, ty))

            # Transform corners
            new_corners = transform_corners(corners, M)

            # Validate transformed corners
            if not is_valid_corners(new_corners, img_size):
                continue

            # Apply transform to image
            transformed_frame = cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

            # Save transformed image
            aug_filename = f"{video_name}_aug{aug_count:03d}.jpg"
            cv2.imwrite(str(output_dir / aug_filename), transformed_frame)

            # Clamp corners to image bounds
            clamped_corners = np.clip(new_corners, 0, [w-1, h-1])
            annotations_map[aug_filename] = [(int(x), int(y)) for x, y in clamped_corners]

            aug_count += 1
            total_generated += 1

        print(f"  {video_path}: generated {aug_count} augmented images")

    print(f"\nTotal synthetic images generated: {total_generated + len(unique_annotations)}")
    return annotations_map


def main():
    script_dir = Path(__file__).parent
    data_dir = (script_dir / "../../training_data").resolve()
    output_dir = script_dir / "synthetic_images"

    # Clean previous synthetic data
    if output_dir.exists():
        shutil.rmtree(output_dir)

    # Load original annotations
    orig_file = script_dir / "plate_annotations.json"
    store = AnnotationStore.load(orig_file)
    print(f"Loaded {len(store)} original annotations")

    # Generate synthetic images
    annotations_map = generate_synthetic_images(
        store,
        data_dir,
        output_dir,
        num_augments_per_video=30,     # 30 augmented versions per video
        scale_range=(0.5, 1.8),         # Moderate scale range
        rotation_range=(-35, 35),       # Rotation range
        translate_range=(-0.25, 0.25),  # Translation range
        balance_video_types=True        # Balance primary/secondary
    )

    # Save synthetic annotations
    synthetic_annotations = {
        "version": "1.0",
        "type": "synthetic_images",
        "annotations": [
            {"image_path": img_path, "corners": corners}
            for img_path, corners in annotations_map.items()
        ]
    }

    ann_file = script_dir / "synthetic_annotations.json"
    with open(ann_file, 'w') as f:
        json.dump(synthetic_annotations, f, indent=2)

    print(f"\nSaved {len(annotations_map)} annotations to {ann_file}")

    # Print distribution stats
    primary_count = sum(1 for k in annotations_map if 'primary' in k)
    secondary_count = sum(1 for k in annotations_map if 'secondary' in k)
    print(f"\nFinal distribution:")
    print(f"  Primary images: {primary_count}")
    print(f"  Secondary images: {secondary_count}")


if __name__ == "__main__":
    main()
