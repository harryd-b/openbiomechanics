"""Generate augmented training data for bat detection.

Creates synthetic samples by:
1. Extracting frames with bat annotations
2. Applying affine transforms (scale, rotation, translation)
3. Adding various augmentations (blur, noise, brightness)
4. Generating synthetic bat overlays on background frames

This significantly expands the training set for better generalization.
"""

import cv2
import numpy as np
from pathlib import Path
import json
from typing import List, Tuple, Dict
import random
import shutil

try:
    from .annotations import BatAnnotationStore, BatAnnotation
except ImportError:
    from annotations import BatAnnotationStore, BatAnnotation


def get_affine_transform_matrix(
    img_size: Tuple[int, int],
    scale: float,
    rotation_deg: float,
    translate: Tuple[float, float]
) -> np.ndarray:
    """Build 2x3 affine transformation matrix."""
    h, w = img_size
    cx, cy = w / 2, h / 2

    angle_rad = np.radians(rotation_deg)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

    tx = translate[0] * w
    ty = translate[1] * h

    M = np.array([
        [cos_a * scale, -sin_a * scale, cx * (1 - cos_a * scale) + cy * sin_a * scale + tx],
        [sin_a * scale,  cos_a * scale, cy * (1 - cos_a * scale) - cx * sin_a * scale + ty]
    ], dtype=np.float32)

    return M


def transform_endpoints(endpoints: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Apply affine transform to endpoints."""
    ones = np.ones((endpoints.shape[0], 1), dtype=np.float32)
    endpoints_h = np.hstack([endpoints, ones])
    return endpoints_h @ M.T


def is_valid_endpoints(endpoints: np.ndarray, img_size: Tuple[int, int], margin: float = 0.1) -> bool:
    """Check if endpoints are within valid image bounds."""
    h, w = img_size

    # Check bounds with margin
    if (endpoints[:, 0].min() < -w * margin or
        endpoints[:, 0].max() > w * (1 + margin) or
        endpoints[:, 1].min() < -h * margin or
        endpoints[:, 1].max() > h * (1 + margin)):
        return False

    # Check bat length (not too small or too large)
    length = np.linalg.norm(endpoints[1] - endpoints[0])
    min_length = min(h, w) * 0.05  # At least 5% of smaller dimension
    max_length = max(h, w) * 0.9   # At most 90% of larger dimension

    return min_length <= length <= max_length


def generate_synthetic_bat_line(
    img_size: Tuple[int, int],
    bat_color: Tuple[int, int, int] = (139, 90, 43),  # Brown
    thickness_range: Tuple[int, int] = (3, 8)
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a synthetic bat line on a black background.

    Returns:
        (image, endpoints) where endpoints is (2, 2) array
    """
    h, w = img_size

    # Random bat position
    cx, cy = random.uniform(0.2, 0.8) * w, random.uniform(0.2, 0.8) * h
    length = random.uniform(0.2, 0.5) * min(h, w)
    angle = random.uniform(0, 2 * np.pi)

    # Calculate endpoints
    dx, dy = np.cos(angle) * length / 2, np.sin(angle) * length / 2
    knob = (int(cx - dx), int(cy - dy))
    tip = (int(cx + dx), int(cy + dy))

    # Draw bat
    img = np.zeros((h, w, 3), dtype=np.uint8)
    thickness = random.randint(*thickness_range)

    # Vary color slightly
    color_var = [c + random.randint(-20, 20) for c in bat_color]
    color_var = tuple(max(0, min(255, c)) for c in color_var)

    cv2.line(img, knob, tip, color_var, thickness)

    # Add slight taper (bat is thicker at barrel)
    barrel_thickness = thickness + random.randint(1, 3)
    mid_point = ((knob[0] + tip[0]) // 2, (knob[1] + tip[1]) // 2)
    cv2.line(img, mid_point, tip, color_var, barrel_thickness)

    endpoints = np.array([knob, tip], dtype=np.float32)

    return img, endpoints


def blend_bat_onto_frame(
    frame: np.ndarray,
    bat_img: np.ndarray,
    endpoints: np.ndarray,
    alpha: float = 0.9
) -> Tuple[np.ndarray, np.ndarray]:
    """Blend a synthetic bat onto a frame.

    Returns:
        (blended_image, transformed_endpoints)
    """
    h, w = frame.shape[:2]
    bat_h, bat_w = bat_img.shape[:2]

    # Random transform for the bat
    scale = random.uniform(0.5, 1.5)
    rotation = random.uniform(-180, 180)
    tx = random.uniform(-0.3, 0.3)
    ty = random.uniform(-0.3, 0.3)

    M = get_affine_transform_matrix((bat_h, bat_w), scale, rotation, (tx, ty))

    # Transform bat image
    bat_transformed = cv2.warpAffine(bat_img, M, (w, h))

    # Transform endpoints
    new_endpoints = transform_endpoints(endpoints, M)

    # Check validity
    if not is_valid_endpoints(new_endpoints, (h, w)):
        return None, None

    # Create mask from bat image (non-black pixels)
    mask = np.any(bat_transformed > 10, axis=2).astype(np.float32)
    mask = cv2.GaussianBlur(mask, (3, 3), 0)
    mask = np.stack([mask] * 3, axis=2)

    # Blend
    blended = frame * (1 - mask * alpha) + bat_transformed * mask * alpha
    blended = blended.astype(np.uint8)

    return blended, new_endpoints


def generate_synthetic_images(
    store: BatAnnotationStore,
    data_dir: Path,
    output_dir: Path,
    num_augments_per_annotation: int = 15,
    num_synthetic_per_video: int = 10,
    scale_range: Tuple[float, float] = (0.6, 1.6),
    rotation_range: Tuple[float, float] = (-45, 45),
    translate_range: Tuple[float, float] = (-0.2, 0.2)
) -> Dict[str, List[Tuple[int, int]]]:
    """Generate synthetic training images.

    Args:
        store: Original annotation store
        data_dir: Base data directory
        output_dir: Directory to save synthetic images
        num_augments_per_annotation: Augmented versions per real annotation
        num_synthetic_per_video: Fully synthetic bat overlays per video
        scale_range: Scale factor range
        rotation_range: Rotation in degrees
        translate_range: Translation as fraction of image

    Returns:
        Dictionary mapping filename to endpoints list
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get visible annotations only
    visible_annotations = store.visible_annotations()
    print(f"Processing {len(visible_annotations)} visible bat annotations")

    annotations_map = {}
    total_generated = 0

    # Group annotations by video
    video_annotations: Dict[str, List[BatAnnotation]] = {}
    for ann in visible_annotations:
        if ann.video_path not in video_annotations:
            video_annotations[ann.video_path] = []
        video_annotations[ann.video_path].append(ann)

    for video_path, anns in video_annotations.items():
        full_path = data_dir / video_path
        if not full_path.exists():
            print(f"Warning: Video not found: {full_path}")
            continue

        cap = cv2.VideoCapture(str(full_path))

        for ann in anns:
            cap.set(cv2.CAP_PROP_POS_FRAMES, ann.frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue

            h, w = frame.shape[:2]
            endpoints = np.array(ann.endpoints, dtype=np.float32)

            # Generate video name prefix
            video_name = video_path.replace('\\', '_').replace('/', '_').replace('.mp4', '')
            base_name = f"{video_name}_f{ann.frame_idx:04d}"

            # Save original
            orig_filename = f"{base_name}_orig.jpg"
            cv2.imwrite(str(output_dir / orig_filename), frame)
            annotations_map[orig_filename] = [tuple(map(int, ep)) for ep in endpoints]

            # Generate augmented versions
            aug_count = 0
            attempts = 0
            max_attempts = num_augments_per_annotation * 5

            while aug_count < num_augments_per_annotation and attempts < max_attempts:
                attempts += 1

                # Random transform
                scale = random.uniform(*scale_range)
                rotation = random.uniform(*rotation_range)
                tx = random.uniform(*translate_range)
                ty = random.uniform(*translate_range)

                M = get_affine_transform_matrix((h, w), scale, rotation, (tx, ty))
                new_endpoints = transform_endpoints(endpoints, M)

                if not is_valid_endpoints(new_endpoints, (h, w)):
                    continue

                # Apply transform to image
                transformed = cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

                # Apply additional augmentations
                if random.random() < 0.3:
                    # Gaussian blur
                    ksize = random.choice([3, 5, 7])
                    transformed = cv2.GaussianBlur(transformed, (ksize, ksize), 0)

                if random.random() < 0.3:
                    # Brightness/contrast
                    alpha = random.uniform(0.7, 1.3)
                    beta = random.randint(-30, 30)
                    transformed = cv2.convertScaleAbs(transformed, alpha=alpha, beta=beta)

                if random.random() < 0.2:
                    # Add noise
                    noise = np.random.normal(0, 15, transformed.shape).astype(np.int16)
                    transformed = np.clip(transformed.astype(np.int16) + noise, 0, 255).astype(np.uint8)

                # Save
                aug_filename = f"{base_name}_aug{aug_count:03d}.jpg"
                cv2.imwrite(str(output_dir / aug_filename), transformed)

                clamped = np.clip(new_endpoints, 0, [w-1, h-1])
                annotations_map[aug_filename] = [tuple(map(int, ep)) for ep in clamped]

                aug_count += 1
                total_generated += 1

            print(f"  {video_path} frame {ann.frame_idx}: {aug_count} augmented images")

        # Generate fully synthetic overlays using frames without annotations
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        bg_frames = []
        annotated_frames = {a.frame_idx for a in anns}

        # Sample background frames
        for _ in range(min(num_synthetic_per_video * 2, frame_count)):
            frame_idx = random.randint(0, frame_count - 1)
            if frame_idx in annotated_frames:
                continue

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                bg_frames.append((frame_idx, frame))

            if len(bg_frames) >= num_synthetic_per_video:
                break

        # Generate synthetic bats on background frames
        for idx, (frame_idx, bg_frame) in enumerate(bg_frames):
            h, w = bg_frame.shape[:2]

            # Generate synthetic bat
            bat_img, bat_endpoints = generate_synthetic_bat_line((h, w))

            # Blend onto background
            blended, new_endpoints = blend_bat_onto_frame(bg_frame, bat_img, bat_endpoints)
            if blended is None:
                continue

            video_name = video_path.replace('\\', '_').replace('/', '_').replace('.mp4', '')
            synth_filename = f"{video_name}_synth{idx:03d}.jpg"
            cv2.imwrite(str(output_dir / synth_filename), blended)

            annotations_map[synth_filename] = [tuple(map(int, ep)) for ep in new_endpoints]
            total_generated += 1

        cap.release()

    print(f"\nTotal synthetic images generated: {total_generated}")
    return annotations_map


def main():
    script_dir = Path(__file__).parent
    data_dir = (script_dir / "../../training_data").resolve()
    output_dir = script_dir / "synthetic_images"

    # Clean previous synthetic data
    if output_dir.exists():
        shutil.rmtree(output_dir)

    # Load original annotations
    orig_file = script_dir / "bat_annotations.json"
    if not orig_file.exists():
        print(f"Error: Annotations file not found: {orig_file}")
        print("Run 'python annotate.py' first to create annotations")
        return

    store = BatAnnotationStore.load(orig_file)
    print(f"Loaded {len(store)} original annotations")
    print(f"  Visible: {len(store.visible_annotations())}")

    # Generate synthetic images
    annotations_map = generate_synthetic_images(
        store,
        data_dir,
        output_dir,
        num_augments_per_annotation=20,
        num_synthetic_per_video=10,
        scale_range=(0.6, 1.6),
        rotation_range=(-45, 45),
        translate_range=(-0.2, 0.2)
    )

    # Save synthetic annotations
    synthetic_annotations = {
        "version": "1.0",
        "type": "synthetic_bat_images",
        "annotations": [
            {"image_path": img_path, "endpoints": endpoints}
            for img_path, endpoints in annotations_map.items()
        ]
    }

    ann_file = script_dir / "synthetic_bat_annotations.json"
    with open(ann_file, 'w') as f:
        json.dump(synthetic_annotations, f, indent=2)

    print(f"\nSaved {len(annotations_map)} annotations to {ann_file}")


if __name__ == "__main__":
    main()
