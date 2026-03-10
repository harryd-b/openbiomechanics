"""Bat detection inference module."""

import cv2
import torch
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List, Dict
import albumentations as A
from albumentations.pytorch import ToTensorV2

try:
    from .model import create_model
except ImportError:
    from model import create_model


class BatDetector:
    """Bat endpoint detector for video frames."""

    def __init__(
        self,
        model_path: Path,
        device: Optional[str] = None,
        img_size: int = 224
    ):
        """
        Args:
            model_path: Path to trained model checkpoint
            device: Device to use (auto-detect if None)
            img_size: Model input size
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.img_size = img_size

        # Load model
        checkpoint = torch.load(model_path, map_location=self.device)
        model_type = checkpoint.get('model_type', 'resnet18')

        self.model = create_model(model_type, pretrained=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()

        print(f"Loaded bat detector: {model_type}")
        print(f"  Checkpoint pixel error: {checkpoint.get('pixel_error', 'N/A'):.2f}px")

        # Preprocessing transform
        self.transform = A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

    def detect(self, frame: np.ndarray) -> Tuple[np.ndarray, float]:
        """Detect bat endpoints in a frame.

        Args:
            frame: BGR image (H, W, 3)

        Returns:
            endpoints: (2, 2) array with [[knob_x, knob_y], [tip_x, tip_y]]
            confidence: Detection confidence (based on model output)
        """
        orig_h, orig_w = frame.shape[:2]

        # Preprocess
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        transformed = self.transform(image=rgb)
        img_tensor = transformed['image'].unsqueeze(0).to(self.device)

        # Inference
        with torch.no_grad():
            output = self.model(img_tensor)  # (1, 4)

        # Convert to pixel coordinates
        output = output.cpu().numpy()[0]  # (4,)
        endpoints = output.reshape(2, 2)  # (2, 2)

        # Scale to original image size
        endpoints[:, 0] *= orig_w
        endpoints[:, 1] *= orig_h

        # Confidence based on how close predictions are to valid range
        # (simple heuristic - could be improved with confidence head)
        in_bounds = (
            (endpoints[:, 0] > 0).all() and
            (endpoints[:, 0] < orig_w).all() and
            (endpoints[:, 1] > 0).all() and
            (endpoints[:, 1] < orig_h).all()
        )
        confidence = 1.0 if in_bounds else 0.5

        return endpoints, confidence

    def detect_video(
        self,
        video_path: Path,
        frame_indices: Optional[List[int]] = None,
        sample_rate: int = 1
    ) -> Dict[int, Tuple[np.ndarray, float]]:
        """Detect bat in multiple frames of a video.

        Args:
            video_path: Path to video file
            frame_indices: Specific frames to process (None = all)
            sample_rate: Process every Nth frame if frame_indices is None

        Returns:
            Dictionary mapping frame index to (endpoints, confidence)
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if frame_indices is None:
            frame_indices = list(range(0, frame_count, sample_rate))

        results = {}

        for idx in frame_indices:
            if idx >= frame_count:
                continue

            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue

            endpoints, confidence = self.detect(frame)
            results[idx] = (endpoints, confidence)

        cap.release()
        return results

    def compute_bat_length(self, endpoints: np.ndarray) -> float:
        """Compute bat length in pixels from endpoints."""
        return float(np.linalg.norm(endpoints[1] - endpoints[0]))

    def find_best_detection(
        self,
        video_path: Path,
        num_samples: int = 20,
        min_length_pixels: float = 50,
        length_consistency_threshold: float = 0.3,
        prefer_early_frames: bool = True
    ) -> Tuple[Optional[np.ndarray], float, int]:
        """Find the best bat detection across multiple frames.

        Prioritizes early frames (stance phase) where the bat is static and
        clearly visible, then samples throughout the video for consistency.

        Args:
            video_path: Path to video file
            num_samples: Number of frames to sample
            min_length_pixels: Minimum valid bat length in pixels
            length_consistency_threshold: Max deviation from median length (fraction)
            prefer_early_frames: Weight early frames higher (bat is static)

        Returns:
            endpoints: Best detection (2, 2) or None if no valid detection
            confidence: Detection confidence (0-1)
            frame_idx: Frame index of best detection
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None, 0.0, -1

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        # Sample more heavily from early frames (stance phase has static bat)
        # First 20% of video gets half the samples, rest gets the other half
        if prefer_early_frames:
            early_end = int(frame_count * 0.2)
            n_early = num_samples // 2
            n_rest = num_samples - n_early

            early_indices = np.linspace(0, early_end, n_early, dtype=int).tolist()
            rest_indices = np.linspace(early_end, frame_count - 1, n_rest, dtype=int).tolist()
            indices = early_indices + rest_indices
        else:
            indices = np.linspace(0, frame_count - 1, num_samples, dtype=int).tolist()

        # Run detection on all sampled frames
        results = self.detect_video(video_path, indices)

        if not results:
            return None, 0.0, -1

        # Filter by minimum length
        valid_results = []
        for idx, (endpoints, conf) in results.items():
            length = self.compute_bat_length(endpoints)
            if length >= min_length_pixels:
                valid_results.append((idx, endpoints, length, conf))

        if not valid_results:
            return None, 0.0, -1

        # Compute median length for consistency check
        lengths = [r[2] for r in valid_results]
        median_length = np.median(lengths)

        # Early frame threshold (first 20% of video = stance phase with static bat)
        early_threshold = frame_count * 0.2

        # Filter by length consistency (outliers likely have wrong detections)
        consistent_results = []
        for idx, endpoints, length, conf in valid_results:
            deviation = abs(length - median_length) / median_length
            if deviation <= length_consistency_threshold:
                # Boost confidence for consistent detections
                adjusted_conf = conf * (1.0 - deviation)

                # Bonus for early frames (static bat, clearer detection)
                if prefer_early_frames and idx < early_threshold:
                    adjusted_conf *= 1.2  # 20% bonus for early frames

                consistent_results.append((idx, endpoints, length, adjusted_conf))

        if not consistent_results:
            # Fall back to all valid results if none are consistent
            consistent_results = [(idx, ep, ln, cf) for idx, ep, ln, cf in valid_results]

        # Pick the detection with highest adjusted confidence
        best = max(consistent_results, key=lambda x: x[3])
        best_idx, best_endpoints, best_length, best_conf = best

        # Final confidence based on:
        # 1. Number of consistent detections (more = better)
        # 2. Best individual confidence
        # 3. Whether best detection is from early (static) frames
        consistency_score = len(consistent_results) / len(valid_results) if valid_results else 0
        early_bonus = 0.1 if best_idx < early_threshold else 0.0
        final_confidence = min(1.0, 0.4 * best_conf + 0.4 * consistency_score + 0.2 + early_bonus)

        return best_endpoints, final_confidence, best_idx

    def estimate_scale(
        self,
        video_path: Path,
        known_bat_length_m: float,
        sample_frames: int = 20
    ) -> Tuple[float, float]:
        """Estimate pixel-to-meter scale using bat detection.

        Uses find_best_detection for robust scale estimation.

        Args:
            video_path: Path to video
            known_bat_length_m: Actual bat length in meters
            sample_frames: Number of frames to sample

        Returns:
            scale: meters per pixel
            confidence: Detection confidence
        """
        endpoints, confidence, frame_idx = self.find_best_detection(
            video_path, num_samples=sample_frames
        )

        if endpoints is None:
            return 0.0, 0.0

        length_pixels = self.compute_bat_length(endpoints)
        scale = known_bat_length_m / length_pixels

        return scale, confidence


def visualize_detection(
    frame: np.ndarray,
    endpoints: np.ndarray,
    confidence: float
) -> np.ndarray:
    """Draw bat detection on frame.

    Args:
        frame: BGR image
        endpoints: (2, 2) array
        confidence: Detection confidence

    Returns:
        Annotated frame
    """
    display = frame.copy()

    knob = tuple(endpoints[0].astype(int))
    tip = tuple(endpoints[1].astype(int))

    # Draw bat line
    color = (0, 255, 0) if confidence > 0.7 else (0, 255, 255)
    cv2.line(display, knob, tip, color, 3)

    # Draw endpoints
    cv2.circle(display, knob, 8, (0, 255, 0), -1)  # Green knob
    cv2.circle(display, tip, 8, (0, 0, 255), -1)   # Red tip

    # Labels
    cv2.putText(display, "KNOB", (knob[0] + 10, knob[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.putText(display, "TIP", (tip[0] + 10, tip[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Bat length
    length = np.linalg.norm(endpoints[1] - endpoints[0])
    mid = ((knob[0] + tip[0]) // 2, (knob[1] + tip[1]) // 2)
    cv2.putText(display, f"{length:.0f}px",
                (mid[0], mid[1] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return display


def main():
    """Demo script for bat detection."""
    import argparse

    parser = argparse.ArgumentParser(description="Bat detection demo")
    parser.add_argument("video", type=Path, help="Path to video file")
    parser.add_argument("--model", type=Path,
                        default=Path(__file__).parent / "checkpoints" / "best_bat_model.pth",
                        help="Path to model checkpoint")
    parser.add_argument("--bat-length", type=float, default=0.84,
                        help="Known bat length in meters")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output video path")
    args = parser.parse_args()

    if not args.model.exists():
        print(f"Model not found: {args.model}")
        print("Train the model first using train.py")
        return

    detector = BatDetector(args.model)

    # Process video
    cap = cv2.VideoCapture(str(args.video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Processing {args.video.name}: {frame_count} frames at {fps:.1f}fps")

    # Output video
    out = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(args.output), fourcc, fps, (width, height))

    # Estimate scale
    scale, conf = detector.estimate_scale(args.video, args.bat_length)
    print(f"Estimated scale: {scale*1000:.3f} mm/pixel (confidence: {conf:.2f})")

    # Process each frame
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        endpoints, confidence = detector.detect(frame)
        display = visualize_detection(frame, endpoints, confidence)

        # Add scale info
        cv2.putText(display, f"Scale: {scale*1000:.2f} mm/px",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if out:
            out.write(display)

        # Show progress
        if frame_idx % 100 == 0:
            print(f"  Frame {frame_idx}/{frame_count}")

        frame_idx += 1

    cap.release()
    if out:
        out.release()
        print(f"Saved output to: {args.output}")


if __name__ == "__main__":
    main()
