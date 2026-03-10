"""Validate 3D bat coordinates using stereo triangulation.

Uses bat detector on both camera views to triangulate bat endpoints,
then compares with pose estimation wrist positions for validation.
"""

import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass

try:
    from .detect import BatDetector
except ImportError:
    from detect import BatDetector


@dataclass
class BatValidationResult:
    """Results from 3D bat validation."""
    knob_3d: np.ndarray  # (3,) Triangulated knob position
    tip_3d: np.ndarray   # (3,) Triangulated tip position
    length_3d: float     # Triangulated bat length in meters
    scale_error: float   # Relative error vs known bat length
    reprojection_error: float  # Mean reprojection error (pixels)
    confidence: float    # Overall validation confidence
    wrist_distance: Optional[float] = None  # Distance from knob to wrist (meters)

    @property
    def is_valid(self) -> bool:
        """Check if validation passed basic thresholds."""
        return (
            self.scale_error < 0.15 and  # Within 15% of expected length
            self.reprojection_error < 20.0 and  # Reasonable reprojection
            self.confidence > 0.5
        )


class BatValidator:
    """Validates 3D bat coordinates using stereo bat detection.

    Uses trained bat detector on both camera views, triangulates
    endpoints to 3D, and validates against known bat length and
    wrist positions from pose estimation.
    """

    def __init__(
        self,
        model_path: Path,
        cameras: List['CameraCalibration'],
        known_bat_length_m: float = 0.84
    ):
        """
        Args:
            model_path: Path to trained bat detector checkpoint
            cameras: List of CameraCalibration objects (from triangulation.py)
            known_bat_length_m: Expected bat length in meters
        """
        self.detector = BatDetector(model_path)
        self.cameras = cameras
        self.known_bat_length_m = known_bat_length_m

        if len(cameras) < 2:
            raise ValueError("Need at least 2 cameras for triangulation")

    def triangulate_point(
        self,
        points_2d: List[np.ndarray],
        confidences: List[float],
        min_confidence: float = 0.3
    ) -> Tuple[np.ndarray, float]:
        """Triangulate a single 3D point from multiple 2D observations.

        Args:
            points_2d: List of (2,) arrays, one per camera
            confidences: Confidence per observation
            min_confidence: Minimum confidence to use view

        Returns:
            (3D point, reprojection error in pixels)
        """
        valid_idx = [i for i, c in enumerate(confidences) if c >= min_confidence]

        if len(valid_idx) < 2:
            return np.zeros(3), float('inf')

        # Build DLT matrix (weighted least squares)
        A = []
        for i in valid_idx:
            P = self.cameras[i].P
            x, y = points_2d[i]
            w = confidences[i]

            A.append(w * (x * P[2] - P[0]))
            A.append(w * (y * P[2] - P[1]))

        A = np.array(A)

        # Solve using SVD
        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1]
        X = X[:3] / X[3]  # Homogeneous to 3D

        # Calculate reprojection error
        errors = []
        for i in valid_idx:
            projected = self.cameras[i].project(X.reshape(1, 3))[0]
            error = np.linalg.norm(projected - points_2d[i])
            errors.append(error)

        return X, float(np.mean(errors))

    def validate_from_frames(
        self,
        frames: List[np.ndarray],
        wrist_3d: Optional[np.ndarray] = None
    ) -> BatValidationResult:
        """Validate bat detection using synchronized frames from multiple cameras.

        Args:
            frames: List of BGR frames, one per camera
            wrist_3d: Optional (3,) array of wrist position from pose estimation

        Returns:
            BatValidationResult with triangulated bat and validation metrics
        """
        if len(frames) != len(self.cameras):
            raise ValueError(f"Got {len(frames)} frames but have {len(self.cameras)} cameras")

        # Detect bat in each view
        endpoints_2d = []
        confidences = []

        for frame in frames:
            endpoints, conf = self.detector.detect(frame)
            endpoints_2d.append(endpoints)
            confidences.append(conf)

        # Triangulate knob (index 0) and tip (index 1)
        knob_points = [ep[0] for ep in endpoints_2d]
        tip_points = [ep[1] for ep in endpoints_2d]

        knob_3d, knob_reproj = self.triangulate_point(knob_points, confidences)
        tip_3d, tip_reproj = self.triangulate_point(tip_points, confidences)

        # Compute 3D bat length
        length_3d = float(np.linalg.norm(tip_3d - knob_3d))

        # Scale error relative to known bat length
        scale_error = abs(length_3d - self.known_bat_length_m) / self.known_bat_length_m

        # Mean reprojection error
        reproj_error = (knob_reproj + tip_reproj) / 2

        # Wrist distance (if provided)
        wrist_distance = None
        if wrist_3d is not None:
            wrist_distance = float(np.linalg.norm(knob_3d - wrist_3d))

        # Overall confidence based on:
        # - 2D detection confidences
        # - Reprojection error
        # - Scale consistency
        mean_det_conf = np.mean(confidences)
        reproj_conf = np.clip(1.0 - reproj_error / 50.0, 0.0, 1.0)
        scale_conf = np.clip(1.0 - scale_error / 0.3, 0.0, 1.0)

        confidence = 0.4 * mean_det_conf + 0.3 * reproj_conf + 0.3 * scale_conf

        return BatValidationResult(
            knob_3d=knob_3d,
            tip_3d=tip_3d,
            length_3d=length_3d,
            scale_error=scale_error,
            reprojection_error=reproj_error,
            confidence=confidence,
            wrist_distance=wrist_distance
        )

    def validate_from_videos(
        self,
        video_paths: List[Path],
        frame_idx: Optional[int] = None,
        wrist_3d: Optional[np.ndarray] = None,
        use_best_detection: bool = True
    ) -> BatValidationResult:
        """Validate bat detection using synchronized videos.

        Args:
            video_paths: List of video paths, one per camera
            frame_idx: Specific frame to use (None = auto-select best)
            wrist_3d: Optional wrist position from pose estimation
            use_best_detection: Use find_best_detection for frame selection

        Returns:
            BatValidationResult
        """
        import cv2

        if len(video_paths) != len(self.cameras):
            raise ValueError(f"Got {len(video_paths)} videos but have {len(self.cameras)} cameras")

        # Find best frame from primary camera
        if frame_idx is None and use_best_detection:
            _, _, frame_idx = self.detector.find_best_detection(video_paths[0])
            if frame_idx < 0:
                frame_idx = 0  # Fallback to first frame

        # Extract frame from each video
        frames = []
        for video_path in video_paths:
            cap = cv2.VideoCapture(str(video_path))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            cap.release()

            if not ret:
                raise RuntimeError(f"Cannot read frame {frame_idx} from {video_path}")

            frames.append(frame)

        return self.validate_from_frames(frames, wrist_3d)

    def compute_calibration_scale(
        self,
        video_paths: List[Path],
        num_samples: int = 5
    ) -> Tuple[float, float]:
        """Compute meters-per-pixel scale using triangulated bat length.

        This validates/refines camera calibration scale using known bat length.

        Args:
            video_paths: List of video paths
            num_samples: Number of frames to sample

        Returns:
            (scale_correction_factor, confidence)
            scale_correction_factor: Multiply current scale by this to correct
        """
        import cv2

        # Sample frames and triangulate bat length
        cap = cv2.VideoCapture(str(video_paths[0]))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        # Sample from early frames (static bat)
        early_end = int(frame_count * 0.2)
        sample_indices = np.linspace(0, early_end, num_samples, dtype=int).tolist()

        lengths = []
        for idx in sample_indices:
            try:
                result = self.validate_from_videos(video_paths, frame_idx=idx)
                if result.reprojection_error < 30.0:
                    lengths.append(result.length_3d)
            except Exception:
                continue

        if len(lengths) < 2:
            return 1.0, 0.0  # Not enough data

        median_length = np.median(lengths)
        scale_correction = self.known_bat_length_m / median_length

        # Confidence based on consistency
        length_std = np.std(lengths)
        consistency = 1.0 - min(1.0, length_std / median_length)

        return float(scale_correction), float(consistency)


def validate_pose_wrist_against_bat(
    wrist_3d: np.ndarray,
    bat_knob_3d: np.ndarray,
    max_distance_m: float = 0.15
) -> Tuple[bool, float]:
    """Check if pose estimation wrist is plausibly holding the bat.

    Args:
        wrist_3d: Wrist position from pose estimation (3,)
        bat_knob_3d: Bat knob position from bat detector triangulation (3,)
        max_distance_m: Maximum acceptable distance (meters)

    Returns:
        (is_valid, distance_m)
    """
    distance = float(np.linalg.norm(wrist_3d - bat_knob_3d))
    return distance <= max_distance_m, distance


def main():
    """Demo bat validation."""
    import argparse
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from pose_estimators.triangulation import CameraCalibration

    parser = argparse.ArgumentParser(description="Bat 3D validation demo")
    parser.add_argument("primary_video", type=Path, help="Primary camera video")
    parser.add_argument("secondary_video", type=Path, help="Secondary camera video")
    parser.add_argument("--model", type=Path,
                        default=Path(__file__).parent / "checkpoints" / "best_bat_model.pth")
    parser.add_argument("--bat-length", type=float, default=0.84,
                        help="Known bat length in meters")
    parser.add_argument("--calibration", type=Path, default=None,
                        help="Camera calibration JSON file")
    args = parser.parse_args()

    # Load or create cameras
    if args.calibration and args.calibration.exists():
        import json
        with open(args.calibration) as f:
            cal_data = json.load(f)
        cameras = [CameraCalibration.from_dict(c) for c in cal_data['cameras']]
    else:
        # Default cameras at 5m distance, 90 degrees apart
        import cv2
        cap = cv2.VideoCapture(str(args.primary_video))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        cameras = [
            CameraCalibration.from_defaults(w, h, 5.0, 0),
            CameraCalibration.from_defaults(w, h, 5.0, 90)
        ]

    validator = BatValidator(
        model_path=args.model,
        cameras=cameras,
        known_bat_length_m=args.bat_length
    )

    result = validator.validate_from_videos(
        [args.primary_video, args.secondary_video]
    )

    print(f"Bat 3D Validation Results:")
    print(f"  Knob position: {result.knob_3d}")
    print(f"  Tip position:  {result.tip_3d}")
    print(f"  Length (3D):   {result.length_3d:.3f} m (expected: {args.bat_length:.2f} m)")
    print(f"  Scale error:   {result.scale_error*100:.1f}%")
    print(f"  Reproj error:  {result.reprojection_error:.1f} px")
    print(f"  Confidence:    {result.confidence:.2f}")
    print(f"  Valid:         {result.is_valid}")

    # Compute scale correction
    scale_corr, corr_conf = validator.compute_calibration_scale(
        [args.primary_video, args.secondary_video]
    )
    print(f"\nCalibration scale correction:")
    print(f"  Factor:     {scale_corr:.3f}")
    print(f"  Confidence: {corr_conf:.2f}")


if __name__ == "__main__":
    main()
