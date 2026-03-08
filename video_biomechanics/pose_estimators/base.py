"""Base class for 3D pose estimators."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np


@dataclass
class Pose3DResult:
    """Result from a 3D pose estimator."""

    # 3D joint positions (17, 3) in H36M format
    joints_3d: np.ndarray

    # Per-joint confidence scores (17,)
    confidences: np.ndarray

    # Frame number
    frame_number: int

    # Timestamp in seconds
    timestamp: float

    # Optional: 2D keypoints used (17, 2)
    joints_2d: Optional[np.ndarray] = None

    # Optional: method-specific metadata
    metadata: Optional[dict] = None


# H36M skeleton joint indices
H36M_JOINTS = {
    'pelvis': 0,
    'right_hip': 1,
    'right_knee': 2,
    'right_ankle': 3,
    'left_hip': 4,
    'left_knee': 5,
    'left_ankle': 6,
    'spine': 7,
    'neck': 8,
    'head': 9,
    'head_top': 10,
    'left_shoulder': 11,
    'left_elbow': 12,
    'left_wrist': 13,
    'right_shoulder': 14,
    'right_elbow': 15,
    'right_wrist': 16,
}

# Bone connections for skeleton visualization and constraints
H36M_BONES = [
    ('pelvis', 'right_hip'),
    ('right_hip', 'right_knee'),
    ('right_knee', 'right_ankle'),
    ('pelvis', 'left_hip'),
    ('left_hip', 'left_knee'),
    ('left_knee', 'left_ankle'),
    ('pelvis', 'spine'),
    ('spine', 'neck'),
    ('neck', 'head'),
    ('head', 'head_top'),
    ('neck', 'left_shoulder'),
    ('left_shoulder', 'left_elbow'),
    ('left_elbow', 'left_wrist'),
    ('neck', 'right_shoulder'),
    ('right_shoulder', 'right_elbow'),
    ('right_elbow', 'right_wrist'),
]


class PoseEstimator3D(ABC):
    """Abstract base class for 3D pose estimators."""

    def __init__(self, name: str):
        """
        Initialize estimator.

        Args:
            name: Identifier for this estimator method
        """
        self.name = name
        self._is_initialized = False

    @abstractmethod
    def initialize(self) -> None:
        """Load models and prepare for inference."""
        pass

    @abstractmethod
    def estimate_frame(self,
                       frame: np.ndarray,
                       frame_number: int,
                       timestamp: float) -> Optional[Pose3DResult]:
        """
        Estimate 3D pose from a single frame.

        Args:
            frame: BGR image (H, W, 3)
            frame_number: Frame index
            timestamp: Time in seconds

        Returns:
            Pose3DResult or None if detection failed
        """
        pass

    def estimate_video(self,
                       video_path: str,
                       max_frames: Optional[int] = None) -> List[Pose3DResult]:
        """
        Estimate 3D poses from a video file.

        Args:
            video_path: Path to video file
            max_frames: Maximum frames to process (None = all)

        Returns:
            List of Pose3DResult, one per frame
        """
        import cv2

        if not self._is_initialized:
            self.initialize()

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if max_frames:
            total_frames = min(total_frames, max_frames)

        results = []
        frame_num = 0

        while frame_num < total_frames:
            ret, frame = cap.read()
            if not ret:
                break

            timestamp = frame_num / fps
            result = self.estimate_frame(frame, frame_num, timestamp)

            if result is not None:
                results.append(result)

            frame_num += 1

        cap.release()

        # Post-process sequence if needed
        results = self.post_process_sequence(results)

        return results

    def estimate_video_multiview(self,
                                 video_paths: List[str],
                                 max_frames: Optional[int] = None) -> List[Pose3DResult]:
        """
        Estimate 3D poses from multiple synchronized video views.

        Default implementation uses only the first video.
        Subclasses can override to use multiple views.

        Args:
            video_paths: List of paths to synchronized videos
            max_frames: Maximum frames to process

        Returns:
            List of Pose3DResult
        """
        # Default: use first video only
        return self.estimate_video(video_paths[0], max_frames)

    def post_process_sequence(self,
                              results: List[Pose3DResult]) -> List[Pose3DResult]:
        """
        Post-process a sequence of poses.

        Override in subclasses for temporal smoothing, etc.

        Args:
            results: List of per-frame results

        Returns:
            Post-processed results
        """
        return results

    def get_confidence_weights(self) -> np.ndarray:
        """
        Get per-joint reliability weights for this method.

        Some methods are better at certain joints. This returns
        a (17,) array of weights indicating relative reliability.

        Default: uniform weights.
        """
        return np.ones(17) / 17

    @staticmethod
    def normalize_skeleton(joints_3d: np.ndarray) -> np.ndarray:
        """
        Normalize skeleton to standard scale (meters).

        Centers on pelvis and scales based on torso length.

        Args:
            joints_3d: (17, 3) joint positions

        Returns:
            Normalized (17, 3) positions
        """
        # Center on pelvis
        pelvis = joints_3d[H36M_JOINTS['pelvis']]
        centered = joints_3d - pelvis

        # Calculate torso length
        hip_center = (joints_3d[H36M_JOINTS['left_hip']] +
                      joints_3d[H36M_JOINTS['right_hip']]) / 2
        shoulder_center = (joints_3d[H36M_JOINTS['left_shoulder']] +
                           joints_3d[H36M_JOINTS['right_shoulder']]) / 2
        torso_length = np.linalg.norm(shoulder_center - hip_center)

        # Expected torso length in meters
        EXPECTED_TORSO = 0.50

        if torso_length > 1e-6:
            scale = EXPECTED_TORSO / torso_length
            centered = centered * scale

        return centered
