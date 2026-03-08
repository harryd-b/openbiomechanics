"""
YOLOv8 2D pose estimation + VideoPose3D lifting.

This is the original/default method:
1. YOLOv8-pose extracts 2D keypoints
2. VideoPose3D lifts 2D to 3D using learned priors
"""

import numpy as np
from typing import List, Optional
import cv2

from .base import PoseEstimator3D, Pose3DResult, H36M_JOINTS


class YOLOLiftingEstimator(PoseEstimator3D):
    """
    YOLOv8 2D detection + VideoPose3D 3D lifting.

    Fast and works with single camera, but depth is estimated.
    """

    def __init__(self,
                 yolo_model: str = 'yolov8m-pose.pt',
                 lifting_model_path: Optional[str] = None):
        """
        Initialize estimator.

        Args:
            yolo_model: YOLOv8 pose model name
            lifting_model_path: Path to VideoPose3D weights (None = use default)
        """
        super().__init__(name='yolo_lifting')
        self.yolo_model_name = yolo_model
        self.lifting_model_path = lifting_model_path
        self.pose_estimator = None
        self.lifter = None

    def initialize(self) -> None:
        """Load YOLO and lifting models."""
        from pose_estimation import PoseEstimator
        from lifting_3d import VideoPose3DLifter

        self.pose_estimator = PoseEstimator(model_name=self.yolo_model_name)
        self.lifter = VideoPose3DLifter(model_path=self.lifting_model_path)
        self._is_initialized = True

    def estimate_frame(self,
                       frame: np.ndarray,
                       frame_number: int,
                       timestamp: float) -> Optional[Pose3DResult]:
        """Estimate 3D pose from single frame."""
        if not self._is_initialized:
            self.initialize()

        # Get 2D keypoints
        pose_frame = self.pose_estimator.process_frame(frame, frame_number, timestamp)

        if pose_frame is None or pose_frame.keypoints is None:
            return None

        keypoints_2d = pose_frame.keypoints  # (17, 3) with confidence

        # Lift to 3D
        pose_3d = self.lifter.lift_single_frame(keypoints_2d[:, :2])

        if pose_3d is None:
            return None

        # Extract confidences from 2D detection
        confidences = keypoints_2d[:, 2] if keypoints_2d.shape[1] > 2 else np.ones(17)

        # Normalize skeleton
        joints_3d = self.normalize_skeleton(pose_3d.joints_3d)

        return Pose3DResult(
            joints_3d=joints_3d,
            confidences=confidences,
            frame_number=frame_number,
            timestamp=timestamp,
            joints_2d=keypoints_2d[:, :2],
            metadata={'method': 'yolo_lifting'}
        )

    def estimate_video(self,
                       video_path: str,
                       max_frames: Optional[int] = None) -> List[Pose3DResult]:
        """
        Estimate 3D poses from video using temporal lifting.

        Uses the full sequence for better temporal consistency.
        """
        if not self._is_initialized:
            self.initialize()

        # Process with YOLO first
        poses_2d = self.pose_estimator.process_video(video_path, max_frames=max_frames)

        if not poses_2d:
            return []

        # Extract keypoints and timestamps
        keypoints_list = [p.keypoints for p in poses_2d]
        timestamps = [p.timestamp for p in poses_2d]

        # Lift entire sequence (uses temporal context)
        poses_3d = self.lifter.lift_sequence(
            [kp[:, :2] for kp in keypoints_list],
            timestamps
        )

        # Convert to Pose3DResult
        results = []
        for i, pose_3d in enumerate(poses_3d):
            confidences = keypoints_list[i][:, 2] if keypoints_list[i].shape[1] > 2 else np.ones(17)

            joints_3d = self.normalize_skeleton(pose_3d.joints_3d)

            results.append(Pose3DResult(
                joints_3d=joints_3d,
                confidences=confidences,
                frame_number=i,
                timestamp=timestamps[i],
                joints_2d=keypoints_list[i][:, :2],
                metadata={'method': 'yolo_lifting'}
            ))

        return results

    def get_confidence_weights(self) -> np.ndarray:
        """
        Get per-joint reliability weights.

        YOLO+Lifting is generally good for large joints but
        less reliable for extremities and depth estimation.
        """
        weights = np.ones(17)

        # Higher confidence for torso/large joints
        weights[H36M_JOINTS['pelvis']] = 1.2
        weights[H36M_JOINTS['spine']] = 1.1
        weights[H36M_JOINTS['neck']] = 1.1

        # Lower confidence for extremities
        weights[H36M_JOINTS['left_wrist']] = 0.8
        weights[H36M_JOINTS['right_wrist']] = 0.8
        weights[H36M_JOINTS['left_ankle']] = 0.9
        weights[H36M_JOINTS['right_ankle']] = 0.9
        weights[H36M_JOINTS['head_top']] = 0.7

        return weights / weights.sum()
