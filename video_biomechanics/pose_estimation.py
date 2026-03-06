"""
Pose estimation module - adapted from OpenBiomechanics computer_vision code.
Uses YOLOv8 for 2D pose estimation from video.
"""

import cv2
import numpy as np
from ultralytics import YOLO
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class PoseFrame:
    """Container for pose data from a single frame."""
    frame_number: int
    timestamp: float
    keypoints: np.ndarray  # Shape: (17, 3) for YOLOv8 - x, y, confidence
    bbox: Optional[np.ndarray] = None


# YOLOv8 pose keypoint indices
class Keypoints:
    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16


class PoseEstimator:
    """Extract 2D pose keypoints from video using YOLOv8."""

    def __init__(self, model_name: str = 'yolov8n-pose.pt'):
        """
        Initialize the pose estimator.

        Args:
            model_name: YOLOv8 pose model to use. Options:
                - 'yolov8n-pose.pt' (fastest, least accurate)
                - 'yolov8s-pose.pt'
                - 'yolov8m-pose.pt'
                - 'yolov8l-pose.pt'
                - 'yolov8x-pose.pt' (slowest, most accurate)
        """
        self.model = YOLO(model_name)

    def process_video(self, video_path: str,
                      max_frames: Optional[int] = None,
                      skip_frames: int = 0) -> List[PoseFrame]:
        """
        Process a video and extract pose keypoints for each frame.

        Args:
            video_path: Path to input video file
            max_frames: Maximum number of frames to process (None = all)
            skip_frames: Number of frames to skip between processed frames

        Returns:
            List of PoseFrame objects
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        poses = []
        frame_count = 0
        processed_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if max_frames and processed_count >= max_frames:
                break

            if frame_count % (skip_frames + 1) == 0:
                pose_frame = self._process_frame(frame, frame_count, fps)
                if pose_frame:
                    poses.append(pose_frame)
                processed_count += 1

            frame_count += 1

        cap.release()
        print(f"Processed {processed_count} frames, detected pose in {len(poses)} frames")
        return poses

    def _process_frame(self, frame: np.ndarray,
                       frame_number: int,
                       fps: float) -> Optional[PoseFrame]:
        """Process a single frame and extract pose."""
        results = self.model.predict(frame, verbose=False)

        if len(results) == 0 or results[0].keypoints is None:
            return None

        # Get the first detected person (could extend to handle multiple)
        keypoints = results[0].keypoints.data.cpu().numpy()

        if len(keypoints) == 0:
            return None

        # Take first person detected
        kp = keypoints[0]  # Shape: (17, 3) - x, y, confidence

        bbox = None
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            bbox = results[0].boxes.xyxy[0].cpu().numpy()

        return PoseFrame(
            frame_number=frame_number,
            timestamp=frame_number / fps,
            keypoints=kp,
            bbox=bbox
        )

    def process_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Process a single frame (for real-time use).

        Returns:
            Keypoints array of shape (17, 3) or None if no pose detected
        """
        results = self.model.predict(frame, verbose=False)

        if len(results) == 0 or results[0].keypoints is None:
            return None

        keypoints = results[0].keypoints.data.cpu().numpy()

        if len(keypoints) == 0:
            return None

        return keypoints[0]


def extract_joint_positions(pose: PoseFrame) -> dict:
    """
    Extract named joint positions from a PoseFrame.

    Returns:
        Dictionary mapping joint names to (x, y, confidence) tuples
    """
    kp = pose.keypoints
    return {
        'nose': kp[Keypoints.NOSE],
        'left_shoulder': kp[Keypoints.LEFT_SHOULDER],
        'right_shoulder': kp[Keypoints.RIGHT_SHOULDER],
        'left_elbow': kp[Keypoints.LEFT_ELBOW],
        'right_elbow': kp[Keypoints.RIGHT_ELBOW],
        'left_wrist': kp[Keypoints.LEFT_WRIST],
        'right_wrist': kp[Keypoints.RIGHT_WRIST],
        'left_hip': kp[Keypoints.LEFT_HIP],
        'right_hip': kp[Keypoints.RIGHT_HIP],
        'left_knee': kp[Keypoints.LEFT_KNEE],
        'right_knee': kp[Keypoints.RIGHT_KNEE],
        'left_ankle': kp[Keypoints.LEFT_ANKLE],
        'right_ankle': kp[Keypoints.RIGHT_ANKLE],
    }


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) < 2:
        print("Usage: python pose_estimation.py <video_path>")
        sys.exit(1)

    video_path = sys.argv[1]

    estimator = PoseEstimator(model_name='yolov8m-pose.pt')
    poses = estimator.process_video(video_path)

    print(f"\nExtracted {len(poses)} pose frames")
    if poses:
        print(f"First frame keypoints shape: {poses[0].keypoints.shape}")
        joints = extract_joint_positions(poses[0])
        print(f"Sample joint positions: {list(joints.keys())}")
