"""
2D to 3D pose lifting module.

Converts 2D pose keypoints to 3D joint positions using pretrained models.
This enables calculation of rotation angles and proper biomechanics.

Supported models:
- VideoPose3D (Facebook Research)
- MotionBERT (optional)
"""

import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass
import os


@dataclass
class Pose3D:
    """Container for 3D pose data."""
    frame_number: int
    timestamp: float
    joints_3d: np.ndarray  # Shape: (num_joints, 3) - x, y, z in meters
    confidence: Optional[np.ndarray] = None


# Joint indices for Human3.6M skeleton (used by most lifting models)
class H36MJoints:
    """Human3.6M joint indices (17 joints)."""
    HIP_CENTER = 0
    RIGHT_HIP = 1
    RIGHT_KNEE = 2
    RIGHT_ANKLE = 3
    LEFT_HIP = 4
    LEFT_KNEE = 5
    LEFT_ANKLE = 6
    SPINE = 7
    NECK = 8
    HEAD = 9
    HEAD_TOP = 10
    LEFT_SHOULDER = 11
    LEFT_ELBOW = 12
    LEFT_WRIST = 13
    RIGHT_SHOULDER = 14
    RIGHT_ELBOW = 15
    RIGHT_WRIST = 16


# Mapping from COCO/YOLOv8 keypoints to H36M
YOLO_TO_H36M = {
    # YOLOv8 index -> H36M index (approximate mapping)
    0: 9,   # nose -> head
    5: 11,  # left_shoulder -> left_shoulder
    6: 14,  # right_shoulder -> right_shoulder
    7: 12,  # left_elbow -> left_elbow
    8: 15,  # right_elbow -> right_elbow
    9: 13,  # left_wrist -> left_wrist
    10: 16, # right_wrist -> right_wrist
    11: 4,  # left_hip -> left_hip
    12: 1,  # right_hip -> right_hip
    13: 5,  # left_knee -> left_knee
    14: 2,  # right_knee -> right_knee
    15: 6,  # left_ankle -> left_ankle
    16: 3,  # right_ankle -> right_ankle
}


class VideoPose3DLifter:
    """
    Lift 2D poses to 3D using VideoPose3D.

    Reference: https://github.com/facebookresearch/VideoPose3D

    This is a simplified implementation. For best results, use the full
    VideoPose3D pipeline with temporal convolutions.
    """

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the lifter.

        Args:
            model_path: Path to pretrained weights. If None, will attempt
                       to download or use a simple baseline.
        """
        self.model = None
        self.model_path = model_path
        self._load_model()

    def _load_model(self):
        """Load the pretrained model."""
        try:
            import torch
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            if self.model_path and os.path.exists(self.model_path):
                # Load custom weights
                self.model = self._create_model()
                checkpoint = torch.load(self.model_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_pos'])
                self.model.eval()
                print(f"Loaded VideoPose3D model from {self.model_path}")
            else:
                # Use simple baseline (triangulation-based estimation)
                print("No pretrained model found. Using geometric estimation.")
                self.model = None

        except ImportError:
            print("PyTorch not installed. Using geometric estimation for 3D lifting.")
            self.model = None

    def _create_model(self):
        """Create the VideoPose3D model architecture."""
        # Simplified - in practice you'd import from VideoPose3D repo
        import torch.nn as nn

        class SimpleLifter(nn.Module):
            def __init__(self, input_dim=34, output_dim=51, hidden_dim=1024):
                super().__init__()
                self.fc = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.25),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.25),
                    nn.Linear(hidden_dim, output_dim)
                )

            def forward(self, x):
                return self.fc(x)

        return SimpleLifter().to(self.device)

    def lift_sequence(self,
                      poses_2d: List[np.ndarray],
                      timestamps: List[float],
                      camera_params: Optional[dict] = None) -> List[Pose3D]:
        """
        Lift a sequence of 2D poses to 3D.

        Args:
            poses_2d: List of 2D keypoint arrays, shape (17, 2) or (17, 3)
            timestamps: Corresponding timestamps
            camera_params: Optional camera intrinsics for better estimation

        Returns:
            List of Pose3D objects
        """
        if self.model is not None:
            return self._lift_with_model(poses_2d, timestamps)
        else:
            return self._lift_geometric(poses_2d, timestamps, camera_params)

    def _lift_with_model(self,
                         poses_2d: List[np.ndarray],
                         timestamps: List[float]) -> List[Pose3D]:
        """Lift using trained neural network."""
        import torch

        poses_3d = []

        for i, (pose_2d, ts) in enumerate(zip(poses_2d, timestamps)):
            # Normalize 2D pose
            pose_normalized = self._normalize_2d(pose_2d)

            # Convert to tensor
            input_tensor = torch.FloatTensor(pose_normalized.flatten()).unsqueeze(0)
            input_tensor = input_tensor.to(self.device)

            # Predict
            with torch.no_grad():
                output = self.model(input_tensor)

            joints_3d = output.cpu().numpy().reshape(-1, 3)

            poses_3d.append(Pose3D(
                frame_number=i,
                timestamp=ts,
                joints_3d=joints_3d
            ))

        return poses_3d

    def _lift_geometric(self,
                        poses_2d: List[np.ndarray],
                        timestamps: List[float],
                        camera_params: Optional[dict] = None) -> List[Pose3D]:
        """
        Lift using geometric estimation (no ML model required).

        Uses bone length constraints and assumes a roughly frontal/side view.
        Less accurate but works without pretrained weights.
        """
        # Typical bone lengths in meters (from Human3.6M statistics)
        BONE_LENGTHS = {
            'upper_arm': 0.28,
            'forearm': 0.25,
            'thigh': 0.42,
            'shin': 0.40,
            'torso': 0.50,
            'shoulder_width': 0.40,
            'hip_width': 0.25,
        }

        poses_3d = []

        for i, (pose_2d, ts) in enumerate(zip(poses_2d, timestamps)):
            # Extract 2D positions (ignore confidence if present)
            if pose_2d.shape[1] >= 3:
                kp_2d = pose_2d[:, :2]
                conf = pose_2d[:, 2]
            else:
                kp_2d = pose_2d
                conf = np.ones(len(pose_2d))

            # Estimate scale from 2D pose
            scale = self._estimate_scale(kp_2d, BONE_LENGTHS)

            # Create 3D pose with estimated depth
            joints_3d = self._estimate_3d_positions(kp_2d, scale, BONE_LENGTHS)

            poses_3d.append(Pose3D(
                frame_number=i,
                timestamp=ts,
                joints_3d=joints_3d,
                confidence=conf
            ))

        return poses_3d

    def _normalize_2d(self, pose_2d: np.ndarray) -> np.ndarray:
        """Normalize 2D pose to [-1, 1] range centered on hip."""
        if pose_2d.shape[1] >= 3:
            kp = pose_2d[:, :2].copy()
        else:
            kp = pose_2d.copy()

        # Center on hip midpoint
        if len(kp) >= 13:  # YOLO format
            hip_center = (kp[11] + kp[12]) / 2
        else:
            hip_center = kp[0]  # H36M format

        kp = kp - hip_center

        # Scale by torso length
        if len(kp) >= 13:
            shoulder_center = (kp[5] + kp[6]) / 2
            torso_length = np.linalg.norm(shoulder_center)
            if torso_length > 1e-6:
                kp = kp / torso_length

        return kp

    def _estimate_scale(self, kp_2d: np.ndarray, bone_lengths: dict) -> float:
        """Estimate the scale factor from 2D keypoints."""
        # Use torso length as reference
        if len(kp_2d) >= 13:  # YOLO format
            shoulder_center = (kp_2d[5] + kp_2d[6]) / 2
            hip_center = (kp_2d[11] + kp_2d[12]) / 2
            torso_2d = np.linalg.norm(shoulder_center - hip_center)

            if torso_2d > 1e-6:
                return bone_lengths['torso'] / torso_2d

        return 0.005  # Default scale (assumes ~200px person height)

    def _estimate_3d_positions(self,
                               kp_2d: np.ndarray,
                               scale: float,
                               bone_lengths: dict) -> np.ndarray:
        """
        Estimate 3D positions from 2D with depth estimation.

        Uses anatomical constraints to estimate depth (z).
        """
        n_joints = 17  # H36M format
        joints_3d = np.zeros((n_joints, 3))

        # Map YOLO keypoints to H36M if needed
        if len(kp_2d) == 17:  # Already H36M or YOLO
            # Assume YOLO format, map to H36M
            for yolo_idx, h36m_idx in YOLO_TO_H36M.items():
                if yolo_idx < len(kp_2d):
                    joints_3d[h36m_idx, 0] = kp_2d[yolo_idx, 0] * scale
                    joints_3d[h36m_idx, 1] = kp_2d[yolo_idx, 1] * scale

        # Estimate hip center
        joints_3d[H36MJoints.HIP_CENTER] = (
            joints_3d[H36MJoints.LEFT_HIP] + joints_3d[H36MJoints.RIGHT_HIP]
        ) / 2

        # Estimate spine and neck
        shoulder_center = (
            joints_3d[H36MJoints.LEFT_SHOULDER] + joints_3d[H36MJoints.RIGHT_SHOULDER]
        ) / 2
        joints_3d[H36MJoints.SPINE] = (
            joints_3d[H36MJoints.HIP_CENTER] + shoulder_center
        ) / 2
        joints_3d[H36MJoints.NECK] = shoulder_center

        # Estimate depth (z) based on joint angles and bone lengths
        joints_3d = self._estimate_depth(joints_3d, bone_lengths)

        return joints_3d

    def _estimate_depth(self,
                        joints_3d: np.ndarray,
                        bone_lengths: dict) -> np.ndarray:
        """
        Estimate depth (z-coordinate) for each joint.

        Uses bone length constraints - if 2D distance is less than
        expected bone length, the depth difference accounts for it.
        """
        # For each limb, estimate depth from bone length constraint
        limbs = [
            (H36MJoints.LEFT_SHOULDER, H36MJoints.LEFT_ELBOW, bone_lengths['upper_arm']),
            (H36MJoints.LEFT_ELBOW, H36MJoints.LEFT_WRIST, bone_lengths['forearm']),
            (H36MJoints.RIGHT_SHOULDER, H36MJoints.RIGHT_ELBOW, bone_lengths['upper_arm']),
            (H36MJoints.RIGHT_ELBOW, H36MJoints.RIGHT_WRIST, bone_lengths['forearm']),
            (H36MJoints.LEFT_HIP, H36MJoints.LEFT_KNEE, bone_lengths['thigh']),
            (H36MJoints.LEFT_KNEE, H36MJoints.LEFT_ANKLE, bone_lengths['shin']),
            (H36MJoints.RIGHT_HIP, H36MJoints.RIGHT_KNEE, bone_lengths['thigh']),
            (H36MJoints.RIGHT_KNEE, H36MJoints.RIGHT_ANKLE, bone_lengths['shin']),
        ]

        for parent, child, expected_length in limbs:
            # Current 2D distance
            dist_2d = np.linalg.norm(joints_3d[child, :2] - joints_3d[parent, :2])

            # If 2D distance is less than bone length, there's depth
            if dist_2d < expected_length:
                depth_diff = np.sqrt(max(0, expected_length**2 - dist_2d**2))
                # Assign depth relative to parent (simplified)
                joints_3d[child, 2] = joints_3d[parent, 2] + depth_diff * 0.5

        return joints_3d


class MotionBERTLifter:
    """
    Alternative lifter using MotionBERT.

    Reference: https://github.com/Walter0807/MotionBERT

    Generally more accurate than VideoPose3D but requires more setup.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.model_path = model_path
        print("MotionBERT lifter initialized (placeholder - implement if needed)")

    def lift_sequence(self,
                      poses_2d: List[np.ndarray],
                      timestamps: List[float]) -> List[Pose3D]:
        """Lift sequence using MotionBERT."""
        # Placeholder - would implement full MotionBERT pipeline
        # For now, fall back to geometric estimation
        fallback = VideoPose3DLifter(model_path=None)
        return fallback._lift_geometric(poses_2d, timestamps, None)


def convert_yolo_to_h36m(yolo_keypoints: np.ndarray) -> np.ndarray:
    """
    Convert YOLOv8 pose keypoints to Human3.6M format.

    Args:
        yolo_keypoints: Array of shape (17, 3) in YOLO format

    Returns:
        Array of shape (17, 3) in H36M format
    """
    h36m = np.zeros((17, 3))

    for yolo_idx, h36m_idx in YOLO_TO_H36M.items():
        if yolo_idx < len(yolo_keypoints):
            h36m[h36m_idx] = yolo_keypoints[yolo_idx]

    # Compute derived joints
    h36m[H36MJoints.HIP_CENTER] = (h36m[H36MJoints.LEFT_HIP] + h36m[H36MJoints.RIGHT_HIP]) / 2
    h36m[H36MJoints.NECK] = (h36m[H36MJoints.LEFT_SHOULDER] + h36m[H36MJoints.RIGHT_SHOULDER]) / 2
    h36m[H36MJoints.SPINE] = (h36m[H36MJoints.HIP_CENTER] + h36m[H36MJoints.NECK]) / 2
    h36m[H36MJoints.HEAD_TOP] = h36m[H36MJoints.HEAD]  # Approximate

    return h36m


if __name__ == "__main__":
    # Test with dummy data
    dummy_2d = np.random.rand(17, 3) * 500  # Fake 2D keypoints
    dummy_2d[:, 2] = 0.9  # High confidence

    lifter = VideoPose3DLifter()
    poses_3d = lifter.lift_sequence([dummy_2d], [0.0])

    print(f"Lifted pose shape: {poses_3d[0].joints_3d.shape}")
    print(f"Sample 3D positions:\n{poses_3d[0].joints_3d[:5]}")
