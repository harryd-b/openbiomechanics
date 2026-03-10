"""
Temporal 2D to 3D pose lifting using VideoPose3D.

This module properly handles the temporal convolution requirements of VideoPose3D,
which needs 243 frames of context for its dilated convolution architecture.
"""

import numpy as np
import torch
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass


@dataclass
class Pose3D:
    """Container for 3D pose data."""
    frame_number: int
    timestamp: float
    joints_3d: np.ndarray  # Shape: (num_joints, 3) - x, y, z in meters
    confidence: Optional[np.ndarray] = None


class VideoPose3DTemporalLifter:
    """
    Lift 2D poses to 3D using the full VideoPose3D temporal model.

    This model uses dilated temporal convolutions with a receptive field of 243 frames.
    Input sequences are padded as needed to meet this requirement.
    """

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the temporal lifter.

        Args:
            model_path: Path to pretrained weights (pretrained_h36m_cpn.bin)
        """
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Default model path
        if model_path is None:
            default_path = Path(__file__).parent / 'models' / 'videopose3d' / 'pretrained_h36m_cpn.bin'
            if default_path.exists():
                model_path = str(default_path)

        self.model_path = model_path
        self.receptive_field = 243  # Based on filter_widths=[3,3,3,3,3]

        self._load_model()

    def _load_model(self):
        """Load the pretrained VideoPose3D temporal model."""
        if self.model_path is None or not Path(self.model_path).exists():
            print("No pretrained model found. Using geometric estimation.")
            return

        try:
            # Import the temporal model architecture
            import importlib.util
            model_dir = Path(self.model_path).parent
            model_py = model_dir / 'model.py'

            if not model_py.exists():
                print(f"Model architecture not found at {model_py}")
                return

            spec = importlib.util.spec_from_file_location("videopose3d_model", model_py)
            vp3d_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(vp3d_module)

            # Create model with same config as pretrained
            # The pretrained model uses: 17 joints, 2 features, filter_widths=[3,3,3,3,3], channels=1024
            self.model = vp3d_module.TemporalModel(
                num_joints_in=17,
                in_features=2,
                num_joints_out=17,
                filter_widths=[3, 3, 3, 3, 3],
                causal=False,
                dropout=0.25,
                channels=1024
            )

            # Load pretrained weights
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(checkpoint['model_pos'])
            self.model.to(self.device)
            self.model.eval()

            # Verify receptive field
            self.receptive_field = self.model.receptive_field()
            print(f"Loaded VideoPose3D temporal model (receptive field: {self.receptive_field} frames)")

        except Exception as e:
            print(f"Failed to load temporal model: {e}")
            import traceback
            traceback.print_exc()
            self.model = None

    def lift_sequence(self,
                      keypoints_2d: np.ndarray,
                      timestamps: Optional[List[float]] = None) -> List[Pose3D]:
        """
        Lift a sequence of 2D poses to 3D.

        Args:
            keypoints_2d: Array of shape (n_frames, 17, 2) - H36M format 2D keypoints
            timestamps: Optional list of timestamps for each frame

        Returns:
            List of Pose3D objects
        """
        n_frames = len(keypoints_2d)

        if timestamps is None:
            timestamps = [float(i) / 30.0 for i in range(n_frames)]  # Assume 30 FPS

        if self.model is None:
            return self._lift_geometric(keypoints_2d, timestamps)

        return self._lift_with_model(keypoints_2d, timestamps)

    def _lift_with_model(self,
                         keypoints_2d: np.ndarray,
                         timestamps: List[float]) -> List[Pose3D]:
        """Lift using the temporal neural network."""
        n_frames = len(keypoints_2d)

        # Normalize 2D keypoints
        keypoints_norm = self._normalize_2d_sequence(keypoints_2d)

        # Pad sequence to meet receptive field requirement
        pad_left = self.receptive_field // 2
        pad_right = self.receptive_field - 1 - pad_left

        # Use edge padding (replicate first/last frames)
        padded = np.pad(
            keypoints_norm,
            ((pad_left, pad_right), (0, 0), (0, 0)),
            mode='edge'
        )

        # Convert to tensor: (1, n_frames_padded, 17, 2)
        input_tensor = torch.FloatTensor(padded).unsqueeze(0).to(self.device)

        # Run model
        with torch.no_grad():
            output = self.model(input_tensor)

        # Output shape: (1, n_frames, 17, 3)
        poses_3d_raw = output.cpu().numpy()[0]

        # Create Pose3D objects
        poses_3d = []
        for i in range(n_frames):
            poses_3d.append(Pose3D(
                frame_number=i,
                timestamp=timestamps[i],
                joints_3d=poses_3d_raw[i]
            ))

        return poses_3d

    def _normalize_2d_sequence(self, keypoints_2d: np.ndarray) -> np.ndarray:
        """
        Normalize 2D keypoints for VideoPose3D.

        Centers on hip and scales based on image coordinates.
        """
        # Copy to avoid modifying original
        kp = keypoints_2d.copy()

        # Center on hip (joint 0 in H36M)
        hip_center = kp[:, 0:1, :]
        kp = kp - hip_center

        # Scale - VideoPose3D expects normalized coordinates
        # Typical approach: divide by image size or use a fixed scale
        # For now, use a scale based on typical 1080p video
        kp = kp / 1000.0  # Rough normalization

        return kp

    def _lift_geometric(self,
                        keypoints_2d: np.ndarray,
                        timestamps: List[float]) -> List[Pose3D]:
        """
        Fallback geometric lifting when no model is available.

        Uses bone length constraints to estimate depth.
        """
        # Typical bone lengths in meters
        BONE_LENGTHS = {
            'upper_arm': 0.28,
            'forearm': 0.25,
            'thigh': 0.42,
            'shin': 0.40,
            'torso': 0.50,
        }

        poses_3d = []

        for i, (kp_2d, ts) in enumerate(zip(keypoints_2d, timestamps)):
            # Estimate scale from torso length
            shoulder_center = (kp_2d[11] + kp_2d[14]) / 2  # L/R shoulder
            hip_center = (kp_2d[1] + kp_2d[4]) / 2  # L/R hip (H36M format)
            torso_2d = np.linalg.norm(shoulder_center - hip_center)

            if torso_2d > 1e-6:
                scale = BONE_LENGTHS['torso'] / torso_2d
            else:
                scale = 0.005

            # Create 3D pose with x,y scaled and z estimated from bone constraints
            joints_3d = np.zeros((17, 3))
            joints_3d[:, :2] = kp_2d * scale

            # Estimate depth from bone length constraints
            joints_3d = self._estimate_depth(joints_3d, BONE_LENGTHS)

            poses_3d.append(Pose3D(
                frame_number=i,
                timestamp=ts,
                joints_3d=joints_3d
            ))

        return poses_3d

    def _estimate_depth(self,
                        joints_3d: np.ndarray,
                        bone_lengths: dict) -> np.ndarray:
        """Estimate depth using bone length constraints."""
        # H36M joint indices
        limbs = [
            (11, 12, bone_lengths['upper_arm']),  # L shoulder -> L elbow
            (12, 13, bone_lengths['forearm']),     # L elbow -> L wrist
            (14, 15, bone_lengths['upper_arm']),  # R shoulder -> R elbow
            (15, 16, bone_lengths['forearm']),     # R elbow -> R wrist
            (4, 5, bone_lengths['thigh']),         # L hip -> L knee
            (5, 6, bone_lengths['shin']),          # L knee -> L ankle
            (1, 2, bone_lengths['thigh']),         # R hip -> R knee
            (2, 3, bone_lengths['shin']),          # R knee -> R ankle
        ]

        for parent, child, expected_length in limbs:
            dist_2d = np.linalg.norm(joints_3d[child, :2] - joints_3d[parent, :2])

            if dist_2d < expected_length:
                depth_diff = np.sqrt(max(0, expected_length**2 - dist_2d**2))
                joints_3d[child, 2] = joints_3d[parent, 2] + depth_diff * 0.5

        return joints_3d


def create_lifter(use_temporal: bool = True, model_path: Optional[str] = None):
    """
    Factory function to create the appropriate lifter.

    Args:
        use_temporal: If True, use full temporal model. If False, use geometric.
        model_path: Optional path to pretrained weights

    Returns:
        Lifter instance
    """
    if use_temporal:
        return VideoPose3DTemporalLifter(model_path)
    else:
        # Import the original geometric lifter
        from lifting_3d import VideoPose3DLifter
        return VideoPose3DLifter(model_path=None)


if __name__ == "__main__":
    # Test with dummy data
    print("Testing VideoPose3D temporal lifter...")

    # Create dummy 2D keypoints (100 frames, 17 joints, 2 coords)
    n_frames = 100
    dummy_2d = np.random.rand(n_frames, 17, 2) * 500 + np.array([540, 360])

    lifter = VideoPose3DTemporalLifter()
    poses_3d = lifter.lift_sequence(dummy_2d)

    print(f"Input: {n_frames} frames")
    print(f"Output: {len(poses_3d)} poses")
    if poses_3d:
        print(f"Sample 3D pose shape: {poses_3d[0].joints_3d.shape}")
        print(f"Sample positions:\n{poses_3d[0].joints_3d[:3]}")
