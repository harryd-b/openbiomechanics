"""
OBP-Trained 2D-to-3D Pose Lifter.

This module provides a drop-in replacement for the geometric lifter,
using a neural network trained on OpenBiomechanics motion capture data.

Usage:
    from obp_lifter import OBPLiftingModel

    lifter = OBPLiftingModel()
    poses_3d = lifter.lift_sequence(poses_2d, timestamps)
"""

import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass
import json

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. OBP lifter will fall back to geometric estimation.")


@dataclass
class Pose3D:
    """Container for 3D pose data."""
    frame_number: int
    timestamp: float
    joints_3d: np.ndarray  # Shape: (num_joints, 3)
    confidence: Optional[np.ndarray] = None


# Joint order (H36M-style 17-joint skeleton)
JOINT_NAMES = [
    'hip_center', 'right_hip', 'right_knee', 'right_ankle',
    'left_hip', 'left_knee', 'left_ankle', 'spine',
    'neck', 'head', 'head_top', 'left_shoulder',
    'left_elbow', 'left_wrist', 'right_shoulder', 'right_elbow', 'right_wrist'
]

# Mapping from YOLO 17-keypoint to our order
YOLO_TO_INTERNAL = {
    0: 9,   # nose -> head
    5: 11,  # left_shoulder
    6: 14,  # right_shoulder
    7: 12,  # left_elbow
    8: 15,  # right_elbow
    9: 13,  # left_wrist
    10: 16, # right_wrist
    11: 4,  # left_hip
    12: 1,  # right_hip
    13: 5,  # left_knee
    14: 2,  # right_knee
    15: 6,  # left_ankle
    16: 3,  # right_ankle
}


class ResidualBlock(torch.nn.Module):
    """Residual block matching training architecture."""

    def __init__(self, dim: int, dropout: float = 0.25):
        super().__init__()
        self.block = torch.nn.Sequential(
            torch.nn.Linear(dim, dim),
            torch.nn.BatchNorm1d(dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(dim, dim),
            torch.nn.BatchNorm1d(dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.block(x)


class Pose2DTo3DNet(torch.nn.Module):
    """Neural network for 2D to 3D lifting."""

    def __init__(self, input_dim=34, output_dim=51, hidden_dim=1024, num_layers=4, dropout=0.25):
        super().__init__()

        layers = []
        layers.append(torch.nn.Linear(input_dim, hidden_dim))
        layers.append(torch.nn.BatchNorm1d(hidden_dim))
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Dropout(dropout))

        for _ in range(num_layers - 2):
            layers.append(ResidualBlock(hidden_dim, dropout))

        layers.append(torch.nn.Linear(hidden_dim, output_dim))

        self.model = torch.nn.Sequential(*layers)
        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, x):
        return self.model(x)


class OBPLiftingModel:
    """
    2D to 3D pose lifter trained on OpenBiomechanics data.

    This provides significantly better accuracy than geometric estimation,
    especially for fast athletic movements like pitching and batting.
    """

    def __init__(self, model_path: Optional[str] = None, device: str = 'auto'):
        """
        Initialize the OBP lifter.

        Args:
            model_path: Path to trained model weights. If None, uses default.
            device: 'cuda', 'cpu', or 'auto' (default)
        """
        self.model = None
        self.norm_params = None
        self.device = device

        if not TORCH_AVAILABLE:
            print("OBP lifter: PyTorch not available, using geometric fallback")
            return

        # Set device
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Find model path
        if model_path is None:
            script_dir = Path(__file__).parent
            model_path = script_dir / 'models' / 'obp_lifter.pt'

        if not Path(model_path).exists():
            print(f"OBP lifter: Model not found at {model_path}, using geometric fallback")
            return

        self._load_model(model_path)

    def _load_model(self, model_path: str):
        """Load the trained model."""
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        self.model = Pose2DTo3DNet(
            input_dim=checkpoint.get('input_dim', 34),
            output_dim=checkpoint.get('output_dim', 51)
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        self.norm_params = {
            'pose_mean': checkpoint['pose_mean'],
            'pose_std': checkpoint['pose_std']
        }

        print(f"OBP lifter: Loaded model from {model_path}")

    @property
    def is_available(self) -> bool:
        """Check if the neural network model is available."""
        return self.model is not None

    def lift_single(self, pose_2d: np.ndarray) -> np.ndarray:
        """
        Lift a single 2D pose to 3D.

        Args:
            pose_2d: (17, 2) or (17, 3) array of 2D keypoints
                     If (17, 3), third column is confidence (ignored)

        Returns:
            (17, 3) array of 3D joint positions in meters
        """
        if not self.is_available:
            return self._geometric_lift(pose_2d)

        # Extract 2D coordinates
        if pose_2d.shape[1] >= 3:
            kp_2d = pose_2d[:, :2].copy()
        else:
            kp_2d = pose_2d.copy()

        # Normalize: center on hip, scale by torso
        hip = kp_2d[0].copy()
        kp_centered = kp_2d - hip

        # Torso = hip_center to neck (index 0 to 8)
        torso_vec = kp_2d[8] - kp_2d[0]
        torso_length = np.linalg.norm(torso_vec)

        if torso_length > 1e-6:
            kp_norm = kp_centered / torso_length
        else:
            kp_norm = kp_centered

        # Run through network
        with torch.no_grad():
            input_tensor = torch.FloatTensor(kp_norm.flatten()).unsqueeze(0)
            input_tensor = input_tensor.to(self.device)

            output = self.model(input_tensor)
            pose_3d = output.cpu().numpy().reshape(17, 3)

        # Denormalize
        pose_3d = pose_3d * self.norm_params['pose_std'] + self.norm_params['pose_mean']

        return pose_3d

    def lift_sequence(self, poses_2d: List[np.ndarray],
                      timestamps: List[float]) -> List[Pose3D]:
        """
        Lift a sequence of 2D poses to 3D.

        Args:
            poses_2d: List of 2D pose arrays
            timestamps: Corresponding timestamps

        Returns:
            List of Pose3D objects
        """
        results = []

        for i, (pose_2d, ts) in enumerate(zip(poses_2d, timestamps)):
            joints_3d = self.lift_single(pose_2d)

            # Extract confidence if available
            if pose_2d.shape[1] >= 3:
                confidence = pose_2d[:, 2]
            else:
                confidence = np.ones(17)

            results.append(Pose3D(
                frame_number=i,
                timestamp=ts,
                joints_3d=joints_3d,
                confidence=confidence
            ))

        return results

    def _geometric_lift(self, pose_2d: np.ndarray) -> np.ndarray:
        """Fallback geometric estimation when model is not available."""
        # Simple depth estimation using bone length constraints
        BONE_LENGTHS = {
            'thigh': 0.45,
            'shin': 0.46,
            'upper_arm': 0.33,
            'forearm': 0.27,
            'torso': 0.54,
        }

        if pose_2d.shape[1] >= 3:
            kp_2d = pose_2d[:, :2]
        else:
            kp_2d = pose_2d

        # Estimate scale
        hip_center = (kp_2d[1] + kp_2d[4]) / 2  # right_hip + left_hip
        neck = (kp_2d[11] + kp_2d[14]) / 2  # shoulders
        torso_2d = np.linalg.norm(neck - hip_center)

        if torso_2d > 1e-6:
            scale = BONE_LENGTHS['torso'] / torso_2d
        else:
            scale = 0.005

        # Convert to 3D with zero depth initially
        joints_3d = np.zeros((17, 3))
        joints_3d[:, :2] = kp_2d * scale

        # Estimate depth using bone length constraints
        limbs = [
            (4, 5, BONE_LENGTHS['thigh']),  # left hip to knee
            (5, 6, BONE_LENGTHS['shin']),   # left knee to ankle
            (1, 2, BONE_LENGTHS['thigh']),  # right hip to knee
            (2, 3, BONE_LENGTHS['shin']),   # right knee to ankle
            (11, 12, BONE_LENGTHS['upper_arm']),  # left shoulder to elbow
            (12, 13, BONE_LENGTHS['forearm']),    # left elbow to wrist
            (14, 15, BONE_LENGTHS['upper_arm']),  # right shoulder to elbow
            (15, 16, BONE_LENGTHS['forearm']),    # right elbow to wrist
        ]

        for parent, child, expected in limbs:
            dist_2d = np.linalg.norm(joints_3d[child, :2] - joints_3d[parent, :2])
            if dist_2d < expected:
                depth_diff = np.sqrt(max(0, expected**2 - dist_2d**2))
                joints_3d[child, 2] = joints_3d[parent, 2] + depth_diff * 0.5

        return joints_3d

    def convert_yolo_to_internal(self, yolo_keypoints: np.ndarray) -> np.ndarray:
        """
        Convert YOLO pose keypoints to internal format.

        Args:
            yolo_keypoints: (17, 3) array from YOLO pose detection

        Returns:
            (17, 3) array in internal joint order
        """
        internal = np.zeros((17, 3 if yolo_keypoints.shape[1] >= 3 else 2))

        for yolo_idx, internal_idx in YOLO_TO_INTERNAL.items():
            if yolo_idx < len(yolo_keypoints):
                internal[internal_idx] = yolo_keypoints[yolo_idx]

        # Compute derived joints
        internal[0] = (internal[1] + internal[4]) / 2  # hip_center
        internal[8] = (internal[11] + internal[14]) / 2  # neck
        internal[7] = (internal[0] + internal[8]) / 2  # spine
        internal[10] = internal[9]  # head_top = head

        return internal


def load_biomechanical_priors(priors_path: Optional[str] = None) -> Dict:
    """
    Load biomechanical priors for pose validation.

    Args:
        priors_path: Path to priors JSON. If None, uses default.

    Returns:
        Dictionary with bone lengths, joint limits, etc.
    """
    if priors_path is None:
        script_dir = Path(__file__).parent
        priors_path = script_dir / 'models' / 'pose_constraints.json'

    if not Path(priors_path).exists():
        print(f"Warning: Priors not found at {priors_path}")
        return {}

    with open(priors_path) as f:
        return json.load(f)


def validate_pose_3d(pose_3d: np.ndarray, priors: Dict) -> Tuple[bool, List[str]]:
    """
    Validate a 3D pose against biomechanical constraints.

    Args:
        pose_3d: (17, 3) array of joint positions
        priors: Biomechanical priors dictionary

    Returns:
        (is_valid, list of violations)
    """
    violations = []

    if 'bone_lengths' not in priors:
        return True, []

    bone_checks = [
        (1, 2, 'thigh_r'),   # right hip to knee
        (2, 3, 'shin_r'),    # right knee to ankle
        (4, 5, 'thigh_l'),   # left hip to knee
        (5, 6, 'shin_l'),    # left knee to ankle
        (14, 15, 'upper_arm_r'),  # right shoulder to elbow
        (15, 16, 'forearm_r'),    # right elbow to wrist
        (11, 12, 'upper_arm_l'),  # left shoulder to elbow
        (12, 13, 'forearm_l'),    # left elbow to wrist
    ]

    for j1, j2, bone_name in bone_checks:
        if bone_name not in priors['bone_lengths']:
            continue

        constraints = priors['bone_lengths'][bone_name]
        actual_length = np.linalg.norm(pose_3d[j2] - pose_3d[j1])

        if actual_length < constraints['min']:
            violations.append(f"{bone_name} too short: {actual_length:.3f}m < {constraints['min']:.3f}m")
        elif actual_length > constraints['max']:
            violations.append(f"{bone_name} too long: {actual_length:.3f}m > {constraints['max']:.3f}m")

    return len(violations) == 0, violations


# Convenience function for quick usage
def lift_pose(pose_2d: np.ndarray, lifter: Optional[OBPLiftingModel] = None) -> np.ndarray:
    """
    Quick function to lift a 2D pose to 3D.

    Args:
        pose_2d: (17, 2) or (17, 3) 2D keypoints
        lifter: Optional pre-initialized lifter (created if None)

    Returns:
        (17, 3) 3D joint positions
    """
    if lifter is None:
        lifter = OBPLiftingModel()

    return lifter.lift_single(pose_2d)


if __name__ == '__main__':
    # Quick test
    print("Testing OBP Lifter...")

    lifter = OBPLiftingModel()

    if lifter.is_available:
        print(f"Model loaded successfully on {lifter.device}")

        # Create dummy 2D pose
        dummy_2d = np.random.rand(17, 3) * 500
        dummy_2d[:, 2] = 0.9  # confidence

        pose_3d = lifter.lift_single(dummy_2d)
        print(f"Input shape: {dummy_2d.shape}")
        print(f"Output shape: {pose_3d.shape}")
        print(f"Sample joint positions (meters):")
        for i, name in enumerate(JOINT_NAMES[:5]):
            print(f"  {name}: {pose_3d[i]}")

        # Load and test priors
        priors = load_biomechanical_priors()
        if priors:
            is_valid, violations = validate_pose_3d(pose_3d, priors)
            print(f"\nPose validation: {'PASS' if is_valid else 'FAIL'}")
            for v in violations[:3]:
                print(f"  - {v}")
    else:
        print("Model not available - using geometric fallback")
