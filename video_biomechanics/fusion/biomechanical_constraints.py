"""Biomechanical constraints for anatomically plausible poses."""

import numpy as np
from typing import Optional, Dict, Tuple


class BiomechanicalConstraints:
    """
    Enforce anatomical plausibility on 3D poses.

    Constraints:
    - Bone length consistency
    - Joint angle limits
    - Symmetric limb lengths
    """

    # H36M skeleton joint indices
    JOINTS = {
        'pelvis': 0, 'right_hip': 1, 'right_knee': 2, 'right_ankle': 3,
        'left_hip': 4, 'left_knee': 5, 'left_ankle': 6,
        'spine': 7, 'neck': 8, 'head': 9, 'head_top': 10,
        'left_shoulder': 11, 'left_elbow': 12, 'left_wrist': 13,
        'right_shoulder': 14, 'right_elbow': 15, 'right_wrist': 16
    }

    # Bone connections with typical lengths (meters)
    BONES = {
        ('pelvis', 'right_hip'): 0.11,
        ('right_hip', 'right_knee'): 0.42,
        ('right_knee', 'right_ankle'): 0.40,
        ('pelvis', 'left_hip'): 0.11,
        ('left_hip', 'left_knee'): 0.42,
        ('left_knee', 'left_ankle'): 0.40,
        ('pelvis', 'spine'): 0.22,
        ('spine', 'neck'): 0.25,
        ('neck', 'head'): 0.12,
        ('head', 'head_top'): 0.10,
        ('neck', 'left_shoulder'): 0.15,
        ('left_shoulder', 'left_elbow'): 0.28,
        ('left_elbow', 'left_wrist'): 0.25,
        ('neck', 'right_shoulder'): 0.15,
        ('right_shoulder', 'right_elbow'): 0.28,
        ('right_elbow', 'right_wrist'): 0.25,
    }

    # Joint angle limits (degrees)
    ANGLE_LIMITS = {
        'knee_flexion': (0, 150),
        'elbow_flexion': (0, 145),
        'shoulder_flexion': (-60, 180),
        'shoulder_abduction': (-30, 180),
        'hip_flexion': (-30, 120),
        'hip_abduction': (-45, 45),
        'neck_flexion': (-60, 60),
        'neck_rotation': (-80, 80),
    }

    def __init__(self,
                 bone_length_tolerance: float = 0.15,
                 enforce_symmetry: bool = True,
                 max_iterations: int = 5):
        """
        Initialize constraints.

        Args:
            bone_length_tolerance: Allowed deviation from expected bone length (fraction)
            enforce_symmetry: Ensure left/right limbs have similar lengths
            max_iterations: Iterations for constraint optimization
        """
        self.bone_tolerance = bone_length_tolerance
        self.enforce_symmetry = enforce_symmetry
        self.max_iterations = max_iterations

        # Reference bone lengths (will be updated from first pose)
        self.reference_bones: Optional[Dict] = None

    def apply_constraints(self,
                          pose: np.ndarray,
                          update_reference: bool = False) -> np.ndarray:
        """
        Apply all biomechanical constraints to a pose.

        Args:
            pose: (17, 3) joint positions
            update_reference: Update reference bone lengths from this pose

        Returns:
            Constrained (17, 3) pose
        """
        pose = pose.copy()

        # Update reference if requested or not set
        if update_reference or self.reference_bones is None:
            self.reference_bones = self._measure_bones(pose)

        # Iteratively apply constraints
        for _ in range(self.max_iterations):
            pose = self._enforce_bone_lengths(pose)

            if self.enforce_symmetry:
                pose = self._enforce_symmetry(pose)

        return pose

    def _measure_bones(self, pose: np.ndarray) -> Dict[tuple, float]:
        """Measure current bone lengths."""
        bones = {}
        for (j1_name, j2_name), expected in self.BONES.items():
            j1 = self.JOINTS[j1_name]
            j2 = self.JOINTS[j2_name]
            length = np.linalg.norm(pose[j2] - pose[j1])
            bones[(j1_name, j2_name)] = length
        return bones

    def _enforce_bone_lengths(self, pose: np.ndarray) -> np.ndarray:
        """Adjust joints to maintain bone length constraints."""
        pose = pose.copy()

        for (j1_name, j2_name), expected in self.BONES.items():
            j1 = self.JOINTS[j1_name]
            j2 = self.JOINTS[j2_name]

            # Use reference length if available
            if self.reference_bones:
                target_length = self.reference_bones.get((j1_name, j2_name), expected)
            else:
                target_length = expected

            current_length = np.linalg.norm(pose[j2] - pose[j1])

            if current_length < 1e-6:
                continue

            # Check if outside tolerance
            min_len = target_length * (1 - self.bone_tolerance)
            max_len = target_length * (1 + self.bone_tolerance)

            if current_length < min_len or current_length > max_len:
                # Scale to target length
                direction = (pose[j2] - pose[j1]) / current_length
                new_pos = pose[j1] + direction * target_length

                # Move child joint (j2) toward correct position
                # Weight by 0.5 to distribute correction
                pose[j2] = pose[j2] * 0.5 + new_pos * 0.5

        return pose

    def _enforce_symmetry(self, pose: np.ndarray) -> np.ndarray:
        """Ensure left/right limb lengths are similar."""
        pose = pose.copy()

        symmetric_pairs = [
            (('pelvis', 'left_hip'), ('pelvis', 'right_hip')),
            (('left_hip', 'left_knee'), ('right_hip', 'right_knee')),
            (('left_knee', 'left_ankle'), ('right_knee', 'right_ankle')),
            (('neck', 'left_shoulder'), ('neck', 'right_shoulder')),
            (('left_shoulder', 'left_elbow'), ('right_shoulder', 'right_elbow')),
            (('left_elbow', 'left_wrist'), ('right_elbow', 'right_wrist')),
        ]

        for (l1, l2), (r1, r2) in symmetric_pairs:
            lj1, lj2 = self.JOINTS[l1], self.JOINTS[l2]
            rj1, rj2 = self.JOINTS[r1], self.JOINTS[r2]

            left_len = np.linalg.norm(pose[lj2] - pose[lj1])
            right_len = np.linalg.norm(pose[rj2] - pose[rj1])

            if left_len < 1e-6 or right_len < 1e-6:
                continue

            # Average length
            avg_len = (left_len + right_len) / 2

            # Scale both toward average
            if abs(left_len - avg_len) > 0.01:
                direction = (pose[lj2] - pose[lj1]) / left_len
                pose[lj2] = pose[lj1] + direction * avg_len

            if abs(right_len - avg_len) > 0.01:
                direction = (pose[rj2] - pose[rj1]) / right_len
                pose[rj2] = pose[rj1] + direction * avg_len

        return pose

    def calculate_angle(self,
                        pose: np.ndarray,
                        j1: int, j2: int, j3: int) -> float:
        """Calculate angle at j2 formed by j1-j2-j3."""
        v1 = pose[j1] - pose[j2]
        v2 = pose[j3] - pose[j2]

        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)

        if v1_norm < 1e-6 or v2_norm < 1e-6:
            return 0

        cos_angle = np.dot(v1, v2) / (v1_norm * v2_norm)
        cos_angle = np.clip(cos_angle, -1, 1)

        return np.degrees(np.arccos(cos_angle))

    def check_constraints(self, pose: np.ndarray) -> Dict:
        """Check all constraints and return violations."""
        violations = {
            'bone_length': [],
            'symmetry': [],
        }

        # Check bone lengths
        for (j1_name, j2_name), expected in self.BONES.items():
            j1 = self.JOINTS[j1_name]
            j2 = self.JOINTS[j2_name]

            length = np.linalg.norm(pose[j2] - pose[j1])
            deviation = abs(length - expected) / expected

            if deviation > self.bone_tolerance:
                violations['bone_length'].append({
                    'bone': (j1_name, j2_name),
                    'expected': expected,
                    'actual': length,
                    'deviation': deviation
                })

        return violations
