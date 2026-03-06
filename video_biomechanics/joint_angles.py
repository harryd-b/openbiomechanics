"""
Joint angle calculations from 3D (or 2D) pose data.

This module calculates joint angles following OpenBiomechanics conventions.
For 2D analysis, only angles in the image plane can be reliably calculated.
For full 3D analysis, you'll need 3D joint positions from a lifting model.
"""

import numpy as np
from typing import Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class JointAngles:
    """Container for calculated joint angles at a single time point."""
    timestamp: float

    # Elbow angles (degrees)
    left_elbow_flexion: Optional[float] = None
    right_elbow_flexion: Optional[float] = None

    # Knee angles (degrees)
    left_knee_flexion: Optional[float] = None
    right_knee_flexion: Optional[float] = None

    # Shoulder angles (degrees) - simplified for 2D
    left_shoulder_abduction: Optional[float] = None
    right_shoulder_abduction: Optional[float] = None

    # Hip angles (degrees) - simplified for 2D
    left_hip_flexion: Optional[float] = None
    right_hip_flexion: Optional[float] = None

    # Torso angle (degrees) - lean from vertical
    torso_lean: Optional[float] = None


def calculate_angle_3points(p1: np.ndarray,
                           p2: np.ndarray,
                           p3: np.ndarray) -> float:
    """
    Calculate the angle at p2 formed by the vectors p1-p2 and p3-p2.

    Args:
        p1: First point (e.g., shoulder)
        p2: Vertex point (e.g., elbow) - angle is measured here
        p3: Third point (e.g., wrist)

    Returns:
        Angle in degrees (0-180)
    """
    v1 = p1 - p2
    v2 = p3 - p2

    # Handle zero-length vectors
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)

    if norm1 < 1e-6 or norm2 < 1e-6:
        return np.nan

    cos_angle = np.dot(v1, v2) / (norm1 * norm2)
    # Clamp to [-1, 1] to handle numerical errors
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    angle_rad = np.arccos(cos_angle)
    return np.degrees(angle_rad)


def calculate_angle_from_vertical(p1: np.ndarray,
                                  p2: np.ndarray) -> float:
    """
    Calculate the angle of the vector p1->p2 from vertical.

    Args:
        p1: Start point (e.g., hip)
        p2: End point (e.g., shoulder)

    Returns:
        Angle in degrees from vertical (0 = straight up)
    """
    v = p2 - p1

    # Vertical vector (in image coordinates, y increases downward)
    # So "up" is negative y
    vertical = np.array([0, -1]) if len(v) == 2 else np.array([0, 0, 1])

    norm_v = np.linalg.norm(v)
    if norm_v < 1e-6:
        return np.nan

    cos_angle = np.dot(v, vertical) / norm_v
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    return np.degrees(np.arccos(cos_angle))


def calculate_elbow_flexion(shoulder: np.ndarray,
                            elbow: np.ndarray,
                            wrist: np.ndarray) -> float:
    """
    Calculate elbow flexion angle.

    Convention (from OBP):
        Flexion (+) / Extension (-)
        Full extension = 0 degrees

    Returns:
        Elbow flexion in degrees
    """
    # The angle at the elbow
    angle = calculate_angle_3points(shoulder, elbow, wrist)

    # Convert: 180 degrees (straight arm) = 0 flexion
    # Bent arm = positive flexion
    flexion = 180.0 - angle
    return flexion


def calculate_knee_flexion(hip: np.ndarray,
                           knee: np.ndarray,
                           ankle: np.ndarray) -> float:
    """
    Calculate knee flexion angle.

    Convention (from OBP):
        Flexion (+) / Extension (-)
        Full extension = 0 degrees

    Returns:
        Knee flexion in degrees
    """
    angle = calculate_angle_3points(hip, knee, ankle)
    flexion = 180.0 - angle
    return flexion


def calculate_shoulder_abduction(hip: np.ndarray,
                                 shoulder: np.ndarray,
                                 elbow: np.ndarray) -> float:
    """
    Calculate shoulder abduction angle (arm raised from side).

    Simplified 2D calculation - measures angle from torso.

    Convention (from OBP):
        Abduction (+) / Adduction (-)
        Arm at side = 0 degrees

    Returns:
        Shoulder abduction in degrees
    """
    # Vector along torso (hip to shoulder)
    torso = shoulder - hip
    # Vector along upper arm (shoulder to elbow)
    upper_arm = elbow - shoulder

    norm_torso = np.linalg.norm(torso)
    norm_arm = np.linalg.norm(upper_arm)

    if norm_torso < 1e-6 or norm_arm < 1e-6:
        return np.nan

    cos_angle = np.dot(torso, upper_arm) / (norm_torso * norm_arm)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    # Angle from torso line
    angle = np.degrees(np.arccos(cos_angle))

    # Abduction is typically measured from arm-at-side position
    # Arm straight down along torso = 0 degrees abduction
    abduction = 180.0 - angle
    return abduction


def calculate_hip_flexion(shoulder: np.ndarray,
                          hip: np.ndarray,
                          knee: np.ndarray) -> float:
    """
    Calculate hip flexion angle.

    Convention (from OBP hitting):
        Flexion (+) / Extension (-)
        Standing straight = ~180 degrees at hip

    Returns:
        Hip flexion in degrees
    """
    angle = calculate_angle_3points(shoulder, hip, knee)
    flexion = 180.0 - angle
    return flexion


def calculate_torso_lean(left_hip: np.ndarray,
                         right_hip: np.ndarray,
                         left_shoulder: np.ndarray,
                         right_shoulder: np.ndarray) -> float:
    """
    Calculate torso lean angle from vertical.

    Uses midpoints of hips and shoulders to define torso vector.

    Returns:
        Angle from vertical in degrees (positive = forward lean in image)
    """
    mid_hip = (left_hip + right_hip) / 2
    mid_shoulder = (left_shoulder + right_shoulder) / 2

    return calculate_angle_from_vertical(mid_hip, mid_shoulder)


class JointAngleCalculator:
    """Calculate joint angles from pose keypoints."""

    def __init__(self, use_3d: bool = False):
        """
        Initialize calculator.

        Args:
            use_3d: Whether input will be 3D coordinates (True) or 2D (False)
        """
        self.use_3d = use_3d

    def calculate_from_keypoints(self,
                                  keypoints: dict,
                                  timestamp: float = 0.0) -> JointAngles:
        """
        Calculate all joint angles from keypoint dictionary.

        Args:
            keypoints: Dictionary mapping joint names to coordinates
                       Expected keys: left_shoulder, right_shoulder,
                       left_elbow, right_elbow, left_wrist, right_wrist,
                       left_hip, right_hip, left_knee, right_knee,
                       left_ankle, right_ankle
            timestamp: Time of this frame

        Returns:
            JointAngles dataclass with calculated angles
        """
        # Extract coordinates (handle both 2D and 3D, with or without confidence)
        def get_pos(name):
            val = keypoints.get(name)
            if val is None:
                return None
            # Take just x, y (and z if 3D), ignore confidence if present
            if len(val) == 3 and not self.use_3d:
                return np.array(val[:2])  # x, y only
            elif len(val) >= 3 and self.use_3d:
                return np.array(val[:3])  # x, y, z
            return np.array(val[:2])

        # Get all joint positions
        ls = get_pos('left_shoulder')
        rs = get_pos('right_shoulder')
        le = get_pos('left_elbow')
        re = get_pos('right_elbow')
        lw = get_pos('left_wrist')
        rw = get_pos('right_wrist')
        lh = get_pos('left_hip')
        rh = get_pos('right_hip')
        lk = get_pos('left_knee')
        rk = get_pos('right_knee')
        la = get_pos('left_ankle')
        ra = get_pos('right_ankle')

        angles = JointAngles(timestamp=timestamp)

        # Calculate elbow angles
        if all(p is not None for p in [ls, le, lw]):
            angles.left_elbow_flexion = calculate_elbow_flexion(ls, le, lw)
        if all(p is not None for p in [rs, re, rw]):
            angles.right_elbow_flexion = calculate_elbow_flexion(rs, re, rw)

        # Calculate knee angles
        if all(p is not None for p in [lh, lk, la]):
            angles.left_knee_flexion = calculate_knee_flexion(lh, lk, la)
        if all(p is not None for p in [rh, rk, ra]):
            angles.right_knee_flexion = calculate_knee_flexion(rh, rk, ra)

        # Calculate shoulder abduction
        if all(p is not None for p in [lh, ls, le]):
            angles.left_shoulder_abduction = calculate_shoulder_abduction(lh, ls, le)
        if all(p is not None for p in [rh, rs, re]):
            angles.right_shoulder_abduction = calculate_shoulder_abduction(rh, rs, re)

        # Calculate hip flexion
        if all(p is not None for p in [ls, lh, lk]):
            angles.left_hip_flexion = calculate_hip_flexion(ls, lh, lk)
        if all(p is not None for p in [rs, rh, rk]):
            angles.right_hip_flexion = calculate_hip_flexion(rs, rh, rk)

        # Calculate torso lean
        if all(p is not None for p in [lh, rh, ls, rs]):
            angles.torso_lean = calculate_torso_lean(lh, rh, ls, rs)

        return angles


def calculate_angular_velocity(angles: List[JointAngles],
                               attribute: str,
                               fps: float) -> np.ndarray:
    """
    Calculate angular velocity for a specific joint angle over time.

    Args:
        angles: List of JointAngles objects
        attribute: Name of the angle attribute (e.g., 'left_elbow_flexion')
        fps: Frames per second of the video

    Returns:
        Array of angular velocities in degrees/second
    """
    values = [getattr(a, attribute) for a in angles]
    values = np.array(values, dtype=float)

    # Handle NaN values
    dt = 1.0 / fps
    velocity = np.gradient(values, dt)

    return velocity


if __name__ == "__main__":
    # Example usage with dummy data
    keypoints = {
        'left_shoulder': [100, 100, 0.9],
        'right_shoulder': [200, 100, 0.9],
        'left_elbow': [80, 150, 0.9],
        'right_elbow': [220, 150, 0.9],
        'left_wrist': [70, 200, 0.9],
        'right_wrist': [230, 200, 0.9],
        'left_hip': [120, 200, 0.9],
        'right_hip': [180, 200, 0.9],
        'left_knee': [110, 280, 0.9],
        'right_knee': [190, 280, 0.9],
        'left_ankle': [100, 350, 0.9],
        'right_ankle': [200, 350, 0.9],
    }

    calculator = JointAngleCalculator(use_3d=False)
    angles = calculator.calculate_from_keypoints(keypoints, timestamp=0.0)

    print("Calculated Joint Angles:")
    print(f"  Left elbow flexion: {angles.left_elbow_flexion:.1f}°")
    print(f"  Right elbow flexion: {angles.right_elbow_flexion:.1f}°")
    print(f"  Left knee flexion: {angles.left_knee_flexion:.1f}°")
    print(f"  Right knee flexion: {angles.right_knee_flexion:.1f}°")
    print(f"  Torso lean: {angles.torso_lean:.1f}°")
