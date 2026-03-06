"""
3D Joint angle calculations.

Calculates full 3D joint angles including rotations that can't be
determined from 2D data alone:
- Internal/external rotation
- Axial rotation (pelvis, torso)
- Hip-shoulder separation (X-factor)
"""

import numpy as np
from typing import Tuple, Optional, List
from dataclasses import dataclass
from scipy.spatial.transform import Rotation


@dataclass
class JointAngles3D:
    """Full 3D joint angles for a single frame - matches UPLIFT output format."""
    timestamp: float

    # Head (degrees)
    head_extension: Optional[float] = None
    head_lateral_flexion_clockwise: Optional[float] = None
    head_twist_clockwise: Optional[float] = None

    # Shoulder (degrees)
    left_shoulder_flexion: Optional[float] = None
    right_shoulder_flexion: Optional[float] = None
    left_shoulder_adduction: Optional[float] = None
    right_shoulder_adduction: Optional[float] = None
    left_shoulder_external_rotation: Optional[float] = None
    right_shoulder_external_rotation: Optional[float] = None
    left_shoulder_horizontal_adduction: Optional[float] = None
    right_shoulder_horizontal_adduction: Optional[float] = None

    # Elbow (degrees)
    left_elbow_flexion: Optional[float] = None
    right_elbow_flexion: Optional[float] = None

    # Hip w.r.t. trunk (degrees)
    left_hip_flexion_with_respect_to_trunk: Optional[float] = None
    right_hip_flexion_with_respect_to_trunk: Optional[float] = None
    left_hip_adduction_with_respect_to_trunk: Optional[float] = None
    right_hip_adduction_with_respect_to_trunk: Optional[float] = None
    left_hip_internal_rotation_with_respect_to_trunk: Optional[float] = None
    right_hip_internal_rotation_with_respect_to_trunk: Optional[float] = None

    # Hip w.r.t. pelvis (degrees)
    left_hip_adduction_with_respect_to_pelvis: Optional[float] = None
    right_hip_adduction_with_respect_to_pelvis: Optional[float] = None
    left_hip_internal_rotation_with_respect_to_pelvis: Optional[float] = None
    right_hip_internal_rotation_with_respect_to_pelvis: Optional[float] = None

    # Knee (degrees)
    left_knee_extension: Optional[float] = None
    right_knee_extension: Optional[float] = None
    left_knee_varus: Optional[float] = None
    right_knee_varus: Optional[float] = None
    left_knee_internal_rotation: Optional[float] = None
    right_knee_internal_rotation: Optional[float] = None

    # Ankle (degrees)
    left_ankle_dorsiflexion: Optional[float] = None
    right_ankle_dorsiflexion: Optional[float] = None

    # Pelvis global (degrees)
    pelvis_global_tilt: Optional[float] = None
    pelvis_global_rotation: Optional[float] = None

    # Trunk global (degrees)
    trunk_global_flexion: Optional[float] = None
    trunk_global_tilt: Optional[float] = None
    trunk_global_rotation: Optional[float] = None
    trunk_twist_clockwise: Optional[float] = None  # X-factor
    trunk_lateral_flexion_right: Optional[float] = None

    # Arm rotation (degrees) - for kinematic sequence
    left_arm_rotation: Optional[float] = None
    right_arm_rotation: Optional[float] = None

    # Center of mass
    trunk_center_of_mass_x: Optional[float] = None
    trunk_center_of_mass_y: Optional[float] = None
    trunk_center_of_mass_z: Optional[float] = None
    whole_body_center_of_mass_x: Optional[float] = None
    whole_body_center_of_mass_y: Optional[float] = None
    whole_body_center_of_mass_z: Optional[float] = None

    # Legacy names (for backwards compatibility with dashboard)
    hip_shoulder_separation: Optional[float] = None
    pelvis_rotation: Optional[float] = None
    pelvis_tilt: Optional[float] = None
    pelvis_obliquity: Optional[float] = None
    torso_rotation: Optional[float] = None
    torso_flexion: Optional[float] = None
    torso_lateral_tilt: Optional[float] = None
    left_knee_flexion: Optional[float] = None
    right_knee_flexion: Optional[float] = None
    left_hip_flexion: Optional[float] = None
    right_hip_flexion: Optional[float] = None
    left_hip_abduction: Optional[float] = None
    right_hip_abduction: Optional[float] = None
    left_shoulder_abduction: Optional[float] = None
    right_shoulder_abduction: Optional[float] = None
    left_shoulder_rotation: Optional[float] = None
    right_shoulder_rotation: Optional[float] = None


def normalize_vector(v: np.ndarray) -> np.ndarray:
    """Normalize a vector, handling zero-length vectors."""
    norm = np.linalg.norm(v)
    if norm < 1e-8:
        return np.zeros_like(v)
    return v / norm


def angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
    """Calculate angle between two vectors in degrees."""
    v1_n = normalize_vector(v1)
    v2_n = normalize_vector(v2)

    dot = np.clip(np.dot(v1_n, v2_n), -1.0, 1.0)
    return np.degrees(np.arccos(dot))


def signed_angle_about_axis(v1: np.ndarray, v2: np.ndarray, axis: np.ndarray) -> float:
    """
    Calculate signed angle from v1 to v2 about the given axis.

    Positive angle = counterclockwise when looking down the axis.
    """
    v1_n = normalize_vector(v1)
    v2_n = normalize_vector(v2)
    axis_n = normalize_vector(axis)

    # Project vectors onto plane perpendicular to axis
    v1_proj = v1_n - np.dot(v1_n, axis_n) * axis_n
    v2_proj = v2_n - np.dot(v2_n, axis_n) * axis_n

    v1_proj = normalize_vector(v1_proj)
    v2_proj = normalize_vector(v2_proj)

    if np.linalg.norm(v1_proj) < 1e-8 or np.linalg.norm(v2_proj) < 1e-8:
        return 0.0

    # Calculate angle
    dot = np.clip(np.dot(v1_proj, v2_proj), -1.0, 1.0)
    angle = np.arccos(dot)

    # Determine sign using cross product
    cross = np.cross(v1_proj, v2_proj)
    if np.dot(cross, axis_n) < 0:
        angle = -angle

    return np.degrees(angle)


def calculate_segment_rotation(proximal: np.ndarray,
                               distal: np.ndarray,
                               reference_axis: np.ndarray,
                               segment_axis: np.ndarray) -> Tuple[float, float, float]:
    """
    Calculate segment orientation as Euler angles.

    Args:
        proximal: Proximal joint position
        distal: Distal joint position
        reference_axis: Reference direction (e.g., vertical)
        segment_axis: Expected segment axis direction

    Returns:
        Tuple of (flexion, abduction, rotation) in degrees
    """
    segment_vec = distal - proximal
    segment_vec = normalize_vector(segment_vec)

    # Calculate rotation to align segment_axis with segment_vec
    # This gives us the segment orientation

    # Flexion: rotation about mediolateral axis
    vertical = np.array([0, 0, 1])
    forward = np.array([1, 0, 0])

    # Project onto sagittal plane
    sagittal_proj = segment_vec.copy()
    sagittal_proj[1] = 0  # Remove lateral component
    flexion = 90 - angle_between_vectors(sagittal_proj, vertical)

    # Abduction: rotation about anteroposterior axis
    frontal_proj = segment_vec.copy()
    frontal_proj[0] = 0  # Remove forward component
    abduction = angle_between_vectors(frontal_proj, vertical) - 90

    # Rotation: about longitudinal axis (harder to determine without markers)
    rotation = 0.0  # Would need more data to calculate

    return flexion, abduction, rotation


class JointAngleCalculator3D:
    """Calculate full 3D joint angles from joint positions."""

    def __init__(self, skeleton_type: str = 'h36m'):
        """
        Initialize calculator.

        Args:
            skeleton_type: Type of skeleton ('h36m' for Human3.6M format)
        """
        self.skeleton_type = skeleton_type

        # Joint indices for H36M skeleton
        if skeleton_type == 'h36m':
            self.joints = {
                'hip_center': 0,
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

    def calculate(self,
                  joints_3d: np.ndarray,
                  timestamp: float = 0.0,
                  global_forward: np.ndarray = None) -> JointAngles3D:
        """
        Calculate all joint angles from 3D positions.

        Args:
            joints_3d: Array of shape (17, 3) with 3D joint positions
            timestamp: Time of this frame
            global_forward: Direction toward target (default: +X)

        Returns:
            JointAngles3D with calculated angles
        """
        if global_forward is None:
            global_forward = np.array([1, 0, 0])  # Toward pitcher

        angles = JointAngles3D(timestamp=timestamp)
        j = self.joints

        # Get joint positions
        def get(name):
            return joints_3d[j[name]]

        # ===== PELVIS =====
        left_hip = get('left_hip')
        right_hip = get('right_hip')
        hip_center = (left_hip + right_hip) / 2

        # Pelvis coordinate system
        # Right vector: from left hip to right hip
        pelvis_right = right_hip - left_hip
        pelvis_right_norm = np.linalg.norm(pelvis_right)
        if pelvis_right_norm > 1e-6:
            pelvis_right = pelvis_right / pelvis_right_norm
        else:
            pelvis_right = np.array([0, 1, 0])

        # Forward vector: perpendicular to right, in horizontal plane
        # Project right onto horizontal plane and rotate 90 degrees
        pelvis_right_horiz = np.array([pelvis_right[0], pelvis_right[1], 0])
        pelvis_right_horiz_norm = np.linalg.norm(pelvis_right_horiz)
        if pelvis_right_horiz_norm > 1e-6:
            pelvis_right_horiz = pelvis_right_horiz / pelvis_right_horiz_norm
            # Rotate 90 degrees CCW in horizontal plane to get forward
            pelvis_forward = np.array([-pelvis_right_horiz[1], pelvis_right_horiz[0], 0])
        else:
            pelvis_forward = np.array([1, 0, 0])

        # Pelvis rotation: angle of pelvis_forward from global_forward, about Z axis
        # Use atan2 for proper quadrant handling
        pelvis_angle = np.arctan2(pelvis_forward[1], pelvis_forward[0])
        global_angle = np.arctan2(global_forward[1], global_forward[0])
        angles.pelvis_rotation = np.degrees(pelvis_angle - global_angle)
        # Normalize to [-180, 180]
        while angles.pelvis_rotation > 180:
            angles.pelvis_rotation -= 360
        while angles.pelvis_rotation < -180:
            angles.pelvis_rotation += 360

        # Pelvis obliquity (lateral tilt)
        hip_axis = right_hip - left_hip
        hip_axis_len = np.linalg.norm(hip_axis)
        if hip_axis_len > 1e-6:
            angles.pelvis_obliquity = np.degrees(np.arcsin(np.clip(hip_axis[2] / hip_axis_len, -1, 1)))
        else:
            angles.pelvis_obliquity = 0.0

        # ===== TORSO =====
        left_shoulder = get('left_shoulder')
        right_shoulder = get('right_shoulder')
        shoulder_center = (left_shoulder + right_shoulder) / 2

        # Torso right vector
        torso_right = right_shoulder - left_shoulder
        torso_right_norm = np.linalg.norm(torso_right)
        if torso_right_norm > 1e-6:
            torso_right = torso_right / torso_right_norm
        else:
            torso_right = np.array([0, 1, 0])

        # Torso forward: perpendicular to right, in horizontal plane
        torso_right_horiz = np.array([torso_right[0], torso_right[1], 0])
        torso_right_horiz_norm = np.linalg.norm(torso_right_horiz)
        if torso_right_horiz_norm > 1e-6:
            torso_right_horiz = torso_right_horiz / torso_right_horiz_norm
            torso_forward = np.array([-torso_right_horiz[1], torso_right_horiz[0], 0])
        else:
            torso_forward = np.array([1, 0, 0])

        # Torso rotation
        torso_angle = np.arctan2(torso_forward[1], torso_forward[0])
        angles.torso_rotation = np.degrees(torso_angle - global_angle)
        while angles.torso_rotation > 180:
            angles.torso_rotation -= 360
        while angles.torso_rotation < -180:
            angles.torso_rotation += 360

        # Torso flexion (forward lean from vertical)
        torso_vec = shoulder_center - hip_center
        torso_vec_norm = np.linalg.norm(torso_vec)
        torso_up = torso_vec / torso_vec_norm if torso_vec_norm > 1e-6 else np.array([0, 0, 1])
        if torso_vec_norm > 1e-6:
            # Angle from vertical (Z axis)
            cos_angle = np.clip(torso_vec[2] / torso_vec_norm, -1, 1)
            angles.torso_flexion = 90 - np.degrees(np.arccos(cos_angle))
        else:
            angles.torso_flexion = 0.0

        # Torso lateral tilt
        shoulder_axis = right_shoulder - left_shoulder
        shoulder_axis_len = np.linalg.norm(shoulder_axis)
        if shoulder_axis_len > 1e-6:
            angles.torso_lateral_tilt = np.degrees(np.arcsin(np.clip(shoulder_axis[2] / shoulder_axis_len, -1, 1)))
        else:
            angles.torso_lateral_tilt = 0.0

        # ===== HIP-SHOULDER SEPARATION (X-Factor) =====
        # Direct calculation: angle between shoulder line and hip line in horizontal plane
        pelvis_angle_deg = np.degrees(np.arctan2(pelvis_right[1], pelvis_right[0]))
        torso_angle_deg = np.degrees(np.arctan2(torso_right[1], torso_right[0]))
        angles.hip_shoulder_separation = torso_angle_deg - pelvis_angle_deg
        # Normalize to [-180, 180]
        while angles.hip_shoulder_separation > 180:
            angles.hip_shoulder_separation -= 360
        while angles.hip_shoulder_separation < -180:
            angles.hip_shoulder_separation += 360
        # UPLIFT naming convention
        angles.trunk_twist_clockwise = angles.hip_shoulder_separation

        # ===== HIPS =====
        vertical = np.array([0, 0, 1])  # Z-up
        for side in ['left', 'right']:
            hip = get(f'{side}_hip')
            knee = get(f'{side}_knee')
            ankle = get(f'{side}_ankle')

            # Hip flexion
            thigh = knee - hip
            flexion = angle_between_vectors(thigh, -vertical)  # Angle from vertical
            setattr(angles, f'{side}_hip_flexion', 180 - flexion)

            # Hip abduction
            # Project thigh onto frontal plane
            thigh_frontal = thigh.copy()
            thigh_frontal[0] = 0
            abduction = signed_angle_about_axis(
                -vertical, thigh_frontal, np.array([1, 0, 0])
            )
            setattr(angles, f'{side}_hip_abduction', abduction)

        # ===== KNEES =====
        for side in ['left', 'right']:
            hip = get(f'{side}_hip')
            knee = get(f'{side}_knee')
            ankle = get(f'{side}_ankle')

            # Knee flexion and extension
            thigh = hip - knee
            shank = ankle - knee
            flexion = 180 - angle_between_vectors(thigh, shank)
            setattr(angles, f'{side}_knee_flexion', flexion)
            setattr(angles, f'{side}_knee_extension', 180 - flexion)

        # ===== SHOULDERS =====
        for side in ['left', 'right']:
            shoulder = get(f'{side}_shoulder')
            elbow = get(f'{side}_elbow')

            upper_arm = elbow - shoulder

            # Shoulder abduction
            abduction = angle_between_vectors(upper_arm, -torso_up) - 90
            setattr(angles, f'{side}_shoulder_abduction', abduction)

            # Shoulder horizontal abduction
            # Project upper arm onto transverse plane
            upper_arm_horiz = upper_arm.copy()
            upper_arm_horiz[2] = 0
            h_abd = signed_angle_about_axis(
                torso_forward, upper_arm_horiz, np.array([0, 0, 1])
            )
            setattr(angles, f'{side}_shoulder_horizontal_abduction', h_abd)

        # ===== ELBOWS =====
        for side in ['left', 'right']:
            shoulder = get(f'{side}_shoulder')
            elbow = get(f'{side}_elbow')
            wrist = get(f'{side}_wrist')

            upper_arm = shoulder - elbow
            forearm = wrist - elbow

            # Elbow flexion
            flexion = 180 - angle_between_vectors(upper_arm, forearm)
            setattr(angles, f'{side}_elbow_flexion', flexion)

        # ===== ARM ROTATION (for kinematic sequence) =====
        # Arm rotation in transverse plane (shoulder to wrist direction)
        for side in ['left', 'right']:
            shoulder = get(f'{side}_shoulder')
            wrist = get(f'{side}_wrist')

            # Project arm onto horizontal plane
            arm_vec = wrist - shoulder
            arm_horiz = np.array([arm_vec[0], arm_vec[1], 0])
            arm_horiz_len = np.linalg.norm(arm_horiz)

            if arm_horiz_len > 1e-6:
                arm_angle = np.degrees(np.arctan2(arm_horiz[1], arm_horiz[0]))
                setattr(angles, f'{side}_arm_rotation', arm_angle)

        # ===== HEAD ANGLES =====
        head = get('head')
        neck = get('neck')
        head_vec = head - neck
        head_vec_norm = np.linalg.norm(head_vec)
        if head_vec_norm > 1e-6:
            head_vec = head_vec / head_vec_norm
            # Head extension (forward/back tilt)
            angles.head_extension = np.degrees(np.arctan2(-head_vec[0], head_vec[2]))
            # Head lateral flexion
            angles.head_lateral_flexion_clockwise = np.degrees(np.arctan2(head_vec[1], head_vec[2]))
            # Head twist (rotation in transverse plane)
            angles.head_twist_clockwise = np.degrees(np.arctan2(head_vec[1], head_vec[0]))

        # ===== ANKLE DORSIFLEXION =====
        for side in ['left', 'right']:
            knee = get(f'{side}_knee')
            ankle = get(f'{side}_ankle')
            # Use shank direction relative to vertical
            shank = knee - ankle
            shank_norm = np.linalg.norm(shank)
            if shank_norm > 1e-6:
                # Dorsiflexion is angle of shank from vertical
                dorsi = np.degrees(np.arctan2(
                    np.sqrt(shank[0]**2 + shank[1]**2), shank[2]
                ))
                setattr(angles, f'{side}_ankle_dorsiflexion', dorsi)

        # ===== HIP ANGLES W.R.T. TRUNK (UPLIFT naming) =====
        for side in ['left', 'right']:
            hip = get(f'{side}_hip')
            knee = get(f'{side}_knee')
            thigh = knee - hip

            # Hip flexion w.r.t. trunk
            flexion = angle_between_vectors(thigh, -torso_up)
            setattr(angles, f'{side}_hip_flexion_with_respect_to_trunk', 180 - flexion)

            # Hip adduction w.r.t. trunk (negative of abduction)
            thigh_frontal = thigh.copy()
            thigh_frontal[0] = 0
            adduction = -signed_angle_about_axis(-vertical, thigh_frontal, np.array([1, 0, 0]))
            setattr(angles, f'{side}_hip_adduction_with_respect_to_trunk', adduction)

            # Hip internal rotation
            # Simplified: rotation of thigh about its long axis
            setattr(angles, f'{side}_hip_internal_rotation_with_respect_to_trunk', 0.0)  # Placeholder

        # ===== HIP ANGLES W.R.T. PELVIS =====
        for side in ['left', 'right']:
            hip = get(f'{side}_hip')
            knee = get(f'{side}_knee')
            thigh = knee - hip
            thigh_frontal = thigh.copy()
            thigh_frontal[0] = 0

            # Adduction relative to pelvis
            pelvis_up = np.cross(pelvis_forward, pelvis_right)
            adduction_pelvis = -signed_angle_about_axis(-pelvis_up, thigh_frontal, pelvis_forward)
            setattr(angles, f'{side}_hip_adduction_with_respect_to_pelvis', adduction_pelvis)
            setattr(angles, f'{side}_hip_internal_rotation_with_respect_to_pelvis', 0.0)  # Placeholder

        # ===== KNEE VARUS AND ROTATION =====
        for side in ['left', 'right']:
            hip = get(f'{side}_hip')
            knee = get(f'{side}_knee')
            ankle = get(f'{side}_ankle')

            thigh = knee - hip
            shank = ankle - knee

            # Knee varus (lateral bowing) - simplified
            # Cross product gives rotation axis, z-component indicates varus/valgus
            knee_cross = np.cross(thigh, shank)
            setattr(angles, f'{side}_knee_varus', np.degrees(np.arctan2(knee_cross[1], np.linalg.norm(shank))))
            setattr(angles, f'{side}_knee_internal_rotation', 0.0)  # Placeholder - needs full bone tracking

        # ===== SHOULDER ANGLES (UPLIFT naming) =====
        for side in ['left', 'right']:
            shoulder = get(f'{side}_shoulder')
            elbow = get(f'{side}_elbow')
            upper_arm = elbow - shoulder

            # Shoulder flexion (arm raised forward)
            arm_sagittal = np.array([upper_arm[0], 0, upper_arm[2]])
            flexion = np.degrees(np.arctan2(arm_sagittal[0], -arm_sagittal[2]))
            setattr(angles, f'{side}_shoulder_flexion', flexion)

            # Shoulder adduction (negative of abduction)
            abduction = angle_between_vectors(upper_arm, -torso_up) - 90
            setattr(angles, f'{side}_shoulder_adduction', -abduction)
            setattr(angles, f'{side}_shoulder_abduction', abduction)  # Legacy name

            # Shoulder external rotation
            setattr(angles, f'{side}_shoulder_external_rotation', 0.0)  # Placeholder
            setattr(angles, f'{side}_shoulder_rotation', 0.0)  # Legacy name

            # Shoulder horizontal adduction
            upper_arm_horiz = upper_arm.copy()
            upper_arm_horiz[2] = 0
            h_add = -signed_angle_about_axis(torso_forward, upper_arm_horiz, np.array([0, 0, 1]))
            setattr(angles, f'{side}_shoulder_horizontal_adduction', h_add)

        # ===== GLOBAL PELVIS/TRUNK (UPLIFT naming) =====
        # Pelvis obliquity (lateral tilt) - computed earlier
        hip_axis = right_hip - left_hip
        hip_axis_len = np.linalg.norm(hip_axis)
        pelvis_obliquity = np.degrees(np.arcsin(np.clip(hip_axis[2] / hip_axis_len, -1, 1))) if hip_axis_len > 1e-6 else 0.0

        # Torso flexion and lateral tilt - computed from torso_vec
        if torso_vec_norm > 1e-6:
            cos_angle = np.clip(torso_vec[2] / torso_vec_norm, -1, 1)
            torso_flexion_val = 90 - np.degrees(np.arccos(cos_angle))
        else:
            torso_flexion_val = 0.0

        shoulder_axis = right_shoulder - left_shoulder
        shoulder_axis_len = np.linalg.norm(shoulder_axis)
        torso_lateral_tilt_val = np.degrees(np.arcsin(np.clip(shoulder_axis[2] / shoulder_axis_len, -1, 1))) if shoulder_axis_len > 1e-6 else 0.0

        # Assign UPLIFT-named attributes
        angles.pelvis_global_tilt = pelvis_obliquity
        angles.pelvis_global_rotation = np.degrees(pelvis_angle - global_angle)
        angles.trunk_global_flexion = torso_flexion_val
        angles.trunk_global_tilt = torso_lateral_tilt_val
        angles.trunk_global_rotation = np.degrees(torso_angle - global_angle)
        angles.trunk_lateral_flexion_right = torso_lateral_tilt_val

        # Legacy names for backwards compatibility with dashboard
        angles.pelvis_rotation = angles.pelvis_global_rotation
        angles.pelvis_tilt = angles.pelvis_global_tilt
        angles.pelvis_obliquity = pelvis_obliquity
        angles.torso_rotation = angles.trunk_global_rotation
        angles.torso_flexion = torso_flexion_val
        angles.torso_lateral_tilt = torso_lateral_tilt_val

        # ===== CENTER OF MASS (simplified) =====
        # Trunk COM: midpoint between hip center and shoulder center
        trunk_com = (hip_center + shoulder_center) / 2
        angles.trunk_center_of_mass_x = trunk_com[0]
        angles.trunk_center_of_mass_y = trunk_com[1]
        angles.trunk_center_of_mass_z = trunk_com[2]

        # Whole body COM: weighted average of segment COMs (simplified)
        # Using just the major segments
        left_knee = get('left_knee')
        right_knee = get('right_knee')
        left_ankle = get('left_ankle')
        right_ankle = get('right_ankle')

        leg_com = (left_knee + right_knee + left_ankle + right_ankle) / 4
        arm_com = (get('left_elbow') + get('right_elbow') + get('left_wrist') + get('right_wrist')) / 4

        # Weights: trunk ~50%, legs ~30%, arms ~10%, head ~10%
        head_pos = get('head')
        whole_body_com = 0.5 * trunk_com + 0.3 * leg_com + 0.1 * arm_com + 0.1 * head_pos
        angles.whole_body_center_of_mass_x = whole_body_com[0]
        angles.whole_body_center_of_mass_y = whole_body_com[1]
        angles.whole_body_center_of_mass_z = whole_body_com[2]

        return angles


def calculate_angular_velocities_3d(angles_list: List[JointAngles3D],
                                    fps: float) -> List[dict]:
    """
    Calculate angular velocities for all angles.

    Args:
        angles_list: List of JointAngles3D over time
        fps: Frames per second

    Returns:
        List of dictionaries with angular velocities (deg/s)
    """
    dt = 1.0 / fps
    velocities = []

    # Get all angle attribute names
    angle_attrs = [attr for attr in dir(angles_list[0])
                   if not attr.startswith('_') and attr != 'timestamp']

    for i, angles in enumerate(angles_list):
        vel_dict = {'timestamp': angles.timestamp}

        for attr in angle_attrs:
            val = getattr(angles, attr)
            if val is not None and i > 0:
                prev_val = getattr(angles_list[i-1], attr)
                if prev_val is not None:
                    vel_dict[f'{attr}_velocity'] = (val - prev_val) / dt
                else:
                    vel_dict[f'{attr}_velocity'] = None
            else:
                vel_dict[f'{attr}_velocity'] = None

        velocities.append(vel_dict)

    return velocities


if __name__ == "__main__":
    # Test with synthetic 3D data
    np.random.seed(42)

    # Create a simple pose (standing)
    joints_3d = np.array([
        [0, 0, 1.0],      # hip_center
        [0.15, 0, 1.0],   # right_hip
        [0.15, 0, 0.5],   # right_knee
        [0.15, 0, 0.0],   # right_ankle
        [-0.15, 0, 1.0],  # left_hip
        [-0.15, 0, 0.5],  # left_knee
        [-0.15, 0, 0.0],  # left_ankle
        [0, 0, 1.2],      # spine
        [0, 0, 1.5],      # neck
        [0, 0, 1.6],      # head
        [0, 0, 1.7],      # head_top
        [-0.2, 0, 1.5],   # left_shoulder
        [-0.2, 0, 1.2],   # left_elbow
        [-0.2, 0, 0.9],   # left_wrist
        [0.2, 0, 1.5],    # right_shoulder
        [0.2, 0, 1.2],    # right_elbow
        [0.2, 0, 0.9],    # right_wrist
    ])

    calculator = JointAngleCalculator3D()
    angles = calculator.calculate(joints_3d, timestamp=0.0)

    print("3D Joint Angles (standing pose):")
    print(f"  Pelvis rotation: {angles.pelvis_rotation:.1f}°")
    print(f"  Torso rotation: {angles.torso_rotation:.1f}°")
    print(f"  Hip-shoulder separation: {angles.hip_shoulder_separation:.1f}°")
    print(f"  Left knee flexion: {angles.left_knee_flexion:.1f}°")
    print(f"  Right knee flexion: {angles.right_knee_flexion:.1f}°")
    print(f"  Left elbow flexion: {angles.left_elbow_flexion:.1f}°")
    print(f"  Right elbow flexion: {angles.right_elbow_flexion:.1f}°")
