"""
Point-of-Interest (POI) metrics for baseball hitting.

Extracted from OpenBiomechanics baseball_hitting/README.md.
These are the key metrics commonly analyzed in hitting biomechanics.
"""

from dataclasses import dataclass, field
from typing import Optional, List
import numpy as np


@dataclass
class HittingPOIMetrics:
    """
    Point-of-interest metrics for a baseball swing.

    Naming convention follows OBP:
        - _fm = at first move
        - _fp = at foot plant
        - _hs = at heel strike
        - _contact = at contact
        - _max = maximum value during swing
        - _x, _y, _z = component directions
    """

    # Identifiers
    session_swing: Optional[str] = None

    # Outcome metrics
    exit_velo_mph: Optional[float] = None
    bat_speed_mph_contact: Optional[float] = None
    attack_angle_contact: Optional[float] = None  # degrees, + = upward

    # Pelvis angles at key events (degrees)
    pelvis_angle_fm_x: Optional[float] = None  # Posterior(+)/Anterior(-) tilt
    pelvis_angle_fm_y: Optional[float] = None  # Lateral tilt
    pelvis_angle_fm_z: Optional[float] = None  # Rotation toward(+)/away(-) from mound
    pelvis_angle_fp_x: Optional[float] = None
    pelvis_angle_fp_y: Optional[float] = None
    pelvis_angle_fp_z: Optional[float] = None

    # Torso angles at key events (degrees)
    torso_angle_fm_x: Optional[float] = None  # Extension(+)/Flexion(-)
    torso_angle_fm_y: Optional[float] = None  # Lateral tilt
    torso_angle_fm_z: Optional[float] = None  # Rotation toward(+)/away(-) from mound
    torso_angle_fp_x: Optional[float] = None
    torso_angle_fp_y: Optional[float] = None
    torso_angle_fp_z: Optional[float] = None

    # X-Factor (torso-pelvis separation) at key events (degrees)
    x_factor_fm_z: Optional[float] = None  # Hip-shoulder separation at first move
    x_factor_fp_z: Optional[float] = None  # Hip-shoulder separation at foot plant
    x_factor_hs_z: Optional[float] = None  # Hip-shoulder separation at heel strike

    # Maximum X-Factor during swing
    torso_pelvis_stride_max_z: Optional[float] = None  # Max separation during stride
    torso_pelvis_swing_max_x: Optional[float] = None   # Max separation during swing

    # Angular velocities (deg/s)
    pelvis_angular_velocity_seq_max: Optional[float] = None  # Peak pelvis rotation speed
    torso_angular_velocity_seq_max: Optional[float] = None   # Peak torso rotation speed

    # Sequencing (pelvis should peak before torso)
    pelvis_angular_velocity_fp: Optional[float] = None  # Pelvis speed at foot plant
    torso_angular_velocity_fp: Optional[float] = None   # Torso speed at foot plant

    # Lead knee angles (degrees)
    lead_knee_launchpos_x: Optional[float] = None  # Lead knee angle at foot plant
    lead_knee_stride_max_x: Optional[float] = None # Max knee angle during stride

    # Rear elbow angles (degrees)
    rear_elbow_fm_x: Optional[float] = None        # Rear elbow flexion at first move
    rear_elbow_launchpos_x: Optional[float] = None # Rear elbow flexion at foot plant

    # Hand speed (mph or deg/s depending on calculation method)
    hand_speed_mag_max: Optional[float] = None     # Maximum hand speed
    hand_speed_mag_fp: Optional[float] = None      # Hand speed at foot plant

    # Center of gravity
    max_cog_velo_x: Optional[float] = None  # Max COG velocity toward pitcher (m/s)

    # Bat-torso connection angle (degrees)
    bat_torso_angle_ds_y: Optional[float] = None  # "Early connection" - vertical bat angle relative to torso


@dataclass
class SwingEvents:
    """Timing of key events in a swing (in seconds from start of capture)."""

    first_move: Optional[float] = None      # Initial movement
    load: Optional[float] = None            # Maximum coil
    heel_strike: Optional[float] = None     # Front heel contact
    foot_plant: Optional[float] = None      # Front foot fully planted
    max_hip_shoulder_sep: Optional[float] = None  # Peak X-factor
    down_swing: Optional[float] = None      # Start of forward swing
    contact: Optional[float] = None         # Ball contact


class HittingMetricsCalculator:
    """
    Calculate hitting POI metrics from time-series joint angle data.

    This is a simplified implementation. Full implementation would require:
    - 3D joint positions
    - Bat tracking
    - Event detection algorithms
    """

    def __init__(self, fps: float, bats: str = 'R'):
        """
        Initialize calculator.

        Args:
            fps: Video frames per second
            bats: Batting side ('L' or 'R')
        """
        self.fps = fps
        self.bats = bats.upper()

        # Map anatomical to functional sides
        if self.bats == 'R':
            self.lead_side = 'left'
            self.rear_side = 'right'
        else:
            self.lead_side = 'right'
            self.rear_side = 'left'

    def detect_events(self,
                      joint_angles: List,
                      joint_velocities: Optional[np.ndarray] = None) -> SwingEvents:
        """
        Detect swing events from kinematic data.

        This is a simplified heuristic implementation.
        Production code would need more sophisticated detection.

        Args:
            joint_angles: List of JointAngles objects
            joint_velocities: Optional array of angular velocities

        Returns:
            SwingEvents with detected timings
        """
        events = SwingEvents()

        if not joint_angles:
            return events

        timestamps = [a.timestamp for a in joint_angles]

        # Extract relevant angles based on batting side
        if self.lead_side == 'left':
            lead_knee = [a.left_knee_flexion or 0 for a in joint_angles]
        else:
            lead_knee = [a.right_knee_flexion or 0 for a in joint_angles]

        lead_knee = np.array(lead_knee)

        # Simple heuristics (these would need tuning for real data):

        # First move: First significant change in posture
        # (simplified: first 10% of swing)
        events.first_move = timestamps[int(len(timestamps) * 0.1)]

        # Foot plant: When lead knee flexion stabilizes after stride
        # (simplified: look for minimum in middle portion)
        mid_start = int(len(lead_knee) * 0.3)
        mid_end = int(len(lead_knee) * 0.7)
        if mid_end > mid_start:
            mid_section = lead_knee[mid_start:mid_end]
            fp_idx = mid_start + np.argmin(mid_section)
            events.foot_plant = timestamps[fp_idx]

        # Contact: End of swing (simplified)
        events.contact = timestamps[-1]

        return events

    def calculate_metrics(self,
                          joint_angles: List,
                          events: SwingEvents) -> HittingPOIMetrics:
        """
        Calculate POI metrics at detected events.

        Args:
            joint_angles: List of JointAngles objects
            events: SwingEvents with detected timings

        Returns:
            HittingPOIMetrics with calculated values
        """
        metrics = HittingPOIMetrics()

        if not joint_angles:
            return metrics

        timestamps = np.array([a.timestamp for a in joint_angles])

        def get_angle_at_time(t: Optional[float], attr: str) -> Optional[float]:
            """Get angle value closest to time t."""
            if t is None:
                return None
            idx = np.argmin(np.abs(timestamps - t))
            return getattr(joint_angles[idx], attr)

        def get_max_angle(attr: str, start_t: float = None, end_t: float = None) -> Optional[float]:
            """Get maximum angle value in time range."""
            values = []
            for a in joint_angles:
                if start_t and a.timestamp < start_t:
                    continue
                if end_t and a.timestamp > end_t:
                    continue
                val = getattr(a, attr)
                if val is not None:
                    values.append(val)
            return max(values) if values else None

        # Calculate metrics at events
        lead_knee_attr = f'{self.lead_side}_knee_flexion'
        rear_elbow_attr = f'{self.rear_side}_elbow_flexion'

        # Lead knee at foot plant
        metrics.lead_knee_launchpos_x = get_angle_at_time(
            events.foot_plant, lead_knee_attr
        )

        # Max lead knee during stride
        metrics.lead_knee_stride_max_x = get_max_angle(
            lead_knee_attr, events.first_move, events.foot_plant
        )

        # Rear elbow at first move and foot plant
        metrics.rear_elbow_fm_x = get_angle_at_time(
            events.first_move, rear_elbow_attr
        )
        metrics.rear_elbow_launchpos_x = get_angle_at_time(
            events.foot_plant, rear_elbow_attr
        )

        # Torso at foot plant (using torso_lean as proxy for torso_x)
        metrics.torso_angle_fp_x = get_angle_at_time(
            events.foot_plant, 'torso_lean'
        )

        return metrics


# Reference: Full list of POI metrics from OBP hitting data
OBP_HITTING_POI_FIELDS = [
    'session_swing',
    'exit_velo_mph_x',
    'blast_bat_speed_mph_x',
    'bat_speed_mph_contact_x',
    'sweet_spot_velo_mph_contact_x',
    'sweet_spot_velo_mph_contact_y',
    'sweet_spot_velo_mph_contact_z',
    'bat_torso_angle_connection_x',
    'attack_angle_contact_x',
    'bat_speed_mph_max_x',
    'bat_speed_xy_max_x',
    'bat_torso_angle_ds_y',
    'hand_speed_blast_bat_mph_max_x',
    'hand_speed_mag_max_x',
    'pelvis_angle_fm_x',
    'pelvis_angle_fm_y',
    'pelvis_angle_fm_z',
    'pelvis_angle_fp_x',
    'pelvis_angle_fp_y',
    'pelvis_angle_fp_z',
    'pelvis_angle_hs_x',
    'pelvis_angle_hs_y',
    'pelvis_angle_hs_z',
    'torso_angle_fm_x',
    'torso_angle_fm_y',
    'torso_angle_fm_z',
    'torso_angle_fp_x',
    'torso_angle_fp_y',
    'torso_angle_fp_z',
    'torso_angle_hs_x',
    'torso_angle_hs_y',
    'torso_angle_hs_z',
    'x_factor_fm_x',
    'x_factor_fm_y',
    'x_factor_fm_z',
    'x_factor_fp_x',
    'x_factor_fp_y',
    'x_factor_fp_z',
    'pelvis_angular_velocity_seq_max_x',
    'torso_angular_velocity_seq_max_x',
    'max_cog_velo_x',
]
