"""
Event detection for baseball hitting.

Detects key events in a swing:
- First Move: Initial movement toward the pitch
- Load: Maximum coil/weight shift back
- Stride: Foot lift and forward movement
- Heel Strike: Front heel contacts ground
- Foot Plant: Front foot fully planted
- Swing Initiation: Start of forward bat movement
- Contact: Bat contacts ball

Uses velocity thresholds and kinematic patterns.
"""

import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass
from scipy.signal import find_peaks, savgol_filter
from scipy.ndimage import gaussian_filter1d


@dataclass
class SwingEvents:
    """Detected swing events with timestamps and confidence."""
    first_move: Optional[float] = None
    first_move_confidence: float = 0.0

    load: Optional[float] = None
    load_confidence: float = 0.0

    heel_strike: Optional[float] = None
    heel_strike_confidence: float = 0.0

    foot_plant: Optional[float] = None
    foot_plant_confidence: float = 0.0

    swing_start: Optional[float] = None
    swing_start_confidence: float = 0.0

    contact: Optional[float] = None
    contact_confidence: float = 0.0

    # Frame indices (useful for indexing into arrays)
    first_move_frame: Optional[int] = None
    load_frame: Optional[int] = None
    heel_strike_frame: Optional[int] = None
    foot_plant_frame: Optional[int] = None
    swing_start_frame: Optional[int] = None
    contact_frame: Optional[int] = None


class SwingEventDetector:
    """
    Detect key events in a baseball swing from kinematic data.

    Uses multiple signals:
    - Hip/pelvis position and velocity
    - Knee angles
    - Hand/wrist position and velocity
    - Center of mass movement
    """

    def __init__(self, fps: float, bats: str = 'R'):
        """
        Initialize detector.

        Args:
            fps: Video frame rate
            bats: Batting side ('L' for left, 'R' for right)
        """
        self.fps = fps
        self.dt = 1.0 / fps
        self.bats = bats.upper()

        # Set lead/rear based on handedness
        if self.bats == 'R':
            self.lead_side = 'left'
            self.rear_side = 'right'
        else:
            self.lead_side = 'right'
            self.rear_side = 'left'

        # Detection thresholds (can be tuned)
        self.velocity_threshold = 0.05  # m/s - minimum velocity for movement
        self.knee_lift_threshold = 10.0  # degrees - minimum knee flexion change

    def detect_events(self,
                      joint_positions: np.ndarray,
                      joint_angles: Optional[np.ndarray] = None,
                      timestamps: Optional[np.ndarray] = None) -> SwingEvents:
        """
        Detect swing events from kinematic data.

        Args:
            joint_positions: Array of shape (n_frames, n_joints, 2 or 3)
                            Joint positions over time
            joint_angles: Optional array of shape (n_frames, n_angles)
                         Pre-calculated joint angles
            timestamps: Optional array of timestamps

        Returns:
            SwingEvents with detected event times
        """
        n_frames = len(joint_positions)

        if timestamps is None:
            timestamps = np.arange(n_frames) * self.dt

        events = SwingEvents()

        # Calculate derived signals
        signals = self._calculate_signals(joint_positions, joint_angles)

        # Detect each event
        events = self._detect_first_move(signals, timestamps, events)
        events = self._detect_load(signals, timestamps, events)
        events = self._detect_foot_events(signals, timestamps, events)
        events = self._detect_swing_start(signals, timestamps, events)
        events = self._detect_contact(signals, timestamps, events)

        return events

    def _calculate_signals(self,
                           positions: np.ndarray,
                           angles: Optional[np.ndarray]) -> dict:
        """Calculate derived signals used for event detection."""
        n_frames = len(positions)

        # Determine joint indices based on skeleton format
        # Assuming YOLO format: 11=L_hip, 12=R_hip, 9=L_wrist, 10=R_wrist, etc.
        if self.lead_side == 'left':
            lead_hip_idx, rear_hip_idx = 11, 12
            lead_knee_idx, rear_knee_idx = 13, 14
            lead_ankle_idx, rear_ankle_idx = 15, 16
            lead_wrist_idx, rear_wrist_idx = 9, 10
        else:
            lead_hip_idx, rear_hip_idx = 12, 11
            lead_knee_idx, rear_knee_idx = 14, 13
            lead_ankle_idx, rear_ankle_idx = 16, 15
            lead_wrist_idx, rear_wrist_idx = 10, 9

        signals = {}

        # Extract positions (handle both 2D and 3D)
        def get_pos(idx):
            if positions.shape[2] >= 2:
                return positions[:, idx, :2]  # x, y
            return positions[:, idx]

        # Hip positions
        lead_hip = get_pos(lead_hip_idx)
        rear_hip = get_pos(rear_hip_idx)
        hip_center = (lead_hip + rear_hip) / 2

        # Ankle positions
        lead_ankle = get_pos(lead_ankle_idx)

        # Wrist positions (hands)
        lead_wrist = get_pos(lead_wrist_idx)
        rear_wrist = get_pos(rear_wrist_idx)
        hand_center = (lead_wrist + rear_wrist) / 2

        # Calculate velocities (smoothed)
        def calc_velocity(pos):
            vel = np.gradient(pos, self.dt, axis=0)
            # Smooth velocity
            vel = gaussian_filter1d(vel, sigma=2, axis=0)
            return vel

        signals['hip_center'] = hip_center
        signals['hip_velocity'] = calc_velocity(hip_center)
        signals['hip_speed'] = np.linalg.norm(signals['hip_velocity'], axis=1)

        signals['lead_ankle'] = lead_ankle
        signals['lead_ankle_velocity'] = calc_velocity(lead_ankle)
        signals['lead_ankle_speed'] = np.linalg.norm(signals['lead_ankle_velocity'], axis=1)

        # Lead ankle height (y-coordinate, lower = higher in image coords)
        signals['lead_ankle_height'] = -lead_ankle[:, 1]  # Negate for intuitive interpretation

        signals['hand_center'] = hand_center
        signals['hand_velocity'] = calc_velocity(hand_center)
        signals['hand_speed'] = np.linalg.norm(signals['hand_velocity'], axis=1)

        # Hip lateral movement (toward pitcher)
        signals['hip_x'] = hip_center[:, 0]
        signals['hip_x_velocity'] = np.gradient(signals['hip_x'], self.dt)
        signals['hip_x_velocity'] = gaussian_filter1d(signals['hip_x_velocity'], sigma=2)

        # Weight shift (rear-to-lead)
        hip_separation = lead_hip - rear_hip
        signals['weight_shift'] = hip_separation[:, 0]  # x-component

        return signals

    def _detect_first_move(self,
                           signals: dict,
                           timestamps: np.ndarray,
                           events: SwingEvents) -> SwingEvents:
        """
        Detect first move - initial movement toward the pitch.

        Looks for first sustained increase in hip velocity.
        """
        hip_speed = signals['hip_speed']

        # Find first point where hip speed exceeds threshold and stays elevated
        threshold = np.percentile(hip_speed[:10], 90)  # Baseline from first 10 frames
        threshold = max(threshold * 2, self.velocity_threshold * 100)  # Scale for pixel units

        for i in range(5, len(hip_speed) - 5):
            # Check if speed increases and stays elevated
            if hip_speed[i] > threshold:
                if np.mean(hip_speed[i:i+5]) > threshold:
                    events.first_move = timestamps[i]
                    events.first_move_frame = i
                    events.first_move_confidence = 0.7
                    break

        # Fallback: use 10% into the sequence
        if events.first_move is None:
            idx = int(len(timestamps) * 0.1)
            events.first_move = timestamps[idx]
            events.first_move_frame = idx
            events.first_move_confidence = 0.3

        return events

    def _detect_load(self,
                     signals: dict,
                     timestamps: np.ndarray,
                     events: SwingEvents) -> SwingEvents:
        """
        Detect load position - maximum coil/weight shift back.

        Looks for peak in rear weight shift before forward movement.
        """
        if events.first_move_frame is None:
            return events

        weight_shift = signals['weight_shift']
        start_idx = events.first_move_frame

        # Look for minimum weight shift (max toward rear) after first move
        # But before the swing (first 40% of remaining frames)
        search_end = start_idx + int((len(timestamps) - start_idx) * 0.4)
        search_region = weight_shift[start_idx:search_end]

        if len(search_region) > 5:
            # Find local minimum (maximum rear shift)
            min_idx = np.argmin(search_region)
            load_idx = start_idx + min_idx

            events.load = timestamps[load_idx]
            events.load_frame = load_idx
            events.load_confidence = 0.6

        return events

    def _detect_foot_events(self,
                            signals: dict,
                            timestamps: np.ndarray,
                            events: SwingEvents) -> SwingEvents:
        """
        Detect heel strike and foot plant.

        Heel strike: Lead foot first contacts ground (end of stride)
        Foot plant: Lead foot fully planted (stable base)
        """
        lead_ankle_height = signals['lead_ankle_height']
        lead_ankle_speed = signals['lead_ankle_speed']

        start_idx = events.load_frame if events.load_frame else int(len(timestamps) * 0.2)

        # Find peak ankle height (top of leg lift) after load
        search_start = start_idx
        search_end = int(len(timestamps) * 0.7)

        if search_end <= search_start:
            search_end = search_start + 10

        search_region = lead_ankle_height[search_start:search_end]

        if len(search_region) > 3:
            # Find when ankle height peaks then drops
            peak_idx = search_start + np.argmax(search_region)

            # Heel strike: ankle velocity drops sharply (foot stops moving down)
            ankle_vel_after_peak = lead_ankle_speed[peak_idx:search_end]

            if len(ankle_vel_after_peak) > 3:
                # Find where velocity drops significantly
                vel_threshold = np.max(ankle_vel_after_peak) * 0.3

                for i, vel in enumerate(ankle_vel_after_peak):
                    if vel < vel_threshold and i > 2:
                        hs_idx = peak_idx + i
                        events.heel_strike = timestamps[hs_idx]
                        events.heel_strike_frame = hs_idx
                        events.heel_strike_confidence = 0.6
                        break

            # Foot plant: shortly after heel strike when ankle is stable
            if events.heel_strike_frame:
                # Look for minimum ankle movement after heel strike
                stable_search = lead_ankle_speed[events.heel_strike_frame:events.heel_strike_frame + 10]
                if len(stable_search) > 2:
                    fp_offset = np.argmin(stable_search)
                    fp_idx = events.heel_strike_frame + fp_offset

                    events.foot_plant = timestamps[fp_idx]
                    events.foot_plant_frame = fp_idx
                    events.foot_plant_confidence = 0.6

        # Fallback: estimate based on swing timing
        if events.foot_plant is None:
            fp_idx = int(len(timestamps) * 0.5)
            events.foot_plant = timestamps[fp_idx]
            events.foot_plant_frame = fp_idx
            events.foot_plant_confidence = 0.3

        return events

    def _detect_swing_start(self,
                            signals: dict,
                            timestamps: np.ndarray,
                            events: SwingEvents) -> SwingEvents:
        """
        Detect swing initiation - start of forward bat/hand movement.

        Looks for rapid increase in hand speed toward the pitcher.
        """
        hand_speed = signals['hand_speed']
        hand_velocity = signals['hand_velocity']

        # Start search around foot plant
        start_idx = events.foot_plant_frame if events.foot_plant_frame else int(len(timestamps) * 0.4)
        start_idx = max(0, start_idx - 5)  # Look slightly before foot plant

        # Find rapid hand acceleration
        hand_accel = np.gradient(hand_speed, self.dt)
        hand_accel = gaussian_filter1d(hand_accel, sigma=2)

        search_region = hand_accel[start_idx:]

        if len(search_region) > 5:
            # Find peak acceleration
            peaks, properties = find_peaks(search_region, height=np.percentile(search_region, 70))

            if len(peaks) > 0:
                # Take first significant peak
                swing_idx = start_idx + peaks[0]
                events.swing_start = timestamps[swing_idx]
                events.swing_start_frame = swing_idx
                events.swing_start_confidence = 0.7

        return events

    def _detect_contact(self,
                        signals: dict,
                        timestamps: np.ndarray,
                        events: SwingEvents) -> SwingEvents:
        """
        Detect contact - bat contacts ball.

        Without bat tracking, estimate from:
        - Peak hand speed (contact is near peak)
        - Sudden deceleration
        """
        hand_speed = signals['hand_speed']

        # Start search after swing start
        start_idx = events.swing_start_frame if events.swing_start_frame else int(len(timestamps) * 0.5)

        search_region = hand_speed[start_idx:]

        if len(search_region) > 3:
            # Contact is typically at or just after peak hand speed
            peak_idx = np.argmax(search_region)
            contact_idx = start_idx + peak_idx

            events.contact = timestamps[contact_idx]
            events.contact_frame = contact_idx
            events.contact_confidence = 0.5  # Lower confidence without ball tracking

        # Fallback: use 85% of sequence
        if events.contact is None:
            contact_idx = int(len(timestamps) * 0.85)
            events.contact = timestamps[contact_idx]
            events.contact_frame = contact_idx
            events.contact_confidence = 0.2

        return events


def detect_swing_events_simple(joint_angles: List,
                                fps: float,
                                bats: str = 'R') -> SwingEvents:
    """
    Simplified event detection using only joint angles.

    For use when full joint positions aren't available.

    Args:
        joint_angles: List of JointAngles objects
        fps: Frames per second
        bats: Batting side

    Returns:
        SwingEvents
    """
    events = SwingEvents()
    dt = 1.0 / fps

    if not joint_angles:
        return events

    timestamps = np.array([a.timestamp for a in joint_angles])
    n_frames = len(timestamps)

    # Extract lead knee flexion
    lead_side = 'left' if bats.upper() == 'R' else 'right'
    knee_attr = f'{lead_side}_knee_flexion'

    knee_angles = np.array([getattr(a, knee_attr) or 0 for a in joint_angles])

    # Smooth
    if len(knee_angles) > 5:
        knee_angles = gaussian_filter1d(knee_angles, sigma=2)

    knee_velocity = np.gradient(knee_angles, dt)

    # First move: 10% into sequence
    fm_idx = int(n_frames * 0.1)
    events.first_move = timestamps[fm_idx]
    events.first_move_frame = fm_idx
    events.first_move_confidence = 0.3

    # Foot plant: when lead knee flexion is minimum (most extended during landing)
    mid_start = int(n_frames * 0.3)
    mid_end = int(n_frames * 0.7)

    if mid_end > mid_start:
        mid_section = knee_angles[mid_start:mid_end]
        fp_local_idx = np.argmin(mid_section)
        fp_idx = mid_start + fp_local_idx

        events.foot_plant = timestamps[fp_idx]
        events.foot_plant_frame = fp_idx
        events.foot_plant_confidence = 0.5

    # Contact: 85% into sequence
    contact_idx = int(n_frames * 0.85)
    events.contact = timestamps[contact_idx]
    events.contact_frame = contact_idx
    events.contact_confidence = 0.2

    return events


if __name__ == "__main__":
    # Test with synthetic data
    n_frames = 100
    fps = 60.0

    # Create fake joint positions (17 YOLO keypoints, 2D)
    positions = np.random.rand(n_frames, 17, 2) * 500 + 100

    # Simulate a swing motion
    t = np.linspace(0, 1, n_frames)

    # Lead ankle (idx 15) lifts then plants
    positions[:, 15, 1] = 400 - 50 * np.sin(np.pi * t)  # Lower y = higher

    # Hands (idx 9, 10) accelerate forward
    positions[:, 9, 0] = 200 + 200 * (t ** 2)
    positions[:, 10, 0] = 210 + 200 * (t ** 2)

    detector = SwingEventDetector(fps=fps, bats='R')
    events = detector.detect_events(positions)

    print("Detected Events:")
    print(f"  First Move: {events.first_move:.3f}s (confidence: {events.first_move_confidence:.2f})")
    print(f"  Load: {events.load}s (confidence: {events.load_confidence:.2f})" if events.load else "  Load: not detected")
    print(f"  Foot Plant: {events.foot_plant:.3f}s (confidence: {events.foot_plant_confidence:.2f})" if events.foot_plant else "  Foot Plant: not detected")
    print(f"  Contact: {events.contact:.3f}s (confidence: {events.contact_confidence:.2f})" if events.contact else "  Contact: not detected")
