"""Temporal filtering for smooth pose trajectories."""

import numpy as np
from typing import List, Optional


class TemporalKalmanFilter:
    """
    Kalman filter for temporally smooth pose trajectories.

    Maintains state estimate (position + velocity) for each joint
    and updates based on observations.
    """

    def __init__(self,
                 n_joints: int = 17,
                 fps: float = 30.0,
                 process_noise: float = 0.1,
                 measurement_noise: float = 0.05,
                 velocity_decay: float = 0.95):
        """
        Initialize Kalman filter.

        Args:
            n_joints: Number of joints to track
            fps: Video frame rate
            process_noise: Process noise (higher = more responsive)
            measurement_noise: Measurement noise (higher = more smoothing)
            velocity_decay: Velocity decay factor (prevents drift)
        """
        self.n_joints = n_joints
        self.dt = 1.0 / fps
        self.velocity_decay = velocity_decay

        # State: [x, y, z, vx, vy, vz] for each joint
        self.state = np.zeros((n_joints, 6))
        self.covariance = np.eye(6)[None, :, :].repeat(n_joints, axis=0) * 0.1

        # Process model: x_new = x + v*dt
        self.F = np.array([
            [1, 0, 0, self.dt, 0, 0],
            [0, 1, 0, 0, self.dt, 0],
            [0, 0, 1, 0, 0, self.dt],
            [0, 0, 0, velocity_decay, 0, 0],
            [0, 0, 0, 0, velocity_decay, 0],
            [0, 0, 0, 0, 0, velocity_decay],
        ])

        # Observation model: measure position only
        self.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
        ])

        # Noise covariances
        self.Q = np.eye(6) * process_noise  # Process noise
        self.R = np.eye(3) * measurement_noise  # Measurement noise

        self.initialized = False

    def reset(self):
        """Reset filter state."""
        self.state = np.zeros((self.n_joints, 6))
        self.covariance = np.eye(6)[None, :, :].repeat(self.n_joints, axis=0) * 0.1
        self.initialized = False

    def predict(self):
        """Predict next state."""
        for j in range(self.n_joints):
            # State prediction
            self.state[j] = self.F @ self.state[j]

            # Covariance prediction
            self.covariance[j] = self.F @ self.covariance[j] @ self.F.T + self.Q

    def update(self, observation: np.ndarray, confidence: Optional[np.ndarray] = None):
        """
        Update state with observation.

        Args:
            observation: (17, 3) measured joint positions
            confidence: (17,) optional per-joint confidence
        """
        if not self.initialized:
            # Initialize state from first observation
            self.state[:, :3] = observation
            self.state[:, 3:] = 0  # Zero initial velocity
            self.initialized = True
            return

        if confidence is None:
            confidence = np.ones(self.n_joints)

        for j in range(self.n_joints):
            # Skip low-confidence observations
            if confidence[j] < 0.1:
                continue

            # Innovation
            y = observation[j] - self.H @ self.state[j]

            # Innovation covariance (scale by inverse confidence)
            R_scaled = self.R / (confidence[j] + 1e-6)
            S = self.H @ self.covariance[j] @ self.H.T + R_scaled

            # Kalman gain
            K = self.covariance[j] @ self.H.T @ np.linalg.inv(S)

            # State update
            self.state[j] = self.state[j] + K @ y

            # Covariance update
            I_KH = np.eye(6) - K @ self.H
            self.covariance[j] = I_KH @ self.covariance[j]

    def get_position(self) -> np.ndarray:
        """Get current position estimate."""
        return self.state[:, :3].copy()

    def get_velocity(self) -> np.ndarray:
        """Get current velocity estimate."""
        return self.state[:, 3:].copy()

    def filter_sequence(self,
                        poses: List[np.ndarray],
                        confidences: Optional[List[np.ndarray]] = None) -> List[np.ndarray]:
        """
        Filter entire pose sequence.

        Args:
            poses: List of (17, 3) pose arrays
            confidences: Optional list of (17,) confidence arrays

        Returns:
            Filtered list of (17, 3) pose arrays
        """
        self.reset()

        if confidences is None:
            confidences = [None] * len(poses)

        filtered = []

        for pose, conf in zip(poses, confidences):
            # Predict
            if self.initialized:
                self.predict()

            # Update
            self.update(pose, conf)

            # Get filtered position
            filtered.append(self.get_position())

        return filtered

    def smooth_sequence(self,
                        poses: List[np.ndarray],
                        confidences: Optional[List[np.ndarray]] = None) -> List[np.ndarray]:
        """
        Smooth sequence using forward-backward filtering.

        This provides better smoothing than single-pass filtering
        by using information from future frames.

        Args:
            poses: List of (17, 3) pose arrays
            confidences: Optional list of (17,) confidence arrays

        Returns:
            Smoothed list of (17, 3) pose arrays
        """
        # Forward pass
        forward = self.filter_sequence(poses, confidences)

        # Backward pass
        self.reset()
        if confidences:
            backward = self.filter_sequence(poses[::-1], confidences[::-1])[::-1]
        else:
            backward = self.filter_sequence(poses[::-1])[::-1]

        # Average forward and backward
        smoothed = [
            (f + b) / 2 for f, b in zip(forward, backward)
        ]

        return smoothed


class SimpleMovingAverage:
    """
    Simple moving average filter as alternative to Kalman.

    Useful when dynamics model assumptions don't hold.
    """

    def __init__(self, window_size: int = 5):
        """
        Initialize moving average filter.

        Args:
            window_size: Number of frames to average
        """
        self.window_size = window_size

    def filter_sequence(self, poses: List[np.ndarray]) -> List[np.ndarray]:
        """Apply moving average filter to sequence."""
        poses_array = np.stack(poses)
        n_frames = len(poses)

        filtered = []

        for i in range(n_frames):
            start = max(0, i - self.window_size // 2)
            end = min(n_frames, i + self.window_size // 2 + 1)

            filtered.append(np.mean(poses_array[start:end], axis=0))

        return filtered
