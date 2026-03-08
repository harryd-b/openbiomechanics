"""
Multi-view triangulation 3D pose estimation.

Uses 2D detections from multiple calibrated cameras
to triangulate actual 3D positions via geometry.

This is the most accurate method when cameras are properly calibrated.
"""

import numpy as np
from typing import List, Optional, Tuple, Dict
from pathlib import Path
import cv2
import json

from .base import PoseEstimator3D, Pose3DResult, H36M_JOINTS


class CameraCalibration:
    """Camera intrinsic and extrinsic parameters."""

    def __init__(self,
                 K: np.ndarray,
                 dist: np.ndarray,
                 R: np.ndarray,
                 t: np.ndarray):
        """
        Initialize camera calibration.

        Args:
            K: Intrinsic matrix (3x3)
            dist: Distortion coefficients (5,) or (8,)
            R: Rotation matrix (3x3) world to camera
            t: Translation vector (3,) world to camera
        """
        self.K = K
        self.dist = dist
        self.R = R
        self.t = t.reshape(3, 1) if t.ndim == 1 else t

        # Projection matrix P = K @ [R|t]
        self.P = K @ np.hstack([R, self.t])

    def project(self, points_3d: np.ndarray) -> np.ndarray:
        """Project 3D points to 2D."""
        points_h = np.hstack([points_3d, np.ones((len(points_3d), 1))])
        projected = (self.P @ points_h.T).T
        return projected[:, :2] / projected[:, 2:3]

    def undistort_points(self, points_2d: np.ndarray) -> np.ndarray:
        """Remove lens distortion from 2D points."""
        points_undist = cv2.undistortPoints(
            points_2d.reshape(-1, 1, 2).astype(np.float32),
            self.K, self.dist, P=self.K
        )
        return points_undist.reshape(-1, 2)

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            'K': self.K.tolist(),
            'dist': self.dist.tolist(),
            'R': self.R.tolist(),
            't': self.t.flatten().tolist()
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'CameraCalibration':
        """Deserialize from dictionary."""
        return cls(
            K=np.array(d['K']),
            dist=np.array(d['dist']),
            R=np.array(d['R']),
            t=np.array(d['t'])
        )

    @classmethod
    def from_defaults(cls,
                      image_width: int,
                      image_height: int,
                      camera_distance: float = 5.0,
                      camera_angle: float = 0.0) -> 'CameraCalibration':
        """
        Create default calibration based on typical camera setup.

        Camera is positioned at given distance and angle, looking at origin.
        Uses standard camera projection: P = K[R | t] where t = -R*C
        and C is the camera center in world coordinates.

        Args:
            image_width: Image width in pixels
            image_height: Image height in pixels
            camera_distance: Distance from subject (meters)
            camera_angle: Angle around Y axis from +Z axis (degrees)
                         0 = camera on +Z axis looking toward origin
                         90 = camera on +X axis looking toward origin
        """
        # Estimate focal length (typical phone camera ~60 degree FOV)
        focal_length = max(image_width, image_height) * 1.2

        K = np.array([
            [focal_length, 0, image_width / 2],
            [0, focal_length, image_height / 2],
            [0, 0, 1]
        ], dtype=np.float64)

        dist = np.zeros(5)

        # Camera position in world coordinates
        angle_rad = np.radians(camera_angle)
        camera_pos = np.array([
            camera_distance * np.sin(angle_rad),  # X
            0,                                      # Y (camera at same height)
            camera_distance * np.cos(angle_rad)   # Z
        ])

        # Camera looks toward origin
        # Camera Z-axis points from camera toward origin (forward direction)
        forward = -camera_pos / np.linalg.norm(camera_pos)  # Unit vector toward origin
        up = np.array([0, 1, 0])  # World up
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)  # Recompute up to ensure orthogonality

        # R transforms world coords to camera coords
        # Camera axes in world coords: right, up, -forward (OpenCV convention: Z forward)
        R = np.array([right, up, -forward])  # Each row is a camera axis in world coords

        # t = -R * camera_pos (transforms origin to camera frame)
        t = -R @ camera_pos

        return cls(K=K, dist=dist, R=R, t=t)


class TriangulationEstimator(PoseEstimator3D):
    """
    Multi-view triangulation 3D pose estimation.

    Requires calibrated cameras for best results.
    Falls back to estimated calibration if not provided.
    """

    def __init__(self,
                 calibration_path: Optional[str] = None,
                 yolo_model: str = 'yolov8m-pose.pt',
                 min_confidence: float = 0.3,
                 ransac_threshold: float = 0.01,
                 keep_absolute: bool = False):
        """
        Initialize triangulation estimator.

        Args:
            calibration_path: Path to calibration JSON file
            yolo_model: YOLOv8 model for 2D detection
            min_confidence: Minimum 2D detection confidence
            ransac_threshold: RANSAC threshold for robust triangulation
            keep_absolute: If True, keep absolute world coordinates (don't normalize)
        """
        super().__init__(name='triangulation')
        self.calibration_path = calibration_path
        self.yolo_model_name = yolo_model
        self.min_confidence = min_confidence
        self.ransac_threshold = ransac_threshold
        self.keep_absolute = keep_absolute

        self.cameras: List[CameraCalibration] = []
        self.pose_estimator = None

    def initialize(self) -> None:
        """Load calibration and 2D detector."""
        from pose_estimation import PoseEstimator

        self.pose_estimator = PoseEstimator(model_name=self.yolo_model_name)

        # Load calibration if provided
        if self.calibration_path and Path(self.calibration_path).exists():
            self.load_calibration(self.calibration_path)

        self._is_initialized = True

    def load_calibration(self, path: str) -> None:
        """Load camera calibration from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)

        self.cameras = [
            CameraCalibration.from_dict(cam_data)
            for cam_data in data['cameras']
        ]
        print(f"Loaded calibration for {len(self.cameras)} cameras")

    def save_calibration(self, path: str) -> None:
        """Save camera calibration to JSON file."""
        data = {
            'cameras': [cam.to_dict() for cam in self.cameras]
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    def calibrate_from_checkerboard(self,
                                    video_paths: List[str],
                                    checkerboard_size: Tuple[int, int] = (9, 6),
                                    square_size: float = 0.025,
                                    max_frames: int = 50) -> bool:
        """
        Calibrate cameras using checkerboard pattern.

        Args:
            video_paths: List of video paths (one per camera)
            checkerboard_size: (columns, rows) of internal corners
            square_size: Size of each square in meters
            max_frames: Maximum frames to use per camera

        Returns:
            True if calibration succeeded
        """
        print("Starting camera calibration...")

        # Prepare object points
        objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
        objp *= square_size

        self.cameras = []

        for cam_idx, video_path in enumerate(video_paths):
            print(f"Calibrating camera {cam_idx + 1}...")

            objpoints = []
            imgpoints = []
            img_size = None

            cap = cv2.VideoCapture(video_path)
            frame_count = 0

            while frame_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break

                if img_size is None:
                    img_size = (frame.shape[1], frame.shape[0])

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)

                if ret:
                    # Refine corners
                    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                    corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

                    objpoints.append(objp)
                    imgpoints.append(corners_refined)

                frame_count += 1

            cap.release()

            if len(objpoints) < 10:
                print(f"  Warning: Only found {len(objpoints)} valid frames")
                if len(objpoints) < 3:
                    print(f"  Failed: Not enough checkerboard detections")
                    continue

            # Calibrate camera
            ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
                objpoints, imgpoints, img_size, None, None
            )

            if ret:
                # Use first frame's extrinsics as reference
                R, _ = cv2.Rodrigues(rvecs[0])
                t = tvecs[0]

                self.cameras.append(CameraCalibration(K, dist, R, t))
                print(f"  Calibrated successfully (reprojection error: {ret:.3f})")
            else:
                print(f"  Calibration failed")

        if len(self.cameras) >= 2:
            print(f"Calibration complete for {len(self.cameras)} cameras")
            return True
        else:
            print("Calibration failed: Need at least 2 cameras")
            return False

    def set_camera_params(self,
                          video_paths: List[str],
                          camera_distances: List[float],
                          camera_angles: List[float] = None) -> None:
        """
        Set camera parameters from known distances and angles.

        Args:
            video_paths: List of video paths
            camera_distances: Distance from subject for each camera (meters)
            camera_angles: Angle from front for each camera (degrees)
        """
        if camera_angles is None:
            # Default: first camera at 0°, second at 90°
            camera_angles = [0, 90][:len(video_paths)]

        self.cameras = []

        for i, video_path in enumerate(video_paths):
            cap = cv2.VideoCapture(video_path)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()

            cam = CameraCalibration.from_defaults(
                width, height,
                camera_distance=camera_distances[i],
                camera_angle=camera_angles[i]
            )
            self.cameras.append(cam)

        print(f"Set parameters for {len(self.cameras)} cameras")

    def triangulate_point(self,
                          points_2d: List[np.ndarray],
                          confidences: List[float]) -> Tuple[np.ndarray, float]:
        """
        Triangulate a single 3D point from multiple 2D observations.

        Args:
            points_2d: List of (2,) arrays, one per camera
            confidences: Confidence per observation

        Returns:
            (3D point, reprojection error)
        """
        # Filter by confidence
        valid_idx = [i for i, c in enumerate(confidences) if c >= self.min_confidence]

        if len(valid_idx) < 2:
            # Not enough views
            return np.zeros(3), float('inf')

        # Build DLT matrix
        A = []
        for i in valid_idx:
            P = self.cameras[i].P
            x, y = points_2d[i]
            w = confidences[i]

            A.append(w * (x * P[2] - P[0]))
            A.append(w * (y * P[2] - P[1]))

        A = np.array(A)

        # Solve using SVD
        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1]
        X = X[:3] / X[3]  # Homogeneous to 3D

        # Calculate reprojection error
        errors = []
        for i in valid_idx:
            projected = self.cameras[i].project(X.reshape(1, 3))[0]
            error = np.linalg.norm(projected - points_2d[i])
            errors.append(error)

        return X, np.mean(errors)

    def triangulate_pose(self,
                         poses_2d: List[np.ndarray],
                         confidences_2d: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Triangulate full 3D pose from multiple 2D poses.

        Args:
            poses_2d: List of (17, 2) arrays, one per camera
            confidences_2d: List of (17,) confidence arrays

        Returns:
            (joints_3d (17, 3), per-joint errors (17,))
        """
        n_joints = 17
        joints_3d = np.zeros((n_joints, 3))
        errors = np.zeros(n_joints)

        for j in range(n_joints):
            points = [pose[j] for pose in poses_2d]
            confs = [conf[j] for conf in confidences_2d]

            joints_3d[j], errors[j] = self.triangulate_point(points, confs)

        return joints_3d, errors

    def estimate_frame(self,
                       frame: np.ndarray,
                       frame_number: int,
                       timestamp: float) -> Optional[Pose3DResult]:
        """
        Not supported for triangulation - use estimate_video_multiview.
        """
        raise NotImplementedError(
            "Triangulation requires multiple views. Use estimate_video_multiview()"
        )

    def estimate_video_multiview(self,
                                 video_paths: List[str],
                                 max_frames: Optional[int] = None) -> List[Pose3DResult]:
        """
        Estimate 3D poses from multiple synchronized videos.

        Args:
            video_paths: List of paths to synchronized videos
            max_frames: Maximum frames to process

        Returns:
            List of Pose3DResult
        """
        if not self._is_initialized:
            self.initialize()

        if len(video_paths) < 2:
            raise ValueError("Triangulation requires at least 2 videos")

        if len(self.cameras) != len(video_paths):
            raise ValueError(
                f"Camera count ({len(self.cameras)}) doesn't match "
                f"video count ({len(video_paths)}). Call set_camera_params() first."
            )

        # Get 2D poses from each camera
        print("Extracting 2D poses from each view...")
        all_poses_2d = []
        timestamps = None

        for i, video_path in enumerate(video_paths):
            print(f"  Processing camera {i + 1}...")
            poses = self.pose_estimator.process_video(video_path, max_frames=max_frames)
            all_poses_2d.append(poses)

            if timestamps is None:
                timestamps = [p.timestamp for p in poses]

        # Align frame counts
        min_frames = min(len(poses) for poses in all_poses_2d)
        print(f"Triangulating {min_frames} frames...")

        results = []

        for frame_idx in range(min_frames):
            # Collect 2D poses from all cameras
            poses_2d = []
            confidences_2d = []

            for cam_poses in all_poses_2d:
                kp = cam_poses[frame_idx].keypoints

                # Undistort points
                points = self.cameras[len(poses_2d)].undistort_points(kp[:, :2])
                poses_2d.append(points)

                conf = kp[:, 2] if kp.shape[1] > 2 else np.ones(17)
                confidences_2d.append(conf)

            # Triangulate
            joints_3d, errors = self.triangulate_pose(poses_2d, confidences_2d)

            # Normalize skeleton (unless keeping absolute coordinates)
            if not self.keep_absolute:
                joints_3d = self.normalize_skeleton(joints_3d)

            # Combined confidence (inverse of reprojection error)
            confidences = 1.0 / (errors + 1e-6)
            confidences = confidences / confidences.max()  # Normalize to [0, 1]

            results.append(Pose3DResult(
                joints_3d=joints_3d,
                confidences=confidences,
                frame_number=frame_idx,
                timestamp=timestamps[frame_idx],
                metadata={
                    'method': 'triangulation',
                    'reprojection_errors': errors.tolist()
                }
            ))

        return results

    def get_confidence_weights(self) -> np.ndarray:
        """
        Get per-joint reliability weights.

        Triangulation is generally reliable for all joints when
        cameras are properly calibrated.
        """
        weights = np.ones(17)

        # Slightly higher confidence for well-visible joints
        weights[H36M_JOINTS['pelvis']] = 1.1
        weights[H36M_JOINTS['spine']] = 1.1
        weights[H36M_JOINTS['neck']] = 1.05

        # Slightly lower for joints that may be occluded
        weights[H36M_JOINTS['left_wrist']] = 0.95
        weights[H36M_JOINTS['right_wrist']] = 0.95

        return weights / weights.sum()
