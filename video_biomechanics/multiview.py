"""
Multi-view video processing for improved 3D reconstruction.

Two camera views enable triangulation-based 3D pose estimation,
which is more accurate than single-view lifting models.

Typical setup:
- Camera 1: Side view (perpendicular to swing direction)
- Camera 2: Front/back view or 45-degree angle
"""

import numpy as np
import cv2
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass
from pathlib import Path

from pose_estimation import PoseEstimator, PoseFrame, Keypoints
from lifting_3d import convert_yolo_to_h36m


@dataclass
class CameraParams:
    """Camera intrinsic and extrinsic parameters."""
    # Intrinsic matrix (3x3)
    K: np.ndarray
    # Distortion coefficients
    dist: np.ndarray
    # Rotation matrix (3x3) - world to camera
    R: Optional[np.ndarray] = None
    # Translation vector (3x1) - world to camera
    t: Optional[np.ndarray] = None
    # Projection matrix (3x4) - K @ [R|t]
    P: Optional[np.ndarray] = None

    @classmethod
    def from_defaults(cls, image_width: int, image_height: int) -> 'CameraParams':
        """Create default camera parameters assuming standard lens."""
        # Estimate focal length (typical for phone cameras)
        focal_length = max(image_width, image_height) * 1.2

        K = np.array([
            [focal_length, 0, image_width / 2],
            [0, focal_length, image_height / 2],
            [0, 0, 1]
        ], dtype=np.float64)

        dist = np.zeros(5)

        return cls(K=K, dist=dist)


@dataclass
class MultiViewFrame:
    """Synchronized frame data from multiple cameras."""
    frame_number: int
    timestamp: float
    poses_2d: List[np.ndarray]  # List of (17, 3) arrays per camera
    pose_3d: Optional[np.ndarray] = None  # Triangulated (17, 3)
    confidences: Optional[np.ndarray] = None


def normalize_skeleton_to_meters(joints_3d: np.ndarray) -> np.ndarray:
    """
    Normalize a 3D skeleton to real-world scale (meters).

    Uses body proportions to estimate scale:
    - Average torso length (hip to shoulder): ~0.50m
    - Average thigh length: ~0.42m
    - Average shin length: ~0.40m

    Args:
        joints_3d: Array of shape (17, 3) in arbitrary units

    Returns:
        Normalized skeleton in meters, centered at hip
    """
    # Joint indices (H36M format)
    HIP_CENTER = 0
    LEFT_HIP = 4
    RIGHT_HIP = 1
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 14
    LEFT_KNEE = 5
    RIGHT_KNEE = 2

    # Calculate current torso length
    hip_center = (joints_3d[LEFT_HIP] + joints_3d[RIGHT_HIP]) / 2
    shoulder_center = (joints_3d[LEFT_SHOULDER] + joints_3d[RIGHT_SHOULDER]) / 2

    current_torso = np.linalg.norm(shoulder_center - hip_center)

    # Also check thigh length for robustness
    left_thigh = np.linalg.norm(joints_3d[LEFT_KNEE] - joints_3d[LEFT_HIP])
    right_thigh = np.linalg.norm(joints_3d[RIGHT_KNEE] - joints_3d[RIGHT_HIP])
    current_thigh = (left_thigh + right_thigh) / 2

    # Expected lengths in meters
    EXPECTED_TORSO = 0.50  # meters
    EXPECTED_THIGH = 0.42  # meters

    # Calculate scale factor (average of torso and thigh estimates)
    if current_torso > 1e-6:
        scale_torso = EXPECTED_TORSO / current_torso
    else:
        scale_torso = 1.0

    if current_thigh > 1e-6:
        scale_thigh = EXPECTED_THIGH / current_thigh
    else:
        scale_thigh = 1.0

    scale = (scale_torso + scale_thigh) / 2

    # Center on hip and scale
    normalized = (joints_3d - hip_center) * scale

    # Ensure Z is up (flip if average Z of shoulders is below hips)
    if normalized[LEFT_SHOULDER, 2] < normalized[LEFT_HIP, 2]:
        normalized[:, 2] = -normalized[:, 2]

    return normalized


class VideoSynchronizer:
    """
    Synchronize multiple video streams.

    Methods:
    - Frame-based: Assume videos start at same time, use frame numbers
    - Audio-based: Use audio waveform correlation (requires audio)
    - Visual-based: Use visual events (e.g., flash, clap) for sync
    """

    def __init__(self, method: str = 'frame'):
        """
        Initialize synchronizer.

        Args:
            method: Sync method - 'frame', 'audio', or 'visual'
        """
        self.method = method
        self.offsets = {}  # Camera ID -> frame offset

    def set_manual_offset(self, camera_id: int, offset_frames: int):
        """Manually set frame offset for a camera."""
        self.offsets[camera_id] = offset_frames

    def sync_by_motion(self,
                       poses_list: List[List[PoseFrame]],
                       fps: float) -> List[int]:
        """
        Automatically sync videos by correlating motion.

        Uses cross-correlation of joint velocities to find offset.

        Args:
            poses_list: List of pose sequences per camera
            fps: Frame rate

        Returns:
            List of frame offsets per camera (first camera is reference)
        """
        if len(poses_list) < 2:
            return [0] * len(poses_list)

        offsets = [0]  # First camera is reference

        # Get reference motion signal (sum of joint velocities)
        ref_signal = self._compute_motion_signal(poses_list[0])

        for i in range(1, len(poses_list)):
            signal = self._compute_motion_signal(poses_list[i])

            # Cross-correlate
            correlation = np.correlate(ref_signal, signal, mode='full')
            offset = np.argmax(correlation) - len(signal) + 1
            offsets.append(offset)

        return offsets

    def _compute_motion_signal(self, poses: List[PoseFrame]) -> np.ndarray:
        """Compute motion signal from pose sequence."""
        if len(poses) < 2:
            return np.zeros(len(poses))

        signal = []
        for i in range(1, len(poses)):
            diff = poses[i].keypoints[:, :2] - poses[i-1].keypoints[:, :2]
            motion = np.sum(np.linalg.norm(diff, axis=1))
            signal.append(motion)

        return np.array(signal)


class MultiViewTriangulator:
    """
    Triangulate 3D points from multiple 2D views.

    Uses Direct Linear Transform (DLT) for triangulation.
    """

    def __init__(self, camera_params: List[CameraParams]):
        """
        Initialize triangulator.

        Args:
            camera_params: List of CameraParams for each camera
        """
        self.cameras = camera_params
        self.n_cameras = len(camera_params)

    def triangulate_point(self,
                          points_2d: List[np.ndarray],
                          confidences: List[float] = None) -> Tuple[np.ndarray, float]:
        """
        Triangulate a single 3D point from multiple 2D observations.

        Args:
            points_2d: List of 2D points (x, y) from each camera
            confidences: Optional confidence weights per camera

        Returns:
            Tuple of (3D point, reprojection error)
        """
        if len(points_2d) < 2:
            raise ValueError("Need at least 2 views for triangulation")

        if confidences is None:
            confidences = [1.0] * len(points_2d)

        # Build DLT matrix
        A = []
        for i, (pt, cam, conf) in enumerate(zip(points_2d, self.cameras, confidences)):
            if cam.P is None:
                continue

            P = cam.P
            x, y = pt[0], pt[1]

            A.append(conf * (x * P[2, :] - P[0, :]))
            A.append(conf * (y * P[2, :] - P[1, :]))

        A = np.array(A)

        # Solve using SVD
        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1]
        X = X[:3] / X[3]  # Homogeneous to 3D

        # Calculate reprojection error
        error = self._reprojection_error(X, points_2d, confidences)

        return X, error

    def triangulate_pose(self,
                         poses_2d: List[np.ndarray],
                         min_confidence: float = 0.3) -> Tuple[np.ndarray, np.ndarray]:
        """
        Triangulate full pose from multiple 2D poses.

        Args:
            poses_2d: List of (17, 3) arrays [x, y, conf] per camera
            min_confidence: Minimum confidence to use a point

        Returns:
            Tuple of (3D pose array (17, 3), per-joint errors)
        """
        n_joints = poses_2d[0].shape[0]
        pose_3d = np.zeros((n_joints, 3))
        errors = np.zeros(n_joints)

        for j in range(n_joints):
            points = []
            confs = []
            valid_cams = []

            for i, pose in enumerate(poses_2d):
                conf = pose[j, 2] if pose.shape[1] > 2 else 1.0
                if conf >= min_confidence:
                    points.append(pose[j, :2])
                    confs.append(conf)
                    valid_cams.append(i)

            if len(points) >= 2:
                # Create subset of cameras for this joint
                sub_triangulator = MultiViewTriangulator(
                    [self.cameras[i] for i in valid_cams]
                )
                pose_3d[j], errors[j] = sub_triangulator.triangulate_point(points, confs)
            else:
                # Not enough views - use single view estimation
                pose_3d[j] = np.array([points[0][0], points[0][1], 0]) if points else np.zeros(3)
                errors[j] = float('inf')

        return pose_3d, errors

    def _reprojection_error(self,
                            point_3d: np.ndarray,
                            points_2d: List[np.ndarray],
                            weights: List[float]) -> float:
        """Calculate weighted reprojection error."""
        errors = []

        X_h = np.append(point_3d, 1)  # Homogeneous

        for pt_2d, cam, w in zip(points_2d, self.cameras, weights):
            if cam.P is None:
                continue

            projected = cam.P @ X_h
            projected = projected[:2] / projected[2]

            error = np.linalg.norm(projected - pt_2d)
            errors.append(w * error)

        return np.mean(errors) if errors else float('inf')


class MultiViewPipeline:
    """
    Process multiple video views for 3D biomechanics extraction.
    """

    def __init__(self,
                 pose_model: str = 'yolov8m-pose.pt',
                 bats: str = 'R',
                 use_ensemble: bool = True,
                 fusion_model_path: str = 'models/fusion_model.pt',
                 angle_model_path: str = 'models/angle_predictor.pt',
                 methods: List[str] = None):
        """
        Initialize multi-view pipeline.

        Args:
            pose_model: YOLOv8 pose model
            bats: Batting side
            use_ensemble: Use trained ensemble pose fusion and angle prediction
            fusion_model_path: Path to trained fusion model
            angle_model_path: Path to trained angle predictor
            methods: List of pose estimation methods to use
        """
        self.pose_estimator = PoseEstimator(model_name=pose_model)
        self.bats = bats
        self.synchronizer = VideoSynchronizer()
        self.cameras: List[CameraParams] = []
        self.fps = None
        self.calibrated = False  # Set to True when using calibrated camera distances
        self.methods = methods or ['yolo_lifting', 'motionbert', 'triangulation']

        # Ensemble pipeline and angle predictor
        self.use_ensemble = use_ensemble
        self.ensemble_pipeline = None
        self.angle_predictor = None

        if use_ensemble:
            self._init_ensemble(fusion_model_path, angle_model_path)

    def _init_ensemble(self, fusion_model_path: str, angle_model_path: str):
        """Initialize ensemble pipeline and angle predictor if models exist."""
        from pathlib import Path
        import os

        # Use absolute path relative to this file's directory
        base_dir = Path(__file__).parent
        fusion_path = base_dir / fusion_model_path
        angle_path = base_dir / angle_model_path

        print(f"[DEBUG] CWD: {os.getcwd()}")
        print(f"[DEBUG] Base dir: {base_dir}")
        print(f"[DEBUG] Fusion path: {fusion_path} (exists: {fusion_path.exists()})")
        print(f"[DEBUG] Angle path: {angle_path} (exists: {angle_path.exists()})")

        if fusion_path.exists() and angle_path.exists():
            try:
                from ensemble_pipeline import EnsemblePosePipeline
                from fusion.learned_angles import AnglePredictor

                print(f"Loading trained ensemble models with methods: {self.methods}")
                self.ensemble_pipeline = EnsemblePosePipeline(
                    methods=self.methods,
                    fusion_model_path=str(fusion_path)
                )

                self.angle_predictor = AnglePredictor()
                self.angle_predictor.load(str(angle_path))
                print("  Ensemble pipeline ready (3 methods including triangulation)")
                print("  Angle predictor ready")

            except Exception as e:
                print(f"Failed to load ensemble models: {e}")
                print("  Falling back to standard processing")
                self.use_ensemble = False
        else:
            if not fusion_path.exists():
                print(f"Fusion model not found: {fusion_path}")
            if not angle_path.exists():
                print(f"Angle model not found: {angle_path}")
            print("  Using standard processing")
            self.use_ensemble = False

    def _process_with_ensemble(self, video_paths: List[str], max_frames: Optional[int] = None) -> Dict:
        """Process videos using trained ensemble pipeline and angle predictor."""
        import pandas as pd
        import numpy as np
        from dataclasses import dataclass

        print("\n[ENSEMBLE MODE] Using trained pose fusion and angle prediction")

        # Set up camera parameters for triangulation if we have 2 videos
        if len(video_paths) >= 2 and 'triangulation' in self.ensemble_pipeline.estimators:
            camera_distances = [
                getattr(self, 'camera_distances', [3.0, 3.0])[0] if hasattr(self, 'camera_distances') else 3.0,
                getattr(self, 'camera_distances', [3.0, 3.0])[1] if hasattr(self, 'camera_distances') and len(self.camera_distances) > 1 else 3.0
            ]
            camera_angles = [0, 90]  # Default: side and back views
            self.ensemble_pipeline.estimators['triangulation'].set_camera_params(
                video_paths, camera_distances, camera_angles
            )
            print(f"  Set up triangulation: distances={camera_distances}m, angles={camera_angles}°")

        # Process with ensemble pipeline
        results = self.ensemble_pipeline.process_videos(video_paths, max_frames=max_frames)
        poses_3d = results.poses_3d
        self.fps = results.fps

        print(f"  Fused {len(poses_3d)} 3D poses")

        # Predict angles using trained model
        print("  Predicting joint angles with trained model...")
        poses_array = np.array(poses_3d)
        predicted_angles_df = self.angle_predictor.predict_dataframe(poses_array)

        # Also calculate angles with rule-based method for comparison/fallback
        from joint_angles_3d import JointAngleCalculator3D
        angle_calc = JointAngleCalculator3D()

        joint_angles = []
        for i, pose in enumerate(poses_3d):
            timestamp = results.timestamps[i] if i < len(results.timestamps) else i / self.fps
            angles = angle_calc.calculate(pose, timestamp=timestamp, frame_number=i)

            # Override with learned predictions for columns we trained on
            for col in self.angle_predictor.angle_names:
                if hasattr(angles, col):
                    setattr(angles, col, predicted_angles_df.iloc[i][col])

            joint_angles.append(angles)

        # Detect events
        print("  Detecting events...")
        from event_detection import SwingEventDetector
        detector = SwingEventDetector(fps=self.fps, bats=self.bats)
        events = detector.detect_events(poses_array)

        # Calculate metrics
        print("  Calculating metrics...")
        metrics = self._calculate_metrics(joint_angles, events)

        # Build timeseries dataframe with learned angle predictions
        timeseries_df = self._build_timeseries_df(joint_angles, poses_3d)

        # Add learned angles to timeseries
        for col in self.angle_predictor.angle_names:
            if col not in timeseries_df.columns:
                timeseries_df[col] = predicted_angles_df[col].values[:len(timeseries_df)]

        # Create multiview frames for compatibility
        multiview_frames = [
            MultiViewFrame(
                frame_number=i,
                timestamp=results.timestamps[i] if i < len(results.timestamps) else i / self.fps,
                poses_2d=[],
                pose_3d=pose
            )
            for i, pose in enumerate(poses_3d)
        ]

        return {
            'poses_2d': [],
            'multiview_frames': multiview_frames,
            'poses_3d': poses_3d,
            'joint_angles': joint_angles,
            'events': events,
            'metrics': metrics,
            'timeseries_df': timeseries_df,
            'fps': self.fps,
            'n_cameras': len(video_paths),
            'sync_offsets': [0] * len(video_paths),
            'method': 'ensemble'
        }

    def add_camera(self, params: CameraParams):
        """Add camera parameters."""
        self.cameras.append(params)

    def estimate_cameras_from_videos(self,
                                     video_paths: List[str],
                                     camera_distance: float = 5.0,
                                     camera_angle: float = 90.0,
                                     camera_distances: List[float] = None) -> List[CameraParams]:
        """
        Set up camera parameters from video properties and calibration data.

        Args:
            video_paths: Paths to video files
            camera_distance: Default distance from subject (meters) - used if camera_distances not provided
            camera_angle: Angle between cameras (degrees)
            camera_distances: List of actual distances for each camera [primary, secondary] in meters

        Returns:
            List of CameraParams
        """
        cameras = []

        # Use provided distances or default
        if camera_distances is None:
            camera_distances = [camera_distance] * len(video_paths)

        for i, path in enumerate(video_paths):
            cap = cv2.VideoCapture(path)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()

            dist = camera_distances[i] if i < len(camera_distances) else camera_distance

            # Create intrinsic params
            cam = CameraParams.from_defaults(width, height)

            # Set extrinsic params based on camera index and calibration
            if i == 0:
                # Camera 1: Primary view (looking along +X axis toward origin)
                R = np.eye(3)
                t = np.array([[dist], [0], [0]])
            else:
                # Camera 2: Secondary view at specified angle
                angle_rad = np.radians(camera_angle)
                R = np.array([
                    [np.cos(angle_rad), 0, np.sin(angle_rad)],
                    [0, 1, 0],
                    [-np.sin(angle_rad), 0, np.cos(angle_rad)]
                ])
                t = np.array([
                    [dist * np.cos(angle_rad)],
                    [0],
                    [dist * np.sin(angle_rad)]
                ])

            cam.R = R
            cam.t = t
            cam.P = cam.K @ np.hstack([R, t])

            cameras.append(cam)

        self.cameras = cameras
        self.calibrated = camera_distances is not None  # Flag for using triangulation
        return cameras

    def process_videos(self,
                       video_paths: List[str],
                       sync_method: str = 'motion',
                       max_frames: Optional[int] = None) -> Dict:
        """
        Process multiple videos and extract 3D biomechanics.

        Args:
            video_paths: List of paths to video files (1-2 videos)
            sync_method: How to sync videos - 'motion', 'manual', or 'none'
            max_frames: Maximum frames to process

        Returns:
            Dictionary with results
        """
        n_videos = len(video_paths)

        if n_videos == 0:
            raise ValueError("At least one video required")

        if n_videos > 2:
            print(f"Warning: Only using first 2 of {n_videos} videos")
            video_paths = video_paths[:2]

        print(f"Processing {len(video_paths)} video(s)...")

        # Get video properties
        cap = cv2.VideoCapture(video_paths[0])
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        # Use ensemble pipeline if available
        if self.use_ensemble and self.ensemble_pipeline is not None:
            return self._process_with_ensemble(video_paths, max_frames)

        # Estimate camera parameters if not provided
        if len(self.cameras) < len(video_paths):
            print("Estimating camera parameters...")
            self.estimate_cameras_from_videos(video_paths)

        # Step 1: Extract 2D poses from each video
        print("\nStep 1: Extracting 2D poses from each view...")
        all_poses = []
        for i, path in enumerate(video_paths):
            print(f"  Processing video {i+1}: {Path(path).name}")
            poses = self.pose_estimator.process_video(path, max_frames=max_frames)
            all_poses.append(poses)
            print(f"    Detected {len(poses)} frames")

        # Step 2: Synchronize videos
        if len(video_paths) > 1 and sync_method != 'none':
            print("\nStep 2: Synchronizing videos...")
            if sync_method == 'motion':
                offsets = self.synchronizer.sync_by_motion(all_poses, self.fps)
                print(f"  Detected offsets: {offsets}")
            else:
                offsets = [self.synchronizer.offsets.get(i, 0) for i in range(len(video_paths))]
        else:
            offsets = [0] * len(video_paths)

        # Step 3: Reconstruct 3D poses
        print("\nStep 3: Reconstructing 3D poses...")

        # Use triangulation if we have 2 calibrated cameras, otherwise use single-view lifting
        use_triangulation = len(video_paths) == 2 and getattr(self, 'calibrated', False)

        if use_triangulation:
            print("  Using calibrated multi-view triangulation...")
            multiview_frames = self._triangulate_sequence(all_poses, offsets)
        else:
            print("  Using single-view 3D lifting...")
            from lifting_3d import VideoPose3DLifter
            lifter = VideoPose3DLifter()

            # Use camera 1 (typically side view) for 3D lifting
            poses_2d = [p.keypoints for p in all_poses[0]]
            timestamps = [p.timestamp for p in all_poses[0]]
            poses_3d = lifter.lift_sequence(poses_2d, timestamps)

            multiview_frames = [
                MultiViewFrame(
                    frame_number=p.frame_number,
                    timestamp=p.timestamp,
                    poses_2d=[all_poses[0][i].keypoints],
                    pose_3d=p.joints_3d
                )
                for i, p in enumerate(poses_3d)
            ]

        print(f"  Reconstructed {len(multiview_frames)} 3D poses")

        # Step 4: Calculate joint angles
        print("\nStep 4: Calculating joint angles...")
        from joint_angles_3d import JointAngleCalculator3D
        angle_calc = JointAngleCalculator3D()

        joint_angles = []
        for frame in multiview_frames:
            if frame.pose_3d is not None:
                angles = angle_calc.calculate(frame.pose_3d, timestamp=frame.timestamp)
                joint_angles.append(angles)

        # Step 5: Detect events
        print("\nStep 5: Detecting events...")
        from event_detection import SwingEventDetector

        # Build position array from 3D poses
        positions = np.array([f.pose_3d for f in multiview_frames if f.pose_3d is not None])
        detector = SwingEventDetector(fps=self.fps, bats=self.bats)
        events = detector.detect_events(positions)

        # Step 6: Calculate metrics
        print("\nStep 6: Calculating metrics...")
        from hitting_metrics import HittingPOIMetrics
        metrics = self._calculate_metrics(joint_angles, events)

        # Build results
        import pandas as pd
        poses_3d_list = [f.pose_3d for f in multiview_frames]
        timeseries_df = self._build_timeseries_df(joint_angles, poses_3d_list)

        results = {
            'poses_2d': all_poses,
            'multiview_frames': multiview_frames,
            'poses_3d': [f.pose_3d for f in multiview_frames],
            'joint_angles': joint_angles,
            'events': events,
            'metrics': metrics,
            'timeseries_df': timeseries_df,
            'fps': self.fps,
            'n_cameras': len(video_paths),
            'sync_offsets': offsets
        }

        return results

    def _triangulate_sequence(self,
                              all_poses: List[List[PoseFrame]],
                              offsets: List[int]) -> List[MultiViewFrame]:
        """Triangulate 3D poses from synchronized 2D sequences."""
        triangulator = MultiViewTriangulator(self.cameras)

        # Align sequences using offsets
        min_len = min(len(poses) for poses in all_poses)
        for i, offset in enumerate(offsets):
            if offset > 0:
                all_poses[i] = all_poses[i][offset:]
            elif offset < 0:
                all_poses[i] = all_poses[i][:offset]

        min_len = min(len(poses) for poses in all_poses)

        frames = []
        for i in range(min_len):
            poses_2d = [all_poses[cam][i].keypoints for cam in range(len(all_poses))]

            pose_3d, errors = triangulator.triangulate_pose(poses_2d)

            # Convert from YOLO keypoint order to H36M skeleton format
            pose_3d_h36m = convert_yolo_to_h36m(pose_3d)

            # Normalize to meters using body proportions
            pose_3d_normalized = normalize_skeleton_to_meters(pose_3d_h36m)

            frames.append(MultiViewFrame(
                frame_number=i,
                timestamp=all_poses[0][i].timestamp,
                poses_2d=poses_2d,
                pose_3d=pose_3d_normalized,
                confidences=1.0 / (errors + 1e-6)
            ))

        return frames

    def _calculate_metrics(self, joint_angles, events):
        """Calculate POI metrics from joint angles."""
        from hitting_metrics import HittingPOIMetrics
        import numpy as np

        metrics = HittingPOIMetrics()

        if not joint_angles:
            return metrics

        timestamps = np.array([a.timestamp for a in joint_angles])

        def get_at_time(t, attr):
            if t is None:
                return None
            idx = np.argmin(np.abs(timestamps - t))
            return getattr(joint_angles[idx], attr, None)

        def get_max(attr, start_t=None, end_t=None):
            values = []
            for a in joint_angles:
                if start_t and a.timestamp < start_t:
                    continue
                if end_t and a.timestamp > end_t:
                    continue
                val = getattr(a, attr, None)
                if val is not None:
                    values.append(val)
            return max(values) if values else None

        lead_side = 'left' if self.bats == 'R' else 'right'
        rear_side = 'right' if self.bats == 'R' else 'left'

        fp_time = getattr(events, 'foot_plant', None)
        fm_time = getattr(events, 'first_move', None)

        metrics.lead_knee_launchpos_x = get_at_time(fp_time, f'{lead_side}_knee_flexion')
        metrics.rear_elbow_launchpos_x = get_at_time(fp_time, f'{rear_side}_elbow_flexion')
        metrics.x_factor_fp_z = get_at_time(fp_time, 'hip_shoulder_separation')
        metrics.torso_pelvis_stride_max_z = get_max('hip_shoulder_separation', fm_time, fp_time)

        # Angular velocities
        if len(joint_angles) > 1:
            dt = 1.0 / self.fps
            pelvis_rot = [a.pelvis_rotation or 0 for a in joint_angles]
            torso_rot = [a.torso_rotation or 0 for a in joint_angles]

            metrics.pelvis_angular_velocity_seq_max = float(np.max(np.abs(np.gradient(pelvis_rot, dt))))
            metrics.torso_angular_velocity_seq_max = float(np.max(np.abs(np.gradient(torso_rot, dt))))

        return metrics

    def _build_timeseries_df(self, joint_angles, poses_3d=None):
        """Build DataFrame from joint angles and 3D positions."""
        import pandas as pd
        import numpy as np

        # H36M joint names for output
        JOINT_NAMES = [
            'pelvis', 'right_hip', 'right_knee', 'right_ankle',
            'left_hip', 'left_knee', 'left_ankle',
            'spine', 'neck', 'head', 'head_top',
            'left_shoulder', 'left_elbow', 'left_wrist',
            'right_shoulder', 'right_elbow', 'right_wrist'
        ]

        records = []
        for i, angles in enumerate(joint_angles):
            record = {'timestamp': angles.timestamp, 'frame': i}

            # Add joint angles
            for attr in dir(angles):
                if attr.startswith('_') or attr == 'timestamp':
                    continue
                val = getattr(angles, attr)
                if isinstance(val, (int, float)) and val is not None:
                    record[attr] = val

            # Add 3D positions if available
            if poses_3d is not None and i < len(poses_3d):
                pose = poses_3d[i]
                if pose is not None and hasattr(pose, 'shape'):
                    for j, name in enumerate(JOINT_NAMES):
                        if j < len(pose):
                            record[f'{name}_3d_x'] = pose[j, 0]
                            record[f'{name}_3d_y'] = pose[j, 1]
                            record[f'{name}_3d_z'] = pose[j, 2]

            records.append(record)

        df = pd.DataFrame(records)

        # Unwrap rotation angles to remove ±180° discontinuities
        rotation_cols = [
            'pelvis_rotation', 'pelvis_global_rotation',
            'torso_rotation', 'trunk_global_rotation',
            'hip_shoulder_separation', 'trunk_twist_clockwise',
            'head_twist_clockwise'
        ]
        for col in rotation_cols:
            if col in df.columns and not df[col].isna().all():
                radians = np.deg2rad(df[col].values)
                unwrapped = np.unwrap(radians)
                df[col] = np.rad2deg(unwrapped)

        if len(df) > 1 and self.fps:
            dt = 1.0 / self.fps
            # Compute velocities for angle columns only (not positions)
            angle_cols = [c for c in df.columns if c not in ['timestamp', 'frame']
                         and '_3d_' not in c and not c.endswith('_velocity')]
            for col in angle_cols:
                if not df[col].isna().all():
                    try:
                        df[f'{col}_velocity'] = np.gradient(
                            df[col].ffill().bfill(), dt
                        )
                    except:
                        pass

        return df


def process_multiview(video_paths: List[str],
                      bats: str = 'R',
                      output_dir: str = './output') -> Dict:
    """
    Convenience function to process multi-view videos.

    Args:
        video_paths: List of 1-2 video paths
        bats: Batting side
        output_dir: Output directory

    Returns:
        Results dictionary
    """
    pipeline = MultiViewPipeline(bats=bats)
    results = pipeline.process_videos(video_paths)

    # Export
    from pathlib import Path
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    results['timeseries_df'].to_csv(out_path / 'timeseries.csv', index=False)

    metrics = results['metrics']
    metrics_dict = {k: v for k, v in metrics.__dict__.items() if v is not None}
    import pandas as pd
    pd.DataFrame([metrics_dict]).to_csv(out_path / 'poi_metrics.csv', index=False)

    print(f"\nResults exported to {output_dir}")

    return results


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python multiview.py <video1> [video2] [-b L|R] [-o output_dir]")
        print("\nExamples:")
        print("  python multiview.py swing_side.mp4")
        print("  python multiview.py swing_side.mp4 swing_front.mp4")
        print("  python multiview.py swing_side.mp4 swing_front.mp4 -b L -o ./results")
        sys.exit(1)

    # Parse arguments
    videos = []
    bats = 'R'
    output = './output'

    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == '-b' and i + 1 < len(sys.argv):
            bats = sys.argv[i + 1]
            i += 2
        elif arg == '-o' and i + 1 < len(sys.argv):
            output = sys.argv[i + 1]
            i += 2
        else:
            videos.append(arg)
            i += 1

    print("=" * 60)
    print("MULTI-VIEW BIOMECHANICS PIPELINE")
    print("=" * 60)
    print(f"Videos: {videos}")
    print(f"Batting side: {bats}")
    print(f"Output: {output}")
    print("=" * 60)

    results = process_multiview(videos, bats=bats, output_dir=output)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Processed {results['n_cameras']} camera(s)")
    print(f"Frames: {len(results['joint_angles'])}")

    m = results['metrics']
    if m.x_factor_fp_z:
        print(f"X-Factor at foot plant: {m.x_factor_fp_z:.1f}°")
    if m.pelvis_angular_velocity_seq_max:
        print(f"Peak pelvis velocity: {m.pelvis_angular_velocity_seq_max:.1f}°/s")
