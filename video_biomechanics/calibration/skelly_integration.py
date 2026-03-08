"""
Integration with skellytracker and aniposelib for 2D pose detection and triangulation.

This module uses:
- skellytracker: For 2D pose detection (YOLO-based)
- aniposelib: For camera math and triangulation

Install dependencies:
    pip install skellytracker aniposelib

Usage:
    from skelly_integration import SkellyPipeline

    pipeline = SkellyPipeline()
    pipeline.calibrate_from_videos(videos, calibration_frame=60)
    poses_3d = pipeline.process_videos(videos)
"""

import sys
import numpy as np
import cv2
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Try to import skellytracker
try:
    from skellytracker import YOLOPoseTracker
    HAS_SKELLYTRACKER = True
except ImportError:
    HAS_SKELLYTRACKER = False
    print("skellytracker not installed. Install with: pip install skellytracker")

# Try to import aniposelib
try:
    from aniposelib.cameras import CameraGroup, Camera
    HAS_ANIPOSELIB = True
except ImportError:
    HAS_ANIPOSELIB = False
    print("aniposelib not installed. Install with: pip install aniposelib")

# Home plate dimensions (meters)
PLATE_WIDTH = 0.4318  # 17 inches
PLATE_DEPTH = 0.2159  # 8.5 inches
PLATE_POINTS_3D = np.array([
    [0, 0, 0],                                    # Front left
    [PLATE_WIDTH, 0, 0],                          # Front right
    [PLATE_WIDTH, 0, PLATE_DEPTH],                # Back right
    [PLATE_WIDTH / 2, 0, PLATE_DEPTH + PLATE_WIDTH / 2],  # Apex
    [0, 0, PLATE_DEPTH],                          # Back left
], dtype=np.float32)


@dataclass
class CameraParams:
    """Camera intrinsic and extrinsic parameters."""
    matrix: np.ndarray      # 3x3 intrinsic matrix
    dist: np.ndarray        # Distortion coefficients
    rvec: np.ndarray        # Rotation vector
    tvec: np.ndarray        # Translation vector
    size: Tuple[int, int]   # (width, height)

    def get_extrinsics_mat(self) -> np.ndarray:
        """Get 3x4 extrinsic matrix [R|t]."""
        R, _ = cv2.Rodrigues(self.rvec)
        return self.matrix @ np.hstack([R, self.tvec])


class SkellyPipeline:
    """
    Motion capture pipeline using skellytracker + aniposelib.

    Combines:
    - Home plate calibration for camera extrinsics
    - skellytracker for 2D pose detection
    - aniposelib triangulation for 3D reconstruction
    """

    def __init__(self, model_size: str = "medium"):
        self.model_size = model_size
        self.cameras: Dict[str, CameraParams] = {}
        self.camera_group: Optional[CameraGroup] = None

        # Initialize tracker if available
        if HAS_SKELLYTRACKER:
            self.tracker = YOLOPoseTracker(model_size=model_size)
        else:
            self.tracker = None

    def detect_plate_corners(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Detect home plate corners in image using color + shape analysis."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # White detection
        lower_white = np.array([0, 0, 180])
        upper_white = np.array([180, 50, 255])
        mask = cv2.inRange(hsv, lower_white, upper_white)

        # Clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
            area = cv2.contourArea(contour)
            if area < 1000:
                continue

            # Approximate polygon
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            if len(approx) == 5:
                corners = approx.reshape(-1, 2).astype(np.float32)
                corners = self._order_pentagon_corners(corners)
                return corners

        return None

    def _order_pentagon_corners(self, corners: np.ndarray) -> np.ndarray:
        """Order pentagon corners: front-left, front-right, back-right, apex, back-left."""
        centroid = corners.mean(axis=0)

        # Find apex (furthest from centroid in Y direction)
        y_dists = corners[:, 1] - centroid[1]
        apex_idx = np.argmax(y_dists)  # Assuming apex is at top of image (higher Y)

        # Reorder starting from front-left
        ordered = []
        angles = np.arctan2(corners[:, 1] - centroid[1], corners[:, 0] - centroid[0])
        sorted_indices = np.argsort(angles)

        # Find where apex is in sorted order
        apex_pos = np.where(sorted_indices == apex_idx)[0][0]

        # Rotate so apex is at position 3 (index 3 in our convention)
        roll = (apex_pos - 3) % 5
        reordered = np.roll(sorted_indices, -roll)

        return corners[reordered]

    def calibrate_camera(self, image: np.ndarray, view_name: str) -> bool:
        """Calibrate a single camera using home plate detection."""
        corners = self.detect_plate_corners(image)
        if corners is None:
            return False

        h, w = image.shape[:2]

        # Estimate camera matrix (assume standard FOV)
        fov = 70  # degrees
        focal_length = w / (2 * np.tan(np.radians(fov / 2)))
        camera_matrix = np.array([
            [focal_length, 0, w / 2],
            [0, focal_length, h / 2],
            [0, 0, 1]
        ], dtype=np.float64)

        dist_coeffs = np.zeros(5)

        # Solve PnP
        success, rvec, tvec = cv2.solvePnP(
            PLATE_POINTS_3D,
            corners,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return False

        self.cameras[view_name] = CameraParams(
            matrix=camera_matrix,
            dist=dist_coeffs,
            rvec=rvec,
            tvec=tvec,
            size=(w, h)
        )

        return True

    def calibrate_from_videos(
        self,
        videos: Dict[str, str],
        calibration_frame: int = 60
    ) -> bool:
        """Calibrate all cameras from video files."""
        self.cameras = {}

        for view_name, video_path in videos.items():
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, calibration_frame)
            ret, frame = cap.read()
            cap.release()

            if not ret:
                print(f"Failed to read frame from {video_path}")
                return False

            if not self.calibrate_camera(frame, view_name):
                print(f"Failed to calibrate {view_name}")
                return False

            print(f"Calibrated {view_name}: {self.cameras[view_name].size}")

        # Build aniposelib CameraGroup if available
        if HAS_ANIPOSELIB and len(self.cameras) >= 2:
            self._build_camera_group()

        return True

    def _build_camera_group(self):
        """Build aniposelib CameraGroup from calibrated cameras."""
        cameras = []

        for name, params in self.cameras.items():
            R, _ = cv2.Rodrigues(params.rvec)

            cam = Camera(
                matrix=params.matrix,
                dist=params.dist,
                size=params.size,
                rvec=params.rvec.flatten(),
                tvec=params.tvec.flatten(),
                name=name
            )
            cameras.append(cam)

        self.camera_group = CameraGroup(cameras)
        print(f"Built CameraGroup with {len(cameras)} cameras")

    def detect_2d_pose(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Detect 2D pose keypoints using skellytracker."""
        if self.tracker is None:
            raise RuntimeError("skellytracker not available")

        self.tracker.process_image(image)
        tracked = self.tracker.tracked_objects.get("tracked_person")

        if tracked is None or tracked.extra.get("landmarks") is None:
            return None

        landmarks = tracked.extra["landmarks"]
        if landmarks.shape[0] == 0:
            return None

        return landmarks[0]  # Shape: (num_keypoints, 2) or (num_keypoints, 3) with conf

    def triangulate(self, points_2d: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Triangulate 3D points from multi-view 2D detections.

        Args:
            points_2d: Dict mapping view names to (N, 2) arrays of 2D points

        Returns:
            (N, 3) array of 3D points
        """
        if HAS_ANIPOSELIB and self.camera_group is not None:
            return self._triangulate_aniposelib(points_2d)
        else:
            return self._triangulate_dlt(points_2d)

    def _triangulate_aniposelib(self, points_2d: Dict[str, np.ndarray]) -> np.ndarray:
        """Triangulate using aniposelib CameraGroup."""
        view_names = list(self.cameras.keys())
        n_points = points_2d[view_names[0]].shape[0]
        n_cams = len(view_names)

        # Stack points: (n_cams, n_points, 2)
        stacked = np.zeros((n_cams, n_points, 2))
        for i, name in enumerate(view_names):
            pts = points_2d[name]
            if pts.shape[1] > 2:
                pts = pts[:, :2]  # Strip confidence if present
            stacked[i] = pts

        # Use aniposelib triangulation
        points_3d = self.camera_group.triangulate(stacked, progress=False)

        return points_3d

    def _triangulate_dlt(self, points_2d: Dict[str, np.ndarray]) -> np.ndarray:
        """Fallback DLT triangulation without aniposelib."""
        view_names = list(self.cameras.keys())
        n_points = points_2d[view_names[0]].shape[0]

        # Build projection matrices
        P_matrices = {}
        for name, params in self.cameras.items():
            P_matrices[name] = params.get_extrinsics_mat()

        points_3d = np.zeros((n_points, 3))

        for i in range(n_points):
            A = []
            for name in view_names:
                P = P_matrices[name]
                pt = points_2d[name][i, :2]

                if np.isnan(pt).any():
                    continue

                x, y = pt
                A.append(x * P[2] - P[0])
                A.append(y * P[2] - P[1])

            if len(A) < 4:
                points_3d[i] = np.nan
                continue

            A = np.array(A)
            _, _, Vt = np.linalg.svd(A)
            X = Vt[-1]
            points_3d[i] = X[:3] / X[3]

        return points_3d

    def process_videos(
        self,
        videos: Dict[str, str],
        max_frames: Optional[int] = None,
        progress_interval: int = 50
    ) -> np.ndarray:
        """
        Process videos and return 3D poses.

        Args:
            videos: Dict mapping view names to video paths
            max_frames: Maximum frames to process (None for all)
            progress_interval: Print progress every N frames

        Returns:
            (n_frames, n_joints, 3) array of 3D poses
        """
        if self.tracker is None:
            raise RuntimeError("skellytracker not available")

        view_names = list(videos.keys())
        caps = {name: cv2.VideoCapture(path) for name, path in videos.items()}

        # Get video info
        n_frames = int(caps[view_names[0]].get(cv2.CAP_PROP_FRAME_COUNT))
        if max_frames:
            n_frames = min(n_frames, max_frames)

        # YOLO pose has 17 keypoints
        n_joints = 17
        all_poses_3d = []

        for frame_idx in range(n_frames):
            # Read frames from all views
            frames = {}
            for name, cap in caps.items():
                ret, frame = cap.read()
                if not ret:
                    break
                frames[name] = frame

            if len(frames) != len(view_names):
                break

            # Detect 2D poses in each view
            poses_2d = {}
            valid = True
            for name, frame in frames.items():
                keypoints = self.detect_2d_pose(frame)
                if keypoints is None:
                    valid = False
                    break
                poses_2d[name] = keypoints[:, :2]  # (17, 2)

            if not valid:
                all_poses_3d.append(np.full((n_joints, 3), np.nan))
                continue

            # Triangulate
            pose_3d = self.triangulate(poses_2d)
            all_poses_3d.append(pose_3d)

            if frame_idx % progress_interval == 0:
                print(f"Processed frame {frame_idx}/{n_frames}")

        # Clean up
        for cap in caps.values():
            cap.release()

        return np.array(all_poses_3d)


def main():
    """Test the pipeline on session_005."""
    print("=" * 70)
    print("SKELLY INTEGRATION TEST")
    print("=" * 70)

    if not HAS_SKELLYTRACKER:
        print("\nInstall skellytracker first:")
        print("  pip install skellytracker")
        return

    session_dir = Path(__file__).parent.parent / 'training_data' / 'session_005'

    if not session_dir.exists():
        print(f"Session not found: {session_dir}")
        return

    videos = {
        'side': str(session_dir / 'side.mp4'),
        'back': str(session_dir / 'back.mp4'),
    }

    # Initialize pipeline
    pipeline = SkellyPipeline(model_size="medium")

    # Calibrate
    print("\nCalibrating cameras...")
    if not pipeline.calibrate_from_videos(videos, calibration_frame=60):
        print("Calibration failed")
        return

    # Process first 100 frames
    print("\nProcessing videos...")
    poses_3d = pipeline.process_videos(videos, max_frames=100, progress_interval=20)

    print(f"\nResult shape: {poses_3d.shape}")
    print(f"Valid frames: {np.sum(~np.isnan(poses_3d[:, 0, 0]))}")

    # Compare with UPLIFT if available
    uplift_csv = session_dir / 'uplift.csv'
    if uplift_csv.exists():
        from fusion.train_fusion import load_uplift_positions
        gt = load_uplift_positions(str(uplift_csv))[:100]

        valid = ~(np.isnan(poses_3d).any(axis=(1, 2)) | np.isnan(gt).any(axis=(1, 2)))
        if valid.any():
            errors = np.linalg.norm(poses_3d[valid] - gt[valid], axis=2)
            mpjpe = np.nanmean(errors) * 100
            print(f"\nMPJPE vs UPLIFT: {mpjpe:.2f} cm")


if __name__ == '__main__':
    main()
