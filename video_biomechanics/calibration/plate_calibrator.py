"""
Plate-based camera calibration for baseball biomechanics.

Uses home plate as a known reference object to calibrate cameras
and applies learned corrections to minimize systematic errors.
"""

import cv2
import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .batters_box_calibrator import BattersBoxCalibrator

# Try to import neural network plate detector
_NN_DETECTOR_AVAILABLE = False
PlateDetector = None
load_detector = None

try:
    from ..ml.plate_detector.inference import PlateDetector, load_detector
    _NN_DETECTOR_AVAILABLE = True
except ImportError:
    # Try absolute import as fallback
    try:
        import sys
        _ml_path = Path(__file__).parent.parent / "ml" / "plate_detector"
        if str(_ml_path) not in sys.path:
            sys.path.insert(0, str(_ml_path))
        from inference import PlateDetector, load_detector
        _NN_DETECTOR_AVAILABLE = True
    except ImportError:
        pass


# Home plate dimensions (MLB official, in meters)
PLATE_FRONT_WIDTH = 0.4318   # 17 inches
PLATE_SIDE_EDGE = 0.2159     # 8.5 inches
PLATE_BACK_EDGE = 0.3048     # 12 inches
PLATE_TOTAL_DEPTH = 0.4315   # Front to back point

# World coordinates for plate corners (origin at front center, Y up, Z toward pitcher)
# Order: Back, Left, Front-Left, Front-Right, Right (clockwise from back)
PLATE_WORLD_COORDS = np.array([
    [0, 0, -PLATE_TOTAL_DEPTH],                    # Back point
    [-PLATE_FRONT_WIDTH/2, 0, -PLATE_SIDE_EDGE],   # Left corner
    [-PLATE_FRONT_WIDTH/2, 0, 0],                  # Front-Left
    [PLATE_FRONT_WIDTH/2, 0, 0],                   # Front-Right
    [PLATE_FRONT_WIDTH/2, 0, -PLATE_SIDE_EDGE],    # Right corner
], dtype=np.float32)

# H36M joint names
JOINT_NAMES = [
    'Pelvis', 'RHip', 'RKnee', 'RAnkle', 'LHip', 'LKnee', 'LAnkle',
    'Spine', 'Thorax', 'Neck', 'Head', 'LShoulder', 'LElbow', 'LWrist',
    'RShoulder', 'RElbow', 'RWrist'
]


@dataclass
class CalibrationConfig:
    """Configuration for plate-based calibration."""

    # Camera scale factors (learned from optimization)
    camera_scales: Dict[str, float] = field(default_factory=lambda: {
        'side': 0.36,
        'back': 0.20,
    })

    # Per-joint bias corrections in cm (subtract from predictions)
    # Learned from comparing triangulated poses to UPLIFT ground truth
    joint_biases_cm: Dict[str, List[float]] = field(default_factory=lambda: {
        'Pelvis': [1.57, -3.65, -2.08],
        'RHip': [0.99, -5.03, -5.30],
        'RKnee': [-3.05, -9.45, -7.60],
        'RAnkle': [-5.20, -13.89, -8.32],
        'LHip': [2.14, -2.31, 1.19],
        'LKnee': [0.22, 2.44, 9.08],
        'LAnkle': [2.06, 9.45, 17.93],
        'Spine': [1.70, -0.66, -1.41],
        'Thorax': [0.87, 4.13, 1.29],
        'Neck': [-2.76, -3.91, -0.08],
        'Head': [-0.90, -5.38, 0.57],
        'LShoulder': [-0.11, 3.79, 5.32],
        'LElbow': [0.50, 1.45, -0.53],
        'LWrist': [-1.06, 4.80, -1.19],
        'RShoulder': [2.88, 4.69, -2.36],
        'RElbow': [2.33, 5.38, -5.59],
        'RWrist': [-2.20, 8.15, -0.91],
    })

    # Fixed alignment transform (learned via Procrustes on calibration data)
    # Transforms raw triangulated poses to UPLIFT coordinate frame
    # Format: R (3x3 rotation), scale (float), t (3, translation)
    alignment_R: Optional[List[List[float]]] = None
    alignment_scale: float = 1.0
    alignment_t: Optional[List[float]] = None

    # Camera FOV assumption (degrees)
    # iPhone portrait mode vertical FOV
    camera_fov: float = 38.8
    # iPhone landscape mode horizontal FOV
    camera_fov_landscape: float = 74.6

    # Person height in cm (for GT-free scale computation)
    # Set this to the athlete's approximate height for accurate scaling
    # Default 170cm assumes average adult; adjust for children or specific athletes
    person_height_cm: float = 170.0

    # Plate detection parameters
    # White detection HSV range - permissive to handle shadows and netting
    # Plate can appear cream/off-white when viewed through netting
    white_hsv_lower: Tuple[int, int, int] = (0, 0, 140)
    white_hsv_upper: Tuple[int, int, int] = (180, 100, 255)
    min_contour_area: int = 500

    def save(self, path: str):
        """Save configuration to JSON."""
        data = {
            'camera_scales': self.camera_scales,
            'joint_biases_cm': self.joint_biases_cm,
            'alignment_R': self.alignment_R,
            'alignment_scale': self.alignment_scale,
            'alignment_t': self.alignment_t,
            'camera_fov': self.camera_fov,
            'camera_fov_landscape': self.camera_fov_landscape,
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'CalibrationConfig':
        """Load configuration from JSON."""
        with open(path, 'r') as f:
            data = json.load(f)
        config = cls()
        config.camera_scales = data.get('camera_scales', config.camera_scales)
        config.joint_biases_cm = data.get('joint_biases_cm', config.joint_biases_cm)
        config.alignment_R = data.get('alignment_R', config.alignment_R)
        config.alignment_scale = data.get('alignment_scale', config.alignment_scale)
        config.alignment_t = data.get('alignment_t', config.alignment_t)
        config.camera_fov = data.get('camera_fov', config.camera_fov)
        config.camera_fov_landscape = data.get('camera_fov_landscape', config.camera_fov_landscape)
        return config


class PlateCalibrator:
    """
    Calibrates cameras using home plate as a reference object.

    The calibration process:
    1. Detect home plate corners in each camera view
    2. Use solvePnP to estimate camera orientation
    3. Apply learned scale factors to correct camera distances
    4. Build projection matrices for triangulation
    """

    def __init__(
        self,
        config: Optional[CalibrationConfig] = None,
        nn_detector_path: Optional[Path] = None,
        use_nn_detector: bool = True,
        camera_distances: Optional[Dict[str, float]] = None
    ):
        """
        Initialize plate calibrator.

        Args:
            config: Calibration configuration
            nn_detector_path: Path to neural network model checkpoint.
                             If None, uses default location.
            use_nn_detector: Whether to use neural network detector (default True)
            camera_distances: Optional dict mapping view type to known distance
                             from plate in meters, e.g. {'side': 3.81, 'back': 2.29}
        """
        self.config = config or CalibrationConfig()
        self.cameras: Dict[str, dict] = {}
        self._plate_detector = BattersBoxCalibrator(fov=self.config.camera_fov)
        self._known_distances = camera_distances or {}

        # Initialize neural network detector if available
        self._nn_detector = None
        if use_nn_detector and _NN_DETECTOR_AVAILABLE:
            if nn_detector_path is not None:
                self._nn_detector = PlateDetector(nn_detector_path)
            else:
                # Try to load from default location
                self._nn_detector = load_detector()

    def detect_plate_corners(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect home plate corners in a video frame.

        Uses red mat to locate plate region, then finds pentagon-shaped white regions.
        Returns ordered corner points (5 points for pentagon) or None if not found.
        """
        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Find red mat first - helps locate the plate region
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 50, 50])
        upper_red2 = np.array([180, 255, 255])
        red_mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)

        red_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, red_kernel)

        red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # If red mat found, restrict search to mat area
        mat_mask = None
        if red_contours:
            mat_contour = max(red_contours, key=cv2.contourArea)
            if cv2.contourArea(mat_contour) > w * h * 0.02:  # At least 2% of frame
                mat_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.drawContours(mat_mask, [mat_contour], -1, 255, -1)

        # White detection - permissive to handle shadows and netting
        lower = np.array(self.config.white_hsv_lower)
        upper = np.array(self.config.white_hsv_upper)
        white_mask = cv2.inRange(hsv, lower, upper)

        # Restrict to mat area if found
        if mat_mask is not None:
            white_mask = white_mask & mat_mask

        # Use small kernel to avoid merging plate with nearby lines
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        white_clean = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel_small)
        white_clean = cv2.morphologyEx(white_clean, cv2.MORPH_OPEN, kernel_small)

        contours, _ = cv2.findContours(white_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Find best pentagon-shaped contour
        best_contour = None
        best_score = 0
        img_center_x = w // 2

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.config.min_contour_area or area > 20000:
                continue

            # Check location
            M = cv2.moments(contour)
            if M['m00'] > 0:
                cx = M['m10'] / M['m00']
                cy = M['m01'] / M['m00']

                # Plate should be in lower 35% of frame and middle 80% horizontally
                if cy < h * 0.65 or cx < w * 0.1 or cx > w * 0.9:
                    continue

            # Filter elongated shapes (lines)
            rect = cv2.boundingRect(contour)
            aspect = rect[2] / rect[3] if rect[3] > 0 else 0
            if aspect > 4 or aspect < 0.25:
                continue

            # Try to approximate to pentagon
            hull = cv2.convexHull(contour)
            for eps in [0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05]:
                approx = cv2.approxPolyDP(hull, eps * cv2.arcLength(hull, True), True)
                if len(approx) == 5:
                    # Score: favor near center and larger area
                    dist_from_center = abs(cx - img_center_x)
                    center_score = 1.0 / (1 + (dist_from_center / 100) ** 2)
                    score = area * center_score * 2  # 5-vertex bonus
                    if score > best_score:
                        best_score = score
                        best_contour = approx
                    break
            else:
                # Accept 4-6 vertices as fallback
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                if 4 <= len(approx) <= 6:
                    score = area
                    if score > best_score:
                        best_score = score
                        best_contour = approx

        if best_contour is None or len(best_contour) < 4:
            return None

        return best_contour.reshape(-1, 2).astype(np.float32)

    def order_corners(self, corners: np.ndarray, view_type: str) -> np.ndarray:
        """
        Order plate corners starting from back point, going clockwise.

        Args:
            corners: Detected corner points
            view_type: 'side' or 'back' - determines how to find back point

        Returns:
            Ordered corners: [Back, Left, Front-Left, Front-Right, Right]
        """
        center = corners.mean(axis=0)

        # Calculate angles from center for clockwise ordering
        angles = np.arctan2(corners[:, 1] - center[1], corners[:, 0] - center[0])
        sorted_indices = np.argsort(-angles)  # Descending for clockwise

        # Find back point based on view type
        if view_type == 'side':
            # Side view: back point is leftmost (smallest X)
            back_idx = np.argmin(corners[:, 0])
        else:
            # Back view: back point is at bottom (largest Y, closest to camera)
            back_idx = np.argmax(corners[:, 1])

        # Rotate order to start from back point
        start_pos = np.where(sorted_indices == back_idx)[0][0]
        ordered_indices = np.roll(sorted_indices, -start_pos)

        return corners[ordered_indices]

    def _reorder_corners_for_view(self, corners: np.ndarray, view_type: str) -> np.ndarray:
        """
        Reorder plate corners based on view type for correct triangulation.

        Args:
            corners: Raw detected corners (5, 2)
            view_type: 'side' or 'back'

        Returns:
            Corners ordered as [FL, FR, BR, Apex, BL] for the given view
        """
        xs = corners[:, 0]
        ys = corners[:, 1]

        if view_type == 'side':
            # Side view: Apex is leftmost, front edge is rightmost
            # FL has larger Y, FR has smaller Y

            # Apex = leftmost
            apex_idx = int(np.argmin(xs))

            # Front corners = two rightmost (excluding apex)
            right_sorted = np.argsort(xs)[::-1]
            front_candidates = [i for i in right_sorted if i != apex_idx][:2]

            if ys[front_candidates[0]] > ys[front_candidates[1]]:
                FL_idx, FR_idx = front_candidates[0], front_candidates[1]
            else:
                FL_idx, FR_idx = front_candidates[1], front_candidates[0]

            # Back corners = remaining
            back_candidates = [i for i in range(5) if i != apex_idx and i not in [FL_idx, FR_idx]]
            if ys[back_candidates[0]] > ys[back_candidates[1]]:
                BL_idx, BR_idx = back_candidates[0], back_candidates[1]
            else:
                BL_idx, BR_idx = back_candidates[1], back_candidates[0]

        elif view_type == 'back':
            # Back view: Apex is bottommost, front edge is topmost
            # FL has larger X (right side of image), FR has smaller X

            # Apex = bottommost
            apex_idx = int(np.argmax(ys))

            # Front corners = two topmost (smallest Y)
            top_sorted = np.argsort(ys)
            front_candidates = [top_sorted[0], top_sorted[1]]

            if xs[front_candidates[0]] > xs[front_candidates[1]]:
                FL_idx, FR_idx = front_candidates[0], front_candidates[1]
            else:
                FL_idx, FR_idx = front_candidates[1], front_candidates[0]

            # Back corners = remaining (not apex, not front)
            back_candidates = [i for i in range(5) if i != apex_idx and i not in [FL_idx, FR_idx]]
            if xs[back_candidates[0]] > xs[back_candidates[1]]:
                BL_idx, BR_idx = back_candidates[0], back_candidates[1]
            else:
                BL_idx, BR_idx = back_candidates[1], back_candidates[0]

        else:
            # Default: return as-is
            return corners

        return np.array([
            corners[FL_idx],
            corners[FR_idx],
            corners[BR_idx],
            corners[apex_idx],
            corners[BL_idx],
        ])

    def _find_best_corner_permutation(self, raw_corners: np.ndarray, K: np.ndarray,
                                        view_type: str) -> Tuple[np.ndarray, tuple]:
        """
        Find the best corner permutation by testing all possibilities.

        For each permutation, we solve PnP and measure reprojection error.
        We also apply camera position constraints based on view type.

        Args:
            raw_corners: Detected corners (5, 2)
            K: Camera intrinsic matrix
            view_type: 'side' or 'back'

        Returns:
            Tuple of (ordered_corners, best_permutation)
        """
        from itertools import permutations

        best_error = float('inf')
        best_perm = None
        best_ordered = None

        for perm in permutations(range(5)):
            ordered = raw_corners[list(perm)]

            success, rvec, tvec = cv2.solvePnP(
                PLATE_WORLD_COORDS,
                ordered,
                K,
                np.zeros(4),
                flags=cv2.SOLVEPNP_ITERATIVE
            )

            if not success:
                continue

            # Compute reprojection error
            projected, _ = cv2.projectPoints(PLATE_WORLD_COORDS, rvec, tvec, K, np.zeros(4))
            projected = projected.reshape(-1, 2)
            errors = [np.linalg.norm(o - p) for o, p in zip(ordered, projected)]
            mean_error = np.mean(errors)

            # Apply camera position and viewing direction constraints
            R, _ = cv2.Rodrigues(rvec)
            cam_pos = -R.T @ tvec.flatten()
            view_dir = R[:, 2]  # Camera optical axis in world coords

            # Constraint 1: Camera should be above ground (Y > 0)
            if cam_pos[1] < 0:
                mean_error += 100.0  # Hard penalty

            # Constraint 2: Camera MUST look toward the plate
            # This is a HARD constraint - reject solutions where camera looks away
            to_plate = -cam_pos / (np.linalg.norm(cam_pos) + 1e-6)
            look_alignment = np.dot(view_dir, to_plate)
            if look_alignment < 0:  # Camera looking away from plate - invalid
                mean_error += 100.0  # Hard penalty to reject this solution

            if mean_error < best_error:
                best_error = mean_error
                best_perm = perm
                best_ordered = ordered

        return best_ordered, best_perm

    def calibrate_camera(self, frame: np.ndarray, view_type: str) -> Optional[dict]:
        """
        Calibrate a single camera from a frame containing the plate.

        Args:
            frame: Video frame with visible home plate
            view_type: 'side' or 'back'

        Returns:
            Camera calibration dict with K, R, t, P matrices, or None if failed
        """
        raw_corners = None

        # Try neural network detector first (most robust)
        if self._nn_detector is not None:
            raw_corners = self._nn_detector.detect(frame)

        # Fallback to BattersBoxCalibrator
        if raw_corners is None:
            result = self._plate_detector.detect_all(frame)
            raw_corners = result.plate_corners if result else None

        # Fallback to direct HSV detection
        if raw_corners is None:
            raw_corners = self.detect_plate_corners(frame)

        if raw_corners is None or len(raw_corners) != 5:
            return None

        h, w = frame.shape[:2]
        is_portrait = h > w

        # Build camera matrix with orientation-aware FOV
        # iPhone has different FOVs based on sensor orientation:
        # - Portrait: 38.8° vertical FOV
        # - Landscape: 74.6° horizontal FOV
        if is_portrait:
            fov_v = self.config.camera_fov
        else:
            fov_h = self.config.camera_fov_landscape
            aspect = w / h
            fov_v = 2 * np.degrees(np.arctan(np.tan(np.radians(fov_h / 2)) / aspect))

        f = h / (2 * np.tan(np.radians(fov_v / 2)))
        K = np.array([
            [f, 0, w/2],
            [0, f, h/2],
            [0, 0, 1]
        ], dtype=np.float32)

        # Find best corner permutation automatically
        # This works for both portrait and landscape, and handles different camera setups
        # (e.g., left-handed vs right-handed batter positions)
        ordered, perm = self._find_best_corner_permutation(raw_corners, K, view_type)
        if ordered is None:
            return None

        n_pts = 5

        # Solve PnP for camera pose
        success, rvec, tvec = cv2.solvePnP(
            PLATE_WORLD_COORDS[:n_pts],
            ordered[:n_pts],
            K,
            np.zeros(4),
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return None

        # Build rotation matrix and projection matrix
        R, _ = cv2.Rodrigues(rvec)

        # Compute camera position in world coordinates
        cam_position = -R.T @ tvec.flatten()

        # If known distance is provided, correct the translation scale
        # solvePnP assumes FOV is correct, but if FOV is wrong, distances are wrong
        if view_type in self._known_distances:
            known_dist = self._known_distances[view_type]
            current_dist = np.linalg.norm(cam_position)
            if current_dist > 0:
                scale_correction = known_dist / current_dist
                tvec = tvec * scale_correction
                cam_position = -R.T @ tvec.flatten()

        P = K @ np.hstack([R, tvec])

        return {
            'K': K,
            'R': R,
            't': tvec,
            'P': P,
            'position': cam_position,
            'view_type': view_type,
            'n_corners': n_pts,
            'ordered_corners': ordered,  # Store for scale computation
        }

    def calibrate_from_videos(self, video_paths: Dict[str, str],
                               calibration_frame: int = 60) -> bool:
        """
        Calibrate all cameras from video files.

        Args:
            video_paths: Dict mapping view type to video path, e.g.,
                         {'side': 'path/to/side.mp4', 'back': 'path/to/back.mp4'}
            calibration_frame: Frame index to use for calibration

        Returns:
            True if all cameras calibrated successfully
        """
        self.cameras = {}
        self._calibration_frames = {}  # Store frames for scale computation

        # Frames to try if initial frame fails (plate might be obscured)
        frames_to_try = [calibration_frame] + [f for f in [15, 30, 45, 60] if f != calibration_frame]

        for view_type, path in video_paths.items():
            cap = cv2.VideoCapture(path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            result = None
            used_frame = None

            for try_frame in frames_to_try:
                if try_frame >= total_frames:
                    continue

                cap.set(cv2.CAP_PROP_POS_FRAMES, try_frame)
                ret, frame = cap.read()

                if not ret:
                    continue

                candidate = self.calibrate_camera(frame, view_type)
                if candidate is None:
                    continue

                # Validate camera pose makes sense for this view type
                pos = candidate['position']
                pose_valid = True

                if view_type == 'side':
                    # Side camera should be far to the side (large |X|), above ground,
                    # and toward centerfield (Z < 0) for good triangulation with back camera
                    if abs(pos[0]) < 1.0 or pos[1] < 0.3 or pos[2] > 0:
                        pose_valid = False
                elif view_type == 'back':
                    # Back camera should be behind centerfield, roughly centered
                    if pos[2] > 0 or abs(pos[0]) > 3.0 or pos[1] < 0:
                        pose_valid = False

                if pose_valid:
                    result = candidate
                    self._calibration_frames[view_type] = frame
                    used_frame = try_frame
                    break
                elif result is None:
                    # Keep first valid calibration as fallback even if pose is bad
                    result = candidate
                    self._calibration_frames[view_type] = frame
                    used_frame = try_frame

            cap.release()

            if result is not None:
                self.cameras[view_type] = result
                frame_info = f" (frame {used_frame})" if used_frame != calibration_frame else ""
                print(f"  {view_type}: Calibrated ({result['n_corners']} corners){frame_info}, "
                      f"position: {result['position']}")
            else:
                print(f"  {view_type}: Calibration failed")

        # Compute auto-scale from plate geometry
        if len(self.cameras) >= 2:
            self._compute_plate_scale()

        return len(self.cameras) >= 2

    def _compute_plate_scale(self):
        """Compute alignment transform using plate corners as ground truth.

        The plate provides 5 known 3D points (PLATE_WORLD_COORDS).
        We triangulate these corners and compute the optimal rigid transform
        (rotation, scale, translation) to align triangulated corners to known.

        This gives us a per-session calibration using ONLY plate geometry.
        """
        self._compute_plate_alignment()

    def _compute_plate_alignment(self):
        """Compute alignment transform using plate corners as ground truth.

        Triangulates all 5 plate corners and computes optimal rigid transform
        (rotation, scale, translation) via Procrustes to align to known coordinates.
        """
        # Check we have ordered corners from both cameras
        if 'ordered_corners' not in self.cameras.get('side', {}):
            print("  WARNING: No ordered corners for side camera")
            self._plate_scale = 1.0
            self._plate_R = np.eye(3)
            self._plate_t = np.zeros(3)
            return

        if 'ordered_corners' not in self.cameras.get('back', {}):
            print("  WARNING: No ordered corners for back camera")
            self._plate_scale = 1.0
            self._plate_R = np.eye(3)
            self._plate_t = np.zeros(3)
            return

        side_corners = self.cameras['side']['ordered_corners']
        back_corners = self.cameras['back']['ordered_corners']

        P_side, P_back = self.get_projection_matrices()

        # Triangulate all 5 plate corners
        triangulated = np.zeros((5, 3))
        for i in range(5):
            triangulated[i] = self.triangulate_point(
                side_corners[i], back_corners[i], P_side, P_back
            )

        # Known plate coordinates (ground truth)
        known = PLATE_WORLD_COORDS.copy()

        # Compute Procrustes alignment: triangulated -> known
        # Find R, s, t such that: known ≈ s * R @ triangulated + t
        tri_center = triangulated.mean(axis=0)
        known_center = known.mean(axis=0)

        tri_centered = triangulated - tri_center
        known_centered = known - known_center

        # Scale
        tri_norm = np.sqrt((tri_centered ** 2).sum())
        known_norm = np.sqrt((known_centered ** 2).sum())
        scale = known_norm / tri_norm

        # Rotation via SVD
        tri_scaled = tri_centered / tri_norm
        known_scaled = known_centered / known_norm

        H = tri_scaled.T @ known_scaled
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        # Ensure proper rotation (det = 1)
        if np.linalg.det(R) < 0:
            Vt[-1] *= -1
            R = Vt.T @ U.T

        # Translation
        t = known_center - scale * (R @ tri_center)

        # CRITICAL: Plate corners all have Y=0, so Procrustes doesn't constrain Y direction.
        # Use camera position to verify Y-up: camera should be above ground (Y > 0).
        # Transform camera position to aligned frame and check Y.
        side_cam_raw = self.cameras['side']['position']
        side_cam_aligned = scale * (R @ side_cam_raw) + t

        if side_cam_aligned[1] < 0:
            # Camera ended up below ground - flip Y axis
            # Flip Y by negating the Y row of R
            R[1, :] = -R[1, :]
            t = known_center - scale * (R @ tri_center)
            side_cam_aligned = scale * (R @ side_cam_raw) + t

        # Store transform
        self._plate_scale = scale
        self._plate_R = R
        self._plate_t = t

        # Verify alignment
        aligned = scale * (triangulated @ R.T) + t
        errors = np.linalg.norm(aligned - known, axis=1) * 100  # cm

        print(f"  Plate alignment calibration:")
        print(f"    Scale factor: {scale:.4f}")
        print(f"    Rotation angle: {np.degrees(np.arccos((np.trace(R)-1)/2)):.1f}°")
        print(f"    Per-corner alignment error: {errors.mean():.2f} cm (max: {errors.max():.2f} cm)")
        print(f"    Camera height (Y): {side_cam_aligned[1]:.2f}m")

    def _validate_with_ground_plane(self, P_side, P_back, initial_scale):
        """Validate scale using ground plane constraint.

        The athlete's feet should be at approximately Y=0 (ground level),
        same as the plate. If they're significantly above/below, adjust scale.

        Returns adjusted scale or None if validation not possible.
        """
        from ultralytics import YOLO

        try:
            model = YOLO('yolov8m-pose.pt')
        except Exception:
            print("    Could not load pose model for ground validation")
            return None

        side_frame = self._calibration_frames['side']
        back_frame = self._calibration_frames['back']

        side_res = model(side_frame, verbose=False)[0]
        back_res = model(back_frame, verbose=False)[0]

        if (len(side_res.keypoints.xy) == 0 or len(back_res.keypoints.xy) == 0):
            print("    No person detected for ground validation")
            return None

        side_kps = side_res.keypoints.xy[0].cpu().numpy()
        back_kps = back_res.keypoints.xy[0].cpu().numpy()

        # COCO indices: 15=L_ankle, 16=R_ankle
        l_ankle_3d = self.triangulate_point(side_kps[15], back_kps[15], P_side, P_back)
        r_ankle_3d = self.triangulate_point(side_kps[16], back_kps[16], P_side, P_back)

        # Apply initial scale
        l_ankle_scaled = l_ankle_3d * initial_scale
        r_ankle_scaled = r_ankle_3d * initial_scale

        # Check ankle Y positions (should be near 0)
        avg_ankle_y = (l_ankle_scaled[1] + r_ankle_scaled[1]) / 2

        print(f"    Left ankle Y: {l_ankle_scaled[1]*100:.1f} cm")
        print(f"    Right ankle Y: {r_ankle_scaled[1]*100:.1f} cm")
        print(f"    Average ankle Y: {avg_ankle_y*100:.1f} cm (should be ~0)")

        # If ankles are significantly off ground, we have a geometry issue
        # This could indicate wrong FOV or lens distortion
        if abs(avg_ankle_y) > 0.30:  # More than 30cm off ground
            print(f"    WARNING: Ankles {avg_ankle_y*100:.1f}cm from ground - geometry issue")
            # Don't adjust scale for geometry issues - that's a different problem
            return None

        # Ankles reasonably close to ground - scale is good
        return initial_scale

    def _validate_anthropometrics(self):
        """Validate triangulated body proportions against expected ranges.

        Uses the person_height_cm as a reference to check if measurements
        are realistic. This is a sanity check, not a calibration.
        """
        from ultralytics import YOLO

        height_m = self.config.person_height_cm / 100.0
        print(f"  Anthropometric validation (expected height: {self.config.person_height_cm:.0f} cm):")

        try:
            model = YOLO('yolov8m-pose.pt')
        except Exception:
            return

        side_frame = self._calibration_frames['side']
        back_frame = self._calibration_frames['back']

        side_res = model(side_frame, verbose=False)[0]
        back_res = model(back_frame, verbose=False)[0]

        if (len(side_res.keypoints.xy) == 0 or len(back_res.keypoints.xy) == 0):
            return

        side_kps = side_res.keypoints.xy[0].cpu().numpy()
        back_kps = back_res.keypoints.xy[0].cpu().numpy()

        P_side, P_back = self.get_projection_matrices()

        # Triangulate key segments and apply scale
        # COCO: 11=L_hip, 13=L_knee, 5=L_shoulder, 7=L_elbow
        l_hip = self.triangulate_point(side_kps[11], back_kps[11], P_side, P_back) * self._plate_scale
        l_knee = self.triangulate_point(side_kps[13], back_kps[13], P_side, P_back) * self._plate_scale
        l_shoulder = self.triangulate_point(side_kps[5], back_kps[5], P_side, P_back) * self._plate_scale
        l_elbow = self.triangulate_point(side_kps[7], back_kps[7], P_side, P_back) * self._plate_scale

        thigh = np.linalg.norm(l_knee - l_hip)
        upper_arm = np.linalg.norm(l_elbow - l_shoulder)

        # Expected proportions (anthropometric ratios)
        expected_thigh = height_m * 0.26
        expected_arm = height_m * 0.19

        # Check if within reasonable range (allow 30% deviation)
        thigh_ratio = thigh / expected_thigh
        arm_ratio = upper_arm / expected_arm

        print(f"    Thigh: {thigh*100:.1f} cm (expected ~{expected_thigh*100:.1f} cm, ratio={thigh_ratio:.2f})")
        print(f"    Upper arm: {upper_arm*100:.1f} cm (expected ~{expected_arm*100:.1f} cm, ratio={arm_ratio:.2f})")

        if 0.7 < thigh_ratio < 1.3 and 0.7 < arm_ratio < 1.3:
            print(f"    Proportions look reasonable for {self.config.person_height_cm:.0f} cm person")
        else:
            print(f"    WARNING: Proportions may be off - check scale or person_height_cm")

    def _compute_scale_from_height(self):
        """Compute scale using limb segment lengths (pose-independent).

        Uses thigh and upper arm lengths which don't change with pose.
        These are estimated from person height using anthropometric ratios.
        """
        from ultralytics import YOLO

        height_m = self.config.person_height_cm / 100.0
        print(f"  Computing scale from person height: {self.config.person_height_cm:.0f} cm")

        # Anthropometric ratios (approximate, based on human proportions)
        # Thigh length ~26% of height, upper arm ~19% of height
        expected_thigh = height_m * 0.26
        expected_upper_arm = height_m * 0.19

        # Load pose model and detect in calibration frame
        model = YOLO('yolov8m-pose.pt')

        side_frame = self._calibration_frames['side']
        back_frame = self._calibration_frames['back']

        side_res = model(side_frame, verbose=False)[0]
        back_res = model(back_frame, verbose=False)[0]

        if (len(side_res.keypoints.xy) == 0 or len(back_res.keypoints.xy) == 0):
            print("  WARNING: Could not detect person, using scale=1.0")
            self._plate_scale = 1.0
            return

        side_kps = side_res.keypoints.xy[0].cpu().numpy()
        back_kps = back_res.keypoints.xy[0].cpu().numpy()

        P_side, P_back = self.get_projection_matrices()

        # COCO indices: 5=L_shoulder, 7=L_elbow, 11=L_hip, 13=L_knee
        #               6=R_shoulder, 8=R_elbow, 12=R_hip, 14=R_knee
        segments = []

        # Left thigh (hip to knee)
        l_hip = self.triangulate_point(side_kps[11], back_kps[11], P_side, P_back)
        l_knee = self.triangulate_point(side_kps[13], back_kps[13], P_side, P_back)
        l_thigh = np.linalg.norm(l_knee - l_hip)
        segments.append(('L_thigh', l_thigh, expected_thigh))

        # Right thigh
        r_hip = self.triangulate_point(side_kps[12], back_kps[12], P_side, P_back)
        r_knee = self.triangulate_point(side_kps[14], back_kps[14], P_side, P_back)
        r_thigh = np.linalg.norm(r_knee - r_hip)
        segments.append(('R_thigh', r_thigh, expected_thigh))

        # Left upper arm (shoulder to elbow)
        l_shoulder = self.triangulate_point(side_kps[5], back_kps[5], P_side, P_back)
        l_elbow = self.triangulate_point(side_kps[7], back_kps[7], P_side, P_back)
        l_arm = np.linalg.norm(l_elbow - l_shoulder)
        segments.append(('L_arm', l_arm, expected_upper_arm))

        # Right upper arm
        r_shoulder = self.triangulate_point(side_kps[6], back_kps[6], P_side, P_back)
        r_elbow = self.triangulate_point(side_kps[8], back_kps[8], P_side, P_back)
        r_arm = np.linalg.norm(r_elbow - r_shoulder)
        segments.append(('R_arm', r_arm, expected_upper_arm))

        # Compute scale from each segment
        scales = []
        for name, measured, expected in segments:
            if measured > 0.05:  # Skip if too small (detection issue)
                scale = expected / measured
                scales.append(scale)
                print(f"    {name}: {measured*100:.1f}cm -> {expected*100:.1f}cm (scale={scale:.3f})")

        if scales:
            self._plate_scale = np.median(scales)
            print(f"  Median scale factor: {self._plate_scale:.4f}")
        else:
            print("  WARNING: Could not compute scale from limbs")
            self._plate_scale = 1.0

    def get_projection_matrices(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Get projection matrices for triangulation."""
        P_side = self.cameras.get('side', {}).get('P')
        P_back = self.cameras.get('back', {}).get('P')
        return P_side, P_back

    def apply_joint_bias_correction(self, poses_3d: np.ndarray) -> np.ndarray:
        """
        Apply per-joint bias correction to triangulated poses.

        Args:
            poses_3d: Array of shape (n_frames, 17, 3) in meters

        Returns:
            Corrected poses with biases removed
        """
        corrected = poses_3d.copy()

        for j, joint_name in enumerate(JOINT_NAMES):
            if joint_name in self.config.joint_biases_cm:
                bias = np.array(self.config.joint_biases_cm[joint_name]) / 100  # cm to m
                corrected[:, j, :] -= bias

        return corrected

    def learn_alignment(self, pred_poses: np.ndarray, gt_poses: np.ndarray) -> None:
        """
        Learn the alignment transform from predicted to ground truth poses.

        Uses Procrustes analysis to find optimal rotation, scale, translation.
        The learned transform is stored in self.config and can be saved/loaded.

        Args:
            pred_poses: (N, 17, 3) predicted poses from triangulation
            gt_poses: (N, 17, 3) ground truth poses from UPLIFT
        """
        n = min(len(pred_poses), len(gt_poses))
        pred, gt = pred_poses[:n], gt_poses[:n]

        # Flatten for Procrustes
        pf = pred.reshape(-1, 3)
        gf = gt.reshape(-1, 3)

        # Remove invalid points
        valid = ~(np.isnan(pf).any(1) | np.isinf(pf).any(1) |
                  np.isnan(gf).any(1) | np.isinf(gf).any(1))
        pv, gv = pf[valid], gf[valid]

        if len(pv) < 10:
            print("WARNING: Not enough valid points to learn alignment")
            return

        # Compute centroids
        pc, gc = pv.mean(0), gv.mean(0)

        # Compute scales
        ps = np.sqrt(((pv - pc)**2).sum() / len(pv))
        gs = np.sqrt(((gv - gc)**2).sum() / len(gv))
        scale = gs / ps

        # Normalize
        pn, gn = (pv - pc) / ps, (gv - gc) / gs

        # SVD for rotation
        U, _, Vt = np.linalg.svd(pn.T @ gn)
        R = Vt.T @ U.T

        # Ensure proper rotation (det = 1)
        if np.linalg.det(R) < 0:
            Vt[-1] *= -1
            R = Vt.T @ U.T

        # Translation
        t = gc - scale * (R @ pc)

        # Store in config
        self.config.alignment_R = R.tolist()
        self.config.alignment_scale = float(scale)
        self.config.alignment_t = t.tolist()

        print(f"Learned alignment: scale={scale:.4f}, "
              f"rotation_angle={np.degrees(np.arccos((np.trace(R)-1)/2)):.1f}°")

    def transform_to_uplift_frame(self, poses_3d: np.ndarray,
                                   apply_bias_correction: bool = False,
                                   center_on_pelvis: bool = True) -> np.ndarray:
        """
        Transform triangulated poses to UPLIFT body-relative convention.

        Our plate-based calibration produces poses in a consistent world frame:
        - Origin: plate center
        - X: toward 1st base
        - Y: up
        - Z: toward catcher (negative toward pitcher)

        UPLIFT uses body-relative axes where the batter faces a consistent direction.
        This method:
        1. Detects handedness from batter position (pelvis X before centering)
        2. Rotates to body-relative frame based on handedness
        3. Centers on pelvis

        Args:
            poses_3d: Array of shape (n_frames, 17, 3) or (17, 3)
            apply_bias_correction: Whether to apply learned joint biases (default False)
            center_on_pelvis: If True, center each frame on the pelvis (default True)

        Returns:
            Body-relative poses in UPLIFT convention
        """
        # Handle both single pose and sequence
        single_pose = poses_3d.ndim == 2
        if single_pose:
            poses_3d = poses_3d[np.newaxis, ...]

        transformed = poses_3d.copy()

        # Detect handedness from hip Z positions at first frame
        # In plate coords: Z+ = toward pitcher
        # Right-handed batter: left hip is "front" (closer to pitcher)
        #   -> left_hip_z > right_hip_z
        # Left-handed batter: right hip is "front" (closer to pitcher)
        #   -> right_hip_z > left_hip_z
        # Joint indices: left_hip=4, right_hip=1
        left_hip_z = transformed[0, 4, 2]   # Z coord of left hip
        right_hip_z = transformed[0, 1, 2]  # Z coord of right hip
        is_right_handed = left_hip_z > right_hip_z

        # Rotate to body-relative frame
        # Right-handed batter stands sideways with left shoulder toward pitcher
        # Their body faces roughly toward +X (first base)
        # Left-handed batter faces roughly toward -X (third base)
        #
        # To get consistent body-relative axes:
        # Right-handed: rotate -90° around Y so body-forward becomes +Z
        # Left-handed: rotate +90° around Y so body-forward becomes +Z
        if is_right_handed:
            # Rotate +90° around Y: (x,y,z) -> (-z, y, x)
            # Right-handed batter faces pitcher, right hip is toward catcher
            rotated = np.zeros_like(transformed)
            rotated[:, :, 0] = -transformed[:, :, 2]  # new X = -old Z
            rotated[:, :, 1] = transformed[:, :, 1]   # Y unchanged
            rotated[:, :, 2] = transformed[:, :, 0]   # new Z = old X
            transformed = rotated
        else:
            # Rotate -90° around Y: (x,y,z) -> (z, y, -x)
            # Left-handed batter faces pitcher, left hip is toward catcher
            rotated = np.zeros_like(transformed)
            rotated[:, :, 0] = transformed[:, :, 2]   # new X = old Z
            rotated[:, :, 1] = transformed[:, :, 1]   # Y unchanged
            rotated[:, :, 2] = -transformed[:, :, 0]  # new Z = -old X
            transformed = rotated

        # Flip Y-axis: Our triangulation produces Y-down, UPLIFT uses Y-up
        # Head should be above pelvis (positive Y), ankles below (negative Y)
        transformed[:, :, 1] = -transformed[:, :, 1]

        # Center on pelvis (body-relative)
        if center_on_pelvis:
            pelvis = transformed[:, 0:1, :].copy()
            transformed = transformed - pelvis

        # Apply bias correction if requested and available
        if apply_bias_correction and self.config.joint_biases_cm:
            transformed = self.apply_joint_bias_correction(transformed)

        # Return in original shape
        if single_pose:
            return transformed[0]
        return transformed

    def process_pose(self, keypoints_side: np.ndarray,
                     keypoints_back: np.ndarray,
                     to_uplift_frame: bool = True) -> np.ndarray:
        """
        Full pipeline: triangulate and transform to UPLIFT frame.

        This is the main method for production use without ground truth.
        It triangulates 2D keypoints and outputs poses in the same coordinate
        frame as UPLIFT ground truth data.

        Args:
            keypoints_side: (17, 2) array of 2D keypoints from side camera
            keypoints_back: (17, 2) array of 2D keypoints from back camera
            to_uplift_frame: If True, transform to UPLIFT coordinates (default)

        Returns:
            (17, 3) array of 3D joint positions in UPLIFT-compatible frame
        """
        # Triangulate
        pose_3d = self.triangulate_pose(keypoints_side, keypoints_back)

        # Transform to UPLIFT frame if requested
        if to_uplift_frame:
            pose_3d = self.transform_to_uplift_frame(pose_3d)

        return pose_3d

    def triangulate_point(self, pt1: np.ndarray, pt2: np.ndarray,
                          P1: np.ndarray, P2: np.ndarray) -> np.ndarray:
        """Triangulate a single 3D point from two 2D observations."""
        A = np.array([
            pt1[0] * P1[2] - P1[0],
            pt1[1] * P1[2] - P1[1],
            pt2[0] * P2[2] - P2[0],
            pt2[1] * P2[2] - P2[1]
        ])

        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1]
        return X[:3] / X[3]

    def triangulate_pose(self, keypoints_side: np.ndarray,
                         keypoints_back: np.ndarray,
                         use_robust: bool = True,
                         person_height: float = 1.75) -> np.ndarray:
        """
        Triangulate a full pose from 2D keypoints in both views.

        Args:
            keypoints_side: (17, 2) array of 2D keypoints from side camera
            keypoints_back: (17, 2) array of 2D keypoints from back camera
            use_robust: If True, use hybrid triangulation with bone constraints
            person_height: Assumed person height in meters (for robust mode)

        Returns:
            (17, 3) array of 3D joint positions in plate coordinate frame
        """
        P_side, P_back = self.get_projection_matrices()

        if P_side is None or P_back is None:
            raise ValueError("Cameras not calibrated")

        if use_robust and 'side' in self.cameras and 'back' in self.cameras:
            # Use hybrid triangulation for better accuracy
            from .robust_triangulation import HybridTriangulator

            hybrid = HybridTriangulator(
                P_side, P_back,
                self.cameras['side']['K'],
                self.cameras['back']['K'],
                self.cameras['side']['R'],
                self.cameras['back']['R'],
                self.cameras['side']['position'],
                self.cameras['back']['position'],
                person_height=person_height
            )

            pose_3d, method = hybrid.triangulate(
                keypoints_side, keypoints_back,
                test_lr_swap=True
            )
        else:
            # Fall back to basic DLT triangulation
            pose_3d = np.zeros((17, 3))
            for j in range(17):
                pose_3d[j] = self.triangulate_point(
                    keypoints_side[j], keypoints_back[j],
                    P_side, P_back
                )

        # Apply plate alignment transform: scaled rotation + translation
        scale = getattr(self, '_plate_scale', 1.0)
        R = getattr(self, '_plate_R', np.eye(3))
        t = getattr(self, '_plate_t', np.zeros(3))

        # Transform: aligned = scale * (R @ point) + t
        pose_3d = scale * (pose_3d @ R.T) + t

        return pose_3d

    def process_videos(self,
                       video_paths: Dict[str, str],
                       pose_model: str = 'yolov8m-pose.pt',
                       max_frames: Optional[int] = None,
                       progress_interval: int = 100) -> dict:
        """
        Process synchronized video pairs and extract 3D poses.

        This is the main production method for batch video processing.

        Args:
            video_paths: Dict mapping view type to video path
            pose_model: YOLO pose model to use
            max_frames: Maximum frames to process (None = all)
            progress_interval: Print progress every N frames

        Returns:
            Dictionary with:
                - 'poses_3d': (N, 17, 3) array in UPLIFT frame
                - 'poses_raw': (N, 17, 3) array before transform
                - 'fps': Frame rate
                - 'n_frames': Number of frames processed
                - 'failed_frames': List of frame indices with detection failures
        """
        from ultralytics import YOLO

        # Calibrate if not already done
        if len(self.cameras) < 2:
            print("Calibrating cameras...")
            if not self.calibrate_from_videos(video_paths):
                raise RuntimeError("Camera calibration failed")

        # Load pose model
        print(f"Loading pose model: {pose_model}")
        model = YOLO(pose_model)

        # Open videos
        caps = {k: cv2.VideoCapture(v) for k, v in video_paths.items()}
        fps = caps['side'].get(cv2.CAP_PROP_FPS)
        total_frames = int(caps['side'].get(cv2.CAP_PROP_FRAME_COUNT))

        if max_frames:
            total_frames = min(total_frames, max_frames)

        print(f"Processing {total_frames} frames at {fps:.1f} FPS...")

        poses_raw = []
        poses_3d = []
        failed_frames = []

        for i in range(total_frames):
            frames = {}
            for view, cap in caps.items():
                ret, frame = cap.read()
                if not ret:
                    break
                frames[view] = frame

            if len(frames) < 2:
                break

            # Detect 2D poses
            results_side = model(frames['side'], verbose=False)
            results_back = model(frames['back'], verbose=False)

            # Check for valid detections
            if (len(results_side) == 0 or len(results_side[0].keypoints) == 0 or
                results_side[0].keypoints[0].xy is None or
                len(results_back) == 0 or len(results_back[0].keypoints) == 0 or
                results_back[0].keypoints[0].xy is None):
                poses_raw.append(np.full((17, 3), np.nan))
                poses_3d.append(np.full((17, 3), np.nan))
                failed_frames.append(i)
                continue

            # Extract keypoints
            kp_side = results_side[0].keypoints[0].xy[0].cpu().numpy()
            kp_back = results_back[0].keypoints[0].xy[0].cpu().numpy()

            # Convert to H36M and triangulate
            h36m_side = coco_to_h36m(kp_side)
            h36m_back = coco_to_h36m(kp_back)

            try:
                pose_raw = self.triangulate_pose(h36m_side, h36m_back)
                pose_3d = self.transform_to_uplift_frame(pose_raw)
                poses_raw.append(pose_raw)
                poses_3d.append(pose_3d)
            except Exception:
                poses_raw.append(np.full((17, 3), np.nan))
                poses_3d.append(np.full((17, 3), np.nan))
                failed_frames.append(i)

            if i % progress_interval == 0:
                print(f"  Frame {i}/{total_frames}")

        # Cleanup
        for cap in caps.values():
            cap.release()

        poses_raw = np.array(poses_raw)
        poses_3d = np.array(poses_3d)

        print(f"Processed {len(poses_3d)} frames ({len(failed_frames)} failures)")

        return {
            'poses_3d': poses_3d,
            'poses_raw': poses_raw,
            'fps': fps,
            'n_frames': len(poses_3d),
            'failed_frames': failed_frames,
        }

    def to_uplift_csv(self,
                      poses_3d: np.ndarray,
                      output_path: str,
                      fps: float = 30.0,
                      metadata: Optional[Dict] = None) -> None:
        """
        Export 3D poses to UPLIFT-compatible CSV format.

        Args:
            poses_3d: (N, 17, 3) array of poses in UPLIFT frame
            output_path: Path for output CSV
            fps: Frame rate for time column
            metadata: Optional dict with athlete_name, sessionid, etc.
        """
        import pandas as pd

        n_frames = len(poses_3d)
        metadata = metadata or {}

        # UPLIFT joint name mapping (H36M index -> UPLIFT column prefix)
        h36m_to_uplift = {
            0: 'pelvis_3d',
            1: 'right_hip_jc_3d',
            2: 'right_knee_jc_3d',
            3: 'right_ankle_jc_3d',
            4: 'left_hip_jc_3d',
            5: 'left_knee_jc_3d',
            6: 'left_ankle_jc_3d',
            7: 'spine_3d',  # interpolated
            8: 'thorax_3d',  # proximal_neck approximate
            9: 'proximal_neck_3d',
            10: 'mid_head_3d',
            11: 'left_shoulder_jc_3d',
            12: 'left_elbow_jc_3d',
            13: 'left_wrist_jc_3d',
            14: 'right_shoulder_jc_3d',
            15: 'right_elbow_jc_3d',
            16: 'right_wrist_jc_3d',
        }

        # Build dataframe
        data = {
            'frame': np.arange(n_frames),
            'time': np.arange(n_frames) / fps,
            'fps': [fps] * n_frames,
        }

        # Add metadata columns
        for key in ['athlete_name', 'athleteid', 'sessionid', 'orgid',
                    'activity', 'movement', 'handedness']:
            data[key] = [metadata.get(key, '')] * n_frames

        # Add joint positions
        for h36m_idx, uplift_prefix in h36m_to_uplift.items():
            data[f'{uplift_prefix}_x'] = poses_3d[:, h36m_idx, 0]
            data[f'{uplift_prefix}_y'] = poses_3d[:, h36m_idx, 1]
            data[f'{uplift_prefix}_z'] = poses_3d[:, h36m_idx, 2]

        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
        print(f"Exported {n_frames} frames to {output_path}")


def coco_to_h36m(coco_keypoints: np.ndarray) -> np.ndarray:
    """
    Convert COCO 17-keypoint format to H36M 17-keypoint format.

    COCO order: nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles
    H36M order: pelvis, hips, knees, ankles, spine, thorax, neck, head, shoulders, elbows, wrists
    """
    h36m = np.zeros((17, 2))

    # Pelvis = midpoint of hips
    h36m[0] = (coco_keypoints[11] + coco_keypoints[12]) / 2

    # Right leg
    h36m[1] = coco_keypoints[12]  # RHip
    h36m[2] = coco_keypoints[14]  # RKnee
    h36m[3] = coco_keypoints[16]  # RAnkle

    # Left leg
    h36m[4] = coco_keypoints[11]  # LHip
    h36m[5] = coco_keypoints[13]  # LKnee
    h36m[6] = coco_keypoints[15]  # LAnkle

    # Torso
    thorax = (coco_keypoints[5] + coco_keypoints[6]) / 2
    h36m[7] = (h36m[0] + thorax) / 2  # Spine
    h36m[8] = thorax                   # Thorax
    h36m[9] = thorax                   # Neck (approximate)
    h36m[10] = coco_keypoints[0]       # Head/Nose

    # Left arm
    h36m[11] = coco_keypoints[5]   # LShoulder
    h36m[12] = coco_keypoints[7]   # LElbow
    h36m[13] = coco_keypoints[9]   # LWrist

    # Right arm
    h36m[14] = coco_keypoints[6]   # RShoulder
    h36m[15] = coco_keypoints[8]   # RElbow
    h36m[16] = coco_keypoints[10]  # RWrist

    return h36m
