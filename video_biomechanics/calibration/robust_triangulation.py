"""
Robust 3D pose triangulation methods.

Two approaches:
1. Skeleton-constrained triangulation - optimize with bone length constraints
2. Per-view 3D lifting with fusion - lift each 2D pose independently then fuse
"""

import numpy as np
import cv2
from scipy.optimize import minimize
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


# Anatomical bone length constraints (in meters)
# Based on adult human proportions, with reasonable ranges
# Using H36M keypoint indices (the format used after coco_to_h36m conversion)
BONE_CONSTRAINTS = {
    # (joint1_idx, joint2_idx): (min_length, max_length, typical_length)
    # H36M: 0=pelvis, 1=rhip, 2=rknee, 3=rankle, 4=lhip, 5=lknee, 6=lankle,
    #       7=spine, 8=thorax, 9=neck, 10=head, 11=lshoulder, 12=lelbow,
    #       13=lwrist, 14=rshoulder, 15=relbow, 16=rwrist
    'right_thigh': (1, 2, 0.35, 0.55, 0.45),      # rhip to rknee
    'left_thigh': (4, 5, 0.35, 0.55, 0.45),       # lhip to lknee
    'right_shin': (2, 3, 0.35, 0.50, 0.42),       # rknee to rankle
    'left_shin': (5, 6, 0.35, 0.50, 0.42),        # lknee to lankle
    'right_upper_arm': (14, 15, 0.25, 0.38, 0.30), # rshoulder to relbow
    'left_upper_arm': (11, 12, 0.25, 0.38, 0.30),  # lshoulder to lelbow
    'right_forearm': (15, 16, 0.20, 0.32, 0.26),   # relbow to rwrist
    'left_forearm': (12, 13, 0.20, 0.32, 0.26),    # lelbow to lwrist
    'shoulders': (11, 14, 0.30, 0.50, 0.40),       # shoulder to shoulder
    'hips': (1, 4, 0.20, 0.35, 0.28),              # hip to hip
    'torso_left': (11, 4, 0.40, 0.60, 0.50),       # lshoulder to lhip
    'torso_right': (14, 1, 0.40, 0.60, 0.50),      # rshoulder to rhip
    'neck': (9, 10, 0.10, 0.25, 0.15),             # neck to head
}

# H36M skeleton for lifting (17 joints)
H36M_SKELETON = [
    (0, 1),   # Pelvis -> RHip
    (1, 2),   # RHip -> RKnee
    (2, 3),   # RKnee -> RAnkle
    (0, 4),   # Pelvis -> LHip
    (4, 5),   # LHip -> LKnee
    (5, 6),   # LKnee -> LAnkle
    (0, 7),   # Pelvis -> Spine
    (7, 8),   # Spine -> Thorax
    (8, 9),   # Thorax -> Neck
    (9, 10),  # Neck -> Head
    (8, 11),  # Thorax -> LShoulder
    (11, 12), # LShoulder -> LElbow
    (12, 13), # LElbow -> LWrist
    (8, 14),  # Thorax -> RShoulder
    (14, 15), # RShoulder -> RElbow
    (15, 16), # RElbow -> RWrist
]


@dataclass
class TriangulationResult:
    """Result of robust triangulation."""
    poses_3d: np.ndarray           # (N, 17, 3) 3D poses
    method: str                     # 'skeleton_constrained' or 'lifting_fusion'
    per_frame_errors: np.ndarray   # Reprojection errors per frame
    bone_length_violations: int    # Number of bone length constraint violations
    confidence: np.ndarray         # Per-joint confidence scores


class SkeletonConstrainedTriangulator:
    """
    Triangulate 3D poses with skeleton constraints.

    Instead of triangulating each joint independently, optimizes for
    a complete skeleton that:
    1. Minimizes reprojection error in both views
    2. Satisfies bone length constraints
    3. Handles left/right ambiguity
    """

    def __init__(self, P1: np.ndarray, P2: np.ndarray,
                 bone_constraints: Dict = None):
        """
        Args:
            P1: Projection matrix for view 1 (3x4)
            P2: Projection matrix for view 2 (3x4)
            bone_constraints: Optional custom bone length constraints
        """
        self.P1 = P1
        self.P2 = P2
        self.bone_constraints = bone_constraints or BONE_CONSTRAINTS

    def triangulate_point(self, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        """Basic DLT triangulation for a single point."""
        pts1 = np.array([[p1[0]], [p1[1]]], dtype=np.float64)
        pts2 = np.array([[p2[0]], [p2[1]]], dtype=np.float64)
        X = cv2.triangulatePoints(self.P1, self.P2, pts1, pts2)
        return X[:3, 0] / X[3, 0]

    def project(self, pt_3d: np.ndarray, P: np.ndarray) -> np.ndarray:
        """Project 3D point to 2D."""
        pt_h = np.append(pt_3d, 1)
        proj = P @ pt_h
        return proj[:2] / proj[2]

    def reprojection_error(self, pts_3d: np.ndarray,
                           kps1: np.ndarray, kps2: np.ndarray) -> float:
        """Compute total reprojection error."""
        error = 0.0
        for i in range(len(pts_3d)):
            proj1 = self.project(pts_3d[i], self.P1)
            proj2 = self.project(pts_3d[i], self.P2)
            error += np.linalg.norm(proj1 - kps1[i])
            error += np.linalg.norm(proj2 - kps2[i])
        return error

    def bone_length_cost(self, pts_3d: np.ndarray) -> float:
        """Compute penalty for bone length violations."""
        cost = 0.0
        for name, (j1, j2, min_len, max_len, typical) in self.bone_constraints.items():
            if j1 >= len(pts_3d) or j2 >= len(pts_3d):
                continue
            length = np.linalg.norm(pts_3d[j1] - pts_3d[j2])
            if length < min_len:
                cost += (min_len - length) ** 2 * 1000  # Strong penalty
            elif length > max_len:
                cost += (length - max_len) ** 2 * 1000
            else:
                # Soft penalty toward typical length
                cost += (length - typical) ** 2 * 10
        return cost

    def optimize_skeleton(self, kps1: np.ndarray, kps2: np.ndarray,
                          initial_3d: np.ndarray = None,
                          test_lr_swap: bool = True,
                          max_iter: int = 50) -> Tuple[np.ndarray, float]:
        """
        Optimize 3D skeleton to minimize reprojection + bone constraints.

        Args:
            kps1: (17, 2) keypoints from view 1
            kps2: (17, 2) keypoints from view 2
            initial_3d: Optional initial 3D estimate
            test_lr_swap: Whether to test left/right swapped configurations
            max_iter: Maximum optimization iterations (default 50 for speed)

        Returns:
            Optimized 3D positions (17, 3) and final cost
        """
        n_joints = len(kps1)

        # Get initial estimate from basic triangulation
        if initial_3d is None:
            initial_3d = np.zeros((n_joints, 3))
            for i in range(n_joints):
                initial_3d[i] = self.triangulate_point(kps1[i], kps2[i])

        def objective(x):
            pts_3d = x.reshape(-1, 3)
            reproj_cost = self.reprojection_error(pts_3d, kps1, kps2)
            bone_cost = self.bone_length_cost(pts_3d)
            return reproj_cost + bone_cost

        # Optimize
        x0 = initial_3d.flatten()
        result = minimize(objective, x0, method='L-BFGS-B',
                         options={'maxiter': max_iter, 'ftol': 1e-5})

        best_pts = result.x.reshape(-1, 3)
        best_cost = result.fun

        # Test left/right swap if requested
        if test_lr_swap:
            # Swap left/right keypoints in view 2
            kps2_swapped = kps2.copy()
            lr_pairs = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]
            for l, r in lr_pairs:
                if l < len(kps2_swapped) and r < len(kps2_swapped):
                    kps2_swapped[l], kps2_swapped[r] = kps2[r].copy(), kps2[l].copy()

            # Re-triangulate with swapped
            initial_swapped = np.zeros((n_joints, 3))
            for i in range(n_joints):
                initial_swapped[i] = self.triangulate_point(kps1[i], kps2_swapped[i])

            def objective_swapped(x):
                pts_3d = x.reshape(-1, 3)
                reproj_cost = self.reprojection_error(pts_3d, kps1, kps2_swapped)
                bone_cost = self.bone_length_cost(pts_3d)
                return reproj_cost + bone_cost

            result_swapped = minimize(objective_swapped, initial_swapped.flatten(),
                                     method='L-BFGS-B',
                                     options={'maxiter': max_iter, 'ftol': 1e-5})

            if result_swapped.fun < best_cost:
                best_pts = result_swapped.x.reshape(-1, 3)
                best_cost = result_swapped.fun

        return best_pts, best_cost

    def triangulate_sequence(self, keypoints1: np.ndarray, keypoints2: np.ndarray,
                             use_temporal: bool = True) -> TriangulationResult:
        """
        Triangulate a sequence of poses.

        Args:
            keypoints1: (N, 17, 2) keypoints from view 1
            keypoints2: (N, 17, 2) keypoints from view 2
            use_temporal: Use previous frame as initialization

        Returns:
            TriangulationResult with 3D poses
        """
        n_frames = len(keypoints1)
        poses_3d = np.zeros((n_frames, 17, 3))
        errors = np.zeros(n_frames)

        prev_pose = None

        for i in range(n_frames):
            initial = prev_pose if use_temporal and prev_pose is not None else None

            pose_3d, cost = self.optimize_skeleton(
                keypoints1[i], keypoints2[i],
                initial_3d=initial,
                test_lr_swap=(i == 0)  # Only test swap on first frame
            )

            poses_3d[i] = pose_3d
            errors[i] = cost
            prev_pose = pose_3d

        # Count bone length violations
        violations = 0
        for i in range(n_frames):
            for name, (j1, j2, min_len, max_len, _) in self.bone_constraints.items():
                if j1 >= 17 or j2 >= 17:
                    continue
                length = np.linalg.norm(poses_3d[i, j1] - poses_3d[i, j2])
                if length < min_len or length > max_len:
                    violations += 1

        return TriangulationResult(
            poses_3d=poses_3d,
            method='skeleton_constrained',
            per_frame_errors=errors,
            bone_length_violations=violations,
            confidence=np.ones((n_frames, 17))  # TODO: compute actual confidence
        )


class LiftingFusionTriangulator:
    """
    Lift 2D poses to 3D independently, then fuse using calibration.

    Uses anthropometric priors to estimate depth from 2D proportions,
    then aligns the two 3D estimates using Procrustes alignment.
    """

    def __init__(self, K1: np.ndarray, K2: np.ndarray,
                 R1: np.ndarray, R2: np.ndarray,
                 cam1_pos: np.ndarray, cam2_pos: np.ndarray,
                 person_height: float = 1.75):
        """
        Args:
            K1, K2: Camera intrinsic matrices
            R1, R2: Camera rotation matrices
            cam1_pos, cam2_pos: Camera positions in world coordinates
            person_height: Assumed person height in meters
        """
        self.K1 = K1
        self.K2 = K2
        self.R1 = R1
        self.R2 = R2
        self.cam1_pos = cam1_pos
        self.cam2_pos = cam2_pos
        self.person_height = person_height

    def lift_2d_to_3d(self, keypoints_2d: np.ndarray,
                       K: np.ndarray, R: np.ndarray,
                       cam_pos: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Lift 2D keypoints to 3D world coordinates using proportional depth.

        Args:
            keypoints_2d: (17, 2) 2D keypoints in image coordinates
            K: Camera intrinsic matrix
            R: Camera rotation matrix
            cam_pos: Camera position in world coordinates

        Returns:
            (17, 3) estimated 3D positions in world coordinates, estimated depth
        """
        # Estimate depth from person's 2D vertical extent
        # Using H36M indices: 10=head, 3=rankle, 6=lankle
        head_y = keypoints_2d[10][1]  # head
        ankle_y = (keypoints_2d[3][1] + keypoints_2d[6][1]) / 2  # average of ankles
        person_2d_height = abs(ankle_y - head_y)

        f = (K[0, 0] + K[1, 1]) / 2  # focal length

        if person_2d_height > 50:
            depth = f * self.person_height / person_2d_height
        else:
            depth = 3.0  # fallback

        # Backproject to camera coordinates
        # In camera frame: Y points down (image convention)
        # After R.T transform, camera Y maps to world via R[:,1]
        # If R[:,1][1] > 0, camera-Y-down maps to world-Y-up (no flip needed)
        # If R[:,1][1] < 0, camera-Y-down maps to world-Y-down (flip to get Y-up)
        cam_y_in_world = R[:, 1]
        flip_y = cam_y_in_world[1] < 0  # Flip when camera Y-axis points DOWN in world

        cx, cy = K[0, 2], K[1, 2]
        pts_cam = np.zeros((17, 3))

        for i in range(17):
            x_2d, y_2d = keypoints_2d[i]
            x_cam = (x_2d - cx) * depth / f
            y_cam = (y_2d - cy) * depth / f
            if flip_y:
                y_cam = -y_cam  # Flip for cameras where image-down maps to world-up
            z_cam = depth
            pts_cam[i] = [x_cam, y_cam, z_cam]

        # Transform to world coordinates
        pts_world = np.zeros((17, 3))
        for i in range(17):
            pts_world[i] = R.T @ pts_cam[i] + cam_pos

        return pts_world, depth

    def procrustes_align(self, source: np.ndarray,
                          target: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Align source skeleton to target using Procrustes (rotation, scale, translation).

        Returns:
            Aligned source points, transform dict with R, scale, t
        """
        src_center = source.mean(axis=0)
        tgt_center = target.mean(axis=0)
        src_centered = source - src_center
        tgt_centered = target - tgt_center

        src_scale = np.sqrt((src_centered ** 2).sum())
        tgt_scale = np.sqrt((tgt_centered ** 2).sum())
        src_normalized = src_centered / src_scale
        tgt_normalized = tgt_centered / tgt_scale

        H = src_normalized.T @ tgt_normalized
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1] *= -1
            R = Vt.T @ U.T

        scale = tgt_scale / src_scale
        aligned = scale * (src_centered @ R.T) + tgt_center

        return aligned, {'R': R, 'scale': scale, 't': tgt_center - scale * (R @ src_center)}

    def triangulate(self, kps1: np.ndarray, kps2: np.ndarray) -> np.ndarray:
        """
        Full pipeline: lift each view to 3D, align with Procrustes, then average.

        Args:
            kps1: (17, 2) keypoints from view 1
            kps2: (17, 2) keypoints from view 2

        Returns:
            (17, 3) fused 3D pose in world coordinates
        """
        # Lift each view independently
        pts1_world, _ = self.lift_2d_to_3d(kps1, self.K1, self.R1, self.cam1_pos)
        pts2_world, _ = self.lift_2d_to_3d(kps2, self.K2, self.R2, self.cam2_pos)

        # Align view 2 to view 1 using Procrustes
        pts2_aligned, _ = self.procrustes_align(pts2_world, pts1_world)

        # Average the aligned estimates
        pts_fused = (pts1_world + pts2_aligned) / 2

        return pts_fused

    def triangulate_sequence(self, keypoints1: np.ndarray,
                              keypoints2: np.ndarray) -> TriangulationResult:
        """
        Triangulate a sequence of poses.

        Args:
            keypoints1: (N, 17, 2) keypoints from view 1
            keypoints2: (N, 17, 2) keypoints from view 2

        Returns:
            TriangulationResult with 3D poses
        """
        n_frames = len(keypoints1)
        poses_3d = np.zeros((n_frames, 17, 3))

        for i in range(n_frames):
            poses_3d[i] = self.triangulate(keypoints1[i], keypoints2[i])

        return TriangulationResult(
            poses_3d=poses_3d,
            method='lifting_fusion',
            per_frame_errors=np.zeros(n_frames),  # Not computed for lifting
            bone_length_violations=0,
            confidence=np.ones((n_frames, 17))
        )


class HybridTriangulator:
    """
    Combines skeleton-constrained and lifting-fusion approaches.

    Uses lifting fusion for initial estimate (good depth), then refines
    with skeleton constraints (good bone lengths).
    """

    def __init__(self, P1: np.ndarray, P2: np.ndarray,
                 K1: np.ndarray, K2: np.ndarray,
                 R1: np.ndarray, R2: np.ndarray,
                 cam1_pos: np.ndarray, cam2_pos: np.ndarray,
                 person_height: float = 1.75):
        """Initialize with full camera parameters."""
        self.P1 = P1
        self.P2 = P2

        self.lifting = LiftingFusionTriangulator(
            K1, K2, R1, R2, cam1_pos, cam2_pos, person_height
        )
        self.skeleton = SkeletonConstrainedTriangulator(P1, P2)

    def triangulate(self, kps1: np.ndarray, kps2: np.ndarray,
                    test_lr_swap: bool = True,
                    fast_mode: bool = True) -> Tuple[np.ndarray, str]:
        """
        Hybrid triangulation combining both methods.

        Fast mode (default): Uses lifting fusion only - good for batch processing.
        Full mode: Tests multiple methods and picks best.

        Returns:
            (17, 3) 3D pose, method used
        """
        # Fast mode: just use lifting fusion (good scale/depth, ~10ms)
        if fast_mode:
            try:
                pts = self.lifting.triangulate(kps1, kps2)
                return pts, 'lifting'
            except Exception:
                # Fallback to basic DLT
                pts_basic = np.zeros((17, 3))
                for i in range(17):
                    pts_basic[i] = self.skeleton.triangulate_point(kps1[i], kps2[i])
                return pts_basic, 'basic_dlt'

        # Full mode: test multiple methods (slower but may be more accurate)
        results = []

        # Method 1: Lifting fusion (fast, good scale)
        try:
            pts_lifting = self.lifting.triangulate(kps1, kps2)
            score = self._score_skeleton(pts_lifting)
            results.append(('lifting', pts_lifting, score))
        except Exception:
            pass

        # Method 2: Lifting + light skeleton refinement
        try:
            pts_lifting = self.lifting.triangulate(kps1, kps2)
            pts_refined, _ = self.skeleton.optimize_skeleton(
                kps1, kps2, initial_3d=pts_lifting, test_lr_swap=test_lr_swap,
                max_iter=30  # Light refinement only
            )
            score = self._score_skeleton(pts_refined)
            results.append(('lifting_refined', pts_refined, score))
        except Exception:
            pass

        if not results:
            # Fallback to basic DLT
            pts_basic = np.zeros((17, 3))
            for i in range(17):
                pts_basic[i] = self.skeleton.triangulate_point(kps1[i], kps2[i])
            return pts_basic, 'basic_dlt'

        # Return best result (highest score = most valid bone lengths)
        best = max(results, key=lambda x: x[2])
        return best[1], best[0]

    def _score_skeleton(self, pts_3d: np.ndarray) -> float:
        """
        Score skeleton by how many bone lengths are within valid ranges.

        Higher score = better.
        """
        score = 0.0
        for name, (j1, j2, min_len, max_len, typical) in BONE_CONSTRAINTS.items():
            if j1 >= len(pts_3d) or j2 >= len(pts_3d):
                continue
            length = np.linalg.norm(pts_3d[j1] - pts_3d[j2])
            if min_len <= length <= max_len:
                score += 1.0  # In range
                # Bonus for being close to typical
                deviation = abs(length - typical) / typical
                score += max(0, 1.0 - deviation)
            else:
                # Penalty for out of range
                if length < min_len:
                    score -= (min_len - length) / min_len
                else:
                    score -= (length - max_len) / max_len
        return score

    def triangulate_sequence(self, keypoints1: np.ndarray,
                              keypoints2: np.ndarray) -> TriangulationResult:
        """Triangulate a sequence using hybrid approach."""
        n_frames = len(keypoints1)
        poses_3d = np.zeros((n_frames, 17, 3))
        methods_used = []

        for i in range(n_frames):
            # Only test LR swap on first frame
            pts, method = self.triangulate(
                keypoints1[i], keypoints2[i],
                test_lr_swap=(i == 0)
            )
            poses_3d[i] = pts
            methods_used.append(method)

        return TriangulationResult(
            poses_3d=poses_3d,
            method=f'hybrid ({methods_used[0]})',
            per_frame_errors=np.zeros(n_frames),
            bone_length_violations=0,
            confidence=np.ones((n_frames, 17))
        )


def triangulate_robust(kps1: np.ndarray, kps2: np.ndarray,
                       P1: np.ndarray, P2: np.ndarray,
                       method: str = 'skeleton_constrained',
                       **kwargs) -> np.ndarray:
    """
    Convenience function for robust triangulation.

    Args:
        kps1: (17, 2) or (N, 17, 2) keypoints from view 1
        kps2: (17, 2) or (N, 17, 2) keypoints from view 2
        P1, P2: Projection matrices
        method: 'skeleton_constrained' or 'lifting_fusion'

    Returns:
        (17, 3) or (N, 17, 3) triangulated 3D poses
    """
    if method == 'skeleton_constrained':
        triangulator = SkeletonConstrainedTriangulator(P1, P2)

        if kps1.ndim == 2:
            # Single frame
            pts_3d, _ = triangulator.optimize_skeleton(kps1, kps2, **kwargs)
            return pts_3d
        else:
            # Sequence
            result = triangulator.triangulate_sequence(kps1, kps2, **kwargs)
            return result.poses_3d

    elif method == 'lifting_fusion':
        raise NotImplementedError("Lifting fusion requires camera intrinsics/extrinsics. "
                                  "Use LiftingFusionTriangulator directly.")

    else:
        raise ValueError(f"Unknown method: {method}")
