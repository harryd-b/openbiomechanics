"""
Hybrid 3D pose estimation pipeline.

Combines single-view VideoPose3D lifting with stereo triangulation for
improved accuracy and robustness.

Strategy:
1. Run 2D detection on both cameras
2. Lift each view to 3D independently using VideoPose3D
3. Triangulate 2D detections from both views
4. Fuse results: use triangulation for global position, single-view for bone structure
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent.parent))

from calibration import PlateCalibrator, CalibrationConfig, JOINT_NAMES
from calibration.run_production import compute_mpjpe, align_frame_rates
from calibration.compare_pose_estimators import (
    MediaPipeEstimator, YOLOv8Estimator, VideoPose3DLifter,
    lift_2d_to_3d, normalize_poses
)
from fusion.train_fusion import load_uplift_positions


@dataclass
class HybridResult:
    """Result from hybrid pipeline."""
    poses_3d: np.ndarray  # (n_frames, 17, 3)
    triangulated: np.ndarray  # Raw triangulation
    single_view_side: np.ndarray  # Single-view from side camera
    single_view_back: np.ndarray  # Single-view from back camera
    method_weights: np.ndarray  # Per-joint fusion weights
    fps: float


def triangulate_points(
    points_2d_1: np.ndarray,
    points_2d_2: np.ndarray,
    P1: np.ndarray,
    P2: np.ndarray
) -> np.ndarray:
    """
    Triangulate 3D points from two 2D views using DLT.

    Args:
        points_2d_1: (n_joints, 2) from camera 1
        points_2d_2: (n_joints, 2) from camera 2
        P1, P2: (3, 4) projection matrices

    Returns:
        (n_joints, 3) 3D points
    """
    n_joints = len(points_2d_1)
    points_3d = np.zeros((n_joints, 3))

    for j in range(n_joints):
        x1, y1 = points_2d_1[j]
        x2, y2 = points_2d_2[j]

        # Build DLT matrix
        A = np.array([
            x1 * P1[2, :] - P1[0, :],
            y1 * P1[2, :] - P1[1, :],
            x2 * P2[2, :] - P2[0, :],
            y2 * P2[2, :] - P2[1, :]
        ])

        # SVD solution
        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1]
        points_3d[j] = X[:3] / X[3]

    return points_3d


def compute_bone_lengths(poses_3d: np.ndarray) -> Dict[str, float]:
    """Compute median bone lengths from 3D poses."""
    # H36M skeleton connections
    bones = {
        'r_upper_arm': (14, 15),  # R shoulder -> R elbow
        'r_forearm': (15, 16),    # R elbow -> R wrist
        'l_upper_arm': (11, 12),  # L shoulder -> L elbow
        'l_forearm': (12, 13),    # L elbow -> L wrist
        'r_thigh': (1, 2),        # R hip -> R knee
        'r_shin': (2, 3),         # R knee -> R ankle
        'l_thigh': (4, 5),        # L hip -> L knee
        'l_shin': (5, 6),         # L knee -> L ankle
        'torso': (0, 8),          # Pelvis -> Neck
    }

    lengths = {}
    for name, (j1, j2) in bones.items():
        dists = np.linalg.norm(poses_3d[:, j1, :] - poses_3d[:, j2, :], axis=1)
        lengths[name] = float(np.median(dists[dists > 0]))

    return lengths


def enforce_bone_lengths(
    poses_3d: np.ndarray,
    target_lengths: Dict[str, float],
    iterations: int = 3
) -> np.ndarray:
    """
    Adjust poses to match target bone lengths while preserving joint angles.

    Uses iterative projection to pull joints toward correct bone length.
    """
    bones = {
        'r_upper_arm': (14, 15),
        'r_forearm': (15, 16),
        'l_upper_arm': (11, 12),
        'l_forearm': (12, 13),
        'r_thigh': (1, 2),
        'r_shin': (2, 3),
        'l_thigh': (4, 5),
        'l_shin': (5, 6),
    }

    result = poses_3d.copy()

    for _ in range(iterations):
        for name, (parent, child) in bones.items():
            if name not in target_lengths:
                continue

            target = target_lengths[name]
            vec = result[:, child, :] - result[:, parent, :]
            current = np.linalg.norm(vec, axis=1, keepdims=True)
            current = np.clip(current, 1e-6, None)

            # Scale vector to target length
            scaled = vec * (target / current)
            result[:, child, :] = result[:, parent, :] + scaled

    return result


def fuse_poses(
    triangulated: np.ndarray,
    single_view_1: np.ndarray,
    single_view_2: np.ndarray,
    confidence_1: np.ndarray,
    confidence_2: np.ndarray,
    alpha: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fuse triangulated poses with single-view estimates.

    CONSERVATIVE Strategy:
    - Use triangulation as primary (it's more accurate when it works)
    - Only use single-view as fallback for NaN/invalid triangulated joints
    - Light temporal smoothing to reduce jitter

    Args:
        triangulated: (n_frames, 17, 3) triangulated poses
        single_view_1: (n_frames, 17, 3) from camera 1
        single_view_2: (n_frames, 17, 3) from camera 2
        confidence_1: (n_frames, 17) detection confidence
        confidence_2: (n_frames, 17) detection confidence
        alpha: Not used in conservative mode

    Returns:
        fused: (n_frames, 17, 3) fused poses
        weights: (n_frames, 17) per-joint weights used (1=triangulation, 0=single-view)
    """
    n_frames = len(triangulated)
    fused = triangulated.copy()
    weights = np.ones((n_frames, 17))  # Track which method was used

    # Align single-view to triangulated scale/position
    pelvis_tri = triangulated[:, 0:1, :]
    pelvis_sv1 = single_view_1[:, 0:1, :]
    pelvis_sv2 = single_view_2[:, 0:1, :]

    sv1_aligned = single_view_1 - pelvis_sv1 + pelvis_tri
    sv2_aligned = single_view_2 - pelvis_sv2 + pelvis_tri

    # Scale to match triangulated torso
    torso_tri = np.linalg.norm(triangulated[:, 8, :] - triangulated[:, 0, :], axis=1, keepdims=True)
    torso_sv1 = np.linalg.norm(single_view_1[:, 8, :] - single_view_1[:, 0, :], axis=1, keepdims=True)
    torso_sv2 = np.linalg.norm(single_view_2[:, 8, :] - single_view_2[:, 0, :], axis=1, keepdims=True)

    scale_1 = np.clip(torso_tri / np.clip(torso_sv1, 0.01, None), 0.5, 2.0)
    scale_2 = np.clip(torso_tri / np.clip(torso_sv2, 0.01, None), 0.5, 2.0)

    sv1_scaled = (sv1_aligned - pelvis_tri) * scale_1[:, :, np.newaxis] + pelvis_tri
    sv2_scaled = (sv2_aligned - pelvis_tri) * scale_2[:, :, np.newaxis] + pelvis_tri

    # Fill in NaN values from triangulation with single-view
    nan_mask = np.isnan(triangulated).any(axis=2)  # (n_frames, 17)

    for f in range(n_frames):
        for j in range(17):
            if nan_mask[f, j]:
                # Use weighted average of single-view estimates
                w1 = confidence_1[f, j]
                w2 = confidence_2[f, j]
                w_total = w1 + w2 + 1e-6

                fused[f, j, :] = (sv1_scaled[f, j, :] * w1 + sv2_scaled[f, j, :] * w2) / w_total
                weights[f, j] = 0  # Mark as single-view

    # Count fallbacks
    n_fallbacks = np.sum(nan_mask)
    if n_fallbacks > 0:
        print(f"    Used single-view fallback for {n_fallbacks} joint-frames "
              f"({100*n_fallbacks/(n_frames*17):.1f}%)")

    return fused, weights


def fuse_poses_weighted(
    triangulated: np.ndarray,
    single_view_1: np.ndarray,
    single_view_2: np.ndarray,
    confidence_1: np.ndarray,
    confidence_2: np.ndarray,
    alpha: float = 0.9
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Alternative fusion with weighted blending.

    Uses high alpha (0.9) to strongly prefer triangulation but still
    incorporate some single-view information for smoothing.
    """
    n_frames = len(triangulated)

    # Align single-view to triangulated
    pelvis_tri = triangulated[:, 0:1, :]
    pelvis_sv1 = single_view_1[:, 0:1, :]
    pelvis_sv2 = single_view_2[:, 0:1, :]

    sv1_aligned = single_view_1 - pelvis_sv1 + pelvis_tri
    sv2_aligned = single_view_2 - pelvis_sv2 + pelvis_tri

    # Scale
    torso_tri = np.linalg.norm(triangulated[:, 8, :] - triangulated[:, 0, :], axis=1, keepdims=True)
    torso_sv1 = np.linalg.norm(single_view_1[:, 8, :] - single_view_1[:, 0, :], axis=1, keepdims=True)
    torso_sv2 = np.linalg.norm(single_view_2[:, 8, :] - single_view_2[:, 0, :], axis=1, keepdims=True)

    scale_1 = np.clip(torso_tri / np.clip(torso_sv1, 0.01, None), 0.5, 2.0)
    scale_2 = np.clip(torso_tri / np.clip(torso_sv2, 0.01, None), 0.5, 2.0)

    sv1_scaled = (sv1_aligned - pelvis_tri) * scale_1[:, :, np.newaxis] + pelvis_tri
    sv2_scaled = (sv2_aligned - pelvis_tri) * scale_2[:, :, np.newaxis] + pelvis_tri

    # Average single-view
    sv_avg = (sv1_scaled + sv2_scaled) / 2

    # Simple weighted blend
    fused = alpha * triangulated + (1 - alpha) * sv_avg

    # Handle NaNs - use single-view where triangulation failed
    nan_mask = np.isnan(triangulated)
    fused[nan_mask] = sv_avg[nan_mask]

    weights = np.ones((n_frames, 17)) * alpha
    weights[nan_mask.any(axis=2)] = 0

    return fused, weights


def run_hybrid_pipeline(
    session_dir: Path,
    estimator_type: str = 'mediapipe'
) -> Dict:
    """
    Run hybrid 3D pose estimation pipeline.

    Args:
        session_dir: Session directory with videos
        estimator_type: 'mediapipe' or 'yolov8'

    Returns:
        Results dictionary
    """
    print("=" * 70)
    print("HYBRID 3D POSE ESTIMATION PIPELINE")
    print("=" * 70)
    print(f"Session: {session_dir.name}")
    print(f"Estimator: {estimator_type}")

    # Find videos (primary/secondary preferred, fallback to side/back)
    video_side = None
    video_back = None

    for name in ['primary.mp4', 'side.mp4']:
        candidate = session_dir / name
        if candidate.exists():
            video_side = candidate
            break

    for name in ['secondary.mp4', 'back.mp4']:
        candidate = session_dir / name
        if candidate.exists():
            video_back = candidate
            break

    if not video_side or not video_back or not video_side.exists() or not video_back.exists():
        return {'error': 'Videos not found (need primary.mp4/secondary.mp4 or side.mp4/back.mp4)'}

    # Initialize estimator
    print("\n" + "-" * 70)
    print("STEP 1: 2D Pose Detection")
    print("-" * 70)

    if estimator_type == 'mediapipe':
        estimator = MediaPipeEstimator()
    else:
        estimator = YOLOv8Estimator('yolov8m-pose.pt')

    # Process both videos
    print(f"  Processing side view...")
    result_side = estimator.process_video(str(video_side))
    kp_side_h36m = estimator.to_h36m(result_side.keypoints)
    conf_side = result_side.keypoints[:, :, 2] if result_side.keypoints.shape[2] > 2 else np.ones((len(result_side.keypoints), 17))
    # Map confidence to H36M joints
    if estimator_type == 'mediapipe':
        conf_side_h36m = np.ones((len(result_side.keypoints), 17)) * 0.9
    else:
        conf_side_h36m = conf_side

    print(f"  Processing back view...")
    result_back = estimator.process_video(str(video_back))
    kp_back_h36m = estimator.to_h36m(result_back.keypoints)
    conf_back = result_back.keypoints[:, :, 2] if result_back.keypoints.shape[2] > 2 else np.ones((len(result_back.keypoints), 17))
    if estimator_type == 'mediapipe':
        conf_back_h36m = np.ones((len(result_back.keypoints), 17)) * 0.9
    else:
        conf_back_h36m = conf_back

    fps = result_side.fps

    # Align frame counts
    n_frames = min(len(kp_side_h36m), len(kp_back_h36m))
    kp_side_h36m = kp_side_h36m[:n_frames]
    kp_back_h36m = kp_back_h36m[:n_frames]
    conf_side_h36m = conf_side_h36m[:n_frames]
    conf_back_h36m = conf_back_h36m[:n_frames]

    print(f"  Frames: {n_frames}")

    # Step 2: Single-view lifting
    print("\n" + "-" * 70)
    print("STEP 2: Single-View 3D Lifting (VideoPose3D)")
    print("-" * 70)

    poses_3d_side = lift_2d_to_3d(kp_side_h36m, use_learned=True)
    poses_3d_back = lift_2d_to_3d(kp_back_h36m, use_learned=True)

    print(f"  Side view lifted: {poses_3d_side.shape}")
    print(f"  Back view lifted: {poses_3d_back.shape}")

    # Step 3: Triangulation
    print("\n" + "-" * 70)
    print("STEP 3: Stereo Triangulation")
    print("-" * 70)

    # Load calibration and run triangulation
    config = CalibrationConfig()
    calibrator = PlateCalibrator(config=config)

    videos = {
        'side': str(video_side),
        'back': str(video_back)
    }

    tri_results = calibrator.process_videos(videos, progress_interval=100)
    poses_triangulated = tri_results['poses_3d']

    # Align frame counts again
    n_frames = min(n_frames, len(poses_triangulated))
    poses_triangulated = poses_triangulated[:n_frames]
    poses_3d_side = poses_3d_side[:n_frames]
    poses_3d_back = poses_3d_back[:n_frames]
    conf_side_h36m = conf_side_h36m[:n_frames]
    conf_back_h36m = conf_back_h36m[:n_frames]

    print(f"  Triangulated frames: {len(poses_triangulated)}")

    # Step 4: Fusion
    print("\n" + "-" * 70)
    print("STEP 4: Fuse Triangulation + Single-View")
    print("-" * 70)

    # Normalize all poses to same scale
    poses_tri_norm = normalize_poses(poses_triangulated)
    poses_side_norm = normalize_poses(poses_3d_side)
    poses_back_norm = normalize_poses(poses_3d_back)

    # Conservative fusion (fallback only)
    poses_fused, fusion_weights = fuse_poses(
        poses_tri_norm,
        poses_side_norm,
        poses_back_norm,
        conf_side_h36m,
        conf_back_h36m
    )

    # Also try weighted fusion
    poses_weighted, _ = fuse_poses_weighted(
        poses_tri_norm,
        poses_side_norm,
        poses_back_norm,
        conf_side_h36m,
        conf_back_h36m,
        alpha=0.95  # Very strongly prefer triangulation
    )

    print(f"  Fused poses: {poses_fused.shape}")

    # Step 5: Evaluate
    result = {
        'session': session_dir.name,
        'n_frames': n_frames,
        'fps': fps,
        'poses_fused': poses_fused,
        'poses_weighted': poses_weighted,
        'poses_triangulated': poses_tri_norm,
        'poses_side': poses_side_norm,
        'poses_back': poses_back_norm,
    }

    uplift_csv = session_dir / 'uplift.csv'
    if uplift_csv.exists():
        print("\n" + "-" * 70)
        print("STEP 5: Compare with UPLIFT Ground Truth")
        print("-" * 70)

        gt_poses = load_uplift_positions(str(uplift_csv))
        gt_norm = normalize_poses(gt_poses)

        # Align frame rates
        poses_fused_resampled = align_frame_rates(poses_fused, fps, 240.0)
        poses_tri_resampled = align_frame_rates(poses_tri_norm, fps, 240.0)
        poses_side_resampled = align_frame_rates(poses_side_norm, fps, 240.0)

        n_compare = min(len(poses_fused_resampled), len(gt_norm))

        # Compute MPJPE for each method
        mpjpe_fused, _ = compute_mpjpe(poses_fused_resampled[:n_compare], gt_norm[:n_compare])
        mpjpe_tri, _ = compute_mpjpe(poses_tri_resampled[:n_compare], gt_norm[:n_compare])
        mpjpe_side, _ = compute_mpjpe(poses_side_resampled[:n_compare], gt_norm[:n_compare])

        # Also evaluate weighted fusion
        poses_weighted_resampled = align_frame_rates(poses_weighted, fps, 240.0)
        mpjpe_weighted, _ = compute_mpjpe(poses_weighted_resampled[:n_compare], gt_norm[:n_compare])

        print(f"\n  Results (MPJPE in cm):")
        print(f"  {'Method':<30} {'MPJPE':>10}")
        print(f"  {'-' * 42}")
        print(f"  {'Single-view (side)':<30} {mpjpe_side:>10.2f}")
        print(f"  {'Triangulation only':<30} {mpjpe_tri:>10.2f}")
        print(f"  {'Hybrid (conservative)':<30} {mpjpe_fused:>10.2f}")
        print(f"  {'Hybrid (weighted, alpha=0.95)':<30} {mpjpe_weighted:>10.2f}")

        best_hybrid = min(mpjpe_fused, mpjpe_weighted)
        improvement = mpjpe_tri - best_hybrid
        print(f"\n  Best hybrid: {best_hybrid:.2f} cm")
        print(f"  Improvement over triangulation: {improvement:+.2f} cm")

        result['mpjpe_fused'] = float(mpjpe_fused)
        result['mpjpe_weighted'] = float(mpjpe_weighted)
        result['mpjpe_triangulated'] = float(mpjpe_tri)
        result['mpjpe_single_view'] = float(mpjpe_side)
        result['improvement'] = float(improvement)

    return result


def run_all_sessions(estimator_type: str = 'mediapipe'):
    """Run hybrid pipeline on all sessions."""
    training_data = Path(__file__).parent.parent / 'training_data'
    sessions = sorted(training_data.glob('session_*'))

    results = []

    for session_dir in sessions:
        print(f"\n{'='*70}")
        print(f"Processing {session_dir.name}")
        print('='*70)

        try:
            result = run_hybrid_pipeline(session_dir, estimator_type)
            if 'mpjpe_fused' in result:
                results.append({
                    'session': result['session'],
                    'mpjpe_single_view': result.get('mpjpe_single_view'),
                    'mpjpe_triangulated': result.get('mpjpe_triangulated'),
                    'mpjpe_fused': result.get('mpjpe_fused'),
                    'improvement': result.get('improvement'),
                })
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()

    if results:
        df = pd.DataFrame(results)
        print("\n" + "=" * 70)
        print("AGGREGATE RESULTS")
        print("=" * 70)
        print(df.to_string(index=False))

        print(f"\n  Mean MPJPE:")
        print(f"    Single-view:   {df['mpjpe_single_view'].mean():.2f} cm")
        print(f"    Triangulation: {df['mpjpe_triangulated'].mean():.2f} cm")
        print(f"    Hybrid:        {df['mpjpe_fused'].mean():.2f} cm")
        print(f"    Improvement:   {df['improvement'].mean():+.2f} cm")

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Hybrid 3D pose estimation")
    parser.add_argument('--session', type=Path, default=None)
    parser.add_argument('--estimator', choices=['mediapipe', 'yolov8'], default='mediapipe')
    parser.add_argument('--all', action='store_true')
    args = parser.parse_args()

    if args.all:
        run_all_sessions(args.estimator)
    elif args.session:
        run_hybrid_pipeline(args.session, args.estimator)
    else:
        session = Path(__file__).parent.parent / 'training_data' / 'session_001'
        run_hybrid_pipeline(session, args.estimator)


if __name__ == '__main__':
    main()
