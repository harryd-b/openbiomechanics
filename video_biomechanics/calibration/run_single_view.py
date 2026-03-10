"""
Single-view 3D pose estimation pipeline.

Uses YOLOv8 2D detection + VideoPose3D lifting on a single camera view.
Runs on both views and compares with UPLIFT ground truth.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))

from pose_estimation import PoseEstimator, PoseFrame
from lifting_3d import VideoPose3DLifter, Pose3D, YOLO_TO_H36M
from calibration import JOINT_NAMES
from calibration.run_production import compute_mpjpe, align_frame_rates
from fusion.train_fusion import load_uplift_positions


def yolo_to_h36m(poses_2d: np.ndarray) -> np.ndarray:
    """
    Convert YOLOv8 keypoints to H36M format for lifting.

    Args:
        poses_2d: (n_frames, 17, 3) YOLOv8 keypoints

    Returns:
        (n_frames, 17, 2) H36M format keypoints
    """
    n_frames = len(poses_2d)
    h36m = np.zeros((n_frames, 17, 2))

    for yolo_idx, h36m_idx in YOLO_TO_H36M.items():
        h36m[:, h36m_idx, :] = poses_2d[:, yolo_idx, :2]

    # Fill in missing joints (spine, neck, hip_center) by interpolation
    # Hip center = midpoint of left/right hip
    h36m[:, 0, :] = (h36m[:, 1, :] + h36m[:, 4, :]) / 2

    # Spine = between hip center and shoulders midpoint
    shoulders_mid = (h36m[:, 11, :] + h36m[:, 14, :]) / 2
    h36m[:, 7, :] = (h36m[:, 0, :] + shoulders_mid) / 2

    # Neck = midpoint of shoulders
    h36m[:, 8, :] = shoulders_mid

    # Head top = slightly above head
    h36m[:, 10, :] = h36m[:, 9, :] + np.array([0, -20])  # 20px above head

    return h36m


def process_single_view(
    video_path: Path,
    lifter: VideoPose3DLifter,
    pose_model: str = 'yolov8m-pose.pt'
) -> Tuple[np.ndarray, float, List[PoseFrame]]:
    """
    Process a single video view to get 3D poses.

    Returns:
        poses_3d: (n_frames, 17, 3) array
        fps: Video frame rate
        poses_2d: List of PoseFrame objects
    """
    print(f"  Processing {video_path.name}...")

    # 2D pose estimation
    estimator = PoseEstimator(model_name=pose_model)
    poses_2d = estimator.process_video(str(video_path))

    if not poses_2d:
        raise ValueError(f"No poses detected in {video_path}")

    # Get video FPS
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    # Convert to array
    keypoints = np.array([p.keypoints for p in poses_2d])
    timestamps = [p.timestamp for p in poses_2d]

    # Convert YOLO to H36M format
    keypoints_h36m = yolo_to_h36m(keypoints)

    # Lift to 3D
    poses_3d_list = lifter.lift_sequence(keypoints_h36m, timestamps)

    # Convert to array
    poses_3d = np.array([p.joints_3d for p in poses_3d_list])

    return poses_3d, fps, poses_2d


def normalize_poses(poses_3d: np.ndarray) -> np.ndarray:
    """
    Normalize poses to be pelvis-centered with consistent scale.

    Args:
        poses_3d: (n_frames, 17, 3)

    Returns:
        Normalized poses
    """
    # Center on pelvis
    pelvis = poses_3d[:, 0:1, :]
    centered = poses_3d - pelvis

    # Normalize scale using torso length (pelvis to neck)
    # This is more stable than using limbs
    torso_lengths = np.linalg.norm(centered[:, 8, :] - centered[:, 0, :], axis=1)
    median_torso = np.median(torso_lengths[torso_lengths > 0])

    if median_torso > 0:
        # Scale to ~0.5m torso (typical human)
        scale = 0.5 / median_torso
        centered = centered * scale

    return centered


def run_single_view_pipeline(
    session_dir: Path,
    view: str = 'side',
    pose_model: str = 'yolov8m-pose.pt'
) -> Dict:
    """
    Run single-view 3D estimation on a session.

    Args:
        session_dir: Session directory
        view: 'side' or 'back' (also accepts 'primary'/'secondary')
        pose_model: YOLOv8 pose model

    Returns:
        Results dictionary
    """
    print("=" * 70)
    print(f"SINGLE-VIEW 3D POSE ESTIMATION")
    print("=" * 70)
    print(f"Session: {session_dir.name}")
    print(f"View: {view}")

    # Find video
    video_map = {
        'side': ['primary.mp4', 'side.mp4'],
        'back': ['secondary.mp4', 'back.mp4'],
        'primary': ['primary.mp4', 'side.mp4'],
        'secondary': ['secondary.mp4', 'back.mp4'],
    }

    video_path = None
    for name in video_map.get(view, [f'{view}.mp4']):
        candidate = session_dir / name
        if candidate.exists():
            video_path = candidate
            break

    if video_path is None:
        return {'error': f'Video not found for view: {view}'}

    print(f"Video: {video_path.name}")

    # Initialize lifter
    lifter = VideoPose3DLifter()

    # Process video
    print("\n" + "-" * 70)
    print("STEP 1: 2D Pose Detection + 3D Lifting")
    print("-" * 70)

    try:
        poses_3d, fps, poses_2d = process_single_view(video_path, lifter, pose_model)
    except Exception as e:
        return {'error': str(e)}

    print(f"  Frames: {len(poses_3d)}")
    print(f"  FPS: {fps:.2f}")

    # Normalize poses
    print("\n" + "-" * 70)
    print("STEP 2: Normalize poses")
    print("-" * 70)

    poses_normalized = normalize_poses(poses_3d)
    print(f"  Centered on pelvis, scaled to ~0.5m torso")

    # Export
    print("\n" + "-" * 70)
    print("STEP 3: Export results")
    print("-" * 70)

    output_csv = session_dir / f'poses_3d_single_{view}.csv'
    export_poses_csv(poses_normalized, fps, str(output_csv))
    print(f"  Saved: {output_csv}")

    result = {
        'view': view,
        'video_path': str(video_path),
        'poses_3d': poses_normalized,
        'fps': fps,
        'n_frames': len(poses_3d),
        'output_csv': str(output_csv),
    }

    # Compare with UPLIFT if available
    uplift_csv = session_dir / 'uplift.csv'
    if uplift_csv.exists():
        print("\n" + "-" * 70)
        print("STEP 4: Compare with UPLIFT")
        print("-" * 70)

        gt_poses = load_uplift_positions(str(uplift_csv))

        # Also normalize ground truth
        gt_normalized = normalize_poses(gt_poses)

        # Align frame rates
        gt_fps = 240.0
        poses_resampled = align_frame_rates(poses_normalized, fps, gt_fps)

        n_compare = min(len(poses_resampled), len(gt_normalized))
        mpjpe, errors = compute_mpjpe(poses_resampled[:n_compare], gt_normalized[:n_compare])

        result['mpjpe_cm'] = float(mpjpe) if not np.isnan(mpjpe) else None
        result['n_frames_compared'] = n_compare

        if not np.isnan(mpjpe):
            print(f"  MPJPE: {mpjpe:.2f} cm")

            # Per-joint breakdown
            per_joint = np.nanmean(errors, axis=0) * 100
            result['per_joint_mpjpe'] = {
                JOINT_NAMES[j]: float(per_joint[j])
                for j in range(min(len(JOINT_NAMES), len(per_joint)))
            }

    return result


def export_poses_csv(poses_3d: np.ndarray, fps: float, output_path: str):
    """Export poses to CSV in UPLIFT-compatible format."""
    records = []

    for i, pose in enumerate(poses_3d):
        record = {
            'frame': i,
            'time': i / fps,
            'fps': fps,
        }

        for j, name in enumerate(JOINT_NAMES):
            if j < len(pose):
                record[f'{name.lower()}_3d_x'] = pose[j, 0]
                record[f'{name.lower()}_3d_y'] = pose[j, 1]
                record[f'{name.lower()}_3d_z'] = pose[j, 2]

        records.append(record)

    pd.DataFrame(records).to_csv(output_path, index=False)
    print(f"Exported {len(records)} frames to {output_path}")


def run_both_views(session_dir: Path, pose_model: str = 'yolov8m-pose.pt') -> Dict:
    """
    Run single-view estimation on both views and return best result.
    """
    results = {}

    for view in ['side', 'back']:
        result = run_single_view_pipeline(session_dir, view, pose_model)
        results[view] = result

    # Determine best view
    side_mpjpe = results.get('side', {}).get('mpjpe_cm')
    back_mpjpe = results.get('back', {}).get('mpjpe_cm')

    if side_mpjpe and back_mpjpe:
        best_view = 'side' if side_mpjpe < back_mpjpe else 'back'
    elif side_mpjpe:
        best_view = 'side'
    elif back_mpjpe:
        best_view = 'back'
    else:
        best_view = None

    return {
        'side': results.get('side', {}),
        'back': results.get('back', {}),
        'best_view': best_view,
        'best_mpjpe': min(filter(None, [side_mpjpe, back_mpjpe])) if any([side_mpjpe, back_mpjpe]) else None
    }


def analyze_single_view_correlations(sessions: List[Path], pose_model: str = 'yolov8m-pose.pt'):
    """
    Analyze single-view performance across all sessions.
    """
    print("\n" + "=" * 70)
    print("SINGLE-VIEW CORRELATION ANALYSIS")
    print("=" * 70)

    results = []

    for session_dir in sessions:
        print(f"\nProcessing {session_dir.name}...")

        try:
            result = run_both_views(session_dir, pose_model)

            side_res = result.get('side', {})
            back_res = result.get('back', {})

            if side_res.get('mpjpe_cm') or back_res.get('mpjpe_cm'):
                results.append({
                    'session': session_dir.name,
                    'side_mpjpe_cm': side_res.get('mpjpe_cm'),
                    'back_mpjpe_cm': back_res.get('mpjpe_cm'),
                    'best_view': result.get('best_view'),
                    'best_mpjpe_cm': result.get('best_mpjpe'),
                    'side_frames': side_res.get('n_frames', 0),
                    'back_frames': back_res.get('n_frames', 0),
                })
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue

    if len(results) < 3:
        print("\nNot enough sessions for analysis")
        return

    df = pd.DataFrame(results)

    print("\n" + "-" * 70)
    print("RESULTS SUMMARY")
    print("-" * 70)
    print(df.to_string(index=False))

    # Stats
    print("\n" + "-" * 70)
    print("AGGREGATE STATS")
    print("-" * 70)

    side_valid = df['side_mpjpe_cm'].dropna()
    back_valid = df['back_mpjpe_cm'].dropna()
    best_valid = df['best_mpjpe_cm'].dropna()

    if len(side_valid) > 0:
        print(f"Side view - Mean MPJPE: {side_valid.mean():.2f} cm (+/- {side_valid.std():.2f})")
    if len(back_valid) > 0:
        print(f"Back view - Mean MPJPE: {back_valid.mean():.2f} cm (+/- {back_valid.std():.2f})")
    if len(best_valid) > 0:
        print(f"Best view - Mean MPJPE: {best_valid.mean():.2f} cm (+/- {best_valid.std():.2f})")

    # View preference
    view_counts = df['best_view'].value_counts()
    print(f"\nBest view distribution:")
    for view, count in view_counts.items():
        print(f"  {view}: {count} sessions")

    return df


def main():
    """Run single-view pipeline."""
    import argparse

    parser = argparse.ArgumentParser(description="Single-view 3D pose estimation")
    parser.add_argument('--session', type=Path, default=None,
                        help='Process single session')
    parser.add_argument('--view', choices=['side', 'back', 'both'], default='both',
                        help='Which view to process')
    parser.add_argument('--all', action='store_true',
                        help='Process all sessions')
    parser.add_argument('--correlations', action='store_true',
                        help='Run correlation analysis')
    parser.add_argument('--model', default='yolov8m-pose.pt',
                        help='YOLOv8 pose model')
    args = parser.parse_args()

    training_data = Path(__file__).parent.parent / 'training_data'

    if args.session:
        if args.view == 'both':
            run_both_views(args.session, args.model)
        else:
            run_single_view_pipeline(args.session, args.view, args.model)
    elif args.all or args.correlations:
        sessions = sorted(training_data.glob('session_*'))
        analyze_single_view_correlations(sessions, args.model)
    else:
        # Default: process first session
        session = training_data / 'session_001'
        run_both_views(session, args.model)


if __name__ == '__main__':
    main()
