"""
Production pipeline for plate-based 3D pose estimation.

Processes video pairs using plate calibration, exports UPLIFT-compatible CSV,
and compares accuracy against ground truth if available.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from calibration import PlateCalibrator, CalibrationConfig, JOINT_NAMES
from fusion.train_fusion import load_uplift_positions


def compute_mpjpe(pred: np.ndarray, gt: np.ndarray) -> tuple:
    """Compute MPJPE between predictions and ground truth."""
    # Handle NaN values
    valid = ~(np.isnan(pred).any(axis=(1, 2)) | np.isnan(gt).any(axis=(1, 2)))
    pred_valid = pred[valid]
    gt_valid = gt[valid]

    if len(pred_valid) == 0:
        return np.nan, np.array([])

    errors = np.linalg.norm(pred_valid - gt_valid, axis=2)
    return np.nanmean(errors) * 100, errors  # cm


def align_frame_rates(poses: np.ndarray, source_fps: float, target_fps: float) -> np.ndarray:
    """Resample poses to match target frame rate."""
    if abs(source_fps - target_fps) < 0.1:
        return poses

    n_source = len(poses)
    n_target = int(n_source * target_fps / source_fps)

    # Simple nearest-neighbor resampling
    indices = np.linspace(0, n_source - 1, n_target).astype(int)
    return poses[indices]


def run_production_pipeline(session_dir: Path, output_dir: Path = None):
    """
    Run the full production pipeline on a session.

    Args:
        session_dir: Directory containing side.mp4, back.mp4, and optionally uplift.csv
        output_dir: Where to save outputs (defaults to session_dir)
    """
    output_dir = output_dir or session_dir
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PLATE-BASED 3D POSE ESTIMATION - PRODUCTION PIPELINE")
    print("=" * 70)
    print(f"Session: {session_dir}")
    print(f"Output:  {output_dir}")
    print(f"Time:    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Video paths - try primary/secondary first, fallback to side/back
    if (session_dir / 'primary.mp4').exists():
        videos = {
            'side': str(session_dir / 'primary.mp4'),
            'back': str(session_dir / 'secondary.mp4'),
        }
    else:
        videos = {
            'side': str(session_dir / 'side.mp4'),
            'back': str(session_dir / 'back.mp4'),
        }

    # Check videos exist
    for name, path in videos.items():
        if not Path(path).exists():
            raise FileNotFoundError(f"Video not found: {path}")

    # Initialize calibrator with alignment config
    print("\n" + "-" * 70)
    print("STEP 1: Initialize calibrator")
    print("-" * 70)

    # Load alignment config (computed from session_005 with UPLIFT ground truth)
    alignment_config_path = Path(__file__).parent / 'alignment_config.json'
    if alignment_config_path.exists():
        print(f"  Loading alignment config from: {alignment_config_path}")
        config = CalibrationConfig.load(str(alignment_config_path))
    else:
        print("  WARNING: No alignment config found, using defaults")
        config = CalibrationConfig()
        config.camera_scales = {'side': 1.0, 'back': 1.0}
        config.joint_biases_cm = {}

    calibrator = PlateCalibrator(config=config)

    print(f"  Camera FOV: {calibrator.config.camera_fov}°")
    print(f"  Alignment scale: {calibrator.config.alignment_scale:.4f}")
    print(f"  Using BattersBoxCalibrator for robust plate detection")

    # Process videos
    print("\n" + "-" * 70)
    print("STEP 2: Process videos")
    print("-" * 70)

    results = calibrator.process_videos(videos, progress_interval=50)

    poses_3d = results['poses_3d']
    fps = results['fps']
    n_frames = results['n_frames']
    n_failed = len(results['failed_frames'])

    print(f"\n  Results:")
    print(f"    Frames processed: {n_frames}")
    print(f"    Failed detections: {n_failed} ({100*n_failed/n_frames:.1f}%)")
    print(f"    Video FPS: {fps:.2f}")

    # Export to UPLIFT CSV
    print("\n" + "-" * 70)
    print("STEP 3: Export to UPLIFT-compatible CSV")
    print("-" * 70)

    output_csv = output_dir / 'poses_3d.csv'
    calibrator.to_uplift_csv(
        poses_3d,
        str(output_csv),
        fps=fps,
        metadata={
            'activity': 'baseball',
            'movement': 'hitting',
            'sessionid': session_dir.name,
        }
    )

    # Compare with ground truth if available
    uplift_csv = session_dir / 'uplift.csv'
    if uplift_csv.exists():
        print("\n" + "-" * 70)
        print("STEP 4: Compare with UPLIFT ground truth")
        print("-" * 70)

        # Load ground truth
        gt_poses = load_uplift_positions(str(uplift_csv))
        print(f"  Ground truth frames: {len(gt_poses)}")

        # Resample to match frame rates (UPLIFT is typically 240 FPS)
        gt_fps = 240.0  # UPLIFT default
        poses_resampled = align_frame_rates(poses_3d, fps, gt_fps)

        # Trim to same length
        n_compare = min(len(poses_resampled), len(gt_poses))
        poses_compare = poses_resampled[:n_compare]
        gt_compare = gt_poses[:n_compare]

        print(f"  Comparing {n_compare} aligned frames")

        # Compute MPJPE
        mpjpe, errors = compute_mpjpe(poses_compare, gt_compare)

        if not np.isnan(mpjpe):
            print(f"\n  MPJPE: {mpjpe:.2f} cm")

            # Per-joint breakdown
            print(f"\n  Per-joint MPJPE:")
            print(f"  {'Joint':<12} {'Error (cm)':>10}")
            print(f"  {'-' * 24}")

            per_joint = np.nanmean(errors, axis=0) * 100
            for j, name in enumerate(JOINT_NAMES):
                print(f"  {name:<12} {per_joint[j]:>10.1f}")

            # Best/worst
            best_idx = np.nanargmin(per_joint)
            worst_idx = np.nanargmax(per_joint)
            print(f"\n  Best:  {JOINT_NAMES[best_idx]} ({per_joint[best_idx]:.1f} cm)")
            print(f"  Worst: {JOINT_NAMES[worst_idx]} ({per_joint[worst_idx]:.1f} cm)")
        else:
            print("  WARNING: Could not compute MPJPE (no valid frames)")
    else:
        print("\n  No ground truth available for comparison")

    # Save calibration config
    config_path = output_dir / 'calibration_config.json'
    calibrator.config.save(str(config_path))
    print(f"\n  Calibration config saved to: {config_path}")

    # Summary
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"""
Outputs:
  - 3D poses CSV: {output_csv}
  - Calibration config: {config_path}

To use these poses in downstream analysis:
  import pandas as pd
  df = pd.read_csv('{output_csv}')
  # Columns: frame, time, fps, <joint>_3d_x/y/z for each joint
""")

    return {
        'poses_3d': poses_3d,
        'fps': fps,
        'output_csv': str(output_csv),
        'config_path': str(config_path),
    }


def main():
    """Run on session_005 by default."""
    session_dir = Path(__file__).parent.parent / 'training_data' / 'session_005'

    if not session_dir.exists():
        print(f"Session directory not found: {session_dir}")
        print("Usage: python run_production.py [session_dir]")
        return

    # Allow command-line override
    if len(sys.argv) > 1:
        session_dir = Path(sys.argv[1])

    run_production_pipeline(session_dir)


if __name__ == '__main__':
    main()
