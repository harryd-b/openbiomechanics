"""
Learn calibration parameters from ground truth data.

This script:
1. Processes videos with plate calibration
2. Compares against UPLIFT ground truth
3. Learns the alignment transform (R, scale, t)
4. Saves the complete calibration config for production use
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from calibration import PlateCalibrator, CalibrationConfig, coco_to_h36m, JOINT_NAMES
from fusion.train_fusion import load_uplift_positions


def compute_mpjpe(pred: np.ndarray, gt: np.ndarray) -> float:
    """Compute MPJPE in cm."""
    valid = ~(np.isnan(pred).any(axis=(1, 2)) | np.isnan(gt).any(axis=(1, 2)))
    if not valid.any():
        return float('nan')
    errors = np.linalg.norm(pred[valid] - gt[valid], axis=2)
    return np.nanmean(errors) * 100


def main():
    print("=" * 70)
    print("CALIBRATION LEARNING")
    print("=" * 70)

    # Session with ground truth
    session_dir = Path(__file__).parent.parent / 'training_data' / 'session_005'
    videos = {
        'side': str(session_dir / 'side.mp4'),
        'back': str(session_dir / 'back.mp4'),
    }
    uplift_csv = session_dir / 'uplift.csv'

    # Load ground truth
    print("\nLoading ground truth...")
    gt_poses = load_uplift_positions(str(uplift_csv))
    print(f"  {len(gt_poses)} frames")

    # Initialize calibrator
    print("\nInitializing calibrator...")
    calibrator = PlateCalibrator()

    # Process videos to get raw triangulated poses
    print("\nProcessing videos...")
    results = calibrator.process_videos(videos, progress_interval=100)
    poses_raw = results['poses_raw']

    print(f"  Processed {len(poses_raw)} frames")

    # Align frame counts
    n = min(len(poses_raw), len(gt_poses))
    poses_raw = poses_raw[:n]
    gt_poses = gt_poses[:n]

    # Step 1: Test without alignment
    print("\n" + "-" * 70)
    print("STEP 1: Baseline (no alignment)")
    print("-" * 70)
    mpjpe_raw = compute_mpjpe(poses_raw, gt_poses)
    print(f"  MPJPE: {mpjpe_raw:.2f} cm")

    # Step 2: Learn alignment transform
    print("\n" + "-" * 70)
    print("STEP 2: Learn alignment transform")
    print("-" * 70)
    calibrator.learn_alignment(poses_raw, gt_poses)

    # Step 3: Apply alignment and test
    print("\n" + "-" * 70)
    print("STEP 3: Test with learned alignment")
    print("-" * 70)
    poses_aligned = calibrator.transform_to_uplift_frame(poses_raw, apply_bias_correction=False)
    mpjpe_aligned = compute_mpjpe(poses_aligned, gt_poses)
    print(f"  MPJPE (aligned, no bias): {mpjpe_aligned:.2f} cm")

    # Step 4: Apply bias correction
    poses_corrected = calibrator.transform_to_uplift_frame(poses_raw, apply_bias_correction=True)
    mpjpe_corrected = compute_mpjpe(poses_corrected, gt_poses)
    print(f"  MPJPE (aligned + bias): {mpjpe_corrected:.2f} cm")

    # Per-joint breakdown
    print("\n  Per-joint MPJPE:")
    print(f"  {'Joint':<12} {'Error (cm)':>10}")
    print(f"  {'-' * 24}")

    valid = ~(np.isnan(poses_corrected).any(axis=(1, 2)) |
              np.isnan(gt_poses).any(axis=(1, 2)))
    errors = np.linalg.norm(poses_corrected[valid] - gt_poses[valid], axis=2) * 100
    per_joint = np.nanmean(errors, axis=0)

    for j, name in enumerate(JOINT_NAMES):
        print(f"  {name:<12} {per_joint[j]:>10.1f}")

    # Save calibration config
    print("\n" + "-" * 70)
    print("STEP 4: Save calibration config")
    print("-" * 70)

    config_path = Path(__file__).parent / 'learned_config.json'
    calibrator.config.save(str(config_path))
    print(f"  Saved to: {config_path}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"""
Calibration learning complete:
  - Raw MPJPE:     {mpjpe_raw:.2f} cm
  - Aligned MPJPE: {mpjpe_aligned:.2f} cm
  - Final MPJPE:   {mpjpe_corrected:.2f} cm (with bias correction)

The learned config includes:
  - Camera scales: side={calibrator.config.camera_scales['side']}, back={calibrator.config.camera_scales['back']}
  - Alignment: R, scale={calibrator.config.alignment_scale:.4f}, t
  - Per-joint bias corrections: {len(calibrator.config.joint_biases_cm)} joints

To use in production:
  from calibration import PlateCalibrator, CalibrationConfig
  config = CalibrationConfig.load('{config_path}')
  calibrator = PlateCalibrator(config=config)
""")

    return calibrator.config


if __name__ == '__main__':
    main()
