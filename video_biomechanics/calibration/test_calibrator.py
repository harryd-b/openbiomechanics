"""
Test script for the plate calibrator module.
Validates accuracy against UPLIFT ground truth.
"""

import sys
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from calibration import PlateCalibrator, CalibrationConfig, coco_to_h36m, JOINT_NAMES
from fusion.train_fusion import load_uplift_positions


def procrustes_align(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Align predictions to ground truth using Procrustes."""
    n = min(len(pred), len(gt))
    pred, gt = pred[:n], gt[:n]

    pf = pred.reshape(-1, 3)
    gf = gt.reshape(-1, 3)
    valid = ~(np.isnan(pf).any(1) | np.isinf(pf).any(1))
    pv, gv = pf[valid], gf[valid]

    pc, gc = pv.mean(0), gv.mean(0)
    ps = np.sqrt(((pv - pc)**2).sum() / len(pv))
    gs = np.sqrt(((gv - gc)**2).sum() / len(gv))
    scale = gs / ps

    pn, gn = (pv - pc) / ps, (gv - gc) / gs
    U, _, Vt = np.linalg.svd(pn.T @ gn)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    t = gc - scale * (R @ pc)

    aligned = np.array([scale * (R @ p) + t for p in pred.reshape(-1, 3)]).reshape(n, 17, 3)
    return aligned


def compute_mpjpe(pred: np.ndarray, gt: np.ndarray) -> tuple:
    """Compute MPJPE between predictions and ground truth."""
    n = min(len(pred), len(gt))
    errors = np.linalg.norm(pred[:n] - gt[:n], axis=2)
    return np.mean(errors) * 100, errors  # cm


def main():
    print("=" * 70)
    print("PLATE CALIBRATOR TEST")
    print("=" * 70)

    # Paths
    session_dir = Path(__file__).parent.parent / 'training_data' / 'session_005'
    videos = {
        'side': str(session_dir / 'side.mp4'),
        'back': str(session_dir / 'back.mp4'),
    }
    uplift_csv = str(session_dir / 'uplift.csv')

    # Load ground truth
    print("\nLoading ground truth...")
    gt_poses = load_uplift_positions(uplift_csv)
    print(f"  {gt_poses.shape[0]} frames")

    # Initialize calibrator
    print("\nInitializing plate calibrator...")
    calibrator = PlateCalibrator()

    # Calibrate cameras
    print("\nCalibrating cameras from plate...")
    success = calibrator.calibrate_from_videos(videos)

    if not success:
        print("ERROR: Calibration failed")
        return

    # Load YOLO for 2D pose detection
    print("\nLoading YOLO model...")
    model = YOLO('yolov8m-pose.pt')

    # Process videos
    print("\nTriangulating poses...")
    poses_3d = []

    cap_side = cv2.VideoCapture(videos['side'])
    cap_back = cv2.VideoCapture(videos['back'])
    n_frames = int(cap_side.get(cv2.CAP_PROP_FRAME_COUNT))

    for i in range(n_frames):
        ret1, frame1 = cap_side.read()
        ret2, frame2 = cap_back.read()

        if not ret1 or not ret2:
            break

        # Detect 2D poses
        results1 = model(frame1, verbose=False)
        results2 = model(frame2, verbose=False)

        if (len(results1) == 0 or len(results1[0].keypoints) == 0 or
            len(results2) == 0 or len(results2[0].keypoints) == 0):
            poses_3d.append(np.zeros((17, 3)))
            continue

        kp1 = results1[0].keypoints[0].xy[0].cpu().numpy()
        kp2 = results2[0].keypoints[0].xy[0].cpu().numpy()

        # Convert to H36M format
        h36m_1 = coco_to_h36m(kp1)
        h36m_2 = coco_to_h36m(kp2)

        # Triangulate
        pose_3d = calibrator.triangulate_pose(h36m_1, h36m_2)
        poses_3d.append(pose_3d)

        if i % 100 == 0:
            print(f"  Frame {i}/{n_frames}")

    cap_side.release()
    cap_back.release()

    poses_3d = np.array(poses_3d)
    print(f"  Triangulated {len(poses_3d)} frames")

    # Evaluate accuracy
    print("\n" + "=" * 70)
    print("ACCURACY EVALUATION")
    print("=" * 70)

    # =========================================================================
    # Method 1: Procrustes alignment (requires ground truth - for validation)
    # =========================================================================
    print("\n--- Method 1: Procrustes Alignment (requires ground truth) ---")

    # Step 1: Procrustes align to ground truth
    poses_aligned = procrustes_align(poses_3d, gt_poses)

    # Without bias correction
    mpjpe_raw, errors_raw = compute_mpjpe(poses_aligned, gt_poses)
    print(f"MPJPE (aligned, no bias correction): {mpjpe_raw:.2f} cm")

    # Step 2: Apply bias correction AFTER alignment
    # (biases were learned in aligned coordinate space)
    poses_corrected = calibrator.apply_joint_bias_correction(poses_aligned)
    mpjpe_corrected, errors_corrected = compute_mpjpe(poses_corrected, gt_poses)
    print(f"MPJPE (aligned + bias correction): {mpjpe_corrected:.2f} cm")

    improvement = mpjpe_raw - mpjpe_corrected
    print(f"Improvement from bias correction: {improvement:.2f} cm ({improvement/mpjpe_raw*100:.1f}%)")

    # =========================================================================
    # Method 2: Direct UPLIFT frame transform (NO ground truth needed)
    # =========================================================================
    print("\n--- Method 2: Direct UPLIFT Transform (production mode) ---")

    # Transform raw triangulated poses directly to UPLIFT frame
    poses_uplift = calibrator.transform_to_uplift_frame(poses_3d, apply_bias_correction=False)
    mpjpe_direct_raw, _ = compute_mpjpe(poses_uplift, gt_poses)
    print(f"MPJPE (direct transform, no bias): {mpjpe_direct_raw:.2f} cm")

    # With bias correction
    poses_uplift_corrected = calibrator.transform_to_uplift_frame(poses_3d, apply_bias_correction=True)
    mpjpe_direct_corrected, errors_direct = compute_mpjpe(poses_uplift_corrected, gt_poses)
    print(f"MPJPE (direct transform + bias): {mpjpe_direct_corrected:.2f} cm")

    # Compare methods
    print(f"\nComparison:")
    print(f"  Procrustes + bias:      {mpjpe_corrected:.2f} cm (optimal, needs ground truth)")
    print(f"  Direct transform + bias: {mpjpe_direct_corrected:.2f} cm (production, no ground truth)")
    diff = mpjpe_direct_corrected - mpjpe_corrected
    print(f"  Difference: {diff:.2f} cm")

    # Per-joint breakdown (using production method)
    print("\nPer-joint MPJPE (production mode with bias correction):")
    print(f"{'Joint':<12} {'Procrustes':>12} {'Production':>12}")
    print("-" * 38)

    per_joint_proc = np.mean(errors_corrected, axis=0) * 100
    per_joint_prod = np.mean(errors_direct, axis=0) * 100
    for j, name in enumerate(JOINT_NAMES):
        print(f"{name:<12} {per_joint_proc[j]:>12.1f} {per_joint_prod[j]:>12.1f}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"""
Plate-based calibration results:

  VALIDATION (with Procrustes alignment to ground truth):
    - MPJPE: {mpjpe_corrected:.2f} cm
    - Best joint:  {JOINT_NAMES[np.argmin(per_joint_proc)]} ({np.min(per_joint_proc):.1f} cm)
    - Worst joint: {JOINT_NAMES[np.argmax(per_joint_proc)]} ({np.max(per_joint_proc):.1f} cm)

  PRODUCTION (direct UPLIFT transform, no ground truth needed):
    - MPJPE: {mpjpe_direct_corrected:.2f} cm
    - Best joint:  {JOINT_NAMES[np.argmin(per_joint_prod)]} ({np.min(per_joint_prod):.1f} cm)
    - Worst joint: {JOINT_NAMES[np.argmax(per_joint_prod)]} ({np.max(per_joint_prod):.1f} cm)

  Configuration:
    - Camera scales: side={calibrator.config.camera_scales['side']}, back={calibrator.config.camera_scales['back']}
    - Camera FOV: {calibrator.config.camera_fov}°
    - Coordinate transform: Z-flip (plate frame -> UPLIFT frame)
    - Per-joint bias correction: enabled
""")

    # Save configuration
    config_path = Path(__file__).parent / 'session_005_config.json'
    calibrator.config.save(str(config_path))
    print(f"Configuration saved to: {config_path}")


if __name__ == '__main__':
    main()
