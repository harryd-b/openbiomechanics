"""
Analyze correlation between pipeline output and UPLIFT ground truth.
Does NOT use UPLIFT as input - only for post-hoc comparison.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))

from calibration import JOINT_NAMES
from fusion.train_fusion import load_uplift_positions


def analyze_session(session_dir: Path):
    """Analyze correlation between pipeline output and UPLIFT."""

    print("=" * 70)
    print(f"CORRELATION ANALYSIS: {session_dir.name}")
    print("=" * 70)

    # Load pipeline output
    poses_csv = session_dir / 'poses_3d.csv'
    if not poses_csv.exists():
        print(f"Pipeline output not found: {poses_csv}")
        return

    df_pred = pd.read_csv(poses_csv)

    # Load UPLIFT ground truth
    uplift_csv = session_dir / 'uplift.csv'
    if not uplift_csv.exists():
        print(f"UPLIFT data not found: {uplift_csv}")
        return

    gt_poses = load_uplift_positions(str(uplift_csv))

    # Extract predicted poses
    n_frames = min(len(df_pred), len(gt_poses))
    pred_poses = np.zeros((n_frames, 17, 3))

    # Map columns to joints
    h36m_to_uplift = {
        0: 'pelvis_3d',
        1: 'right_hip_jc_3d',
        2: 'right_knee_jc_3d',
        3: 'right_ankle_jc_3d',
        4: 'left_hip_jc_3d',
        5: 'left_knee_jc_3d',
        6: 'left_ankle_jc_3d',
        7: 'spine_3d',
        8: 'thorax_3d',
        9: 'proximal_neck_3d',
        10: 'mid_head_3d',
        11: 'left_shoulder_jc_3d',
        12: 'left_elbow_jc_3d',
        13: 'left_wrist_jc_3d',
        14: 'right_shoulder_jc_3d',
        15: 'right_elbow_jc_3d',
        16: 'right_wrist_jc_3d',
    }

    for j, prefix in h36m_to_uplift.items():
        for k, axis in enumerate(['x', 'y', 'z']):
            col = f'{prefix}_{axis}'
            if col in df_pred.columns:
                pred_poses[:n_frames, j, k] = df_pred[col].values[:n_frames]

    gt_poses = gt_poses[:n_frames]

    print(f"\nFrames compared: {n_frames}")

    # Per-axis correlation
    print("\n" + "-" * 70)
    print("PER-AXIS CORRELATION (Pearson r)")
    print("-" * 70)
    print(f"{'Joint':<12} {'X':>8} {'Y':>8} {'Z':>8}")
    print("-" * 40)

    correlations = np.zeros((17, 3))
    for j, name in enumerate(JOINT_NAMES):
        for k, axis in enumerate(['X', 'Y', 'Z']):
            pred = pred_poses[:, j, k].flatten()
            gt = gt_poses[:, j, k].flatten()

            # Remove NaN
            valid = ~(np.isnan(pred) | np.isnan(gt))
            if valid.sum() > 10:
                r, _ = stats.pearsonr(pred[valid], gt[valid])
                correlations[j, k] = r
            else:
                correlations[j, k] = np.nan

        print(f"{name:<12} {correlations[j, 0]:>8.3f} {correlations[j, 1]:>8.3f} {correlations[j, 2]:>8.3f}")

    # Overall correlation
    print("-" * 40)
    mean_corr = np.nanmean(correlations, axis=0)
    print(f"{'Mean':<12} {mean_corr[0]:>8.3f} {mean_corr[1]:>8.3f} {mean_corr[2]:>8.3f}")

    # Motion tracking - does velocity correlate?
    print("\n" + "-" * 70)
    print("VELOCITY CORRELATION (motion tracking)")
    print("-" * 70)

    pred_vel = np.diff(pred_poses, axis=0)
    gt_vel = np.diff(gt_poses, axis=0)

    vel_correlations = np.zeros((17, 3))
    for j, name in enumerate(JOINT_NAMES):
        for k in range(3):
            pred_v = pred_vel[:, j, k].flatten()
            gt_v = gt_vel[:, j, k].flatten()
            valid = ~(np.isnan(pred_v) | np.isnan(gt_v))
            if valid.sum() > 10:
                r, _ = stats.pearsonr(pred_v[valid], gt_v[valid])
                vel_correlations[j, k] = r

    mean_vel_corr = np.nanmean(vel_correlations)
    print(f"Mean velocity correlation: {mean_vel_corr:.3f}")

    # Check for systematic offset
    print("\n" + "-" * 70)
    print("SYSTEMATIC OFFSET ANALYSIS")
    print("-" * 70)

    offset = np.nanmean(pred_poses - gt_poses, axis=0)
    print(f"\nMean offset per joint (pred - gt) in meters:")
    print(f"{'Joint':<12} {'X':>10} {'Y':>10} {'Z':>10}")
    print("-" * 44)

    for j, name in enumerate(JOINT_NAMES):
        print(f"{name:<12} {offset[j, 0]:>10.3f} {offset[j, 1]:>10.3f} {offset[j, 2]:>10.3f}")

    mean_offset = np.nanmean(offset, axis=0)
    print("-" * 44)
    print(f"{'Overall':<12} {mean_offset[0]:>10.3f} {mean_offset[1]:>10.3f} {mean_offset[2]:>10.3f}")

    # What if we correct the offset?
    print("\n" + "-" * 70)
    print("IF WE CORRECTED THE GLOBAL OFFSET:")
    print("-" * 70)

    corrected = pred_poses - mean_offset
    errors_corrected = np.linalg.norm(corrected - gt_poses, axis=2)
    mpjpe_corrected = np.nanmean(errors_corrected) * 100
    print(f"MPJPE after offset correction: {mpjpe_corrected:.2f} cm")

    # What about per-frame Procrustes?
    print("\n" + "-" * 70)
    print("PROCRUSTES ALIGNMENT (optimal, needs GT):")
    print("-" * 70)

    # Compute Procrustes
    pf = pred_poses.reshape(-1, 3)
    gf = gt_poses.reshape(-1, 3)
    valid = ~(np.isnan(pf).any(1) | np.isnan(gf).any(1))
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

    # Apply Procrustes
    aligned = np.zeros_like(pred_poses)
    for i in range(len(pred_poses)):
        for j in range(17):
            aligned[i, j] = scale * (R @ pred_poses[i, j]) + t

    errors_aligned = np.linalg.norm(aligned - gt_poses, axis=2)
    mpjpe_aligned = np.nanmean(errors_aligned) * 100
    print(f"MPJPE after Procrustes: {mpjpe_aligned:.2f} cm")
    print(f"Procrustes scale: {scale:.4f}")
    print(f"Procrustes rotation angle: {np.degrees(np.arccos((np.trace(R)-1)/2)):.1f}°")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"""
Position correlation: X={mean_corr[0]:.3f}, Y={mean_corr[1]:.3f}, Z={mean_corr[2]:.3f}
Velocity correlation: {mean_vel_corr:.3f}

MPJPE with learned session_005 alignment: 69.02 cm
MPJPE after global offset correction: {mpjpe_corrected:.2f} cm
MPJPE after Procrustes (optimal): {mpjpe_aligned:.2f} cm

Interpretation:
- High correlation = poses track correctly but in different coordinate frame
- Low correlation = fundamental mismatch in pose estimation
""")

    return {
        'position_corr': mean_corr,
        'velocity_corr': mean_vel_corr,
        'mpjpe_raw': 69.02,
        'mpjpe_offset_corrected': mpjpe_corrected,
        'mpjpe_procrustes': mpjpe_aligned,
    }


def main():
    session_dir = Path(__file__).parent.parent / 'training_data' / 'session_001'
    if len(sys.argv) > 1:
        session_dir = Path(sys.argv[1])

    analyze_session(session_dir)


if __name__ == '__main__':
    main()
