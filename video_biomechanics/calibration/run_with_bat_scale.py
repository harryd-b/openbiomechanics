"""
Run production pipeline with bat-derived scale calibration.

Uses the bat detector to estimate meters/pixel scale from bat length,
then runs the 3D pose pipeline and compares with UPLIFT ground truth.
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))

from calibration import PlateCalibrator, CalibrationConfig, JOINT_NAMES
from calibration.run_production import compute_mpjpe, align_frame_rates
from fusion.train_fusion import load_uplift_positions


@dataclass
class BatScaleResult:
    """Results from bat-based scale estimation."""
    scale_m_per_px: float
    confidence: float
    bat_length_px: float
    frame_idx: int
    video_name: str


def estimate_scale_from_bat(
    video_path: Path,
    model_path: Path,
    known_bat_length_m: float = 0.84
) -> Optional[BatScaleResult]:
    """
    Estimate scale using bat detector.

    Args:
        video_path: Path to video
        model_path: Path to bat detector model
        known_bat_length_m: Known bat length in meters (default 33" = 0.84m)

    Returns:
        BatScaleResult or None if detection failed
    """
    try:
        from ml.bat_detector import BatDetector
    except ImportError:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from ml.bat_detector import BatDetector

    if not model_path.exists():
        print(f"  WARNING: Bat model not found: {model_path}")
        return None

    detector = BatDetector(model_path)

    # Find best detection (prioritizes early/static frames)
    endpoints, confidence, frame_idx = detector.find_best_detection(
        video_path,
        num_samples=20,
        prefer_early_frames=True
    )

    if endpoints is None:
        print(f"  WARNING: No bat detected in {video_path.name}")
        return None

    bat_length_px = detector.compute_bat_length(endpoints)
    scale = known_bat_length_m / bat_length_px

    return BatScaleResult(
        scale_m_per_px=scale,
        confidence=confidence,
        bat_length_px=bat_length_px,
        frame_idx=frame_idx,
        video_name=video_path.name
    )


def triangulate_bat_endpoints(
    video_paths: Dict[str, Path],
    bat_model_path: Path,
    calibrator: 'PlateCalibrator'
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float]:
    """
    Triangulate bat endpoints from both camera views.

    Returns:
        knob_3d, tip_3d, confidence (or None, None, 0 if failed)
    """
    import cv2

    try:
        from ml.bat_detector import BatDetector
    except ImportError:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from ml.bat_detector import BatDetector

    detector = BatDetector(bat_model_path)

    # Get best frame from primary view
    primary_path = video_paths.get('side') or video_paths.get('primary')
    endpoints_primary, conf_primary, frame_idx = detector.find_best_detection(
        primary_path, num_samples=20, prefer_early_frames=True
    )

    if endpoints_primary is None:
        return None, None, 0.0

    # Detect in secondary view at same frame
    secondary_path = video_paths.get('back') or video_paths.get('secondary')
    cap = cv2.VideoCapture(str(secondary_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return None, None, 0.0

    endpoints_secondary, conf_secondary = detector.detect(frame)

    # Need both cameras for triangulation
    if conf_secondary < 0.5:
        return None, None, 0.0

    # Triangulate each endpoint using calibrator's cameras
    # Get projection matrices from calibrator
    if not hasattr(calibrator, 'cameras') or len(calibrator.cameras) < 2:
        return None, None, 0.0

    P_side = calibrator.cameras['side']['P']
    P_back = calibrator.cameras['back']['P']

    def triangulate_point(pt_side, pt_back):
        """DLT triangulation for a single point."""
        A = np.array([
            pt_side[0] * P_side[2] - P_side[0],
            pt_side[1] * P_side[2] - P_side[1],
            pt_back[0] * P_back[2] - P_back[0],
            pt_back[1] * P_back[2] - P_back[1],
        ])
        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1]
        return X[:3] / X[3]

    knob_3d = triangulate_point(endpoints_primary[0], endpoints_secondary[0])
    tip_3d = triangulate_point(endpoints_primary[1], endpoints_secondary[1])

    confidence = (conf_primary + conf_secondary) / 2

    return knob_3d, tip_3d, confidence


def run_with_bat_scale(
    session_dir: Path,
    bat_model_path: Path,
    known_bat_length_m: float = 0.84,
    output_dir: Optional[Path] = None,
    apply_scale_correction: bool = True
) -> Dict:
    """
    Run production pipeline using bat-derived scale correction.

    Args:
        session_dir: Session directory with videos
        bat_model_path: Path to trained bat detector
        known_bat_length_m: Known bat length in meters
        output_dir: Where to save results
        apply_scale_correction: Whether to apply bat-derived scale to poses

    Returns:
        Results dictionary with poses, scale info, and accuracy metrics
    """
    output_dir = output_dir or session_dir
    output_dir = Path(output_dir)

    print("=" * 70)
    print("BAT-CALIBRATED 3D POSE ESTIMATION")
    print("=" * 70)
    print(f"Session: {session_dir.name}")
    print(f"Known bat length: {known_bat_length_m:.3f} m ({known_bat_length_m/0.0254:.1f}\")")

    # Find videos
    if (session_dir / 'primary.mp4').exists():
        videos = {
            'side': session_dir / 'primary.mp4',
            'back': session_dir / 'secondary.mp4',
        }
    else:
        videos = {
            'side': session_dir / 'side.mp4',
            'back': session_dir / 'back.mp4',
        }

    # Estimate scale from each view using bat
    print("\n" + "-" * 70)
    print("STEP 1: Bat-based scale estimation")
    print("-" * 70)

    bat_results = {}
    for view_name, video_path in videos.items():
        print(f"\n  Processing {view_name} view...")
        result = estimate_scale_from_bat(
            video_path,
            bat_model_path,
            known_bat_length_m
        )
        if result:
            bat_results[view_name] = result
            print(f"    Bat length: {result.bat_length_px:.1f} px")
            print(f"    Scale: {result.scale_m_per_px*1000:.3f} mm/px")
            print(f"    Confidence: {result.confidence:.2f}")
            print(f"    Best frame: {result.frame_idx}")

    if not bat_results:
        print("  ERROR: No bat detected in any view")
        return {'error': 'No bat detected'}

    # Average scale across views (weighted by confidence)
    total_conf = sum(r.confidence for r in bat_results.values())
    avg_scale = sum(
        r.scale_m_per_px * r.confidence
        for r in bat_results.values()
    ) / total_conf

    print(f"\n  Combined 2D scale: {avg_scale*1000:.3f} mm/px")

    # Load or create calibration config
    print("\n" + "-" * 70)
    print("STEP 2: Initialize calibration")
    print("-" * 70)

    base_config_path = Path(__file__).parent / 'alignment_config.json'
    if base_config_path.exists():
        config = CalibrationConfig.load(str(base_config_path))
        print(f"  Base alignment scale: {config.alignment_scale:.4f}")
    else:
        config = CalibrationConfig()

    calibrator = PlateCalibrator(config=config)

    # Process videos first to get camera calibration
    print("\n" + "-" * 70)
    print("STEP 3: Process videos")
    print("-" * 70)

    video_paths = {k: str(v) for k, v in videos.items()}
    results = calibrator.process_videos(video_paths, progress_interval=50)

    poses_3d = results['poses_3d']
    fps = results['fps']

    print(f"\n  Frames: {len(poses_3d)}")
    print(f"  FPS: {fps:.2f}")

    # Triangulate bat to compute 3D scale correction
    scale_correction = 1.0
    bat_length_3d = None

    if apply_scale_correction and len(bat_results) >= 2:
        print("\n" + "-" * 70)
        print("STEP 4: Triangulate bat for 3D scale correction")
        print("-" * 70)

        knob_3d, tip_3d, bat_conf = triangulate_bat_endpoints(
            videos, bat_model_path, calibrator
        )

        if knob_3d is not None:
            bat_length_3d = float(np.linalg.norm(tip_3d - knob_3d))
            scale_correction = known_bat_length_m / bat_length_3d

            print(f"  Triangulated bat length: {bat_length_3d:.3f} m")
            print(f"  Expected bat length: {known_bat_length_m:.3f} m")
            print(f"  Scale correction factor: {scale_correction:.4f}")

            # Only apply scale correction if bat length is reasonable AND
            # the correction would shrink the poses (scale < 1.0)
            # Empirically: scale < 1 helps, scale > 1 hurts (triangulation is too large)
            # Valid range: 1.0x to 4.0x of expected (scale_correction 0.25 to 1.0)
            min_valid_length = known_bat_length_m * 1.0  # 0.84m
            max_valid_length = known_bat_length_m * 4.0  # 3.36m

            if min_valid_length <= bat_length_3d <= max_valid_length and scale_correction <= 1.0:
                poses_3d = poses_3d * scale_correction
                print(f"  Applied scale correction to {len(poses_3d)} frames")
            else:
                print(f"  WARNING: Bat length outside valid range ({min_valid_length:.2f}-{max_valid_length:.2f}m)")
                print(f"  Skipping scale correction (triangulation likely failed)")
                scale_correction = 1.0  # Reset to no correction
        else:
            print("  WARNING: Could not triangulate bat endpoints")

    # Export
    print("\n" + "-" * 70)
    print(f"STEP {'5' if apply_scale_correction else '4'}: Export results")
    print("-" * 70)

    output_csv = output_dir / 'poses_3d_bat_scaled.csv'
    calibrator.to_uplift_csv(
        poses_3d,
        str(output_csv),
        fps=fps,
        metadata={
            'scale_method': 'bat_triangulation' if bat_length_3d else 'bat_2d',
            'bat_scale_mm_per_px': avg_scale * 1000,
            'bat_length_3d_m': bat_length_3d,
            'scale_correction': scale_correction,
        }
    )
    print(f"  Saved: {output_csv}")

    # Compare with ground truth
    result_dict = {
        'bat_results': {k: vars(v) for k, v in bat_results.items()},
        'avg_scale_m_per_px': avg_scale,
        'bat_length_3d_m': bat_length_3d,
        'scale_correction': scale_correction,
        'poses_3d': poses_3d,
        'fps': fps,
        'output_csv': str(output_csv),
    }

    uplift_csv = session_dir / 'uplift.csv'
    if uplift_csv.exists():
        step_num = 6 if (apply_scale_correction and bat_length_3d) else 5
        print("\n" + "-" * 70)
        print(f"STEP {step_num}: Compare with UPLIFT ground truth")
        print("-" * 70)

        gt_poses = load_uplift_positions(str(uplift_csv))

        # Align frame rates
        gt_fps = 240.0
        poses_resampled = align_frame_rates(poses_3d, fps, gt_fps)

        n_compare = min(len(poses_resampled), len(gt_poses))
        mpjpe, errors = compute_mpjpe(poses_resampled[:n_compare], gt_poses[:n_compare])

        result_dict['mpjpe_cm'] = float(mpjpe) if not np.isnan(mpjpe) else None
        result_dict['n_frames_compared'] = n_compare

        if not np.isnan(mpjpe):
            print(f"  MPJPE: {mpjpe:.2f} cm")

            # Per-joint breakdown
            per_joint = np.nanmean(errors, axis=0) * 100
            result_dict['per_joint_mpjpe'] = {
                JOINT_NAMES[j]: float(per_joint[j])
                for j in range(len(JOINT_NAMES))
            }

    return result_dict


def analyze_bat_uplift_correlations(sessions: List[Path], bat_model_path: Path):
    """
    Analyze correlations between bat-derived metrics and UPLIFT accuracy.

    This helps understand:
    1. Does bat confidence predict pose accuracy?
    2. Does bat-derived scale improve accuracy?
    3. Are there systematic patterns?
    """
    print("\n" + "=" * 70)
    print("BAT-UPLIFT CORRELATION ANALYSIS")
    print("=" * 70)

    results = []

    for session_dir in sessions:
        print(f"\nProcessing {session_dir.name}...")

        try:
            result = run_with_bat_scale(session_dir, bat_model_path, apply_scale_correction=True)

            if 'error' not in result and result.get('mpjpe_cm'):
                results.append({
                    'session': session_dir.name,
                    'bat_confidence': result['bat_results'].get('side', {}).get('confidence', 0),
                    'bat_scale_mm_px': result['avg_scale_m_per_px'] * 1000,
                    'bat_length_3d': result.get('bat_length_3d_m'),
                    'scale_correction': result.get('scale_correction', 1.0),
                    'mpjpe_cm': result['mpjpe_cm'],
                })
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue

    if len(results) < 3:
        print("\nNot enough sessions for correlation analysis")
        return

    df = pd.DataFrame(results)

    print("\n" + "-" * 70)
    print("CORRELATION RESULTS")
    print("-" * 70)

    # Bat confidence vs MPJPE
    r_conf, p_conf = stats.pearsonr(df['bat_confidence'], df['mpjpe_cm'])
    print(f"\nBat confidence vs MPJPE:")
    print(f"  Pearson r = {r_conf:.3f} (p = {p_conf:.3f})")
    print(f"  Interpretation: {'Higher confidence = better accuracy' if r_conf < 0 else 'No clear relationship'}")

    # Bat scale vs MPJPE
    r_scale, p_scale = stats.pearsonr(df['bat_scale_mm_px'], df['mpjpe_cm'])
    print(f"\nBat 2D scale vs MPJPE:")
    print(f"  Pearson r = {r_scale:.3f} (p = {p_scale:.3f})")

    # Scale correction vs MPJPE
    if 'scale_correction' in df.columns and df['scale_correction'].notna().sum() >= 3:
        valid = df['scale_correction'].notna()
        r_corr, p_corr = stats.pearsonr(df.loc[valid, 'scale_correction'], df.loc[valid, 'mpjpe_cm'])
        print(f"\nScale correction factor vs MPJPE:")
        print(f"  Pearson r = {r_corr:.3f} (p = {p_corr:.3f})")
        print(f"  Interpretation: {'Larger corrections = worse accuracy' if r_corr > 0 else 'Scale correction helps'}")

    # Summary table
    print("\n" + "-" * 70)
    print("SESSION SUMMARY")
    print("-" * 70)
    print(df.to_string(index=False))

    # Stats
    print("\n" + "-" * 70)
    print("AGGREGATE STATS")
    print("-" * 70)
    print(f"Mean MPJPE: {df['mpjpe_cm'].mean():.2f} cm (+/- {df['mpjpe_cm'].std():.2f})")
    print(f"Mean bat confidence: {df['bat_confidence'].mean():.2f}")
    print(f"Mean scale: {df['bat_scale_mm_px'].mean():.3f} mm/px")

    return df


def main():
    """Run bat-calibrated pipeline on all sessions."""
    import argparse

    parser = argparse.ArgumentParser(description="Bat-calibrated 3D pose pipeline")
    parser.add_argument('--session', type=Path, default=None,
                        help='Process single session')
    parser.add_argument('--all', action='store_true',
                        help='Process all sessions')
    parser.add_argument('--correlations', action='store_true',
                        help='Run correlation analysis')
    parser.add_argument('--bat-length', type=float, default=0.84,
                        help='Known bat length in meters (default: 0.84m = 33")')
    args = parser.parse_args()

    # Find bat model
    bat_model_path = Path(__file__).parent.parent / 'ml' / 'bat_detector' / 'checkpoints' / 'best_bat_model.pth'
    if not bat_model_path.exists():
        print(f"ERROR: Bat model not found: {bat_model_path}")
        print("Train the bat detector first: cd ml/bat_detector && python train.py")
        return

    training_data = Path(__file__).parent.parent / 'training_data'

    if args.session:
        run_with_bat_scale(args.session, bat_model_path, args.bat_length)
    elif args.all or args.correlations:
        sessions = sorted(training_data.glob('session_*'))

        if args.correlations:
            analyze_bat_uplift_correlations(sessions, bat_model_path)
        else:
            for session in sessions:
                try:
                    run_with_bat_scale(session, bat_model_path, args.bat_length)
                except Exception as e:
                    print(f"ERROR processing {session}: {e}")
    else:
        # Default: process first session
        session = training_data / 'session_001'
        run_with_bat_scale(session, bat_model_path, args.bat_length)


if __name__ == '__main__':
    main()
