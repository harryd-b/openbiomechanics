"""Debug camera calibration for a session."""

import sys
import cv2
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from calibration import PlateCalibrator


def debug_session(session_dir: Path):
    """Debug camera calibration for a session."""
    print("=" * 70)
    print(f"CALIBRATION DEBUG: {session_dir.name}")
    print("=" * 70)

    calibrator = PlateCalibrator()

    videos = {
        'side': str(session_dir / 'side.mp4'),
        'back': str(session_dir / 'back.mp4'),
    }

    # Try different calibration frames
    for frame_idx in [30, 60, 90, 120]:
        print(f"\n--- Calibration frame {frame_idx} ---")

        calibrator.cameras = {}
        success = calibrator.calibrate_from_videos(videos, calibration_frame=frame_idx)

        if success:
            for view, cam in calibrator.cameras.items():
                pos = cam['position']
                n_corners = cam['n_corners']
                print(f"  {view}: pos={pos}, corners={n_corners}")

            # Compute baseline (distance between cameras)
            if 'side' in calibrator.cameras and 'back' in calibrator.cameras:
                p1 = calibrator.cameras['side']['position']
                p2 = calibrator.cameras['back']['position']
                baseline = np.linalg.norm(p1 - p2)
                print(f"  Baseline: {baseline:.3f} m")
        else:
            print("  Calibration failed")

    # Visualize plate detection
    print("\n--- Plate Detection Visualization ---")
    for view in ['side', 'back']:
        cap = cv2.VideoCapture(videos[view])
        cap.set(cv2.CAP_PROP_POS_FRAMES, 60)
        ret, frame = cap.read()
        cap.release()

        if ret:
            corners = calibrator.detect_plate_corners(frame)
            if corners is not None:
                print(f"  {view}: Detected {len(corners)} corners")
                print(f"    Corners: {corners.tolist()}")
            else:
                print(f"  {view}: No plate detected")


def main():
    session_dir = Path(__file__).parent.parent / 'training_data' / 'session_001'
    if len(sys.argv) > 1:
        session_dir = Path(sys.argv[1])

    debug_session(session_dir)

    # Also debug session_005 for comparison
    session_005 = Path(__file__).parent.parent / 'training_data' / 'session_005'
    if session_005.exists():
        print("\n")
        debug_session(session_005)


if __name__ == '__main__':
    main()
