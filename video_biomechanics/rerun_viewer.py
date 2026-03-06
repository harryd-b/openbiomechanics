"""
Rerun-based skeleton visualization for biomechanics analysis.

Uses Rerun's native skeleton support for smooth, real-time visualization.
"""

import numpy as np
from typing import List, Optional, Dict
from pathlib import Path

import rerun as rr
import rerun.blueprint as rrb


# H36M skeleton connections (parent -> child)
H36M_SKELETON = [
    # Spine
    (0, 7),   # Hip Center -> Spine
    (7, 8),   # Spine -> Neck
    (8, 9),   # Neck -> Head
    (9, 10),  # Head -> Head Top
    # Left arm
    (8, 11),  # Neck -> Left Shoulder
    (11, 12), # Left Shoulder -> Left Elbow
    (12, 13), # Left Elbow -> Left Wrist
    # Right arm
    (8, 14),  # Neck -> Right Shoulder
    (14, 15), # Right Shoulder -> Right Elbow
    (15, 16), # Right Elbow -> Right Wrist
    # Left leg
    (0, 4),   # Hip Center -> Left Hip
    (4, 5),   # Left Hip -> Left Knee
    (5, 6),   # Left Knee -> Left Ankle
    # Right leg
    (0, 1),   # Hip Center -> Right Hip
    (1, 2),   # Right Hip -> Right Knee
    (2, 3),   # Right Knee -> Right Ankle
]

JOINT_NAMES = [
    'Hip Center', 'Right Hip', 'Right Knee', 'Right Ankle',
    'Left Hip', 'Left Knee', 'Left Ankle',
    'Spine', 'Neck', 'Head', 'Head Top',
    'Left Shoulder', 'Left Elbow', 'Left Wrist',
    'Right Shoulder', 'Right Elbow', 'Right Wrist'
]


def fix_skeleton_coordinates(pose: np.ndarray) -> np.ndarray:
    """
    Fix skeleton coordinates - handle outliers and coordinate system.

    Args:
        pose: Array of shape (17, 3) with joint positions

    Returns:
        Fixed pose array
    """
    pose = pose.copy()

    # Fix outlier joints (pixel coords instead of meters)
    max_reasonable = 5.0  # meters

    # First pass: identify valid joints
    valid_mask = np.all(np.abs(pose) < max_reasonable, axis=1)

    # If we have valid joints, use them to estimate bad ones
    if np.any(valid_mask):
        # Get center of valid joints
        valid_center = pose[valid_mask].mean(axis=0)

        # Fix invalid joints
        for j in range(len(pose)):
            if not valid_mask[j]:
                # Estimate from nearby valid joints
                if j == 9 or j == 10:  # HEAD or HEAD_TOP
                    # Estimate from neck (8) if valid
                    if valid_mask[8]:
                        pose[j] = pose[8] + np.array([0, 0.15, 0])
                    else:
                        pose[j] = valid_center + np.array([0, 0.5, 0])
                else:
                    pose[j] = valid_center

    # Swap Y and Z for proper orientation (Y up)
    # Original: X=lateral, Y=down (image), Z=depth
    # Target: X=lateral, Y=up (height), Z=depth
    fixed = np.zeros_like(pose)
    fixed[:, 0] = pose[:, 0]       # X stays
    fixed[:, 1] = -pose[:, 1]      # Y = -old_Y (flip vertical)
    fixed[:, 2] = pose[:, 2]       # Z stays

    # Center on ground
    fixed[:, 1] = fixed[:, 1] - fixed[:, 1].min() + 0.02

    return fixed


def log_skeleton_sequence(
    poses_3d: List[np.ndarray],
    fps: float = 30.0,
    timeseries_df=None,
    video_path: Optional[str] = None
):
    """
    Log a sequence of skeletons to Rerun.

    Args:
        poses_3d: List of (17, 3) arrays with joint positions per frame
        fps: Frames per second
        timeseries_df: Optional DataFrame with joint angles for graphs
        video_path: Optional path to video file to show alongside
    """
    # Initialize Rerun and start gRPC server
    rr.init("biomechanics_viewer")

    # Start gRPC server and web viewer
    server_uri = rr.serve_grpc()
    print(f"Rerun gRPC server at: {server_uri}")
    rr.serve_web_viewer(open_browser=True, connect_to=server_uri)
    print("Web viewer opened in browser")

    # Log video if available
    if video_path and Path(video_path).exists():
        rr.log("video", rr.AssetVideo(path=video_path))

    # Log each frame
    for frame_idx, pose in enumerate(poses_3d):
        if pose is None:
            continue

        # Set time
        time_sec = frame_idx / fps
        rr.set_time("time", duration=time_sec)
        rr.set_time("frame", sequence=frame_idx)

        # Fix coordinates
        fixed_pose = fix_skeleton_coordinates(pose)

        # Log skeleton as points and lines
        # Points (joints)
        rr.log(
            "skeleton/joints",
            rr.Points3D(
                fixed_pose,
                colors=[[255, 100, 100]] * len(fixed_pose),
                radii=[0.02] * len(fixed_pose),
                labels=JOINT_NAMES
            )
        )

        # Lines (bones)
        lines = []
        for start, end in H36M_SKELETON:
            if start < len(fixed_pose) and end < len(fixed_pose):
                lines.append([fixed_pose[start], fixed_pose[end]])

        if lines:
            rr.log(
                "skeleton/bones",
                rr.LineStrips3D(
                    lines,
                    colors=[[255, 68, 68]] * len(lines),
                    radii=[0.01] * len(lines)
                )
            )

        # Log metrics if available
        if timeseries_df is not None and frame_idx < len(timeseries_df):
            row = timeseries_df.iloc[frame_idx]

            if 'pelvis_rotation_velocity' in row:
                rr.log("metrics/pelvis_velocity", rr.Scalars(row['pelvis_rotation_velocity']))
            if 'torso_rotation_velocity' in row:
                rr.log("metrics/trunk_velocity", rr.Scalars(row['torso_rotation_velocity']))
            if 'right_arm_rotation_velocity' in row:
                rr.log("metrics/arm_velocity", rr.Scalars(row['right_arm_rotation_velocity']))

    print(f"Logged {len(poses_3d)} frames to Rerun viewer")


def launch_viewer(results: Dict, video_path: Optional[str] = None):
    """
    Launch the Rerun viewer with biomechanics data.

    Args:
        results: Results dictionary from processing pipeline
        video_path: Optional path to the video file
    """
    poses_3d = results.get('poses_3d', [])
    if not poses_3d:
        print("No 3D poses to visualize")
        return

    fps = results.get('fps', 30.0)
    timeseries_df = results.get('timeseries_df', None)

    log_skeleton_sequence(
        poses_3d=poses_3d,
        fps=fps,
        timeseries_df=timeseries_df,
        video_path=video_path
    )


if __name__ == "__main__":
    # Test with saved data
    import pandas as pd

    output_dir = Path(__file__).parent / "output"

    # Load poses
    poses_path = output_dir / "poses_3d.npy"
    if poses_path.exists():
        poses_3d = np.load(poses_path)
        poses_3d = [poses_3d[i] for i in range(len(poses_3d))]

        # Load timeseries
        ts_path = output_dir / "timeseries.csv"
        timeseries_df = pd.read_csv(ts_path) if ts_path.exists() else None

        log_skeleton_sequence(poses_3d, fps=30.0, timeseries_df=timeseries_df)
    else:
        print(f"No poses found at {poses_path}")
