"""
Training script for learned fusion network.

Uses UPLIFT data as ground truth to learn optimal fusion weights.

Coordinate System Strategy:
- Single-view methods (YOLO+Lifting, MotionBERT): Normalized, pelvis-centered
- Triangulation: Absolute world coordinates (anchor for position)
- UPLIFT ground truth: Absolute world coordinates (Y-up, meters)
- Fusion output: Absolute world coordinates matching UPLIFT

The fusion model learns to:
1. Use triangulation to anchor the pose in absolute world space
2. Combine YOLO+MotionBERT normalized poses for joint detail/smoothness
3. Output absolute world coordinates that match UPLIFT format

This requires 2+ camera views for triangulation to work.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional, Tuple
import json

try:
    import torch
    from torch.utils.data import Dataset, DataLoader, random_split
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .learned_fusion import FusionNetwork, FusionTrainer


# H36M joint names in order
H36M_JOINT_NAMES = [
    'pelvis', 'right_hip', 'right_knee', 'right_ankle',
    'left_hip', 'left_knee', 'left_ankle',
    'spine', 'neck', 'head', 'head_top',
    'left_shoulder', 'left_elbow', 'left_wrist',
    'right_shoulder', 'right_elbow', 'right_wrist'
]

# UPLIFT column name mapping to H36M
UPLIFT_TO_H36M = {
    # Core body
    'pelvis_3d': 'pelvis',
    'proximal_neck_3d': 'neck',
    'mid_head_3d': 'head',
    # Right leg
    'right_hip_jc_3d': 'right_hip',
    'right_knee_jc_3d': 'right_knee',
    'right_ankle_jc_3d': 'right_ankle',
    # Left leg
    'left_hip_jc_3d': 'left_hip',
    'left_knee_jc_3d': 'left_knee',
    'left_ankle_jc_3d': 'left_ankle',
    # Left arm
    'left_shoulder_jc_3d': 'left_shoulder',
    'left_elbow_jc_3d': 'left_elbow',
    'left_wrist_jc_3d': 'left_wrist',
    # Right arm
    'right_shoulder_jc_3d': 'right_shoulder',
    'right_elbow_jc_3d': 'right_elbow',
    'right_wrist_jc_3d': 'right_wrist',
}


class UPLIFTDataset(Dataset):
    """Dataset for training fusion from UPLIFT ground truth."""

    def __init__(self,
                 method_poses: List[np.ndarray],
                 method_confidences: List[np.ndarray],
                 ground_truth: np.ndarray):
        """
        Initialize dataset.

        Args:
            method_poses: List of (N, 17, 3) arrays, one per method
            method_confidences: List of (N, 17) arrays, one per method
            ground_truth: (N, 17, 3) ground truth positions
        """
        self.method_poses = torch.tensor(
            np.stack(method_poses, axis=1), dtype=torch.float32
        )  # (N, n_methods, 17, 3)

        self.method_confidences = torch.tensor(
            np.stack(method_confidences, axis=1), dtype=torch.float32
        )  # (N, n_methods, 17)

        self.ground_truth = torch.tensor(
            ground_truth, dtype=torch.float32
        )  # (N, 17, 3)

    def __len__(self):
        return len(self.ground_truth)

    def __getitem__(self, idx):
        return (
            self.method_poses[idx],
            self.method_confidences[idx],
            self.ground_truth[idx]
        )


def load_uplift_positions(uplift_csv: str) -> np.ndarray:
    """
    Load 3D joint positions from UPLIFT CSV.

    Args:
        uplift_csv: Path to UPLIFT export CSV

    Returns:
        (N, 17, 3) array of joint positions
    """
    df = pd.read_csv(uplift_csv)

    n_frames = len(df)
    positions = np.zeros((n_frames, 17, 3))

    # Map available UPLIFT columns to H36M joints
    for uplift_prefix, h36m_name in UPLIFT_TO_H36M.items():
        h36m_idx = H36M_JOINT_NAMES.index(h36m_name)

        x_col = f'{uplift_prefix}_x'
        y_col = f'{uplift_prefix}_y'
        z_col = f'{uplift_prefix}_z'

        if x_col in df.columns:
            positions[:, h36m_idx, 0] = df[x_col].values
            positions[:, h36m_idx, 1] = df[y_col].values
            positions[:, h36m_idx, 2] = df[z_col].values

    # For missing joints, interpolate from nearby joints
    positions = interpolate_missing_joints(positions)

    return positions


def interpolate_missing_joints(positions: np.ndarray) -> np.ndarray:
    """Interpolate missing joint positions from skeleton structure."""
    # Check for missing joints (all zeros)
    for j in range(17):
        if np.allclose(positions[:, j], 0):
            # Interpolate based on skeleton structure
            if j == 7:  # spine - midpoint of pelvis and neck
                positions[:, j] = (positions[:, 0] + positions[:, 8]) / 2
            elif j == 9:  # head - above neck
                positions[:, j] = positions[:, 8] + np.array([0, 0.12, 0])
            elif j == 10:  # head_top - above head
                positions[:, j] = positions[:, 9] + np.array([0, 0.1, 0])
            # Add more interpolation rules as needed

    return positions


def normalize_ground_truth(positions: np.ndarray) -> np.ndarray:
    """
    Normalize UPLIFT ground truth the same way we normalize method outputs.

    Centers on pelvis and scales based on torso length to match method outputs.

    Args:
        positions: (N, 17, 3) ground truth positions

    Returns:
        Normalized (N, 17, 3) positions
    """
    EXPECTED_TORSO = 0.50  # Same as in pose_estimators/base.py

    normalized = np.zeros_like(positions)

    for i in range(len(positions)):
        pose = positions[i]

        # Center on pelvis (joint 0)
        pelvis = pose[0]
        centered = pose - pelvis

        # Calculate torso length (hips to shoulders)
        hip_center = (pose[1] + pose[4]) / 2  # right_hip + left_hip
        shoulder_center = (pose[14] + pose[11]) / 2  # right_shoulder + left_shoulder
        torso_length = np.linalg.norm(shoulder_center - hip_center)

        # Scale to expected torso length
        if torso_length > 1e-6:
            scale = EXPECTED_TORSO / torso_length
            centered = centered * scale

        normalized[i] = centered

    return normalized


def align_sequences(our_poses: List[np.ndarray],
                    uplift_poses: np.ndarray,
                    our_fps: float,
                    uplift_fps: float) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Align our pose sequences with UPLIFT ground truth.

    Uses interpolation to match frame counts.
    """
    from scipy.interpolate import interp1d

    # Create time arrays
    our_n = len(our_poses[0])
    uplift_n = len(uplift_poses)

    our_time = np.linspace(0, 1, our_n)
    uplift_time = np.linspace(0, 1, uplift_n)

    # Use the shorter sequence length as target
    target_n = min(our_n, uplift_n)
    target_time = np.linspace(0, 1, target_n)

    # Interpolate our poses
    aligned_poses = []
    for method_poses in our_poses:
        aligned_method = np.zeros((target_n, 17, 3))
        for j in range(17):
            for c in range(3):
                f = interp1d(our_time, method_poses[:, j, c],
                             kind='linear', fill_value='extrapolate')
                aligned_method[:, j, c] = f(target_time)
        aligned_poses.append(aligned_method)

    # Interpolate UPLIFT poses
    aligned_uplift = np.zeros((target_n, 17, 3))
    for j in range(17):
        for c in range(3):
            f = interp1d(uplift_time, uplift_poses[:, j, c],
                         kind='linear', fill_value='extrapolate')
            aligned_uplift[:, j, c] = f(target_time)

    return aligned_poses, aligned_uplift


def prepare_training_data(video_paths: List[str],
                          uplift_csv: str,
                          methods: List[str] = None,
                          use_triangulation: bool = True) -> Tuple:
    """
    Prepare training data from videos and UPLIFT ground truth.

    Args:
        video_paths: List of video paths
        uplift_csv: Path to UPLIFT CSV export
        methods: Methods to use for pose estimation
        use_triangulation: Include triangulation with absolute coords as anchor

    Returns:
        Tuple of (method_poses, method_confidences, ground_truth)
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from ensemble_pipeline import EnsemblePosePipeline

    if methods is None:
        methods = ['yolo_lifting', 'motionbert']

    # Add triangulation for absolute coordinate anchor (requires 2+ videos)
    if use_triangulation and len(video_paths) >= 2 and 'triangulation' not in methods:
        methods = methods + ['triangulation']
        print("Adding triangulation as absolute coordinate anchor")

    print("Running pose estimation methods...")
    pipeline = EnsemblePosePipeline(methods=methods)

    # Set up camera parameters for triangulation if we have 2+ videos
    if 'triangulation' in methods and len(video_paths) >= 2:
        # Default camera setup: both at 3m, 90 degrees apart
        camera_distances = [3.0, 3.0]
        camera_angles = [0, 90]  # Side and back views
        pipeline.estimators['triangulation'].set_camera_params(
            video_paths, camera_distances, camera_angles
        )
        print(f"Set up triangulation with cameras at {camera_distances}m, angles {camera_angles}°")

    results = pipeline.process_videos(video_paths)

    # Get method poses and confidences
    method_poses = []
    method_confidences = []

    for method in methods:
        poses = results.method_poses[method]
        method_poses.append(np.stack(poses))

        # Use uniform confidence if not available
        confidences = np.ones((len(poses), 17))
        method_confidences.append(confidences)

    # Load UPLIFT ground truth (keep in absolute coordinates)
    print("Loading UPLIFT ground truth...")
    ground_truth = load_uplift_positions(uplift_csv)
    # Note: NOT normalizing - keeping absolute coordinates to match triangulation output

    # Align sequences
    print("Aligning sequences...")
    method_poses, ground_truth = align_sequences(
        method_poses, ground_truth,
        results.fps, 240  # UPLIFT typically 240 FPS
    )

    print(f"Prepared {len(ground_truth)} training frames with {len(methods)} methods")

    return method_poses, method_confidences, ground_truth, len(methods)


def prepare_multi_session_data(sessions: List[dict],
                                methods: List[str] = None) -> Tuple:
    """
    Prepare training data from multiple video+UPLIFT sessions.

    Args:
        sessions: List of dicts with 'videos' and 'uplift' keys
                  e.g., [{'videos': ['side1.mp4', 'back1.mp4'], 'uplift': 'uplift1.csv'},
                         {'videos': ['side2.mp4', 'back2.mp4'], 'uplift': 'uplift2.csv'}]
        methods: Pose estimation methods to use

    Returns:
        Tuple of (method_poses, method_confidences, ground_truth)
    """
    if methods is None:
        methods = ['yolo_lifting', 'motionbert']

    all_method_poses = None  # Will be initialized after first session
    all_method_confidences = None
    all_ground_truth = []
    actual_n_methods = None

    print(f"\nProcessing {len(sessions)} training sessions...")

    for i, session in enumerate(sessions):
        print(f"\n--- Session {i+1}/{len(sessions)} ---")
        video_paths = session['videos']
        uplift_csv = session['uplift']

        try:
            method_poses, method_confidences, ground_truth, n_methods = prepare_training_data(
                video_paths, uplift_csv, methods
            )

            # Initialize arrays on first successful session
            if all_method_poses is None:
                actual_n_methods = n_methods
                all_method_poses = [[] for _ in range(n_methods)]
                all_method_confidences = [[] for _ in range(n_methods)]

            # Append to combined dataset
            for m in range(n_methods):
                all_method_poses[m].append(method_poses[m])
                all_method_confidences[m].append(method_confidences[m])
            all_ground_truth.append(ground_truth)

            print(f"  Added {len(ground_truth)} frames")

        except Exception as e:
            print(f"  Warning: Failed to process session {i+1}: {e}")
            continue

    # Concatenate all sessions
    combined_poses = [np.concatenate(poses, axis=0) for poses in all_method_poses]
    combined_confidences = [np.concatenate(confs, axis=0) for confs in all_method_confidences]
    combined_ground_truth = np.concatenate(all_ground_truth, axis=0)

    # Actual number of methods (may include triangulation added by prepare_training_data)
    actual_n_methods = len(combined_poses)

    print(f"\nTotal training frames: {len(combined_ground_truth)}")
    print(f"Actual methods used: {actual_n_methods}")

    return combined_poses, combined_confidences, combined_ground_truth, actual_n_methods


def train_fusion_model(video_paths: List[str] = None,
                       uplift_csv: str = None,
                       sessions: List[dict] = None,
                       output_path: str = 'models/fusion_model.pt',
                       methods: List[str] = None,
                       epochs: int = 100,
                       batch_size: int = 32,
                       learning_rate: float = 1e-4,
                       val_split: float = 0.2):
    """
    Train fusion model on UPLIFT ground truth.

    Supports two modes:
    1. Single session: Provide video_paths and uplift_csv
    2. Multiple sessions: Provide sessions list

    Args:
        video_paths: List of video paths (single session mode)
        uplift_csv: Path to UPLIFT CSV (single session mode)
        sessions: List of session dicts (multi-session mode)
                  e.g., [{'videos': ['side.mp4'], 'uplift': 'data.csv'}, ...]
        output_path: Path to save trained model
        methods: Pose estimation methods to use
        epochs: Training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        val_split: Validation split fraction
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch required for training")

    if methods is None:
        methods = ['yolo_lifting', 'motionbert']

    # Prepare data - support both single and multi-session
    if sessions is not None:
        method_poses, method_confidences, ground_truth, n_methods = prepare_multi_session_data(
            sessions, methods
        )
    elif video_paths is not None and uplift_csv is not None:
        method_poses, method_confidences, ground_truth, n_methods = prepare_training_data(
            video_paths, uplift_csv, methods
        )
    else:
        raise ValueError("Provide either (video_paths, uplift_csv) or sessions")

    # Create dataset
    dataset = UPLIFTDataset(method_poses, method_confidences, ground_truth)

    # Split into train/val
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    print(f"\nTraining: {train_size} frames")
    print(f"Validation: {val_size} frames")

    # Create model with actual number of methods (may include triangulation)
    model = FusionNetwork(n_methods=n_methods)
    trainer = FusionTrainer(model, learning_rate=learning_rate)

    print(f"\nTraining fusion network ({n_methods} methods)...")
    print(f"Device: {trainer.device}")

    best_val_mpjpe = float('inf')

    for epoch in range(epochs):
        # Train
        train_loss = trainer.train_epoch(train_loader)

        # Validate
        val_metrics = trainer.validate(val_loader)

        # Log progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1}/{epochs}")
            print(f"  Train loss: {train_loss:.6f}")
            print(f"  Val MPJPE: {val_metrics['mpjpe_cm']:.2f} cm")

        # Save best model
        if val_metrics['mpjpe'] < best_val_mpjpe:
            best_val_mpjpe = val_metrics['mpjpe']
            trainer.save_checkpoint(output_path, epoch, val_metrics)

    print(f"\nTraining complete!")
    print(f"Best validation MPJPE: {best_val_mpjpe * 100:.2f} cm")
    print(f"Model saved to: {output_path}")

    return trainer.model


def main():
    """Command-line training interface."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Train fusion network on UPLIFT data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single session:
  python -m fusion.train_fusion --videos side.mp4 back.mp4 --uplift data.csv

  # Multiple sessions (via JSON file):
  python -m fusion.train_fusion --sessions-file training_sessions.json

  # training_sessions.json format:
  {
    "sessions": [
      {"videos": ["session1_side.mp4", "session1_back.mp4"], "uplift": "session1.csv"},
      {"videos": ["session2_side.mp4", "session2_back.mp4"], "uplift": "session2.csv"}
    ]
  }
        """
    )

    # Single session mode
    parser.add_argument('--videos', nargs='+',
                        help='Video paths (single session mode)')
    parser.add_argument('--uplift',
                        help='Path to UPLIFT CSV (single session mode)')

    # Multi-session mode
    parser.add_argument('--sessions-file',
                        help='JSON file with multiple training sessions')

    # Common options
    parser.add_argument('--output', '-o', default='models/fusion_model.pt',
                        help='Output model path')
    parser.add_argument('--methods', nargs='+',
                        default=['yolo_lifting', 'motionbert'],
                        help='Pose estimation methods')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')

    args = parser.parse_args()

    # Create output directory
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # Determine mode
    if args.sessions_file:
        # Multi-session mode
        with open(args.sessions_file, 'r') as f:
            config = json.load(f)
        sessions = config['sessions']
        print(f"Loading {len(sessions)} training sessions from {args.sessions_file}")

        train_fusion_model(
            sessions=sessions,
            output_path=args.output,
            methods=args.methods,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr
        )

    elif args.videos and args.uplift:
        # Single session mode
        train_fusion_model(
            video_paths=args.videos,
            uplift_csv=args.uplift,
            output_path=args.output,
            methods=args.methods,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr
        )

    else:
        parser.error("Provide either (--videos + --uplift) or --sessions-file")


if __name__ == '__main__':
    main()
