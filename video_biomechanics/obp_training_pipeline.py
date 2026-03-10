"""
Synthetic Training Data Pipeline for 2D→3D Pose Lifting.

Uses OpenBiomechanics 3D motion capture data to generate:
1. Synthetic 2D projections with camera augmentation
2. Paired (2D, 3D) training data
3. Domain-specific training for baseball/athletic movements

This creates a supervised dataset for training 2D→3D lifting models
that are specialized for fast rotational movements.
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Generator
from dataclasses import dataclass
import json
import torch
from torch.utils.data import Dataset, DataLoader

from obp_data_loader import OBPDataLoader, OBPSequence, STANDARD_JOINT_ORDER


@dataclass
class CameraParams:
    """Virtual camera parameters for 3D→2D projection."""
    focal_length: float = 1000.0  # pixels
    center_x: float = 960.0  # image center
    center_y: float = 540.0
    # Camera position relative to subject (meters)
    distance: float = 5.0  # distance from subject
    azimuth: float = 0.0  # horizontal angle (radians)
    elevation: float = 0.0  # vertical angle (radians)
    roll: float = 0.0  # camera roll (radians)

    def get_rotation_matrix(self) -> np.ndarray:
        """Get camera rotation matrix."""
        # Rotation around Y (azimuth)
        ca, sa = np.cos(self.azimuth), np.sin(self.azimuth)
        Ry = np.array([
            [ca, 0, sa],
            [0, 1, 0],
            [-sa, 0, ca]
        ])

        # Rotation around X (elevation)
        ce, se = np.cos(self.elevation), np.sin(self.elevation)
        Rx = np.array([
            [1, 0, 0],
            [0, ce, -se],
            [0, se, ce]
        ])

        # Rotation around Z (roll)
        cr, sr = np.cos(self.roll), np.sin(self.roll)
        Rz = np.array([
            [cr, -sr, 0],
            [sr, cr, 0],
            [0, 0, 1]
        ])

        return Rz @ Rx @ Ry

    def get_camera_position(self) -> np.ndarray:
        """Get camera position in world coordinates."""
        # Camera looks at origin from distance
        x = self.distance * np.sin(self.azimuth) * np.cos(self.elevation)
        y = self.distance * np.sin(self.elevation)
        z = self.distance * np.cos(self.azimuth) * np.cos(self.elevation)
        return np.array([x, y, z])


def project_3d_to_2d(points_3d: np.ndarray,
                     camera: CameraParams,
                     add_noise: bool = True,
                     noise_std: float = 2.0) -> np.ndarray:
    """
    Project 3D points to 2D using perspective projection.

    Args:
        points_3d: (N, 3) array of 3D points
        camera: Camera parameters
        add_noise: Whether to add Gaussian noise to 2D points
        noise_std: Standard deviation of noise in pixels

    Returns:
        (N, 2) array of 2D pixel coordinates
    """
    R = camera.get_rotation_matrix()
    t = camera.get_camera_position()

    # Transform points to camera coordinates
    points_cam = (points_3d - t) @ R.T

    # Perspective projection
    # Note: In camera frame, Z points forward, X right, Y down
    x_cam = points_cam[:, 0]
    y_cam = points_cam[:, 2]  # Swap Y and Z for typical camera orientation
    z_cam = points_cam[:, 1]

    # Avoid division by zero
    z_cam = np.maximum(z_cam, 0.1)

    # Project
    x_2d = camera.focal_length * x_cam / z_cam + camera.center_x
    y_2d = camera.focal_length * y_cam / z_cam + camera.center_y

    points_2d = np.stack([x_2d, y_2d], axis=1)

    if add_noise:
        noise = np.random.normal(0, noise_std, points_2d.shape)
        points_2d += noise

    return points_2d


def augment_camera(base_camera: CameraParams,
                   azimuth_range: Tuple[float, float] = (-np.pi/3, np.pi/3),
                   elevation_range: Tuple[float, float] = (-np.pi/6, np.pi/6),
                   distance_range: Tuple[float, float] = (3.0, 8.0),
                   roll_range: Tuple[float, float] = (-np.pi/12, np.pi/12),
                   focal_range: Tuple[float, float] = (800, 1200)) -> CameraParams:
    """
    Create augmented camera parameters.

    Args:
        base_camera: Base camera to augment
        *_range: Min/max ranges for each parameter

    Returns:
        New CameraParams with randomized values
    """
    return CameraParams(
        focal_length=np.random.uniform(*focal_range),
        center_x=base_camera.center_x + np.random.uniform(-50, 50),
        center_y=base_camera.center_y + np.random.uniform(-50, 50),
        distance=np.random.uniform(*distance_range),
        azimuth=np.random.uniform(*azimuth_range),
        elevation=np.random.uniform(*elevation_range),
        roll=np.random.uniform(*roll_range)
    )


class SyntheticPoseDataset(Dataset):
    """
    PyTorch Dataset for 2D→3D pose lifting training.

    Generates synthetic 2D projections from OBP 3D data with augmentation.
    """

    def __init__(self,
                 loader: OBPDataLoader,
                 dataset: str = 'baseball_pitching',
                 max_sequences: int = 100,
                 samples_per_sequence: int = 50,
                 augmentations_per_sample: int = 5,
                 normalize: bool = True):
        """
        Initialize the dataset.

        Args:
            loader: OBP data loader
            dataset: Which dataset to use
            max_sequences: Maximum sequences to load
            samples_per_sequence: Frames to sample per sequence
            augmentations_per_sample: Camera augmentations per frame
            normalize: Whether to normalize poses
        """
        self.loader = loader
        self.normalize = normalize

        print(f"Building synthetic dataset from {dataset}...")

        # Collect all 3D poses
        self.poses_3d: List[np.ndarray] = []
        self.sequence_ids: List[str] = []

        for i, seq in enumerate(loader.iter_sequences(dataset, max_sequences)):
            if i % 20 == 0:
                print(f"  Loading sequence {i+1}...")

            # Sample frames evenly
            n_frames = len(seq.frames)
            if n_frames < samples_per_sequence:
                indices = list(range(n_frames))
            else:
                indices = np.linspace(0, n_frames-1, samples_per_sequence, dtype=int)

            for idx in indices:
                frame = seq.frames[idx]
                pose_3d = loader.to_h36m_format(frame)

                # Skip if too many missing joints
                if np.sum(np.all(pose_3d == 0, axis=1)) > 3:
                    continue

                self.poses_3d.append(pose_3d)
                self.sequence_ids.append(seq.session_pitch)

        self.n_poses = len(self.poses_3d)
        self.augmentations_per_sample = augmentations_per_sample

        print(f"  Loaded {self.n_poses} 3D poses")
        print(f"  Total samples (with augmentation): {len(self)}")

        # Compute normalization stats
        if self.normalize:
            all_poses = np.array(self.poses_3d)
            self.pose_mean = np.mean(all_poses, axis=(0, 1))
            self.pose_std = np.std(all_poses) + 1e-6
        else:
            self.pose_mean = np.zeros(3)
            self.pose_std = 1.0

    def __len__(self) -> int:
        return self.n_poses * self.augmentations_per_sample

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a (2D input, 3D target) pair."""
        pose_idx = idx // self.augmentations_per_sample
        pose_3d = self.poses_3d[pose_idx].copy()

        # Create augmented camera
        base_camera = CameraParams()
        camera = augment_camera(base_camera)

        # Project to 2D
        pose_2d = project_3d_to_2d(pose_3d, camera, add_noise=True)

        # Normalize 2D (center and scale)
        pose_2d_norm = self._normalize_2d(pose_2d)

        # Normalize 3D
        if self.normalize:
            pose_3d_norm = (pose_3d - self.pose_mean) / self.pose_std
        else:
            pose_3d_norm = pose_3d

        return (
            torch.FloatTensor(pose_2d_norm.flatten()),
            torch.FloatTensor(pose_3d_norm.flatten())
        )

    def _normalize_2d(self, pose_2d: np.ndarray) -> np.ndarray:
        """Normalize 2D pose: center on hip, scale by torso."""
        # Center on hip (joint 0)
        hip = pose_2d[0].copy()
        pose_centered = pose_2d - hip

        # Scale by torso length (hip to neck, joints 0 and 8)
        torso_length = np.linalg.norm(pose_2d[8] - pose_2d[0])
        if torso_length > 1e-6:
            pose_centered = pose_centered / torso_length

        return pose_centered


class Pose2DTo3DLifter(torch.nn.Module):
    """
    Neural network for lifting 2D poses to 3D.

    Architecture based on Martinez et al. "A simple yet effective baseline
    for 3d human pose estimation" but adapted for sports movements.
    """

    def __init__(self,
                 input_dim: int = 34,  # 17 joints * 2
                 output_dim: int = 51,  # 17 joints * 3
                 hidden_dim: int = 1024,
                 num_layers: int = 4,
                 dropout: float = 0.25):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        # Build residual blocks
        layers = []

        # Input layer
        layers.append(torch.nn.Linear(input_dim, hidden_dim))
        layers.append(torch.nn.BatchNorm1d(hidden_dim))
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Dropout(dropout))

        # Residual blocks
        for _ in range(num_layers - 2):
            layers.append(ResidualBlock(hidden_dim, dropout))

        # Output layer
        layers.append(torch.nn.Linear(hidden_dim, output_dim))

        self.model = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class ResidualBlock(torch.nn.Module):
    """Residual block with batch norm and dropout."""

    def __init__(self, dim: int, dropout: float = 0.25):
        super().__init__()
        self.block = torch.nn.Sequential(
            torch.nn.Linear(dim, dim),
            torch.nn.BatchNorm1d(dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(dim, dim),
            torch.nn.BatchNorm1d(dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


def train_lifter(dataset: SyntheticPoseDataset,
                 epochs: int = 50,
                 batch_size: int = 64,
                 learning_rate: float = 1e-3,
                 device: str = 'cuda') -> Pose2DTo3DLifter:
    """
    Train the 2D→3D lifter model.

    Args:
        dataset: Training dataset
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        device: Device to train on

    Returns:
        Trained model
    """
    if not torch.cuda.is_available() and device == 'cuda':
        device = 'cpu'
        print("CUDA not available, using CPU")

    # Split into train/val
    n_train = int(0.9 * len(dataset))
    n_val = len(dataset) - n_train
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [n_train, n_val]
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Create model
    model = Pose2DTo3DLifter().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )
    criterion = torch.nn.MSELoss()

    best_val_loss = float('inf')
    best_model_state = None

    print(f"\nTraining 2D-to-3D lifter on {device}...")
    print(f"  Train samples: {n_train}")
    print(f"  Val samples: {n_val}")

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0

        for batch_2d, batch_3d in train_loader:
            batch_2d = batch_2d.to(device)
            batch_3d = batch_3d.to(device)

            optimizer.zero_grad()
            pred_3d = model(batch_2d)
            loss = criterion(pred_3d, batch_3d)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch_2d, batch_3d in val_loader:
                batch_2d = batch_2d.to(device)
                batch_3d = batch_3d.to(device)

                pred_3d = model(batch_2d)
                loss = criterion(pred_3d, batch_3d)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()

        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"  Epoch {epoch+1}/{epochs}: "
                  f"train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    print(f"\nTraining complete. Best val_loss: {best_val_loss:.6f}")

    return model


def save_model(model: Pose2DTo3DLifter,
               dataset: SyntheticPoseDataset,
               path: Path):
    """Save model with normalization parameters."""
    torch.save({
        'model_state_dict': model.state_dict(),
        'pose_mean': dataset.pose_mean,
        'pose_std': dataset.pose_std,
        'input_dim': model.input_dim,
        'output_dim': model.output_dim,
    }, path)
    print(f"Model saved to: {path}")


def load_model(path: Path, device: str = 'cpu') -> Tuple[Pose2DTo3DLifter, dict]:
    """Load model and normalization parameters."""
    checkpoint = torch.load(path, map_location=device)
    model = Pose2DTo3DLifter(
        input_dim=checkpoint['input_dim'],
        output_dim=checkpoint['output_dim']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    norm_params = {
        'pose_mean': checkpoint['pose_mean'],
        'pose_std': checkpoint['pose_std']
    }

    return model, norm_params


if __name__ == '__main__':
    script_dir = Path(__file__).parent
    data_root = script_dir.parent

    print(f"Loading OBP data from: {data_root}")

    # Create data loader
    loader = OBPDataLoader(data_root)

    # Create synthetic dataset
    dataset = SyntheticPoseDataset(
        loader,
        dataset='baseball_pitching',
        max_sequences=80,
        samples_per_sequence=40,
        augmentations_per_sample=10
    )

    # Train the model
    model = train_lifter(
        dataset,
        epochs=30,
        batch_size=128,
        learning_rate=1e-3
    )

    # Save model
    output_dir = script_dir / 'models'
    output_dir.mkdir(exist_ok=True)
    save_model(model, dataset, output_dir / 'obp_lifter.pt')

    # Quick test
    print("\n--- Quick test ---")
    model.eval()
    model = model.cpu()  # Move to CPU for testing
    test_input, test_target = dataset[0]
    with torch.no_grad():
        test_pred = model(test_input.unsqueeze(0))

    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {test_pred.shape}")
    print(f"Target shape: {test_target.shape}")

    # Compute MPJPE (Mean Per Joint Position Error)
    pred_joints = test_pred.reshape(17, 3).numpy()
    target_joints = test_target.reshape(17, 3).numpy()
    mpjpe = np.mean(np.linalg.norm(pred_joints - target_joints, axis=1))
    print(f"Test MPJPE: {mpjpe:.4f} (normalized units)")
