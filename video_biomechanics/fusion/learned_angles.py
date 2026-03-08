"""
Learned angle prediction from 3D poses.

Trains a neural network to predict joint angles that match UPLIFT conventions
directly from 3D skeleton positions.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader, random_split
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:

    class AnglePredictionNetwork(nn.Module):
        """
        Neural network to predict joint angles from 3D skeleton.

        Takes 3D joint positions and outputs angles matching UPLIFT conventions.
        """

        def __init__(self,
                     n_joints: int = 17,
                     n_angles: int = 50,
                     hidden_dim: int = 256,
                     use_temporal: bool = True,
                     temporal_window: int = 5):
            """
            Initialize angle prediction network.

            Args:
                n_joints: Number of skeleton joints (H36M = 17)
                n_angles: Number of output angles to predict
                hidden_dim: Hidden layer dimension
                use_temporal: Use temporal context
                temporal_window: Frames of context (if temporal)
            """
            super().__init__()

            self.n_joints = n_joints
            self.n_angles = n_angles
            self.use_temporal = use_temporal
            self.temporal_window = temporal_window

            # Input: 3D positions (17 joints × 3 coords)
            # Plus relative positions and bone vectors for richer features
            input_dim = n_joints * 3  # Raw positions
            input_dim += n_joints * 3  # Pelvis-relative positions
            input_dim += 16 * 3  # Bone vectors (16 bones)

            if use_temporal:
                input_dim *= temporal_window

            # Feature encoder
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
            )

            # Angle prediction heads (separate for different angle types)
            self.angle_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, n_angles),
            )

            # Store angle names for output mapping
            self.angle_names: List[str] = []

        def extract_features(self, poses: torch.Tensor) -> torch.Tensor:
            """
            Extract rich features from 3D poses.

            Args:
                poses: (B, T, 17, 3) or (B, 17, 3) joint positions

            Returns:
                (B, feature_dim) feature vector
            """
            if poses.dim() == 3:
                poses = poses.unsqueeze(1)  # Add temporal dim

            B, T, J, C = poses.shape

            features = []

            for t in range(T):
                frame_poses = poses[:, t]  # (B, 17, 3)

                # Raw positions (flattened)
                features.append(frame_poses.reshape(B, -1))

                # Pelvis-relative positions
                pelvis = frame_poses[:, 0:1, :]  # (B, 1, 3)
                relative = frame_poses - pelvis
                features.append(relative.reshape(B, -1))

                # Bone vectors
                bone_pairs = [
                    (0, 1), (1, 2), (2, 3),  # Right leg
                    (0, 4), (4, 5), (5, 6),  # Left leg
                    (0, 7), (7, 8), (8, 9), (9, 10),  # Spine/head
                    (8, 11), (11, 12), (12, 13),  # Left arm
                    (8, 14), (14, 15), (15, 16),  # Right arm
                ]
                bones = []
                for j1, j2 in bone_pairs:
                    bone = frame_poses[:, j2] - frame_poses[:, j1]
                    bones.append(bone)
                bones = torch.stack(bones, dim=1)  # (B, 16, 3)
                features.append(bones.reshape(B, -1))

            # Concatenate all features
            return torch.cat(features, dim=-1)

        def forward(self, poses: torch.Tensor) -> torch.Tensor:
            """
            Predict angles from poses.

            Args:
                poses: (B, 17, 3) or (B, T, 17, 3) joint positions

            Returns:
                (B, n_angles) predicted angles in degrees
            """
            features = self.extract_features(poses)
            encoded = self.encoder(features)
            angles = self.angle_head(encoded)
            return angles

        def set_angle_names(self, names: List[str]):
            """Set the output angle names."""
            self.angle_names = names

        def predict_dict(self, poses: torch.Tensor) -> Dict[str, torch.Tensor]:
            """Predict angles and return as dictionary."""
            angles = self.forward(poses)
            return {name: angles[:, i] for i, name in enumerate(self.angle_names)}


    class AngleDataset(Dataset):
        """Dataset for training angle prediction from poses."""

        def __init__(self,
                     poses_3d: np.ndarray,
                     angles: np.ndarray,
                     temporal_window: int = 1):
            """
            Initialize dataset.

            Args:
                poses_3d: (N, 17, 3) array of 3D poses
                angles: (N, n_angles) array of target angles
                temporal_window: Number of frames for context
            """
            self.poses = torch.tensor(poses_3d, dtype=torch.float32)
            self.angles = torch.tensor(angles, dtype=torch.float32)
            self.temporal_window = temporal_window
            self.pad = temporal_window // 2

        def __len__(self):
            return len(self.poses)

        def __getitem__(self, idx):
            if self.temporal_window > 1:
                # Get temporal context
                start = max(0, idx - self.pad)
                end = min(len(self.poses), idx + self.pad + 1)

                window = self.poses[start:end]

                # Pad if needed
                if len(window) < self.temporal_window:
                    pad_size = self.temporal_window - len(window)
                    if idx < self.pad:
                        # Pad at start
                        window = torch.cat([window[:1].repeat(pad_size, 1, 1), window])
                    else:
                        # Pad at end
                        window = torch.cat([window, window[-1:].repeat(pad_size, 1, 1)])

                return window, self.angles[idx]
            else:
                return self.poses[idx], self.angles[idx]


    class AnglePredictor:
        """Trainer and inference wrapper for angle prediction."""

        def __init__(self,
                     model: Optional[AnglePredictionNetwork] = None,
                     device: str = 'auto'):
            """
            Initialize predictor.

            Args:
                model: Pre-trained model or None to create new
                device: 'cuda', 'cpu', or 'auto'
            """
            if device == 'auto':
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            else:
                self.device = torch.device(device)

            self.model = model
            if model is not None:
                self.model = model.to(self.device)

            self.angle_names: List[str] = []
            self.angle_mean: Optional[np.ndarray] = None
            self.angle_std: Optional[np.ndarray] = None

        def train(self,
                  poses_3d: np.ndarray,
                  angles_df,
                  epochs: int = 100,
                  batch_size: int = 32,
                  learning_rate: float = 1e-4,
                  val_split: float = 0.2) -> Dict:
            """
            Train angle prediction model.

            Args:
                poses_3d: (N, 17, 3) array of 3D poses
                angles_df: DataFrame with angle columns from UPLIFT
                epochs: Training epochs
                batch_size: Batch size
                learning_rate: Learning rate
                val_split: Validation split fraction

            Returns:
                Training metrics
            """
            # Extract actual angle columns (biomechanical joint angles)
            angle_keywords = ['extension', 'flexion', 'rotation', 'adduction',
                              'abduction', 'twist', 'tilt', 'lateral', 'varus',
                              'valgus', 'dorsiflexion']
            skip_suffixes = ['_velocity', '_at_launch', '_at_contact']
            skip_prefixes = ['max_', 'min_', 'peak_', 'excessive_']

            angle_cols = []
            for col in angles_df.columns:
                # Must contain an angle keyword
                if not any(kw in col.lower() for kw in angle_keywords):
                    continue
                # Skip velocity and event-specific columns
                if any(col.endswith(s) for s in skip_suffixes):
                    continue
                if any(col.startswith(s) for s in skip_prefixes):
                    continue
                if angles_df[col].dtype in ['float64', 'int64', 'float32']:
                    # Must have valid data and reasonable range for angles
                    if not angles_df[col].isna().all():
                        max_val = angles_df[col].abs().max()
                        if max_val < 400:  # Angles should be < 360 degrees
                            angle_cols.append(col)

            self.angle_names = angle_cols
            print(f"Training on {len(angle_cols)} angle columns")

            # Extract angle values
            angles = angles_df[angle_cols].values.astype(np.float32)

            # Handle NaN values
            angles = np.nan_to_num(angles, nan=0.0)

            # Normalize angles for training
            self.angle_mean = angles.mean(axis=0)
            self.angle_std = angles.std(axis=0) + 1e-6
            angles_norm = (angles - self.angle_mean) / self.angle_std

            # Ensure poses and angles are aligned
            n_samples = min(len(poses_3d), len(angles_norm))
            poses_3d = poses_3d[:n_samples]
            angles_norm = angles_norm[:n_samples]

            print(f"Training samples: {n_samples}")

            # Create model
            self.model = AnglePredictionNetwork(
                n_joints=17,
                n_angles=len(angle_cols),
                hidden_dim=256,
                use_temporal=False
            ).to(self.device)
            self.model.set_angle_names(angle_cols)

            # Create dataset
            dataset = AngleDataset(poses_3d, angles_norm)

            # Split
            val_size = int(len(dataset) * val_split)
            train_size = len(dataset) - val_size
            train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)

            # Training
            optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=10
            )

            best_val_loss = float('inf')

            for epoch in range(epochs):
                # Train
                self.model.train()
                train_loss = 0.0
                for poses, targets in train_loader:
                    poses = poses.to(self.device)
                    targets = targets.to(self.device)

                    optimizer.zero_grad()
                    predictions = self.model(poses)
                    loss = F.mse_loss(predictions, targets)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()

                    train_loss += loss.item()

                train_loss /= len(train_loader)

                # Validate
                self.model.eval()
                val_loss = 0.0
                val_mae = 0.0
                with torch.no_grad():
                    for poses, targets in val_loader:
                        poses = poses.to(self.device)
                        targets = targets.to(self.device)

                        predictions = self.model(poses)
                        val_loss += F.mse_loss(predictions, targets).item()

                        # Denormalize for MAE in degrees
                        pred_deg = predictions.cpu().numpy() * self.angle_std + self.angle_mean
                        target_deg = targets.cpu().numpy() * self.angle_std + self.angle_mean
                        val_mae += np.mean(np.abs(pred_deg - target_deg))

                val_loss /= len(val_loader)
                val_mae /= len(val_loader)
                scheduler.step(val_loss)

                if (epoch + 1) % 10 == 0 or epoch == 0:
                    print(f"Epoch {epoch + 1}/{epochs}")
                    print(f"  Train loss: {train_loss:.6f}")
                    print(f"  Val MAE: {val_mae:.2f} deg")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.best_mae = val_mae

            return {'best_val_mae': self.best_mae, 'n_angles': len(angle_cols)}

        def predict(self, poses_3d: np.ndarray) -> np.ndarray:
            """
            Predict angles from 3D poses.

            Args:
                poses_3d: (N, 17, 3) array of 3D poses

            Returns:
                (N, n_angles) predicted angles in degrees
            """
            self.model.eval()

            poses_tensor = torch.tensor(poses_3d, dtype=torch.float32).to(self.device)

            with torch.no_grad():
                predictions = self.model(poses_tensor)

            # Denormalize
            pred_deg = predictions.cpu().numpy() * self.angle_std + self.angle_mean

            return pred_deg

        def predict_dataframe(self, poses_3d: np.ndarray):
            """Predict angles and return as DataFrame."""
            import pandas as pd

            angles = self.predict(poses_3d)
            return pd.DataFrame(angles, columns=self.angle_names)

        def save(self, path: str):
            """Save model and normalization parameters."""
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'angle_names': self.angle_names,
                'angle_mean': self.angle_mean,
                'angle_std': self.angle_std,
                'n_angles': len(self.angle_names),
            }, path)
            print(f"Saved angle predictor to {path}")

        def load(self, path: str):
            """Load model and normalization parameters."""
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)

            self.angle_names = checkpoint['angle_names']
            self.angle_mean = checkpoint['angle_mean']
            self.angle_std = checkpoint['angle_std']

            self.model = AnglePredictionNetwork(
                n_joints=17,
                n_angles=checkpoint['n_angles'],
                hidden_dim=256,
                use_temporal=False
            ).to(self.device)

            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.set_angle_names(self.angle_names)
            self.model.eval()

            print(f"Loaded angle predictor from {path}")
            print(f"  {len(self.angle_names)} angle outputs")


else:
    class AnglePredictionNetwork:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch required for angle prediction")

    class AnglePredictor:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch required for angle prediction")
