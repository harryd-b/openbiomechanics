"""
Learned fusion network for combining multiple pose estimates.

Trained on UPLIFT ground truth data to learn optimal
per-joint weighting based on pose configuration.
"""

import numpy as np
from typing import Tuple, Optional, List
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:

    class FusionNetwork(nn.Module):
        """
        Neural network for learning optimal pose fusion.

        Architecture:
        1. Context encoder: Understands pose configuration
        2. Per-joint weight predictors: Learns method weighting per joint
        3. Residual predictor: Fine-tunes fused result
        """

        def __init__(self,
                     n_methods: int = 3,
                     n_joints: int = 17,
                     hidden_dim: int = 256,
                     use_residual: bool = True):
            """
            Initialize fusion network.

            Args:
                n_methods: Number of pose estimation methods
                n_joints: Number of joints (H36M = 17)
                hidden_dim: Hidden layer dimension
                use_residual: Add residual correction
            """
            super().__init__()

            self.n_methods = n_methods
            self.n_joints = n_joints
            self.use_residual = use_residual

            # Input: concatenated poses + confidences from all methods
            input_dim = n_methods * n_joints * 3 + n_methods * n_joints

            # Context encoder
            self.context_encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.LayerNorm(hidden_dim // 2),
                nn.ReLU(),
            )

            # Per-joint weight predictors
            joint_input_dim = hidden_dim // 2 + n_methods * 3 + n_methods
            self.joint_weight_heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(joint_input_dim, 64),
                    nn.ReLU(),
                    nn.Linear(64, n_methods),
                    nn.Softmax(dim=-1)
                )
                for _ in range(n_joints)
            ])

            # Residual correction (optional refinement)
            if use_residual:
                residual_input_dim = hidden_dim // 2 + n_joints * 3
                self.residual_predictor = nn.Sequential(
                    nn.Linear(residual_input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_dim, n_joints * 3),
                )
                self.residual_scale = nn.Parameter(torch.tensor(0.1))

        def forward(self,
                    method_poses: torch.Tensor,
                    confidences: torch.Tensor) -> torch.Tensor:
            """
            Forward pass.

            Args:
                method_poses: (B, n_methods, 17, 3) poses from each method
                confidences: (B, n_methods, 17) confidence scores

            Returns:
                fused_pose: (B, 17, 3) fused pose
            """
            B = method_poses.shape[0]

            # Flatten inputs for context encoder
            poses_flat = method_poses.reshape(B, -1)
            conf_flat = confidences.reshape(B, -1)
            combined = torch.cat([poses_flat, conf_flat], dim=-1)

            # Get pose context
            context = self.context_encoder(combined)

            # Predict weights and combine for each joint
            fused_pose = torch.zeros(B, self.n_joints, 3, device=method_poses.device)

            for j in range(self.n_joints):
                # Get method predictions for this joint
                joint_poses = method_poses[:, :, j, :]  # (B, n_methods, 3)
                joint_confs = confidences[:, :, j]       # (B, n_methods)

                # Combine context with joint-specific info
                joint_input = torch.cat([
                    context,
                    joint_poses.reshape(B, -1),
                    joint_confs
                ], dim=-1)

                # Predict weights
                weights = self.joint_weight_heads[j](joint_input)  # (B, n_methods)

                # Weighted combination
                weighted = (joint_poses * weights.unsqueeze(-1)).sum(dim=1)
                fused_pose[:, j] = weighted

            # Add residual correction
            if self.use_residual:
                residual_input = torch.cat([
                    context,
                    fused_pose.reshape(B, -1)
                ], dim=-1)
                residual = self.residual_predictor(residual_input)
                residual = residual.reshape(B, self.n_joints, 3)
                fused_pose = fused_pose + self.residual_scale * residual

            return fused_pose

        def get_joint_weights(self,
                              method_poses: torch.Tensor,
                              confidences: torch.Tensor) -> torch.Tensor:
            """
            Get learned weights for each joint and method.

            Returns:
                weights: (B, n_joints, n_methods) weight tensor
            """
            B = method_poses.shape[0]

            poses_flat = method_poses.reshape(B, -1)
            conf_flat = confidences.reshape(B, -1)
            combined = torch.cat([poses_flat, conf_flat], dim=-1)

            context = self.context_encoder(combined)

            weights = torch.zeros(B, self.n_joints, self.n_methods,
                                  device=method_poses.device)

            for j in range(self.n_joints):
                joint_poses = method_poses[:, :, j, :]
                joint_confs = confidences[:, :, j]

                joint_input = torch.cat([
                    context,
                    joint_poses.reshape(B, -1),
                    joint_confs
                ], dim=-1)

                weights[:, j] = self.joint_weight_heads[j](joint_input)

            return weights


    class FusionTrainer:
        """Trainer for fusion network."""

        def __init__(self,
                     model: FusionNetwork,
                     learning_rate: float = 1e-4,
                     bone_loss_weight: float = 0.1,
                     device: str = 'auto'):
            """
            Initialize trainer.

            Args:
                model: FusionNetwork to train
                learning_rate: Learning rate
                bone_loss_weight: Weight for bone length consistency loss
                device: 'cuda', 'cpu', or 'auto'
            """
            if device == 'auto':
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            else:
                self.device = torch.device(device)

            self.model = model.to(self.device)
            self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=10
            )
            self.bone_loss_weight = bone_loss_weight

            self.mse_loss = nn.MSELoss()

            # Bone pairs for consistency loss (H36M skeleton)
            self.bone_pairs = [
                (0, 1), (1, 2), (2, 3),  # Right leg
                (0, 4), (4, 5), (5, 6),  # Left leg
                (0, 7), (7, 8), (8, 9), (9, 10),  # Spine/head
                (8, 11), (11, 12), (12, 13),  # Left arm
                (8, 14), (14, 15), (15, 16),  # Right arm
            ]

        def bone_length_loss(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            """Calculate bone length consistency loss."""
            loss = 0.0

            for j1, j2 in self.bone_pairs:
                pred_len = torch.norm(predicted[:, j2] - predicted[:, j1], dim=-1)
                target_len = torch.norm(target[:, j2] - target[:, j1], dim=-1)
                loss = loss + F.mse_loss(pred_len, target_len)

            return loss / len(self.bone_pairs)

        def train_epoch(self, dataloader) -> float:
            """Train for one epoch."""
            self.model.train()
            total_loss = 0.0

            for batch in dataloader:
                method_poses, confidences, ground_truth = [
                    x.to(self.device) for x in batch
                ]

                self.optimizer.zero_grad()

                # Forward pass
                predicted = self.model(method_poses, confidences)

                # Position loss
                pos_loss = self.mse_loss(predicted, ground_truth)

                # Bone length loss
                bone_loss = self.bone_length_loss(predicted, ground_truth)

                # Total loss
                loss = pos_loss + self.bone_loss_weight * bone_loss

                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            self.scheduler.step(avg_loss)

            return avg_loss

        def validate(self, dataloader) -> dict:
            """Validate model."""
            self.model.eval()
            total_mse = 0.0
            total_mpjpe = 0.0
            n_samples = 0

            with torch.no_grad():
                for batch in dataloader:
                    method_poses, confidences, ground_truth = [
                        x.to(self.device) for x in batch
                    ]

                    predicted = self.model(method_poses, confidences)

                    # MSE
                    mse = F.mse_loss(predicted, ground_truth, reduction='sum')
                    total_mse += mse.item()

                    # MPJPE (Mean Per-Joint Position Error)
                    per_joint_error = torch.norm(predicted - ground_truth, dim=-1)
                    total_mpjpe += per_joint_error.sum().item()

                    n_samples += predicted.shape[0] * predicted.shape[1]

            return {
                'mse': total_mse / n_samples,
                'mpjpe': total_mpjpe / n_samples,
                'mpjpe_cm': (total_mpjpe / n_samples) * 100,
            }

        def save_checkpoint(self, path: str, epoch: int, metrics: dict):
            """Save model checkpoint."""
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'n_methods': self.model.n_methods,
                'metrics': metrics,
            }, path)

        def load_checkpoint(self, path: str) -> int:
            """Load model checkpoint. Returns epoch number."""
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            return checkpoint.get('epoch', 0)


else:
    # Placeholder when PyTorch not available
    class FusionNetwork:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch required for learned fusion")

    class FusionTrainer:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch required for learned fusion")
