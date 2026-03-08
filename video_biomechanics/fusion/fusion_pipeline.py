"""Main fusion pipeline combining all strategies."""

import numpy as np
from typing import List, Dict, Optional
from pathlib import Path

from .outlier_rejection import OutlierRejector
from .weighted_average import WeightedAverageFusion
from .biomechanical_constraints import BiomechanicalConstraints
from .temporal_filter import TemporalKalmanFilter


class FusionPipeline:
    """
    Complete fusion pipeline for combining multiple 3D pose estimates.

    Applies fusion strategies in order:
    1. Outlier rejection
    2. Confidence-weighted average
    3. Biomechanical constraints
    4. Learned fusion (optional)
    5. Temporal filtering
    """

    def __init__(self,
                 outlier_threshold_m: float = 0.10,
                 method_weights: Optional[Dict[str, float]] = None,
                 use_biomech_constraints: bool = True,
                 use_temporal_filter: bool = True,
                 learned_fusion_path: Optional[str] = None,
                 fps: float = 30.0):
        """
        Initialize fusion pipeline.

        Args:
            outlier_threshold_m: Outlier rejection threshold (meters)
            method_weights: Per-method reliability weights
            use_biomech_constraints: Apply biomechanical constraints
            use_temporal_filter: Apply temporal smoothing
            learned_fusion_path: Path to trained fusion network weights
            fps: Video frame rate for temporal filter
        """
        self.outlier_rejector = OutlierRejector(threshold_m=outlier_threshold_m)
        self.weighted_fusion = WeightedAverageFusion(method_weights=method_weights)
        self.biomech = BiomechanicalConstraints() if use_biomech_constraints else None
        self.temporal_filter = TemporalKalmanFilter(fps=fps) if use_temporal_filter else None

        # Load learned fusion if provided
        self.learned_fusion = None
        if learned_fusion_path and Path(learned_fusion_path).exists():
            self.learned_fusion = self._load_learned_fusion(learned_fusion_path)

    def _load_learned_fusion(self, path: str):
        """Load trained fusion network."""
        try:
            import torch
            from .learned_fusion import FusionNetwork

            # Determine number of methods from saved model
            checkpoint = torch.load(path, map_location='cpu')
            n_methods = checkpoint.get('n_methods', 3)

            model = FusionNetwork(n_methods=n_methods)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()

            print(f"Loaded learned fusion model from {path}")
            return model

        except Exception as e:
            print(f"Failed to load learned fusion: {e}")
            return None

    def fuse_frame(self,
                   poses: List[np.ndarray],
                   confidences: List[np.ndarray],
                   method_names: Optional[List[str]] = None) -> np.ndarray:
        """
        Fuse a single frame from multiple methods.

        Args:
            poses: List of (17, 3) pose arrays, one per method
            confidences: List of (17,) confidence arrays
            method_names: Optional method identifiers

        Returns:
            Fused (17, 3) pose
        """
        # Step 1: Outlier rejection
        poses, valid_mask = self.outlier_rejector.reject_outliers(poses, confidences)

        # Step 2: Weighted average
        fused = self.weighted_fusion.fuse(poses, confidences, valid_mask, method_names)

        # Step 3: Learned fusion (if available)
        if self.learned_fusion is not None:
            fused = self._apply_learned_fusion(poses, confidences, fused)

        # Step 4: Biomechanical constraints
        if self.biomech is not None:
            fused = self.biomech.apply_constraints(fused)

        return fused

    def _apply_learned_fusion(self,
                              poses: List[np.ndarray],
                              confidences: List[np.ndarray],
                              initial_fused: np.ndarray) -> np.ndarray:
        """Apply learned fusion refinement."""
        import torch

        # Stack inputs
        poses_tensor = torch.tensor(np.stack(poses), dtype=torch.float32).unsqueeze(0)
        conf_tensor = torch.tensor(np.stack(confidences), dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            refined = self.learned_fusion(poses_tensor, conf_tensor)

        return refined.squeeze(0).numpy()

    def fuse_sequence(self,
                      all_poses: List[List[np.ndarray]],
                      all_confidences: List[List[np.ndarray]],
                      method_names: Optional[List[str]] = None) -> List[np.ndarray]:
        """
        Fuse a sequence of frames from multiple methods.

        Args:
            all_poses: all_poses[method_idx][frame_idx] = (17, 3) array
            all_confidences: all_confidences[method_idx][frame_idx] = (17,) array
            method_names: Optional method identifiers

        Returns:
            List of fused (17, 3) poses
        """
        n_methods = len(all_poses)
        n_frames = min(len(poses) for poses in all_poses)

        # Fuse each frame
        fused_sequence = []

        for frame_idx in range(n_frames):
            frame_poses = [all_poses[m][frame_idx] for m in range(n_methods)]
            frame_confs = [all_confidences[m][frame_idx] for m in range(n_methods)]

            fused = self.fuse_frame(frame_poses, frame_confs, method_names)
            fused_sequence.append(fused)

        # Step 5: Temporal filtering
        if self.temporal_filter is not None:
            # Extract confidences for temporal filter
            avg_confidences = []
            for frame_idx in range(n_frames):
                frame_confs = [all_confidences[m][frame_idx] for m in range(n_methods)]
                avg_conf = np.mean(frame_confs, axis=0)
                avg_confidences.append(avg_conf)

            fused_sequence = self.temporal_filter.smooth_sequence(
                fused_sequence, avg_confidences
            )

        return fused_sequence

    def get_fusion_stats(self,
                         poses: List[np.ndarray],
                         confidences: List[np.ndarray]) -> Dict:
        """Get statistics about the fusion process."""
        _, valid_mask = self.outlier_rejector.reject_outliers(poses, confidences)

        outlier_stats = self.outlier_rejector.get_outlier_stats(poses, valid_mask)

        # Method agreement statistics
        n_joints = poses[0].shape[0]
        agreement = np.zeros(n_joints)

        for j in range(n_joints):
            joint_pos = np.array([p[j] for p in poses])
            spread = np.std(joint_pos, axis=0).mean()
            agreement[j] = 1.0 / (spread + 0.01)  # Higher = more agreement

        agreement = agreement / agreement.max()

        return {
            'outlier_stats': outlier_stats,
            'per_joint_agreement': agreement.tolist(),
            'mean_agreement': float(agreement.mean()),
        }


def create_default_pipeline(fps: float = 30.0,
                            learned_fusion_path: Optional[str] = None) -> FusionPipeline:
    """
    Create a fusion pipeline with sensible defaults.

    Args:
        fps: Video frame rate
        learned_fusion_path: Optional path to trained fusion weights

    Returns:
        Configured FusionPipeline
    """
    # Default method weights (can be tuned based on empirical results)
    method_weights = {
        'yolo_lifting': 1.0,
        'motionbert': 1.2,  # Slightly higher - good temporal consistency
        'triangulation': 1.5,  # Highest when calibrated
    }

    return FusionPipeline(
        outlier_threshold_m=0.10,
        method_weights=method_weights,
        use_biomech_constraints=True,
        use_temporal_filter=True,
        learned_fusion_path=learned_fusion_path,
        fps=fps
    )
