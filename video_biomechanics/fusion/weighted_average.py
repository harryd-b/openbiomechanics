"""Confidence-weighted averaging for pose fusion."""

import numpy as np
from typing import List, Optional


class WeightedAverageFusion:
    """
    Combine multiple pose estimates using confidence-weighted averaging.

    Each method's contribution is weighted by its confidence score,
    optionally multiplied by method-specific reliability weights.
    """

    def __init__(self,
                 method_weights: Optional[dict] = None,
                 confidence_power: float = 1.0):
        """
        Initialize weighted fusion.

        Args:
            method_weights: Per-method global weights {'method_name': weight}
            confidence_power: Exponent for confidence (>1 emphasizes high confidence)
        """
        self.method_weights = method_weights or {}
        self.confidence_power = confidence_power

    def fuse(self,
             poses: List[np.ndarray],
             confidences: List[np.ndarray],
             valid_mask: np.ndarray,
             method_names: Optional[List[str]] = None) -> np.ndarray:
        """
        Fuse multiple pose estimates.

        Args:
            poses: List of (17, 3) arrays
            confidences: List of (17,) arrays
            valid_mask: (n_methods, 17) bool array
            method_names: Optional method identifiers

        Returns:
            fused_pose: (17, 3) array
        """
        n_joints = poses[0].shape[0]
        fused = np.zeros((n_joints, 3))

        for j in range(n_joints):
            weights = []
            positions = []

            for m, (pose, conf) in enumerate(zip(poses, confidences)):
                if not valid_mask[m, j]:
                    continue

                # Base weight from confidence
                w = conf[j] ** self.confidence_power

                # Apply method-specific weight if available
                if method_names and method_names[m] in self.method_weights:
                    w *= self.method_weights[method_names[m]]

                weights.append(w)
                positions.append(pose[j])

            if weights:
                weights = np.array(weights)
                weights = weights / weights.sum()  # Normalize
                fused[j] = np.average(positions, weights=weights, axis=0)
            else:
                # Fallback: median of all methods
                fused[j] = np.median([p[j] for p in poses], axis=0)

        return fused

    def fuse_with_uncertainty(self,
                              poses: List[np.ndarray],
                              confidences: List[np.ndarray],
                              valid_mask: np.ndarray) -> tuple:
        """
        Fuse poses and estimate uncertainty.

        Returns:
            Tuple of (fused_pose (17, 3), uncertainty (17,))
        """
        fused = self.fuse(poses, confidences, valid_mask)

        # Estimate uncertainty as weighted std of predictions
        n_joints = poses[0].shape[0]
        uncertainty = np.zeros(n_joints)

        for j in range(n_joints):
            valid_poses = [p[j] for m, p in enumerate(poses) if valid_mask[m, j]]

            if len(valid_poses) > 1:
                # Distance of each prediction from fused result
                distances = [np.linalg.norm(p - fused[j]) for p in valid_poses]
                uncertainty[j] = np.mean(distances)
            else:
                uncertainty[j] = 0.1  # Default uncertainty

        return fused, uncertainty
