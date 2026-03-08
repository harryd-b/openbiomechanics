"""Outlier rejection for multi-method pose fusion."""

import numpy as np
from typing import List, Tuple


class OutlierRejector:
    """
    Reject outlier joint predictions that disagree with majority.

    For each joint, if one method's prediction is far from the
    median of all methods, it's excluded from fusion.
    """

    def __init__(self,
                 threshold_m: float = 0.10,
                 min_methods: int = 2):
        """
        Initialize outlier rejector.

        Args:
            threshold_m: Distance threshold in meters
            min_methods: Minimum methods that must agree
        """
        self.threshold = threshold_m
        self.min_methods = min_methods

    def reject_outliers(self,
                        poses: List[np.ndarray],
                        confidences: List[np.ndarray]) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Identify and mask outlier predictions.

        Args:
            poses: List of (17, 3) arrays from each method
            confidences: List of (17,) confidence arrays

        Returns:
            Tuple of:
            - Original poses (unchanged)
            - valid_mask: (n_methods, 17) bool array, True = valid
        """
        n_methods = len(poses)
        n_joints = poses[0].shape[0]

        valid_mask = np.ones((n_methods, n_joints), dtype=bool)

        for j in range(n_joints):
            # Get all predictions for this joint
            joint_positions = np.array([p[j] for p in poses])

            # Calculate median position
            median_pos = np.median(joint_positions, axis=0)

            # Check each method's distance from median
            for m in range(n_methods):
                dist = np.linalg.norm(joint_positions[m] - median_pos)

                if dist > self.threshold:
                    valid_mask[m, j] = False

            # Ensure minimum methods remain valid
            valid_count = valid_mask[:, j].sum()
            if valid_count < self.min_methods:
                # Reset to keep all methods (no clear outlier)
                valid_mask[:, j] = True

        return poses, valid_mask

    def get_outlier_stats(self,
                          poses: List[np.ndarray],
                          valid_mask: np.ndarray) -> dict:
        """Get statistics about rejected outliers."""
        n_methods, n_joints = valid_mask.shape
        total_predictions = n_methods * n_joints
        rejected = (~valid_mask).sum()

        per_method_rejected = (~valid_mask).sum(axis=1)
        per_joint_rejected = (~valid_mask).sum(axis=0)

        return {
            'total_predictions': total_predictions,
            'rejected_count': rejected,
            'rejection_rate': rejected / total_predictions,
            'per_method_rejected': per_method_rejected.tolist(),
            'per_joint_rejected': per_joint_rejected.tolist(),
        }
