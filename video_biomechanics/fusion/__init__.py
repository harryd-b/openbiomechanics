"""
Fusion module for combining multiple 3D pose estimates.

Provides multiple fusion strategies:
- OutlierRejector: Remove joints where methods disagree
- WeightedAverageFusion: Combine based on confidence scores
- BiomechanicalConstraints: Enforce anatomical plausibility
- TemporalFilter: Smooth trajectories over time
- LearnedFusion: Neural network trained on ground truth
"""

from .outlier_rejection import OutlierRejector
from .weighted_average import WeightedAverageFusion
from .biomechanical_constraints import BiomechanicalConstraints
from .temporal_filter import TemporalKalmanFilter
from .fusion_pipeline import FusionPipeline, create_default_pipeline

__all__ = [
    'OutlierRejector',
    'WeightedAverageFusion',
    'BiomechanicalConstraints',
    'TemporalKalmanFilter',
    'FusionPipeline',
    'create_default_pipeline',
]
