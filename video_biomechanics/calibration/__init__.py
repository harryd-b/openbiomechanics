"""Camera calibration module for video biomechanics."""

from .plate_calibrator import (
    PlateCalibrator,
    CalibrationConfig,
    coco_to_h36m,
    JOINT_NAMES,
    PLATE_WORLD_COORDS,
)

from .robust_triangulation import (
    SkeletonConstrainedTriangulator,
    LiftingFusionTriangulator,
    HybridTriangulator,
    triangulate_robust,
    BONE_CONSTRAINTS,
    TriangulationResult,
)

__all__ = [
    'PlateCalibrator',
    'CalibrationConfig',
    'coco_to_h36m',
    'JOINT_NAMES',
    'PLATE_WORLD_COORDS',
    'SkeletonConstrainedTriangulator',
    'LiftingFusionTriangulator',
    'HybridTriangulator',
    'triangulate_robust',
    'BONE_CONSTRAINTS',
    'TriangulationResult',
]
