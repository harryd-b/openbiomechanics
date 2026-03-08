"""
Pose estimation methods for 3D human pose extraction.

Available estimators:
- YOLOLiftingEstimator: YOLOv8 2D pose + VideoPose3D lifting (fast, default)
- MotionBERTEstimator: Transformer-based direct 3D estimation (accurate)
- TriangulationEstimator: Multi-view geometric triangulation (most accurate when calibrated)
"""

from .base import PoseEstimator3D, Pose3DResult
from .yolo_lifting import YOLOLiftingEstimator
from .motionbert import MotionBERTEstimator
from .triangulation import TriangulationEstimator

__all__ = [
    'PoseEstimator3D',
    'Pose3DResult',
    'YOLOLiftingEstimator',
    'MotionBERTEstimator',
    'TriangulationEstimator',
]
