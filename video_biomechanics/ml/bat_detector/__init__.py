"""Bat endpoint detection module.

This module provides tools for detecting baseball bat endpoints (knob and tip)
in video frames. The detector uses a ResNet-based CNN trained on annotated
swing videos plus synthetic data augmentation.

Usage:
    # First, prepare training data:
    # 1. Extract frames: python extract_frames.py
    # 2. Annotate (uses shared annotator):
    #    cd ../plate_detector && python annotate.py --mode bat
    # 3. Generate synthetic data: python augment_annotations.py
    # 4. Train: python train.py

    # Then use for inference:
    from ml.bat_detector import BatDetector

    detector = BatDetector("checkpoints/best_bat_model.pth")
    endpoints, confidence = detector.detect(frame)
    bat_length_pixels = detector.compute_bat_length(endpoints)
"""

from .model import BatKeypointModel, create_model, BatKeypointLoss
from .annotations import BatAnnotation, BatAnnotationStore
from .detect import BatDetector, visualize_detection

__all__ = [
    'BatKeypointModel',
    'create_model',
    'BatKeypointLoss',
    'BatAnnotation',
    'BatAnnotationStore',
    'BatDetector',
    'visualize_detection',
]
