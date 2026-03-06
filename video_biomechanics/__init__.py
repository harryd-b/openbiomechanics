"""
Video Biomechanics Pipeline

Extract biomechanical data from video of baseball hitters.
Based on conventions from the OpenBiomechanics Project.

Quick Start (Web App):
    python -m video_biomechanics.app
    # Then open http://localhost:8050 and upload videos

Single Video:
    from video_biomechanics import VideoBiomechanicsPipeline

    pipeline = VideoBiomechanicsPipeline(bats='R', use_3d=True)
    results = pipeline.process_video('swing.mp4')

Multi-View (2 cameras):
    from video_biomechanics import MultiViewPipeline

    pipeline = MultiViewPipeline(bats='R')
    results = pipeline.process_videos(['side.mp4', 'front.mp4'])

    # Access results
    print(results['metrics'].hip_shoulder_separation)
    print(results['timeseries_df'].head())
"""

# Core pipelines
from .pipeline import VideoBiomechanicsPipeline
from .multiview import MultiViewPipeline, process_multiview, CameraParams

# Pose estimation
from .pose_estimation import (
    PoseEstimator,
    PoseFrame,
    Keypoints,
    extract_joint_positions
)

# 2D joint angles
from .joint_angles import (
    JointAngleCalculator,
    JointAngles,
    calculate_angular_velocity,
    calculate_angle_3points,
    calculate_elbow_flexion,
    calculate_knee_flexion
)

# 3D lifting and angles
from .lifting_3d import (
    VideoPose3DLifter,
    Pose3D,
    H36MJoints,
    convert_yolo_to_h36m
)
from .joint_angles_3d import (
    JointAngleCalculator3D,
    JointAngles3D
)

# Event detection
from .event_detection import (
    SwingEventDetector,
    SwingEvents,
    detect_swing_events_simple
)

# Hitting metrics
from .hitting_metrics import (
    HittingMetricsCalculator,
    HittingPOIMetrics,
    OBP_HITTING_POI_FIELDS
)

# Conventions and definitions
from .conventions import (
    HITTING_JOINT_CONVENTIONS,
    HittingEvents,
    PitchingEvents,
    Units,
    get_side_mapping,
    mph_to_ms,
    ms_to_mph,
    deg_to_rad,
    rad_to_deg
)

# Validation
from .validate import (
    OBPDataLoader,
    PipelineValidator,
    ValidationResult,
    compare_with_obp
)

# Visualization
from .visualize import (
    plot_joint_angles,
    plot_angular_velocities,
    plot_pose_skeleton,
    create_summary_report
)

__version__ = '0.3.0'
__all__ = [
    # Main pipelines
    'VideoBiomechanicsPipeline',
    'MultiViewPipeline',
    'process_multiview',
    'CameraParams',

    # Pose estimation
    'PoseEstimator',
    'PoseFrame',
    'Keypoints',

    # Joint angles (2D and 3D)
    'JointAngleCalculator',
    'JointAngles',
    'JointAngleCalculator3D',
    'JointAngles3D',

    # 3D lifting
    'VideoPose3DLifter',
    'Pose3D',
    'H36MJoints',

    # Events
    'SwingEventDetector',
    'SwingEvents',

    # Metrics
    'HittingMetricsCalculator',
    'HittingPOIMetrics',

    # Conventions
    'HITTING_JOINT_CONVENTIONS',
    'HittingEvents',
    'Units',

    # Validation
    'OBPDataLoader',
    'PipelineValidator',
    'compare_with_obp',

    # Visualization
    'plot_joint_angles',
    'create_summary_report',
]


def run_app(port: int = 8050, debug: bool = False):
    """Launch the web application for video upload and analysis."""
    from .app import run_app as _run_app
    _run_app(debug=debug, port=port)
