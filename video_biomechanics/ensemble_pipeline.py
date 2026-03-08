"""
Ensemble pose estimation pipeline.

Combines multiple pose estimation methods with intelligent fusion
for maximum accuracy.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass

from pose_estimators import (
    PoseEstimator3D, Pose3DResult,
    YOLOLiftingEstimator, MotionBERTEstimator, TriangulationEstimator
)
from fusion import FusionPipeline, create_default_pipeline
from joint_angles_3d import JointAngleCalculator3D, JointAngles3D


@dataclass
class EnsembleResult:
    """Result from ensemble pipeline."""
    # Fused 3D poses
    poses_3d: List[np.ndarray]

    # Per-method poses (for analysis)
    method_poses: Dict[str, List[np.ndarray]]

    # Fusion statistics
    fusion_stats: Dict

    # Timestamps
    timestamps: List[float]

    # FPS
    fps: float


class EnsemblePosePipeline:
    """
    Ensemble pipeline combining multiple pose estimation methods.

    Usage:
        pipeline = EnsemblePosePipeline()
        results = pipeline.process_videos(['side.mp4', 'back.mp4'])
        joint_angles = pipeline.calculate_angles(results)
    """

    AVAILABLE_METHODS = {
        'yolo_lifting': YOLOLiftingEstimator,
        'motionbert': MotionBERTEstimator,
        'triangulation': TriangulationEstimator,
    }

    def __init__(self,
                 methods: Optional[List[str]] = None,
                 fusion_model_path: Optional[str] = None,
                 fps: float = 30.0,
                 camera_distances: Optional[List[float]] = None,
                 camera_angles: Optional[List[float]] = None):
        """
        Initialize ensemble pipeline.

        Args:
            methods: List of methods to use. Default: ['yolo_lifting', 'motionbert']
            fusion_model_path: Path to trained fusion network weights
            fps: Video frame rate
            camera_distances: Camera distances for triangulation (meters)
            camera_angles: Camera angles for triangulation (degrees)
        """
        if methods is None:
            methods = ['yolo_lifting', 'motionbert']

        self.method_names = methods
        self.estimators: Dict[str, PoseEstimator3D] = {}
        self.fps = fps
        self.camera_distances = camera_distances
        self.camera_angles = camera_angles

        # Initialize estimators
        for method in methods:
            if method not in self.AVAILABLE_METHODS:
                raise ValueError(f"Unknown method: {method}. "
                                 f"Available: {list(self.AVAILABLE_METHODS.keys())}")

            # Triangulation keeps absolute world coordinates for fusion anchor
            if method == 'triangulation':
                self.estimators[method] = self.AVAILABLE_METHODS[method](keep_absolute=True)
            else:
                self.estimators[method] = self.AVAILABLE_METHODS[method]()

        # Initialize fusion pipeline
        self.fusion = create_default_pipeline(
            fps=fps,
            learned_fusion_path=fusion_model_path
        )

        # Angle calculator
        self.angle_calculator = JointAngleCalculator3D()

        self._initialized = False

    def initialize(self):
        """Initialize all estimators."""
        print("Initializing pose estimators...")
        for name, estimator in self.estimators.items():
            print(f"  Loading {name}...")
            estimator.initialize()

        self._initialized = True

    def process_videos(self,
                       video_paths: List[str],
                       max_frames: Optional[int] = None) -> EnsembleResult:
        """
        Process videos through ensemble pipeline.

        Args:
            video_paths: List of video paths (1-2 videos)
            max_frames: Maximum frames to process

        Returns:
            EnsembleResult with fused poses
        """
        if not self._initialized:
            self.initialize()

        import cv2

        # Get video info
        cap = cv2.VideoCapture(video_paths[0])
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        if max_frames:
            total_frames = min(total_frames, max_frames)

        print(f"\nProcessing {len(video_paths)} video(s) with {len(self.estimators)} methods")
        print(f"  FPS: {self.fps:.1f}")
        print(f"  Frames: {total_frames}")

        # Run each method
        method_results: Dict[str, List[Pose3DResult]] = {}

        for name, estimator in self.estimators.items():
            print(f"\nRunning {name}...")

            if name == 'triangulation' and len(video_paths) >= 2:
                # Set up camera parameters for triangulation
                if self.camera_distances:
                    estimator.set_camera_params(
                        video_paths,
                        self.camera_distances,
                        self.camera_angles
                    )
                results = estimator.estimate_video_multiview(video_paths, max_frames)
            else:
                # Single-view methods
                results = estimator.estimate_video(video_paths[0], max_frames)

            method_results[name] = results
            print(f"  Got {len(results)} poses")

        # Align frame counts
        n_frames = min(len(results) for results in method_results.values())
        print(f"\nAligning to {n_frames} frames")

        # Extract poses and confidences for fusion
        all_poses = []
        all_confidences = []

        for name in self.method_names:
            results = method_results[name][:n_frames]
            poses = [r.joints_3d for r in results]
            confs = [r.confidences for r in results]
            all_poses.append(poses)
            all_confidences.append(confs)

        # Get timestamps
        timestamps = [method_results[self.method_names[0]][i].timestamp
                      for i in range(n_frames)]

        # Single method mode - skip fusion, use raw poses
        if len(self.method_names) == 1:
            print(f"\nSingle method mode: using raw {self.method_names[0]} poses")
            fused_poses = all_poses[0]
            fusion_stats = {'single_method': self.method_names[0]}
        else:
            # Fuse poses from multiple methods
            print("\nFusing poses...")
            fused_poses = self.fusion.fuse_sequence(
                all_poses, all_confidences, self.method_names
            )

            # Calculate fusion stats from first frame
            if all_poses[0]:
                frame_poses = [all_poses[m][0] for m in range(len(self.method_names))]
                frame_confs = [all_confidences[m][0] for m in range(len(self.method_names))]
                fusion_stats = self.fusion.get_fusion_stats(frame_poses, frame_confs)
            else:
                fusion_stats = {}

            print(f"  Mean method agreement: {fusion_stats.get('mean_agreement', 'N/A'):.2f}")

        # Prepare method poses for output
        method_poses_out = {
            name: [r.joints_3d for r in results[:n_frames]]
            for name, results in method_results.items()
        }

        return EnsembleResult(
            poses_3d=fused_poses,
            method_poses=method_poses_out,
            fusion_stats=fusion_stats,
            timestamps=timestamps,
            fps=self.fps
        )

    def calculate_angles(self,
                         results: EnsembleResult) -> List[JointAngles3D]:
        """
        Calculate joint angles from fused poses.

        Args:
            results: EnsembleResult from process_videos()

        Returns:
            List of JointAngles3D, one per frame
        """
        joint_angles = []

        for i, (pose, timestamp) in enumerate(zip(results.poses_3d, results.timestamps)):
            angles = self.angle_calculator.calculate(
                pose,
                timestamp=timestamp,
                frame_number=i
            )
            joint_angles.append(angles)

        return joint_angles

    def process_and_analyze(self,
                            video_paths: List[str],
                            max_frames: Optional[int] = None) -> Tuple[EnsembleResult, List[JointAngles3D]]:
        """
        Process videos and calculate joint angles.

        Convenience method combining process_videos() and calculate_angles().

        Returns:
            Tuple of (EnsembleResult, List[JointAngles3D])
        """
        results = self.process_videos(video_paths, max_frames)
        joint_angles = self.calculate_angles(results)
        return results, joint_angles

    def compare_methods(self,
                        results: EnsembleResult) -> Dict:
        """
        Compare outputs from different methods.

        Returns statistics about method agreement and differences.
        """
        stats = {
            'n_frames': len(results.poses_3d),
            'methods': list(results.method_poses.keys()),
            'per_joint_agreement': {},
            'method_vs_fused': {},
        }

        # Calculate per-joint method agreement
        n_joints = 17
        joint_names = [
            'pelvis', 'right_hip', 'right_knee', 'right_ankle',
            'left_hip', 'left_knee', 'left_ankle',
            'spine', 'neck', 'head', 'head_top',
            'left_shoulder', 'left_elbow', 'left_wrist',
            'right_shoulder', 'right_elbow', 'right_wrist'
        ]

        for j in range(n_joints):
            # Get all method predictions for this joint
            all_preds = []
            for method, poses in results.method_poses.items():
                joint_trajectory = np.array([p[j] for p in poses])
                all_preds.append(joint_trajectory)

            all_preds = np.stack(all_preds)  # (n_methods, n_frames, 3)

            # Calculate std across methods
            std = np.std(all_preds, axis=0).mean()
            stats['per_joint_agreement'][joint_names[j]] = float(std)

        # Calculate each method's difference from fused
        fused = np.array(results.poses_3d)

        for method, poses in results.method_poses.items():
            poses = np.array(poses)
            diff = np.linalg.norm(poses - fused, axis=-1).mean()
            stats['method_vs_fused'][method] = float(diff)

        return stats


def main():
    """Example usage."""
    import argparse

    parser = argparse.ArgumentParser(description='Ensemble pose estimation')
    parser.add_argument('videos', nargs='+', help='Video path(s)')
    parser.add_argument('--methods', nargs='+',
                        default=['yolo_lifting', 'motionbert'],
                        help='Methods to use')
    parser.add_argument('--max-frames', type=int, default=None,
                        help='Maximum frames to process')
    parser.add_argument('--output', '-o', default='./output',
                        help='Output directory')

    args = parser.parse_args()

    print("=" * 60)
    print("ENSEMBLE POSE ESTIMATION PIPELINE")
    print("=" * 60)

    pipeline = EnsemblePosePipeline(methods=args.methods)
    results, joint_angles = pipeline.process_and_analyze(
        args.videos, max_frames=args.max_frames
    )

    # Compare methods
    comparison = pipeline.compare_methods(results)
    print("\nMethod comparison:")
    for method, diff in comparison['method_vs_fused'].items():
        print(f"  {method}: {diff*100:.2f} cm avg diff from fused")

    # Export results
    import pandas as pd
    from pathlib import Path

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Export joint angles
    records = []
    for angles in joint_angles:
        record = {'timestamp': angles.timestamp, 'frame': angles.frame}
        for attr in dir(angles):
            if not attr.startswith('_') and attr not in ['timestamp', 'frame']:
                val = getattr(angles, attr)
                if isinstance(val, (int, float)) and val is not None:
                    record[attr] = val
        records.append(record)

    df = pd.DataFrame(records)
    df.to_csv(output_dir / 'joint_angles.csv', index=False)

    print(f"\nResults exported to {output_dir}")
    print("Done!")


if __name__ == '__main__':
    main()
