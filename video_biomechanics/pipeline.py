"""
Main pipeline for extracting biomechanics from video.

This connects the pose estimation, 3D lifting, joint angle calculation,
event detection, and metrics extraction.

Supports two modes:
- 2D mode: Fast, but limited to image-plane angles
- 3D mode: Uses lifting model for full 3D biomechanics
"""

import numpy as np
import pandas as pd
from typing import Optional, List
from pathlib import Path
from scipy.signal import savgol_filter

from pose_estimation import PoseEstimator, extract_joint_positions, PoseFrame
from joint_angles import JointAngleCalculator, JointAngles
from joint_angles_3d import JointAngleCalculator3D, JointAngles3D
from lifting_3d import VideoPose3DLifter, Pose3D
from event_detection import SwingEventDetector, SwingEvents
from hitting_metrics import HittingMetricsCalculator, HittingPOIMetrics
from conventions import get_side_mapping


class VideoBiomechanicsPipeline:
    """
    End-to-end pipeline for extracting biomechanics from video.

    Modes:
    - 2D: Fast processing, limited to image-plane angles
    - 3D: Uses 2D->3D lifting for full biomechanics including rotations
    """

    def __init__(self,
                 pose_model: str = 'yolov8m-pose.pt',
                 bats: str = 'R',
                 use_3d: bool = True,
                 lifting_model_path: Optional[str] = None):
        """
        Initialize the pipeline.

        Args:
            pose_model: YOLOv8 pose model to use
            bats: Batting handedness ('L' or 'R')
            use_3d: Whether to use 3D lifting (slower but more accurate)
            lifting_model_path: Path to pretrained lifting model weights
        """
        self.pose_estimator = PoseEstimator(model_name=pose_model)
        self.bats = bats
        self.use_3d = use_3d
        self.fps = None

        if use_3d:
            self.lifter = VideoPose3DLifter(model_path=lifting_model_path)
            self.angle_calculator = JointAngleCalculator3D()
        else:
            self.lifter = None
            self.angle_calculator = JointAngleCalculator(use_3d=False)

    def process_video(self,
                      video_path: str,
                      max_frames: Optional[int] = None) -> dict:
        """
        Process a video and extract biomechanical data.

        Args:
            video_path: Path to input video
            max_frames: Maximum frames to process (None = all)

        Returns:
            Dictionary containing:
            - 'poses': List of PoseFrame objects (2D)
            - 'poses_3d': List of Pose3D objects (if use_3d=True)
            - 'joint_angles': List of JointAngles/JointAngles3D objects
            - 'events': SwingEvents object
            - 'metrics': HittingPOIMetrics object
            - 'timeseries_df': DataFrame with time-series data
            - 'fps': Video frame rate
        """
        import cv2

        # Get video FPS
        cap = cv2.VideoCapture(video_path)
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        print(f"Processing video: {video_path}")
        print(f"  FPS: {self.fps:.1f}")
        print(f"  Total frames: {total_frames}")
        print(f"  Mode: {'3D' if self.use_3d else '2D'}")

        results = {'fps': self.fps}

        # Step 1: Extract 2D poses
        print("\nStep 1/5: Extracting 2D poses...")
        poses = self.pose_estimator.process_video(video_path, max_frames=max_frames)
        results['poses'] = poses

        if not poses:
            raise ValueError("No poses detected in video")

        print(f"  Detected poses in {len(poses)} frames")

        # Step 2: Lift to 3D (if enabled)
        if self.use_3d:
            print("\nStep 2/5: Lifting to 3D...")
            poses_2d = [p.keypoints for p in poses]
            timestamps = [p.timestamp for p in poses]
            poses_3d = self.lifter.lift_sequence(poses_2d, timestamps)
            results['poses_3d'] = poses_3d
            print(f"  Lifted {len(poses_3d)} poses to 3D")
        else:
            print("\nStep 2/5: Skipping 3D lifting (2D mode)")
            results['poses_3d'] = None

        # Step 3: Calculate joint angles
        print("\nStep 3/5: Calculating joint angles...")
        if self.use_3d and results['poses_3d']:
            joint_angles = []
            for i, pose_3d in enumerate(results['poses_3d']):
                angles = self.angle_calculator.calculate(
                    pose_3d.joints_3d,
                    timestamp=pose_3d.timestamp,
                    frame_number=i
                )
                joint_angles.append(angles)
        else:
            joint_angles = []
            for pose in poses:
                keypoints = extract_joint_positions(pose)
                angles = self.angle_calculator.calculate_from_keypoints(
                    keypoints, timestamp=pose.timestamp
                )
                joint_angles.append(angles)

        results['joint_angles'] = joint_angles
        print(f"  Calculated angles for {len(joint_angles)} frames")

        # Step 4: Detect events
        print("\nStep 4/5: Detecting swing events...")
        events = self._detect_events(poses, joint_angles)
        results['events'] = events

        print("  Detected events:")
        if events.first_move:
            print(f"    First move: {events.first_move:.3f}s (conf: {events.first_move_confidence:.2f})")
        if events.foot_plant:
            print(f"    Foot plant: {events.foot_plant:.3f}s (conf: {events.foot_plant_confidence:.2f})")
        if events.contact:
            print(f"    Contact: {events.contact:.3f}s (conf: {events.contact_confidence:.2f})")

        # Step 5: Calculate POI metrics
        print("\nStep 5/5: Calculating POI metrics...")
        metrics = self._calculate_metrics(joint_angles, events)
        results['metrics'] = metrics

        # Build time-series DataFrame
        results['timeseries_df'] = self._build_timeseries_df(joint_angles)

        return results

    def _detect_events(self,
                       poses: List[PoseFrame],
                       joint_angles: List) -> SwingEvents:
        """Detect swing events from pose data."""
        # Build position array for event detector
        positions = np.array([p.keypoints[:, :2] for p in poses])

        detector = SwingEventDetector(fps=self.fps, bats=self.bats)
        events = detector.detect_events(positions)

        return events

    def _calculate_metrics(self,
                           joint_angles: List,
                           events: SwingEvents) -> HittingPOIMetrics:
        """Calculate POI metrics from joint angles and events."""
        metrics_calc = HittingMetricsCalculator(fps=self.fps, bats=self.bats)

        # Convert SwingEvents from event_detection to hitting_metrics format
        from hitting_metrics import SwingEvents as HMSwingEvents
        hm_events = HMSwingEvents(
            first_move=events.first_move,
            foot_plant=events.foot_plant,
            contact=events.contact
        )

        # Calculate metrics
        # Need to adapt based on whether we have 2D or 3D angles
        if isinstance(joint_angles[0], JointAngles3D):
            metrics = self._calculate_3d_metrics(joint_angles, events)
        else:
            metrics = metrics_calc.calculate_metrics(joint_angles, hm_events)

        return metrics

    def _calculate_3d_metrics(self,
                              angles_3d: List[JointAngles3D],
                              events: SwingEvents) -> HittingPOIMetrics:
        """Calculate metrics from 3D joint angles."""
        metrics = HittingPOIMetrics()

        if not angles_3d:
            return metrics

        timestamps = np.array([a.timestamp for a in angles_3d])

        def get_at_time(t: Optional[float], attr: str):
            if t is None:
                return None
            idx = np.argmin(np.abs(timestamps - t))
            return getattr(angles_3d[idx], attr)

        def get_max(attr: str, start_t: float = None, end_t: float = None):
            values = []
            for a in angles_3d:
                if start_t and a.timestamp < start_t:
                    continue
                if end_t and a.timestamp > end_t:
                    continue
                val = getattr(a, attr)
                if val is not None:
                    values.append(val)
            return max(values) if values else None

        # Determine lead/rear based on batting side
        lead_side = 'left' if self.bats == 'R' else 'right'
        rear_side = 'right' if self.bats == 'R' else 'left'

        # Knee angles
        metrics.lead_knee_launchpos_x = get_at_time(
            events.foot_plant, f'{lead_side}_knee_flexion'
        )
        metrics.lead_knee_stride_max_x = get_max(
            f'{lead_side}_knee_flexion', events.first_move, events.foot_plant
        )

        # Elbow angles
        metrics.rear_elbow_fm_x = get_at_time(
            events.first_move, f'{rear_side}_elbow_flexion'
        )
        metrics.rear_elbow_launchpos_x = get_at_time(
            events.foot_plant, f'{rear_side}_elbow_flexion'
        )

        # Pelvis angles at foot plant
        metrics.pelvis_angle_fp_x = get_at_time(events.foot_plant, 'pelvis_tilt')
        metrics.pelvis_angle_fp_y = get_at_time(events.foot_plant, 'pelvis_obliquity')
        metrics.pelvis_angle_fp_z = get_at_time(events.foot_plant, 'pelvis_rotation')

        # Torso angles at foot plant
        metrics.torso_angle_fp_x = get_at_time(events.foot_plant, 'torso_flexion')
        metrics.torso_angle_fp_y = get_at_time(events.foot_plant, 'torso_lateral_tilt')
        metrics.torso_angle_fp_z = get_at_time(events.foot_plant, 'torso_rotation')

        # X-factor (hip-shoulder separation)
        metrics.x_factor_fp_z = get_at_time(events.foot_plant, 'hip_shoulder_separation')
        metrics.torso_pelvis_stride_max_z = get_max(
            'hip_shoulder_separation', events.first_move, events.foot_plant
        )

        # Angular velocities
        if len(angles_3d) > 1:
            dt = 1.0 / self.fps

            # Pelvis rotation velocity
            pelvis_rot = [a.pelvis_rotation or 0 for a in angles_3d]
            pelvis_velo = np.gradient(pelvis_rot, dt)
            metrics.pelvis_angular_velocity_seq_max = float(np.max(np.abs(pelvis_velo)))

            # Torso rotation velocity
            torso_rot = [a.torso_rotation or 0 for a in angles_3d]
            torso_velo = np.gradient(torso_rot, dt)
            metrics.torso_angular_velocity_seq_max = float(np.max(np.abs(torso_velo)))

        return metrics

    def _build_timeseries_df(self, joint_angles: List) -> pd.DataFrame:
        """Build a DataFrame from joint angles time series."""
        records = []

        for angles in joint_angles:
            record = {'timestamp': angles.timestamp}

            # Get all numeric attributes
            for attr in dir(angles):
                if attr.startswith('_') or attr == 'timestamp':
                    continue
                val = getattr(angles, attr)
                if isinstance(val, (int, float)) and val is not None:
                    record[attr] = val

            records.append(record)

        df = pd.DataFrame(records)

        # Unwrap rotation angles to remove ±180° discontinuities
        # These columns use arctan2 and can wrap around
        rotation_cols = [
            'pelvis_rotation', 'pelvis_global_rotation',
            'torso_rotation', 'trunk_global_rotation',
            'hip_shoulder_separation', 'trunk_twist_clockwise',
            'head_twist_clockwise'
        ]
        for col in rotation_cols:
            if col in df.columns and not df[col].isna().all():
                # Convert to radians, unwrap, convert back to degrees
                radians = np.deg2rad(df[col].values)
                unwrapped = np.unwrap(radians)
                df[col] = np.rad2deg(unwrapped)

        # Add angular velocities using Savitzky-Golay filter for smooth derivatives
        # This matches UPLIFT's velocity calculation more closely
        if len(df) > 1 and self.fps:
            dt = 1.0 / self.fps
            # Use window size that's odd and less than data length
            window_length = min(7, len(df) if len(df) % 2 == 1 else len(df) - 1)
            if window_length < 3:
                window_length = 3

            for col in df.columns:
                if col != 'timestamp' and not df[col].isna().all():
                    try:
                        # Fill NaN values with interpolation
                        values = df[col].interpolate(method='linear').ffill().bfill()

                        if len(values) >= window_length:
                            # Use Savitzky-Golay filter for smooth derivative
                            # deriv=1 calculates first derivative, delta=dt gives units of per-second
                            df[f'{col}_velocity'] = savgol_filter(
                                values, window_length, polyorder=2, deriv=1, delta=dt
                            )
                        else:
                            # Fallback to numpy gradient for very short sequences
                            df[f'{col}_velocity'] = np.gradient(values, dt)
                    except Exception:
                        pass

        return df

    def export_results(self,
                       results: dict,
                       output_dir: str,
                       prefix: str = 'swing') -> None:
        """
        Export results to CSV files.

        Args:
            results: Output from process_video()
            output_dir: Directory to save files
            prefix: Filename prefix
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Export time series
        results['timeseries_df'].to_csv(
            output_path / f'{prefix}_timeseries.csv', index=False
        )

        # Export POI metrics
        metrics = results['metrics']
        metrics_dict = {k: v for k, v in metrics.__dict__.items() if v is not None}
        pd.DataFrame([metrics_dict]).to_csv(
            output_path / f'{prefix}_poi_metrics.csv', index=False
        )

        # Export events
        events = results['events']
        events_dict = {k: v for k, v in events.__dict__.items()
                      if v is not None and not k.endswith('_frame')}
        pd.DataFrame([events_dict]).to_csv(
            output_path / f'{prefix}_events.csv', index=False
        )

        # Export 3D poses if available
        if results.get('poses_3d'):
            poses_records = []
            for p in results['poses_3d']:
                record = {'frame': p.frame_number, 'timestamp': p.timestamp}
                for i, pos in enumerate(p.joints_3d):
                    record[f'joint_{i}_x'] = pos[0]
                    record[f'joint_{i}_y'] = pos[1]
                    record[f'joint_{i}_z'] = pos[2]
                poses_records.append(record)
            pd.DataFrame(poses_records).to_csv(
                output_path / f'{prefix}_poses_3d.csv', index=False
            )

        print(f"Results exported to {output_path}")


def main():
    """Example usage of the pipeline."""
    import sys
    import argparse

    parser = argparse.ArgumentParser(description='Extract biomechanics from video')
    parser.add_argument('video_path', help='Path to video file')
    parser.add_argument('-o', '--output', default='./output', help='Output directory')
    parser.add_argument('-b', '--bats', default='R', choices=['L', 'R'],
                        help='Batting side (L=left, R=right)')
    parser.add_argument('--2d', dest='use_2d', action='store_true',
                        help='Use 2D mode (faster, less accurate)')
    parser.add_argument('--model', default='yolov8m-pose.pt',
                        help='YOLOv8 pose model to use')

    args = parser.parse_args()

    print("=" * 60)
    print("VIDEO BIOMECHANICS PIPELINE")
    print("=" * 60)

    pipeline = VideoBiomechanicsPipeline(
        pose_model=args.model,
        bats=args.bats,
        use_3d=not args.use_2d
    )

    results = pipeline.process_video(args.video_path)

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    print(f"\nFrames processed: {len(results['poses'])}")
    print(f"Duration: {results['poses'][-1].timestamp:.2f} seconds")

    print("\nKey Metrics:")
    metrics = results['metrics']

    if metrics.lead_knee_launchpos_x:
        print(f"  Lead knee at foot plant: {metrics.lead_knee_launchpos_x:.1f}°")
    if metrics.rear_elbow_launchpos_x:
        print(f"  Rear elbow at foot plant: {metrics.rear_elbow_launchpos_x:.1f}°")
    if metrics.x_factor_fp_z:
        print(f"  Hip-shoulder separation at FP: {metrics.x_factor_fp_z:.1f}°")
    if metrics.torso_pelvis_stride_max_z:
        print(f"  Max hip-shoulder separation: {metrics.torso_pelvis_stride_max_z:.1f}°")
    if metrics.pelvis_angular_velocity_seq_max:
        print(f"  Peak pelvis rotation velocity: {metrics.pelvis_angular_velocity_seq_max:.1f}°/s")
    if metrics.torso_angular_velocity_seq_max:
        print(f"  Peak torso rotation velocity: {metrics.torso_angular_velocity_seq_max:.1f}°/s")

    # Export
    pipeline.export_results(results, args.output)

    print("\nDone!")


if __name__ == "__main__":
    main()
