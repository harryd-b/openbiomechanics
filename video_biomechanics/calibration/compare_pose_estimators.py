"""
Compare pose estimation methods on single-view videos.

Compares: MediaPipe, YOLOv8-pose
Evaluates against UPLIFT ground truth.
"""

import sys
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod

sys.path.insert(0, str(Path(__file__).parent.parent))

from calibration import JOINT_NAMES
from calibration.run_production import compute_mpjpe, align_frame_rates
from fusion.train_fusion import load_uplift_positions


@dataclass
class PoseResult:
    """Result from a pose estimator."""
    keypoints: np.ndarray  # (n_frames, n_joints, 3) - x, y, confidence
    fps: float
    estimator_name: str
    n_joints: int


class PoseEstimatorBase(ABC):
    """Abstract base class for pose estimators."""

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def n_joints(self) -> int:
        pass

    @abstractmethod
    def process_video(self, video_path: str) -> PoseResult:
        pass

    @abstractmethod
    def to_h36m(self, keypoints: np.ndarray) -> np.ndarray:
        """Convert keypoints to H36M 17-joint format."""
        pass


class MediaPipeEstimator(PoseEstimatorBase):
    """MediaPipe Pose estimator (new Tasks API)."""

    def __init__(self):
        import mediapipe as mp
        self.mp = mp

        # Download model if needed
        self.model_path = self._get_model_path()
        self.landmarker = None  # Created per-video to reset timestamp tracking

    def _create_landmarker(self):
        """Create a fresh landmarker (needed for each video due to timestamp tracking)."""
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        return vision.PoseLandmarker.create_from_options(options)

    def _get_model_path(self) -> str:
        """Download MediaPipe pose model if not present."""
        import urllib.request
        model_dir = Path(__file__).parent / 'models'
        model_dir.mkdir(exist_ok=True)
        model_path = model_dir / 'pose_landmarker_heavy.task'

        if not model_path.exists():
            print("  Downloading MediaPipe pose model...")
            url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
            urllib.request.urlretrieve(url, model_path)
            print("  Model downloaded.")

        return str(model_path)

    @property
    def name(self) -> str:
        return "MediaPipe"

    @property
    def n_joints(self) -> int:
        return 33  # MediaPipe has 33 landmarks

    def process_video(self, video_path: str) -> PoseResult:
        from mediapipe.tasks.python import vision

        # Create fresh landmarker for each video (resets timestamp tracking)
        landmarker = self._create_landmarker()

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        keypoints_list = []
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # MediaPipe expects RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)

            # Timestamp in milliseconds
            timestamp_ms = int(frame_idx * 1000 / fps)

            results = landmarker.detect_for_video(mp_image, timestamp_ms)

            h, w = frame.shape[:2]
            if results.pose_landmarks and len(results.pose_landmarks) > 0:
                kp = np.array([
                    [lm.x * w, lm.y * h, lm.visibility]
                    for lm in results.pose_landmarks[0]
                ])
            else:
                kp = np.zeros((33, 3))

            keypoints_list.append(kp)
            frame_idx += 1

        cap.release()

        return PoseResult(
            keypoints=np.array(keypoints_list),
            fps=fps,
            estimator_name=self.name,
            n_joints=self.n_joints
        )

    def to_h36m(self, keypoints: np.ndarray) -> np.ndarray:
        """Convert MediaPipe 33 joints to H36M 17 joints."""
        # MediaPipe landmark indices:
        # 0: nose, 11: left_shoulder, 12: right_shoulder, 13: left_elbow,
        # 14: right_elbow, 15: left_wrist, 16: right_wrist, 23: left_hip,
        # 24: right_hip, 25: left_knee, 26: right_knee, 27: left_ankle,
        # 28: right_ankle

        mp_to_h36m = {
            0: 9,   # nose -> head
            11: 11, # left_shoulder
            12: 14, # right_shoulder
            13: 12, # left_elbow
            14: 15, # right_elbow
            15: 13, # left_wrist
            16: 16, # right_wrist
            23: 4,  # left_hip
            24: 1,  # right_hip
            25: 5,  # left_knee
            26: 2,  # right_knee
            27: 6,  # left_ankle
            28: 3,  # right_ankle
        }

        n_frames = len(keypoints)
        h36m = np.zeros((n_frames, 17, 2))

        for mp_idx, h36m_idx in mp_to_h36m.items():
            h36m[:, h36m_idx, :] = keypoints[:, mp_idx, :2]

        # Compute derived joints
        # Hip center (pelvis)
        h36m[:, 0, :] = (h36m[:, 1, :] + h36m[:, 4, :]) / 2
        # Spine
        shoulders_mid = (h36m[:, 11, :] + h36m[:, 14, :]) / 2
        h36m[:, 7, :] = (h36m[:, 0, :] + shoulders_mid) / 2
        # Neck
        h36m[:, 8, :] = shoulders_mid
        # Head top
        h36m[:, 10, :] = h36m[:, 9, :] + np.array([0, -20])

        return h36m


class YOLOv8Estimator(PoseEstimatorBase):
    """YOLOv8-pose estimator."""

    def __init__(self, model_name: str = 'yolov8m-pose.pt'):
        from ultralytics import YOLO
        self.model = YOLO(model_name)
        self.model_name = model_name

    @property
    def name(self) -> str:
        return f"YOLOv8 ({self.model_name.replace('.pt', '')})"

    @property
    def n_joints(self) -> int:
        return 17

    def process_video(self, video_path: str) -> PoseResult:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        keypoints_list = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = self.model.predict(frame, verbose=False)

            if results[0].keypoints is not None and len(results[0].keypoints.data) > 0:
                kp = results[0].keypoints.data[0].cpu().numpy()  # First person
            else:
                kp = np.zeros((17, 3))

            keypoints_list.append(kp)

        cap.release()

        return PoseResult(
            keypoints=np.array(keypoints_list),
            fps=fps,
            estimator_name=self.name,
            n_joints=self.n_joints
        )

    def to_h36m(self, keypoints: np.ndarray) -> np.ndarray:
        """Convert YOLO 17 joints to H36M format."""
        # YOLO and H36M have similar but not identical joint order
        yolo_to_h36m = {
            0: 9,   # nose -> head
            5: 11,  # left_shoulder
            6: 14,  # right_shoulder
            7: 12,  # left_elbow
            8: 15,  # right_elbow
            9: 13,  # left_wrist
            10: 16, # right_wrist
            11: 4,  # left_hip
            12: 1,  # right_hip
            13: 5,  # left_knee
            14: 2,  # right_knee
            15: 6,  # left_ankle
            16: 3,  # right_ankle
        }

        n_frames = len(keypoints)
        h36m = np.zeros((n_frames, 17, 2))

        for yolo_idx, h36m_idx in yolo_to_h36m.items():
            h36m[:, h36m_idx, :] = keypoints[:, yolo_idx, :2]

        # Compute derived joints
        h36m[:, 0, :] = (h36m[:, 1, :] + h36m[:, 4, :]) / 2  # pelvis
        shoulders_mid = (h36m[:, 11, :] + h36m[:, 14, :]) / 2
        h36m[:, 7, :] = (h36m[:, 0, :] + shoulders_mid) / 2  # spine
        h36m[:, 8, :] = shoulders_mid  # neck
        h36m[:, 10, :] = h36m[:, 9, :] + np.array([0, -20])  # head top

        return h36m


class VideoPose3DLifter:
    """VideoPose3D 2D->3D lifting with pretrained model."""

    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        import torch
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.receptive_field = 243  # Default for filter_widths=[3,3,3,3,3]
        self._load_model()

    def _load_model(self):
        """Load pretrained VideoPose3D model."""
        import torch
        model_path = Path(__file__).parent.parent / 'models' / 'videopose3d' / 'pretrained_h36m_cpn.bin'

        if not model_path.exists():
            print(f"    VideoPose3D model not found")
            return

        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("videopose3d_model", model_path.parent / "model.py")
            vp3d_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(vp3d_module)
            TemporalModel = vp3d_module.TemporalModel

            self.model = TemporalModel(
                num_joints_in=17, in_features=2, num_joints_out=17,
                filter_widths=[3, 3, 3, 3, 3], causal=False, dropout=0.25, channels=1024
            )

            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_pos'])
            self.model.to(self.device)
            self.model.eval()

            self.receptive_field = self.model.receptive_field()
            print(f"    Loaded VideoPose3D (receptive field: {self.receptive_field} frames)")

        except Exception as e:
            print(f"    Failed to load VideoPose3D: {e}")
            import traceback
            traceback.print_exc()
            self.model = None

    def lift(self, keypoints_2d: np.ndarray) -> np.ndarray:
        """
        Lift normalized 2D keypoints to 3D.

        Processes the full sequence at once with appropriate padding.
        The model uses dilated convolutions with receptive field of 243 frames.
        """
        import torch

        if self.model is None:
            return None

        n_frames = len(keypoints_2d)
        pad = self.receptive_field // 2

        # Pad sequence with edge replication
        # This ensures we have enough temporal context for all frames
        padded = np.pad(keypoints_2d, ((pad, pad), (0, 0), (0, 0)), mode='edge')

        # Process entire sequence in one forward pass
        # Input shape: (batch=1, frames, joints=17, features=2)
        batch = torch.FloatTensor(padded).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(batch)

        # Output shape: (1, n_frames, 17, 3)
        # The model outputs exactly n_frames (input - receptive_field + 1)
        poses_3d = output[0].cpu().numpy()

        return poses_3d


def lift_2d_to_3d_geometric(keypoints_2d: np.ndarray) -> np.ndarray:
    """Simple geometric 2D->3D lifting (fallback)."""
    n_frames = len(keypoints_2d)
    poses_3d = np.zeros((n_frames, 17, 3))
    poses_3d[:, :, :2] = keypoints_2d

    # Simple depth estimation
    depth_map = {0: 0, 1: 0, 2: 0.1, 3: 0, 4: 0, 5: 0.1, 6: 0,
                 7: 0.05, 8: 0.1, 9: 0.12, 10: 0.12, 11: 0.08, 12: 0.15,
                 13: 0.2, 14: 0.08, 15: 0.15, 16: 0.2}
    for j, z in depth_map.items():
        poses_3d[:, j, 2] = z

    return poses_3d


def lift_2d_to_3d(keypoints_2d: np.ndarray, use_learned: bool = True) -> np.ndarray:
    """Lift 2D to 3D using best available method."""
    if use_learned:
        lifter = VideoPose3DLifter.get_instance()
        if lifter.model is not None:
            # Normalize: center on hip, scale by torso
            hip_center = (keypoints_2d[:, 1, :] + keypoints_2d[:, 4, :]) / 2
            centered = keypoints_2d - hip_center[:, np.newaxis, :]

            torso = keypoints_2d[:, 8, :] - keypoints_2d[:, 0, :]
            torso_length = np.linalg.norm(torso, axis=1, keepdims=True)
            torso_length = np.clip(torso_length, 1, None)
            normalized = centered / torso_length[:, np.newaxis, :]

            result = lifter.lift(normalized)
            if result is not None:
                return result

    return lift_2d_to_3d_geometric(keypoints_2d)


def normalize_poses(poses_3d: np.ndarray) -> np.ndarray:
    """Normalize poses: center on pelvis, scale to standard torso."""
    # Center on pelvis
    pelvis = poses_3d[:, 0:1, :]
    centered = poses_3d - pelvis

    # Scale using torso length
    torso_lengths = np.linalg.norm(centered[:, 8, :] - centered[:, 0, :], axis=1)
    median_torso = np.median(torso_lengths[torso_lengths > 0])

    if median_torso > 0:
        scale = 0.5 / median_torso
        centered = centered * scale

    return centered


def compare_estimators(
    video_path: Path,
    estimators: List[PoseEstimatorBase],
    gt_poses: Optional[np.ndarray] = None
) -> pd.DataFrame:
    """
    Compare multiple pose estimators on a single video.

    Args:
        video_path: Path to video
        estimators: List of estimator instances
        gt_poses: Optional ground truth (n_frames, 17, 3)

    Returns:
        DataFrame with comparison metrics
    """
    results = []

    for estimator in estimators:
        print(f"\n  Processing with {estimator.name}...")

        try:
            # Run estimator
            pose_result = estimator.process_video(str(video_path))

            # Convert to H36M format
            keypoints_h36m = estimator.to_h36m(pose_result.keypoints)

            # Lift to 3D (uses VideoPose3D if available)
            poses_3d = lift_2d_to_3d(keypoints_h36m, use_learned=True)

            # Normalize
            poses_normalized = normalize_poses(poses_3d)

            n_frames = len(poses_normalized)
            n_detected = np.sum(pose_result.keypoints[:, :, 2].mean(axis=1) > 0.1)

            result = {
                'estimator': estimator.name,
                'n_frames': n_frames,
                'n_detected': n_detected,
                'detection_rate': n_detected / n_frames if n_frames > 0 else 0,
                'fps': pose_result.fps,
            }

            # Compare with ground truth if available
            if gt_poses is not None:
                gt_normalized = normalize_poses(gt_poses)

                # Align frame rates
                poses_resampled = align_frame_rates(
                    poses_normalized, pose_result.fps, 240.0
                )

                n_compare = min(len(poses_resampled), len(gt_normalized))
                mpjpe, errors = compute_mpjpe(
                    poses_resampled[:n_compare],
                    gt_normalized[:n_compare]
                )

                result['mpjpe_cm'] = float(mpjpe) if not np.isnan(mpjpe) else None
                result['n_compared'] = n_compare

                # Per-joint errors
                if not np.isnan(mpjpe):
                    per_joint = np.nanmean(errors, axis=0) * 100
                    for j, name in enumerate(JOINT_NAMES[:len(per_joint)]):
                        result[f'{name}_error_cm'] = float(per_joint[j])

            results.append(result)

        except Exception as e:
            print(f"    ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'estimator': estimator.name,
                'error': str(e)
            })

    return pd.DataFrame(results)


def run_comparison(
    session_dir: Path,
    view: str = 'side'
) -> Dict:
    """
    Run pose estimator comparison on a session.

    Args:
        session_dir: Session directory
        view: 'side' or 'back'

    Returns:
        Results dictionary
    """
    print("=" * 70)
    print("POSE ESTIMATOR COMPARISON")
    print("=" * 70)
    print(f"Session: {session_dir.name}")
    print(f"View: {view}")

    # Find video
    video_names = {
        'side': ['primary.mp4', 'side.mp4'],
        'back': ['secondary.mp4', 'back.mp4'],
    }

    video_path = None
    for name in video_names.get(view, []):
        candidate = session_dir / name
        if candidate.exists():
            video_path = candidate
            break

    if video_path is None:
        return {'error': f'Video not found for view: {view}'}

    print(f"Video: {video_path.name}")

    # Load ground truth if available
    gt_poses = None
    uplift_csv = session_dir / 'uplift.csv'
    if uplift_csv.exists():
        print("Loading UPLIFT ground truth...")
        gt_poses = load_uplift_positions(str(uplift_csv))
        print(f"  Ground truth frames: {len(gt_poses)}")

    # Initialize estimators
    print("\nInitializing estimators...")
    estimators = [
        MediaPipeEstimator(),
        YOLOv8Estimator('yolov8m-pose.pt'),
        YOLOv8Estimator('yolov8l-pose.pt'),  # Larger model
    ]

    # Run comparison
    print("\n" + "-" * 70)
    print("Running comparison...")
    print("-" * 70)

    df = compare_estimators(video_path, estimators, gt_poses)

    # Display results
    print("\n" + "-" * 70)
    print("RESULTS")
    print("-" * 70)

    # Summary columns
    summary_cols = ['estimator', 'detection_rate', 'mpjpe_cm']
    available_cols = [c for c in summary_cols if c in df.columns]
    print(df[available_cols].to_string(index=False))

    return {
        'session': session_dir.name,
        'view': view,
        'results': df.to_dict('records'),
        'best_estimator': df.loc[df['mpjpe_cm'].idxmin(), 'estimator'] if 'mpjpe_cm' in df.columns and df['mpjpe_cm'].notna().any() else None
    }


def run_all_sessions(view: str = 'side'):
    """Run comparison on all sessions."""
    training_data = Path(__file__).parent.parent / 'training_data'
    sessions = sorted(training_data.glob('session_*'))

    all_results = []

    for session_dir in sessions:
        print(f"\n{'='*70}")
        print(f"Processing {session_dir.name}")
        print('='*70)

        result = run_comparison(session_dir, view)

        if 'results' in result:
            for r in result['results']:
                r['session'] = session_dir.name
                all_results.append(r)

    # Aggregate results
    df = pd.DataFrame(all_results)

    print("\n" + "=" * 70)
    print("AGGREGATE RESULTS")
    print("=" * 70)

    # Group by estimator
    if 'mpjpe_cm' in df.columns:
        summary = df.groupby('estimator').agg({
            'mpjpe_cm': ['mean', 'std', 'min', 'max'],
            'detection_rate': 'mean'
        }).round(2)
        print(summary)

        # Best estimator per session
        print("\nBest estimator per session:")
        best_per_session = df.loc[df.groupby('session')['mpjpe_cm'].idxmin()]
        print(best_per_session[['session', 'estimator', 'mpjpe_cm']].to_string(index=False))

    return df


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Compare pose estimators")
    parser.add_argument('--session', type=Path, default=None)
    parser.add_argument('--view', choices=['side', 'back'], default='side')
    parser.add_argument('--all', action='store_true')
    args = parser.parse_args()

    if args.all:
        run_all_sessions(args.view)
    elif args.session:
        run_comparison(args.session, args.view)
    else:
        # Default: session_001
        session = Path(__file__).parent.parent / 'training_data' / 'session_001'
        run_comparison(session, args.view)


if __name__ == '__main__':
    main()
