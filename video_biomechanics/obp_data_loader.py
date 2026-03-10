"""
OpenBiomechanics Project (OBP) Data Loader.

Loads and processes motion capture data from the OBP dataset for:
- Training domain-specific 2D->3D pose lifters
- Extracting biomechanical priors (bone lengths, joint ranges)
- Benchmarking pose estimation accuracy

Dataset: https://github.com/drivelineresearch/openbiomechanics
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import zipfile
import io


@dataclass
class OBPSkeleton:
    """3D skeleton from OBP motion capture data."""
    session_pitch: str
    time: float
    joints: Dict[str, np.ndarray]  # joint_name -> (x, y, z) in meters
    event_times: Dict[str, float] = field(default_factory=dict)

    def get_joint(self, name: str) -> Optional[np.ndarray]:
        """Get joint position by name."""
        return self.joints.get(name)

    def to_array(self, joint_order: List[str]) -> np.ndarray:
        """Convert to array with specified joint order."""
        result = np.zeros((len(joint_order), 3))
        for i, name in enumerate(joint_order):
            if name in self.joints:
                result[i] = self.joints[name]
        return result


@dataclass
class OBPSequence:
    """A complete motion sequence (one pitch/swing)."""
    session_pitch: str
    frames: List[OBPSkeleton]
    metadata: Dict = field(default_factory=dict)

    @property
    def num_frames(self) -> int:
        return len(self.frames)

    @property
    def duration(self) -> float:
        if len(self.frames) < 2:
            return 0.0
        return self.frames[-1].time - self.frames[0].time

    def get_phase(self, start_event: str, end_event: str) -> List[OBPSkeleton]:
        """Extract frames between two events."""
        if not self.frames:
            return []

        event_times = self.frames[0].event_times
        start_time = event_times.get(start_event, self.frames[0].time)
        end_time = event_times.get(end_event, self.frames[-1].time)

        return [f for f in self.frames if start_time <= f.time <= end_time]


# OBP joint names to standardized names mapping
OBP_JOINT_MAP = {
    # Lower body
    'rear_ankle_jc': 'right_ankle',
    'rear_knee_jc': 'right_knee',
    'rear_hip': 'right_hip',
    'lead_ankle_jc': 'left_ankle',
    'lead_knee_jc': 'left_knee',
    'lead_hip': 'left_hip',
    # Upper body - throwing side
    'shoulder_jc': 'right_shoulder',
    'elbow_jc': 'right_elbow',
    'wrist_jc': 'right_wrist',
    'hand_jc': 'right_hand',
    # Upper body - glove side
    'glove_shoulder_jc': 'left_shoulder',
    'glove_elbow_jc': 'left_elbow',
    'glove_wrist_jc': 'left_wrist',
    'glove_hand_jc': 'left_hand',
    # Torso
    'thorax_prox': 'thorax_top',
    'thorax_dist': 'thorax_mid',
    'thorax_ap': 'thorax_front',
    'centerofmass': 'center_of_mass',
}

# Standard joint order for array conversion (17 joints like H36M)
STANDARD_JOINT_ORDER = [
    'hip_center',        # 0 - computed as midpoint
    'right_hip',         # 1
    'right_knee',        # 2
    'right_ankle',       # 3
    'left_hip',          # 4
    'left_knee',         # 5
    'left_ankle',        # 6
    'spine',             # 7 - computed
    'neck',              # 8 - thorax_top
    'head',              # 9 - estimated
    'head_top',          # 10 - estimated
    'left_shoulder',     # 11
    'left_elbow',        # 12
    'left_wrist',        # 13
    'right_shoulder',    # 14
    'right_elbow',       # 15
    'right_wrist',       # 16
]


class OBPDataLoader:
    """Load and process OpenBiomechanics Project data."""

    def __init__(self, data_root: str):
        """
        Initialize the data loader.

        Args:
            data_root: Path to the openbiomechanics repository root
        """
        self.data_root = Path(data_root)
        self._landmarks_cache: Optional[pd.DataFrame] = None
        self._joint_angles_cache: Optional[pd.DataFrame] = None
        self._metadata_cache: Dict[str, pd.DataFrame] = {}

    def _get_dataset_path(self, dataset: str) -> Path:
        """Get path to a dataset (pitching or hitting)."""
        return self.data_root / dataset / 'data'

    def load_landmarks(self, dataset: str = 'baseball_pitching',
                       use_cache: bool = True) -> pd.DataFrame:
        """
        Load 3D landmark positions.

        Args:
            dataset: 'baseball_pitching' or 'baseball_hitting'
            use_cache: Whether to cache the loaded data

        Returns:
            DataFrame with 3D joint positions over time
        """
        if use_cache and self._landmarks_cache is not None:
            return self._landmarks_cache

        zip_path = self._get_dataset_path(dataset) / 'full_sig' / 'landmarks.zip'

        with zipfile.ZipFile(zip_path, 'r') as zf:
            with zf.open('landmarks.csv') as f:
                df = pd.read_csv(f)

        if use_cache:
            self._landmarks_cache = df

        return df

    def load_joint_angles(self, dataset: str = 'baseball_pitching',
                          use_cache: bool = True) -> pd.DataFrame:
        """Load joint angle data."""
        if use_cache and self._joint_angles_cache is not None:
            return self._joint_angles_cache

        zip_path = self._get_dataset_path(dataset) / 'full_sig' / 'joint_angles.zip'

        with zipfile.ZipFile(zip_path, 'r') as zf:
            with zf.open('joint_angles.csv') as f:
                df = pd.read_csv(f)

        if use_cache:
            self._joint_angles_cache = df

        return df

    def load_metadata(self, dataset: str = 'baseball_pitching') -> pd.DataFrame:
        """Load athlete/session metadata."""
        if dataset in self._metadata_cache:
            return self._metadata_cache[dataset]

        csv_path = self._get_dataset_path(dataset) / 'metadata.csv'
        df = pd.read_csv(csv_path)
        self._metadata_cache[dataset] = df
        return df

    def get_sequence(self, session_pitch: str,
                     dataset: str = 'baseball_pitching') -> OBPSequence:
        """
        Load a single motion sequence.

        Args:
            session_pitch: Unique pitch/swing identifier (e.g., '1031_2')
            dataset: Dataset to load from

        Returns:
            OBPSequence containing all frames
        """
        landmarks = self.load_landmarks(dataset)
        seq_data = landmarks[landmarks['session_pitch'] == session_pitch]

        if seq_data.empty:
            raise ValueError(f"Sequence {session_pitch} not found in {dataset}")

        # Extract event times from first row
        event_cols = ['pkh_time', 'fp_10_time', 'fp_100_time',
                      'MER_time', 'BR_time', 'MIR_time']
        event_times = {}
        first_row = seq_data.iloc[0]
        for col in event_cols:
            if col in seq_data.columns:
                event_times[col.replace('_time', '')] = first_row[col]

        # Parse each frame
        frames = []
        for _, row in seq_data.iterrows():
            joints = self._parse_joints(row)
            skeleton = OBPSkeleton(
                session_pitch=session_pitch,
                time=row['time'],
                joints=joints,
                event_times=event_times
            )
            frames.append(skeleton)

        # Load metadata if available
        try:
            metadata_df = self.load_metadata(dataset)
            meta_row = metadata_df[
                metadata_df['session_pitch'] == session_pitch
            ]
            if not meta_row.empty:
                metadata = meta_row.iloc[0].to_dict()
            else:
                metadata = {}
        except Exception:
            metadata = {}

        return OBPSequence(
            session_pitch=session_pitch,
            frames=frames,
            metadata=metadata
        )

    def _parse_joints(self, row: pd.Series) -> Dict[str, np.ndarray]:
        """Parse joint positions from a DataFrame row."""
        joints = {}

        # Parse OBP joint columns
        for obp_name, std_name in OBP_JOINT_MAP.items():
            x_col = f'{obp_name}_x'
            y_col = f'{obp_name}_y'
            z_col = f'{obp_name}_z'

            if all(c in row.index for c in [x_col, y_col, z_col]):
                joints[std_name] = np.array([
                    row[x_col], row[y_col], row[z_col]
                ])

        # Compute derived joints
        if 'right_hip' in joints and 'left_hip' in joints:
            joints['hip_center'] = (joints['right_hip'] + joints['left_hip']) / 2

        if 'right_shoulder' in joints and 'left_shoulder' in joints:
            joints['neck'] = (joints['right_shoulder'] + joints['left_shoulder']) / 2

        if 'hip_center' in joints and 'neck' in joints:
            joints['spine'] = (joints['hip_center'] + joints['neck']) / 2

        # Estimate head position (above neck)
        if 'neck' in joints:
            joints['head'] = joints['neck'] + np.array([0, 0, 0.15])
            joints['head_top'] = joints['neck'] + np.array([0, 0, 0.25])

        return joints

    def get_all_sequence_ids(self, dataset: str = 'baseball_pitching') -> List[str]:
        """Get all available sequence IDs."""
        landmarks = self.load_landmarks(dataset)
        return landmarks['session_pitch'].unique().tolist()

    def to_h36m_format(self, skeleton: OBPSkeleton) -> np.ndarray:
        """
        Convert OBP skeleton to Human3.6M format (17 joints).

        Returns:
            Array of shape (17, 3) with joint positions
        """
        return skeleton.to_array(STANDARD_JOINT_ORDER)

    def iter_sequences(self, dataset: str = 'baseball_pitching',
                       max_sequences: Optional[int] = None):
        """
        Iterate over all sequences in a dataset.

        Args:
            dataset: Dataset to iterate
            max_sequences: Maximum number of sequences to yield

        Yields:
            OBPSequence objects
        """
        landmarks = self.load_landmarks(dataset)
        sequence_ids = landmarks['session_pitch'].unique()

        if max_sequences:
            sequence_ids = sequence_ids[:max_sequences]

        for seq_id in sequence_ids:
            try:
                yield self.get_sequence(seq_id, dataset)
            except Exception as e:
                print(f"Warning: Failed to load sequence {seq_id}: {e}")
                continue


class BiomechanicalPriors:
    """Extract and store biomechanical priors from OBP data."""

    def __init__(self):
        self.bone_lengths: Dict[str, List[float]] = {}
        self.joint_angles: Dict[str, Dict[str, List[float]]] = {}
        self.velocities: Dict[str, List[float]] = {}

    def add_bone_length(self, bone_name: str, length: float):
        """Record a bone length observation."""
        if bone_name not in self.bone_lengths:
            self.bone_lengths[bone_name] = []
        self.bone_lengths[bone_name].append(length)

    def add_joint_angle(self, joint_name: str, axis: str, angle: float):
        """Record a joint angle observation."""
        if joint_name not in self.joint_angles:
            self.joint_angles[joint_name] = {}
        if axis not in self.joint_angles[joint_name]:
            self.joint_angles[joint_name][axis] = []
        self.joint_angles[joint_name][axis].append(angle)

    def get_bone_stats(self, bone_name: str) -> Dict[str, float]:
        """Get statistics for a bone length."""
        if bone_name not in self.bone_lengths:
            return {}
        lengths = np.array(self.bone_lengths[bone_name])
        return {
            'mean': float(np.mean(lengths)),
            'std': float(np.std(lengths)),
            'min': float(np.min(lengths)),
            'max': float(np.max(lengths)),
            'count': len(lengths)
        }

    def get_angle_stats(self, joint_name: str, axis: str) -> Dict[str, float]:
        """Get statistics for a joint angle."""
        if joint_name not in self.joint_angles:
            return {}
        if axis not in self.joint_angles[joint_name]:
            return {}
        angles = np.array(self.joint_angles[joint_name][axis])
        return {
            'mean': float(np.mean(angles)),
            'std': float(np.std(angles)),
            'min': float(np.min(angles)),
            'max': float(np.max(angles)),
            'p5': float(np.percentile(angles, 5)),
            'p95': float(np.percentile(angles, 95)),
            'count': len(angles)
        }

    def to_dict(self) -> Dict:
        """Export all priors as a dictionary."""
        result = {
            'bone_lengths': {},
            'joint_angles': {}
        }

        for bone in self.bone_lengths:
            result['bone_lengths'][bone] = self.get_bone_stats(bone)

        for joint in self.joint_angles:
            result['joint_angles'][joint] = {}
            for axis in self.joint_angles[joint]:
                result['joint_angles'][joint][axis] = self.get_angle_stats(joint, axis)

        return result


def extract_priors(loader: OBPDataLoader,
                   dataset: str = 'baseball_pitching',
                   max_sequences: int = 50) -> BiomechanicalPriors:
    """
    Extract biomechanical priors from OBP dataset.

    Args:
        loader: OBP data loader
        dataset: Dataset to analyze
        max_sequences: Maximum sequences to process

    Returns:
        BiomechanicalPriors with statistics
    """
    priors = BiomechanicalPriors()

    # Define bones to measure
    bones = [
        ('right_hip', 'right_knee', 'thigh'),
        ('right_knee', 'right_ankle', 'shin'),
        ('left_hip', 'left_knee', 'thigh'),
        ('left_knee', 'left_ankle', 'shin'),
        ('right_shoulder', 'right_elbow', 'upper_arm'),
        ('right_elbow', 'right_wrist', 'forearm'),
        ('left_shoulder', 'left_elbow', 'upper_arm'),
        ('left_elbow', 'left_wrist', 'forearm'),
        ('right_hip', 'left_hip', 'hip_width'),
        ('right_shoulder', 'left_shoulder', 'shoulder_width'),
        ('hip_center', 'neck', 'torso'),
    ]

    print(f"Extracting priors from {dataset}...")

    for i, seq in enumerate(loader.iter_sequences(dataset, max_sequences)):
        if i % 10 == 0:
            print(f"  Processing sequence {i+1}...")

        # Sample frames (don't need every frame)
        sample_indices = range(0, len(seq.frames), 10)

        for idx in sample_indices:
            frame = seq.frames[idx]

            # Measure bone lengths
            for joint1, joint2, bone_name in bones:
                pos1 = frame.get_joint(joint1)
                pos2 = frame.get_joint(joint2)

                if pos1 is not None and pos2 is not None:
                    length = np.linalg.norm(pos2 - pos1)
                    if 0.05 < length < 1.0:  # Sanity check
                        priors.add_bone_length(bone_name, length)

    return priors


if __name__ == '__main__':
    import json

    # Find the data root
    script_dir = Path(__file__).parent
    data_root = script_dir.parent  # openbiomechanics repo root

    print(f"Loading OBP data from: {data_root}")

    loader = OBPDataLoader(data_root)

    # Test loading a sequence
    print("\n--- Testing sequence loading ---")
    seq_ids = loader.get_all_sequence_ids('baseball_pitching')
    print(f"Found {len(seq_ids)} pitching sequences")

    if seq_ids:
        seq = loader.get_sequence(seq_ids[0])
        print(f"Loaded sequence: {seq.session_pitch}")
        print(f"  Frames: {seq.num_frames}")
        print(f"  Duration: {seq.duration:.3f}s")

        if seq.frames:
            frame = seq.frames[len(seq.frames)//2]  # Middle frame
            print(f"  Sample joints: {list(frame.joints.keys())[:5]}...")

            h36m = loader.to_h36m_format(frame)
            print(f"  H36M format shape: {h36m.shape}")

    # Extract priors
    print("\n--- Extracting biomechanical priors ---")
    priors = extract_priors(loader, max_sequences=30)

    print("\nBone length statistics (meters):")
    for bone in ['thigh', 'shin', 'upper_arm', 'forearm', 'torso', 'shoulder_width', 'hip_width']:
        stats = priors.get_bone_stats(bone)
        if stats:
            print(f"  {bone}: {stats['mean']:.3f} ± {stats['std']:.3f} "
                  f"(range: {stats['min']:.3f} - {stats['max']:.3f})")

    # Save priors to JSON
    output_path = script_dir / 'models' / 'obp_priors.json'
    output_path.parent.mkdir(exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(priors.to_dict(), f, indent=2)

    print(f"\nPriors saved to: {output_path}")
