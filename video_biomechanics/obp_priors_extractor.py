"""
Extract comprehensive biomechanical priors from OpenBiomechanics data.

This module analyzes the OBP dataset to extract:
- Bone length distributions (per athlete height)
- Joint angle ranges during different motion phases
- Joint velocity limits
- Temporal patterns

These priors can be used to:
- Constrain 3D pose estimation
- Validate pose predictions
- Train better 2D->3D lifting models
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import json
from obp_data_loader import OBPDataLoader, OBPSequence


@dataclass
class ComprehensivePriors:
    """Comprehensive biomechanical priors from OBP data."""

    # Bone lengths normalized by height
    bone_length_ratios: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Absolute bone lengths (meters)
    bone_lengths_abs: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Joint angle ranges by phase
    joint_angle_ranges: Dict[str, Dict[str, Dict[str, float]]] = field(default_factory=dict)

    # Joint velocity limits (deg/s)
    joint_velocity_limits: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Phase timing statistics
    phase_timings: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            'bone_length_ratios': self.bone_length_ratios,
            'bone_lengths_abs': self.bone_lengths_abs,
            'joint_angle_ranges': self.joint_angle_ranges,
            'joint_velocity_limits': self.joint_velocity_limits,
            'phase_timings': self.phase_timings
        }

    def save(self, path: Path):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> 'ComprehensivePriors':
        with open(path, 'r') as f:
            data = json.load(f)
        priors = cls()
        priors.bone_length_ratios = data.get('bone_length_ratios', {})
        priors.bone_lengths_abs = data.get('bone_lengths_abs', {})
        priors.joint_angle_ranges = data.get('joint_angle_ranges', {})
        priors.joint_velocity_limits = data.get('joint_velocity_limits', {})
        priors.phase_timings = data.get('phase_timings', {})
        return priors


def compute_stats(values: List[float]) -> Dict[str, float]:
    """Compute statistics for a list of values."""
    if not values:
        return {}
    arr = np.array(values)
    return {
        'mean': float(np.mean(arr)),
        'std': float(np.std(arr)),
        'min': float(np.min(arr)),
        'max': float(np.max(arr)),
        'p5': float(np.percentile(arr, 5)),
        'p25': float(np.percentile(arr, 25)),
        'p50': float(np.percentile(arr, 50)),
        'p75': float(np.percentile(arr, 75)),
        'p95': float(np.percentile(arr, 95)),
        'n': len(arr)
    }


class PriorsExtractor:
    """Extract biomechanical priors from OBP data."""

    BONES = [
        ('right_hip', 'right_knee', 'thigh_r'),
        ('right_knee', 'right_ankle', 'shin_r'),
        ('left_hip', 'left_knee', 'thigh_l'),
        ('left_knee', 'left_ankle', 'shin_l'),
        ('right_shoulder', 'right_elbow', 'upper_arm_r'),
        ('right_elbow', 'right_wrist', 'forearm_r'),
        ('left_shoulder', 'left_elbow', 'upper_arm_l'),
        ('left_elbow', 'left_wrist', 'forearm_l'),
        ('right_hip', 'left_hip', 'hip_width'),
        ('right_shoulder', 'left_shoulder', 'shoulder_width'),
        ('hip_center', 'neck', 'torso'),
        ('neck', 'head', 'neck_length'),
    ]

    # Joint angle columns in OBP data
    JOINT_ANGLE_COLS = {
        'elbow': ['elbow_angle_x', 'elbow_angle_y', 'elbow_angle_z'],
        'shoulder': ['shoulder_angle_x', 'shoulder_angle_y', 'shoulder_angle_z'],
        'rear_hip': ['rear_hip_angle_x', 'rear_hip_angle_y', 'rear_hip_angle_z'],
        'lead_hip': ['lead_hip_angle_x', 'lead_hip_angle_y', 'lead_hip_angle_z'],
        'rear_knee': ['rear_knee_angle_x', 'rear_knee_angle_y', 'rear_knee_angle_z'],
        'lead_knee': ['lead_knee_angle_x', 'lead_knee_angle_y', 'lead_knee_angle_z'],
        'pelvis': ['pelvis_angle_x', 'pelvis_angle_y', 'pelvis_angle_z'],
        'torso': ['torso_angle_x', 'torso_angle_y', 'torso_angle_z'],
    }

    def __init__(self, loader: OBPDataLoader):
        self.loader = loader

    def extract_all(self, dataset: str = 'baseball_pitching',
                    max_sequences: int = 100) -> ComprehensivePriors:
        """Extract all priors from the dataset."""
        priors = ComprehensivePriors()

        # Collect raw data
        bone_lengths: Dict[str, List[float]] = {}
        bone_ratios: Dict[str, List[float]] = {}
        joint_angles: Dict[str, Dict[str, List[float]]] = {}
        joint_velocities: Dict[str, List[float]] = {}
        phase_times: Dict[str, List[float]] = {}

        print(f"Extracting priors from {dataset}...")

        # Load metadata for heights
        try:
            metadata = self.loader.load_metadata(dataset)
            height_lookup = dict(zip(
                metadata['session_pitch'].astype(str),
                metadata['session_height_m']
            ))
        except Exception:
            height_lookup = {}

        # Load joint angles
        try:
            angles_df = self.loader.load_joint_angles(dataset)
            has_angles = True
        except Exception:
            angles_df = None
            has_angles = False

        for i, seq in enumerate(self.loader.iter_sequences(dataset, max_sequences)):
            if i % 20 == 0:
                print(f"  Processing sequence {i+1}...")

            # Get athlete height (default to average if not found)
            height = height_lookup.get(seq.session_pitch, 1.83)

            # Sample frames
            sample_rate = max(1, len(seq.frames) // 50)
            sample_frames = seq.frames[::sample_rate]

            for frame in sample_frames:
                # Extract bone lengths
                for joint1, joint2, bone_name in self.BONES:
                    pos1 = frame.get_joint(joint1)
                    pos2 = frame.get_joint(joint2)

                    if pos1 is not None and pos2 is not None:
                        length = np.linalg.norm(pos2 - pos1)
                        if 0.01 < length < 1.5:  # Sanity check
                            if bone_name not in bone_lengths:
                                bone_lengths[bone_name] = []
                                bone_ratios[bone_name] = []
                            bone_lengths[bone_name].append(length)
                            bone_ratios[bone_name].append(length / height)

            # Extract joint angles if available
            if has_angles and angles_df is not None:
                seq_angles = angles_df[angles_df['session_pitch'] == seq.session_pitch]

                for joint, cols in self.JOINT_ANGLE_COLS.items():
                    if joint not in joint_angles:
                        joint_angles[joint] = {'x': [], 'y': [], 'z': []}

                    for axis, col in zip(['x', 'y', 'z'], cols):
                        if col in seq_angles.columns:
                            values = seq_angles[col].dropna().values
                            joint_angles[joint][axis].extend(values.tolist())

                            # Compute velocities
                            if len(values) > 1:
                                dt = 1.0 / 360.0  # 360 Hz
                                velocities = np.abs(np.diff(values)) / dt
                                vel_key = f'{joint}_{axis}'
                                if vel_key not in joint_velocities:
                                    joint_velocities[vel_key] = []
                                # Use 95th percentile of velocities for this sequence
                                if len(velocities) > 0:
                                    joint_velocities[vel_key].append(
                                        float(np.percentile(velocities, 95))
                                    )

            # Extract phase timing
            if seq.frames and seq.frames[0].event_times:
                events = seq.frames[0].event_times
                if 'fp_100' in events and 'BR' in events:
                    duration = events['BR'] - events['fp_100']
                    if 0.05 < duration < 0.5:
                        if 'fp_to_br' not in phase_times:
                            phase_times['fp_to_br'] = []
                        phase_times['fp_to_br'].append(duration)

        # Compute statistics
        print("\nComputing statistics...")

        for bone, lengths in bone_lengths.items():
            priors.bone_lengths_abs[bone] = compute_stats(lengths)

        for bone, ratios in bone_ratios.items():
            priors.bone_length_ratios[bone] = compute_stats(ratios)

        for joint, axes in joint_angles.items():
            priors.joint_angle_ranges[joint] = {}
            for axis, values in axes.items():
                if values:
                    priors.joint_angle_ranges[joint][axis] = compute_stats(values)

        for vel_key, velocities in joint_velocities.items():
            priors.joint_velocity_limits[vel_key] = compute_stats(velocities)

        for phase, times in phase_times.items():
            priors.phase_timings[phase] = compute_stats(times)

        return priors


def print_priors_summary(priors: ComprehensivePriors):
    """Print a summary of extracted priors."""
    print("\n" + "="*60)
    print("BIOMECHANICAL PRIORS SUMMARY")
    print("="*60)

    print("\n--- Bone Lengths (meters) ---")
    for bone, stats in priors.bone_lengths_abs.items():
        print(f"  {bone:20s}: {stats['mean']:.3f} ± {stats['std']:.3f} "
              f"[{stats['p5']:.3f} - {stats['p95']:.3f}]")

    print("\n--- Bone Length Ratios (% of height) ---")
    for bone, stats in priors.bone_length_ratios.items():
        print(f"  {bone:20s}: {stats['mean']*100:.1f}% ± {stats['std']*100:.1f}%")

    print("\n--- Joint Angle Ranges (degrees) ---")
    for joint, axes in priors.joint_angle_ranges.items():
        print(f"  {joint}:")
        for axis, stats in axes.items():
            print(f"    {axis}: [{stats['p5']:.1f}° to {stats['p95']:.1f}°] "
                  f"(mean: {stats['mean']:.1f}°)")

    print("\n--- Peak Joint Velocities (deg/s) ---")
    for vel_key, stats in list(priors.joint_velocity_limits.items())[:10]:
        print(f"  {vel_key:20s}: {stats['mean']:.0f} ± {stats['std']:.0f} deg/s")

    if priors.phase_timings:
        print("\n--- Phase Timings (seconds) ---")
        for phase, stats in priors.phase_timings.items():
            print(f"  {phase}: {stats['mean']:.3f} ± {stats['std']:.3f}s")


def create_constraint_config(priors: ComprehensivePriors) -> Dict:
    """
    Create a configuration dictionary for pose estimation constraints.

    This can be used to validate/constrain 3D pose predictions.
    """
    config = {
        'bone_lengths': {},
        'joint_limits': {},
        'velocity_limits': {}
    }

    # Bone length constraints (mean ± 3 std)
    for bone, stats in priors.bone_lengths_abs.items():
        config['bone_lengths'][bone] = {
            'expected': stats['mean'],
            'min': max(0.01, stats['mean'] - 3 * stats['std']),
            'max': stats['mean'] + 3 * stats['std']
        }

    # Joint angle limits (5th to 95th percentile with margin)
    for joint, axes in priors.joint_angle_ranges.items():
        config['joint_limits'][joint] = {}
        for axis, stats in axes.items():
            margin = 10  # degrees of margin
            config['joint_limits'][joint][axis] = {
                'min': stats['p5'] - margin,
                'max': stats['p95'] + margin
            }

    # Velocity limits (95th percentile with margin)
    for vel_key, stats in priors.joint_velocity_limits.items():
        config['velocity_limits'][vel_key] = {
            'max': stats['p95'] * 1.2  # 20% margin
        }

    return config


if __name__ == '__main__':
    script_dir = Path(__file__).parent
    data_root = script_dir.parent

    print(f"Loading OBP data from: {data_root}")

    loader = OBPDataLoader(data_root)
    extractor = PriorsExtractor(loader)

    # Extract priors from both datasets
    pitching_priors = extractor.extract_all('baseball_pitching', max_sequences=80)
    print_priors_summary(pitching_priors)

    # Save priors
    output_dir = script_dir / 'models'
    output_dir.mkdir(exist_ok=True)

    pitching_priors.save(output_dir / 'pitching_priors.json')
    print(f"\nPriors saved to: {output_dir / 'pitching_priors.json'}")

    # Create constraint config
    constraints = create_constraint_config(pitching_priors)
    with open(output_dir / 'pose_constraints.json', 'w') as f:
        json.dump(constraints, f, indent=2)
    print(f"Constraints saved to: {output_dir / 'pose_constraints.json'}")

    # Also extract from hitting data
    print("\n" + "="*60)
    print("Extracting hitting priors...")
    hitting_priors = extractor.extract_all('baseball_hitting', max_sequences=80)
    hitting_priors.save(output_dir / 'hitting_priors.json')
    print(f"Hitting priors saved to: {output_dir / 'hitting_priors.json'}")
