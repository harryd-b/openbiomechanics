"""
Benchmark for 3D Pose Estimation.

Compares different 2D-to-3D lifting approaches against OBP ground truth:
1. Geometric estimation (bone length constraints)
2. OBP-trained neural network
3. (Future) Other methods like MotionBERT

Metrics:
- MPJPE: Mean Per Joint Position Error (mm)
- PA-MPJPE: Procrustes-Aligned MPJPE (mm)
- Per-joint breakdown
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json
import torch

from obp_data_loader import OBPDataLoader, STANDARD_JOINT_ORDER
from obp_training_pipeline import (
    SyntheticPoseDataset, Pose2DTo3DLifter, load_model,
    project_3d_to_2d, CameraParams
)
from lifting_3d import VideoPose3DLifter, H36MJoints


@dataclass
class BenchmarkResult:
    """Results from benchmarking a method."""
    method_name: str
    mpjpe_mm: float  # Mean Per Joint Position Error in mm
    pa_mpjpe_mm: float  # Procrustes-aligned MPJPE in mm
    per_joint_mpjpe: Dict[str, float]  # Per-joint errors
    n_samples: int


def procrustes_align(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
    Align predicted poses to target using Procrustes analysis.

    This removes global rotation, translation, and scale differences,
    measuring only the shape error.

    Args:
        predicted: (N, 3) predicted joint positions
        target: (N, 3) ground truth positions

    Returns:
        Aligned predicted positions
    """
    # Center both
    pred_centered = predicted - np.mean(predicted, axis=0)
    target_centered = target - np.mean(target, axis=0)

    # Scale
    pred_scale = np.linalg.norm(pred_centered)
    target_scale = np.linalg.norm(target_centered)

    if pred_scale > 1e-6:
        pred_centered /= pred_scale

    if target_scale > 1e-6:
        target_centered /= target_scale

    # Optimal rotation using SVD
    H = pred_centered.T @ target_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Handle reflection
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Apply transformation
    aligned = pred_centered @ R * target_scale + np.mean(target, axis=0)

    return aligned


def compute_mpjpe(predicted: np.ndarray, target: np.ndarray) -> float:
    """Compute Mean Per Joint Position Error in mm."""
    errors = np.linalg.norm(predicted - target, axis=1)
    return float(np.mean(errors) * 1000)  # Convert to mm


def compute_per_joint_mpjpe(predicted: np.ndarray,
                            target: np.ndarray,
                            joint_names: List[str]) -> Dict[str, float]:
    """Compute MPJPE per joint."""
    errors = np.linalg.norm(predicted - target, axis=1) * 1000
    return {name: float(errors[i]) for i, name in enumerate(joint_names)}


class GeometricLifterWrapper:
    """Wrapper for the geometric (non-ML) lifter for benchmarking."""

    def __init__(self):
        self.lifter = VideoPose3DLifter(model_path=None)

    def lift(self, pose_2d: np.ndarray, normalize_params: dict = None) -> np.ndarray:
        """Lift 2D pose to 3D."""
        # The geometric lifter expects (17, 2) or (17, 3) input
        if pose_2d.ndim == 1:
            pose_2d = pose_2d.reshape(17, 2)

        result = self.lifter._lift_geometric(
            [np.hstack([pose_2d, np.ones((17, 1)) * 0.9])],
            [0.0],
            None
        )

        return result[0].joints_3d


class OBPLifterWrapper:
    """Wrapper for the OBP-trained neural network lifter."""

    def __init__(self, model_path: Path):
        self.model, self.norm_params = load_model(model_path)
        self.model.eval()

    def lift(self, pose_2d: np.ndarray, normalize_params: dict = None) -> np.ndarray:
        """Lift 2D pose to 3D."""
        if pose_2d.ndim == 2:
            pose_2d = pose_2d.flatten()

        with torch.no_grad():
            input_tensor = torch.FloatTensor(pose_2d).unsqueeze(0)
            output = self.model(input_tensor)

        pose_3d = output.numpy().reshape(17, 3)

        # Denormalize
        if normalize_params:
            pose_3d = pose_3d * normalize_params['pose_std'] + normalize_params['pose_mean']
        else:
            pose_3d = pose_3d * self.norm_params['pose_std'] + self.norm_params['pose_mean']

        return pose_3d


def run_benchmark(loader: OBPDataLoader,
                  methods: Dict[str, object],
                  dataset: str = 'baseball_pitching',
                  n_sequences: int = 20,
                  samples_per_sequence: int = 20) -> Dict[str, BenchmarkResult]:
    """
    Run benchmark comparing different lifting methods.

    Args:
        loader: OBP data loader
        methods: Dictionary of method_name -> lifter wrapper
        dataset: Dataset to benchmark on
        n_sequences: Number of sequences to test
        samples_per_sequence: Frames to sample per sequence

    Returns:
        Dictionary of method_name -> BenchmarkResult
    """
    print(f"Running benchmark on {dataset}...")
    print(f"  Sequences: {n_sequences}")
    print(f"  Samples per sequence: {samples_per_sequence}")

    # Initialize error accumulators
    errors = {name: {'mpjpe': [], 'pa_mpjpe': [], 'per_joint': []}
              for name in methods}

    # Create a standard camera for 2D projection
    camera = CameraParams(
        distance=5.0,
        azimuth=np.pi/4,  # 45 degrees to the side
        elevation=np.pi/12  # slight elevation
    )

    sequence_count = 0
    for seq in loader.iter_sequences(dataset, n_sequences):
        sequence_count += 1
        if sequence_count % 5 == 0:
            print(f"  Processing sequence {sequence_count}/{n_sequences}...")

        # Sample frames
        n_frames = len(seq.frames)
        if n_frames < samples_per_sequence:
            indices = list(range(n_frames))
        else:
            indices = np.linspace(0, n_frames - 1, samples_per_sequence, dtype=int)

        for idx in indices:
            frame = seq.frames[idx]

            # Get ground truth 3D pose
            gt_3d = loader.to_h36m_format(frame)

            # Skip if too many missing joints
            if np.sum(np.all(gt_3d == 0, axis=1)) > 3:
                continue

            # Project to 2D
            pose_2d = project_3d_to_2d(gt_3d, camera, add_noise=True, noise_std=3.0)

            # Normalize 2D pose for input
            hip = pose_2d[0].copy()
            pose_2d_centered = pose_2d - hip
            torso_length = np.linalg.norm(pose_2d[8] - pose_2d[0])
            if torso_length > 1e-6:
                pose_2d_norm = pose_2d_centered / torso_length
            else:
                pose_2d_norm = pose_2d_centered

            # Test each method
            for method_name, lifter in methods.items():
                try:
                    pred_3d = lifter.lift(pose_2d_norm.flatten())

                    # Compute MPJPE
                    mpjpe = compute_mpjpe(pred_3d, gt_3d)
                    errors[method_name]['mpjpe'].append(mpjpe)

                    # Compute PA-MPJPE (Procrustes-aligned)
                    aligned_pred = procrustes_align(pred_3d, gt_3d)
                    pa_mpjpe = compute_mpjpe(aligned_pred, gt_3d)
                    errors[method_name]['pa_mpjpe'].append(pa_mpjpe)

                    # Per-joint errors
                    per_joint = compute_per_joint_mpjpe(pred_3d, gt_3d, STANDARD_JOINT_ORDER)
                    errors[method_name]['per_joint'].append(per_joint)

                except Exception as e:
                    print(f"Warning: {method_name} failed on a sample: {e}")
                    continue

    # Aggregate results
    results = {}
    for method_name in methods:
        if not errors[method_name]['mpjpe']:
            print(f"Warning: No valid samples for {method_name}")
            continue

        # Average per-joint errors
        avg_per_joint = {}
        if errors[method_name]['per_joint']:
            joint_errors = {}
            for pj in errors[method_name]['per_joint']:
                for joint, err in pj.items():
                    if joint not in joint_errors:
                        joint_errors[joint] = []
                    joint_errors[joint].append(err)

            avg_per_joint = {j: float(np.mean(e)) for j, e in joint_errors.items()}

        results[method_name] = BenchmarkResult(
            method_name=method_name,
            mpjpe_mm=float(np.mean(errors[method_name]['mpjpe'])),
            pa_mpjpe_mm=float(np.mean(errors[method_name]['pa_mpjpe'])),
            per_joint_mpjpe=avg_per_joint,
            n_samples=len(errors[method_name]['mpjpe'])
        )

    return results


def print_benchmark_results(results: Dict[str, BenchmarkResult]):
    """Print benchmark results in a nice format."""
    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)

    # Summary table
    print("\n{:<25} {:>12} {:>12} {:>10}".format(
        "Method", "MPJPE (mm)", "PA-MPJPE", "Samples"
    ))
    print("-" * 60)

    for name, result in sorted(results.items(), key=lambda x: x[1].mpjpe_mm):
        print("{:<25} {:>12.1f} {:>12.1f} {:>10}".format(
            name, result.mpjpe_mm, result.pa_mpjpe_mm, result.n_samples
        ))

    # Per-joint breakdown for best method
    best_method = min(results.values(), key=lambda x: x.mpjpe_mm)
    print(f"\n--- Per-Joint MPJPE for {best_method.method_name} ---")

    joints_sorted = sorted(
        best_method.per_joint_mpjpe.items(),
        key=lambda x: x[1],
        reverse=True
    )

    for joint, error in joints_sorted[:10]:
        bar_len = int(error / 10)  # Scale for display
        bar = "#" * min(bar_len, 50)
        print(f"  {joint:20s}: {error:6.1f} mm  {bar}")


def save_benchmark_results(results: Dict[str, BenchmarkResult], path: Path):
    """Save benchmark results to JSON."""
    data = {}
    for name, result in results.items():
        data[name] = {
            'mpjpe_mm': result.mpjpe_mm,
            'pa_mpjpe_mm': result.pa_mpjpe_mm,
            'per_joint_mpjpe': result.per_joint_mpjpe,
            'n_samples': result.n_samples
        }

    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"\nResults saved to: {path}")


if __name__ == '__main__':
    script_dir = Path(__file__).parent
    data_root = script_dir.parent

    print(f"Loading data from: {data_root}")

    loader = OBPDataLoader(data_root)

    # Initialize methods to compare
    methods = {
        'Geometric (baseline)': GeometricLifterWrapper(),
    }

    # Add OBP-trained model if available
    obp_model_path = script_dir / 'models' / 'obp_lifter.pt'
    if obp_model_path.exists():
        methods['OBP-trained NN'] = OBPLifterWrapper(obp_model_path)
        print(f"Loaded OBP-trained model from: {obp_model_path}")
    else:
        print(f"Warning: OBP model not found at {obp_model_path}")

    # Run benchmark on pitching data
    print("\n" + "=" * 70)
    print("PITCHING BENCHMARK")
    print("=" * 70)

    pitching_results = run_benchmark(
        loader,
        methods,
        dataset='baseball_pitching',
        n_sequences=30,
        samples_per_sequence=30
    )
    print_benchmark_results(pitching_results)

    # Run benchmark on hitting data
    print("\n" + "=" * 70)
    print("HITTING BENCHMARK")
    print("=" * 70)

    hitting_results = run_benchmark(
        loader,
        methods,
        dataset='baseball_hitting',
        n_sequences=30,
        samples_per_sequence=30
    )
    print_benchmark_results(hitting_results)

    # Save results
    output_dir = script_dir / 'models'
    save_benchmark_results(pitching_results, output_dir / 'benchmark_pitching.json')
    save_benchmark_results(hitting_results, output_dir / 'benchmark_hitting.json')

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if 'OBP-trained NN' in pitching_results and 'Geometric (baseline)' in pitching_results:
        geo_pitch = pitching_results['Geometric (baseline)'].mpjpe_mm
        obp_pitch = pitching_results['OBP-trained NN'].mpjpe_mm
        improvement = (geo_pitch - obp_pitch) / geo_pitch * 100

        print(f"\nPitching improvement: {improvement:.1f}% reduction in MPJPE")
        print(f"  Geometric: {geo_pitch:.1f} mm")
        print(f"  OBP-trained: {obp_pitch:.1f} mm")

    if 'OBP-trained NN' in hitting_results and 'Geometric (baseline)' in hitting_results:
        geo_hit = hitting_results['Geometric (baseline)'].mpjpe_mm
        obp_hit = hitting_results['OBP-trained NN'].mpjpe_mm
        improvement = (geo_hit - obp_hit) / geo_hit * 100

        print(f"\nHitting improvement: {improvement:.1f}% reduction in MPJPE")
        print(f"  Geometric: {geo_hit:.1f} mm")
        print(f"  OBP-trained: {obp_hit:.1f} mm")
