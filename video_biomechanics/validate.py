"""
Validation module for comparing pipeline output against OpenBiomechanics data.

Use this to:
1. Load OBP ground truth data
2. Compare your calculated metrics against OBP
3. Identify systematic errors in your pipeline
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import os


@dataclass
class ValidationResult:
    """Results from comparing calculated vs ground truth values."""
    metric_name: str
    calculated: float
    ground_truth: float
    error: float
    percent_error: float
    within_tolerance: bool


class OBPDataLoader:
    """Load and parse OpenBiomechanics Project data."""

    def __init__(self, obp_root: str):
        """
        Initialize loader.

        Args:
            obp_root: Path to openbiomechanics repository root
        """
        self.root = Path(obp_root)
        self.hitting_path = self.root / 'baseball_hitting' / 'data'
        self.pitching_path = self.root / 'baseball_pitching' / 'data'

    def load_hitting_poi(self) -> pd.DataFrame:
        """Load hitting point-of-interest metrics."""
        poi_path = self.hitting_path / 'poi' / 'poi.csv'

        if poi_path.exists():
            return pd.read_csv(poi_path)

        # Try alternate location
        alt_path = self.hitting_path / 'poi' / 'POI.csv'
        if alt_path.exists():
            return pd.read_csv(alt_path)

        raise FileNotFoundError(f"POI file not found at {poi_path}")

    def load_hitting_joint_angles(self) -> pd.DataFrame:
        """Load hitting full signal joint angles."""
        angles_path = self.hitting_path / 'full_sig' / 'joint_angles.csv'

        if angles_path.exists():
            return pd.read_csv(angles_path)

        raise FileNotFoundError(f"Joint angles file not found at {angles_path}")

    def load_hitting_metadata(self) -> pd.DataFrame:
        """Load hitting session metadata."""
        meta_path = self.hitting_path / 'metadata.csv'

        if meta_path.exists():
            return pd.read_csv(meta_path)

        raise FileNotFoundError(f"Metadata file not found at {meta_path}")

    def load_pitching_poi(self) -> pd.DataFrame:
        """Load pitching point-of-interest metrics."""
        poi_path = self.pitching_path / 'poi' / 'poi_metrics.csv'

        if poi_path.exists():
            return pd.read_csv(poi_path)

        raise FileNotFoundError(f"POI file not found at {poi_path}")

    def get_swing_data(self, session_swing: str) -> Dict:
        """
        Get all data for a specific swing.

        Args:
            session_swing: Unique swing identifier

        Returns:
            Dictionary with POI metrics, time series data, and metadata
        """
        data = {}

        try:
            poi_df = self.load_hitting_poi()
            swing_poi = poi_df[poi_df['session_swing'] == session_swing]
            if not swing_poi.empty:
                data['poi'] = swing_poi.iloc[0].to_dict()
        except FileNotFoundError:
            pass

        try:
            angles_df = self.load_hitting_joint_angles()
            swing_angles = angles_df[angles_df['session_swing'] == session_swing]
            if not swing_angles.empty:
                data['joint_angles'] = swing_angles
        except FileNotFoundError:
            pass

        return data


class PipelineValidator:
    """Validate pipeline output against ground truth data."""

    def __init__(self, tolerance_degrees: float = 5.0, tolerance_percent: float = 10.0):
        """
        Initialize validator.

        Args:
            tolerance_degrees: Absolute tolerance for angle comparisons (degrees)
            tolerance_percent: Relative tolerance for other metrics (%)
        """
        self.tolerance_degrees = tolerance_degrees
        self.tolerance_percent = tolerance_percent

        # Mapping from pipeline metric names to OBP metric names
        self.metric_mapping = {
            # Lead knee
            'lead_knee_launchpos_x': 'lead_knee_launchpos_x',
            'lead_knee_stride_max_x': 'lead_knee_stride_max_x',

            # Rear elbow
            'rear_elbow_fm_x': 'rear_elbow_fm_x',
            'rear_elbow_launchpos_x': 'rear_elbow_launchpos_x',

            # Pelvis
            'pelvis_angle_fp_x': 'pelvis_angle_fp_x',
            'pelvis_angle_fp_y': 'pelvis_angle_fp_y',
            'pelvis_angle_fp_z': 'pelvis_angle_fp_z',

            # Torso
            'torso_angle_fp_x': 'torso_angle_fp_x',
            'torso_angle_fp_y': 'torso_angle_fp_y',
            'torso_angle_fp_z': 'torso_angle_fp_z',

            # X-factor
            'x_factor_fp_z': 'x_factor_fp_z',
            'torso_pelvis_stride_max_z': 'torso_pelvis_stride_max_z',

            # Angular velocities
            'pelvis_angular_velocity_seq_max_x': 'pelvis_angular_velocity_seq_max_x',
            'torso_angular_velocity_seq_max_x': 'torso_angular_velocity_seq_max_x',
        }

    def validate_poi_metrics(self,
                             calculated: Dict,
                             ground_truth: Dict) -> List[ValidationResult]:
        """
        Validate calculated POI metrics against ground truth.

        Args:
            calculated: Dictionary of calculated metrics
            ground_truth: Dictionary of ground truth metrics from OBP

        Returns:
            List of ValidationResult objects
        """
        results = []

        for calc_name, obp_name in self.metric_mapping.items():
            calc_value = calculated.get(calc_name)
            gt_value = ground_truth.get(obp_name)

            if calc_value is None or gt_value is None:
                continue

            if pd.isna(calc_value) or pd.isna(gt_value):
                continue

            error = calc_value - gt_value
            percent_error = abs(error / gt_value) * 100 if gt_value != 0 else float('inf')

            # Determine if within tolerance
            is_angle = 'angle' in calc_name or 'elbow' in calc_name or 'knee' in calc_name
            if is_angle:
                within_tol = abs(error) <= self.tolerance_degrees
            else:
                within_tol = percent_error <= self.tolerance_percent

            results.append(ValidationResult(
                metric_name=calc_name,
                calculated=calc_value,
                ground_truth=gt_value,
                error=error,
                percent_error=percent_error,
                within_tolerance=within_tol
            ))

        return results

    def validate_time_series(self,
                             calculated_df: pd.DataFrame,
                             ground_truth_df: pd.DataFrame,
                             time_col: str = 'timestamp',
                             angle_cols: List[str] = None) -> Dict:
        """
        Validate time series data against ground truth.

        Args:
            calculated_df: DataFrame with calculated joint angles over time
            ground_truth_df: DataFrame with ground truth from OBP
            time_col: Name of time column
            angle_cols: List of angle columns to compare

        Returns:
            Dictionary with RMSE and correlation for each angle
        """
        if angle_cols is None:
            # Find common columns
            calc_cols = set(calculated_df.columns)
            gt_cols = set(ground_truth_df.columns)
            angle_cols = list(calc_cols & gt_cols - {time_col})

        results = {}

        for col in angle_cols:
            if col not in calculated_df.columns or col not in ground_truth_df.columns:
                continue

            # Interpolate to common time base
            calc_values = calculated_df[col].values
            gt_values = ground_truth_df[col].values

            # Resample if different lengths
            if len(calc_values) != len(gt_values):
                # Interpolate calculated to match ground truth length
                calc_interp = np.interp(
                    np.linspace(0, 1, len(gt_values)),
                    np.linspace(0, 1, len(calc_values)),
                    calc_values
                )
                calc_values = calc_interp

            # Remove NaN values
            mask = ~(np.isnan(calc_values) | np.isnan(gt_values))
            calc_clean = calc_values[mask]
            gt_clean = gt_values[mask]

            if len(calc_clean) < 2:
                continue

            # Calculate metrics
            rmse = np.sqrt(np.mean((calc_clean - gt_clean) ** 2))
            correlation = np.corrcoef(calc_clean, gt_clean)[0, 1]
            mean_error = np.mean(calc_clean - gt_clean)

            results[col] = {
                'rmse': rmse,
                'correlation': correlation,
                'mean_error': mean_error,
                'n_points': len(calc_clean)
            }

        return results

    def generate_report(self,
                        poi_results: List[ValidationResult],
                        timeseries_results: Dict = None) -> str:
        """
        Generate a validation report.

        Args:
            poi_results: Results from validate_poi_metrics
            timeseries_results: Results from validate_time_series

        Returns:
            Formatted report string
        """
        lines = []
        lines.append("=" * 60)
        lines.append("VALIDATION REPORT")
        lines.append("=" * 60)

        # POI Metrics
        lines.append("\nPOINT-OF-INTEREST METRICS")
        lines.append("-" * 40)

        if poi_results:
            passed = sum(1 for r in poi_results if r.within_tolerance)
            total = len(poi_results)
            lines.append(f"Passed: {passed}/{total} ({100*passed/total:.1f}%)")
            lines.append("")

            lines.append(f"{'Metric':<35} {'Calc':>8} {'GT':>8} {'Error':>8} {'Pass':>6}")
            lines.append("-" * 70)

            for r in poi_results:
                status = "✓" if r.within_tolerance else "✗"
                lines.append(f"{r.metric_name:<35} {r.calculated:>8.1f} {r.ground_truth:>8.1f} {r.error:>8.1f} {status:>6}")
        else:
            lines.append("No POI metrics to validate")

        # Time Series
        if timeseries_results:
            lines.append("\n\nTIME SERIES VALIDATION")
            lines.append("-" * 40)

            lines.append(f"{'Angle':<35} {'RMSE':>8} {'Corr':>8} {'Bias':>8}")
            lines.append("-" * 60)

            for col, metrics in timeseries_results.items():
                lines.append(f"{col:<35} {metrics['rmse']:>8.2f} {metrics['correlation']:>8.3f} {metrics['mean_error']:>8.2f}")

        lines.append("\n" + "=" * 60)

        return "\n".join(lines)


def compare_with_obp(pipeline_output: Dict,
                     obp_root: str,
                     session_swing: str = None) -> str:
    """
    Convenience function to compare pipeline output with OBP data.

    Args:
        pipeline_output: Output dictionary from VideoBiomechanicsPipeline
        obp_root: Path to openbiomechanics repository
        session_swing: Specific swing to compare against (optional)

    Returns:
        Validation report string
    """
    loader = OBPDataLoader(obp_root)
    validator = PipelineValidator()

    # Extract calculated metrics
    calculated_poi = {}
    if 'metrics' in pipeline_output:
        metrics = pipeline_output['metrics']
        calculated_poi = {k: v for k, v in metrics.__dict__.items() if v is not None}

    # Load ground truth
    ground_truth_poi = {}
    if session_swing:
        swing_data = loader.get_swing_data(session_swing)
        ground_truth_poi = swing_data.get('poi', {})
    else:
        # Use first swing from POI as example
        try:
            poi_df = loader.load_hitting_poi()
            if not poi_df.empty:
                ground_truth_poi = poi_df.iloc[0].to_dict()
        except FileNotFoundError:
            pass

    # Validate
    poi_results = validator.validate_poi_metrics(calculated_poi, ground_truth_poi)

    # Generate report
    report = validator.generate_report(poi_results)

    return report


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) < 2:
        # Use default OBP path relative to this file
        obp_root = Path(__file__).parent.parent
    else:
        obp_root = sys.argv[1]

    print(f"Loading OBP data from: {obp_root}")

    loader = OBPDataLoader(str(obp_root))

    try:
        poi_df = loader.load_hitting_poi()
        print(f"Loaded {len(poi_df)} swings from POI data")
        print(f"Columns: {list(poi_df.columns[:10])}...")

        # Show sample swing
        if not poi_df.empty:
            sample = poi_df.iloc[0]
            print(f"\nSample swing: {sample.get('session_swing', 'N/A')}")
            print(f"  Exit velocity: {sample.get('exit_velo_mph_x', 'N/A')} mph")
            print(f"  Bat speed: {sample.get('bat_speed_mph_contact_x', 'N/A')} mph")

    except FileNotFoundError as e:
        print(f"Could not load data: {e}")
        print("Make sure the OBP data files are present in the repository.")
