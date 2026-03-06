"""
Visualization utilities for biomechanics data.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional
import pandas as pd


def plot_joint_angles(df: pd.DataFrame,
                      angles: List[str] = None,
                      events: dict = None,
                      title: str = "Joint Angles Over Time",
                      save_path: Optional[str] = None) -> None:
    """
    Plot joint angles over time.

    Args:
        df: DataFrame with timestamp and angle columns
        angles: List of angle column names to plot (default: all angle columns)
        events: Dictionary of event names to timestamps for vertical lines
        title: Plot title
        save_path: Path to save figure (None = show)
    """
    if angles is None:
        # Plot all angle columns (exclude velocities and timestamp)
        angles = [c for c in df.columns
                  if c != 'timestamp' and 'velocity' not in c]

    fig, ax = plt.subplots(figsize=(12, 6))

    for angle in angles:
        if angle in df.columns:
            ax.plot(df['timestamp'], df[angle], label=angle, linewidth=1.5)

    # Add event markers
    if events:
        colors = plt.cm.Set1(np.linspace(0, 1, len(events)))
        for (name, time), color in zip(events.items(), colors):
            if time is not None:
                ax.axvline(x=time, color=color, linestyle='--',
                           label=f'{name}: {time:.3f}s', alpha=0.7)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Angle (degrees)')
    ax.set_title(title)
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved plot to {save_path}")
    else:
        plt.show()


def plot_angular_velocities(df: pd.DataFrame,
                            joints: List[str] = None,
                            events: dict = None,
                            title: str = "Angular Velocities Over Time",
                            save_path: Optional[str] = None) -> None:
    """
    Plot angular velocities over time.

    Args:
        df: DataFrame with timestamp and velocity columns
        joints: List of joint names (will append _velocity)
        events: Dictionary of event names to timestamps
        title: Plot title
        save_path: Path to save figure
    """
    if joints is None:
        velocity_cols = [c for c in df.columns if 'velocity' in c]
    else:
        velocity_cols = [f'{j}_velocity' for j in joints if f'{j}_velocity' in df.columns]

    fig, ax = plt.subplots(figsize=(12, 6))

    for col in velocity_cols:
        label = col.replace('_velocity', '').replace('_', ' ')
        ax.plot(df['timestamp'], df[col], label=label, linewidth=1.5)

    if events:
        colors = plt.cm.Set1(np.linspace(0, 1, len(events)))
        for (name, time), color in zip(events.items(), colors):
            if time is not None:
                ax.axvline(x=time, color=color, linestyle='--',
                           label=f'{name}', alpha=0.7)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Angular Velocity (deg/s)')
    ax.set_title(title)
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
    else:
        plt.show()


def plot_pose_skeleton(keypoints: np.ndarray,
                       image: np.ndarray = None,
                       figsize: tuple = (8, 8),
                       save_path: Optional[str] = None) -> None:
    """
    Plot a skeleton from pose keypoints.

    Args:
        keypoints: Array of shape (17, 3) with x, y, confidence
        image: Optional background image
        figsize: Figure size
        save_path: Path to save figure
    """
    # YOLOv8 skeleton connections
    skeleton = [
        (0, 1), (0, 2), (1, 3), (2, 4),  # Head
        (5, 6),  # Shoulders
        (5, 7), (7, 9),  # Left arm
        (6, 8), (8, 10),  # Right arm
        (5, 11), (6, 12),  # Torso
        (11, 12),  # Hips
        (11, 13), (13, 15),  # Left leg
        (12, 14), (14, 16),  # Right leg
    ]

    fig, ax = plt.subplots(figsize=figsize)

    if image is not None:
        ax.imshow(image)

    # Plot connections
    for start, end in skeleton:
        if keypoints[start, 2] > 0.5 and keypoints[end, 2] > 0.5:
            ax.plot([keypoints[start, 0], keypoints[end, 0]],
                    [keypoints[start, 1], keypoints[end, 1]],
                    'b-', linewidth=2)

    # Plot keypoints
    for i, (x, y, conf) in enumerate(keypoints):
        if conf > 0.5:
            color = 'red' if i < 5 else 'green'  # Head vs body
            ax.plot(x, y, 'o', color=color, markersize=6)

    ax.set_aspect('equal')
    if image is None:
        ax.invert_yaxis()  # Image coordinates

    ax.set_title('Pose Skeleton')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
    else:
        plt.show()


def create_summary_report(results: dict, output_path: str) -> None:
    """
    Create a summary visualization report.

    Args:
        results: Output from VideoBiomechanicsPipeline.process_video()
        output_path: Path to save report image
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    df = results['timeseries_df']
    events = results['events']
    events_dict = {k: v for k, v in events.__dict__.items() if v is not None}

    # Plot 1: Knee angles
    ax = axes[0, 0]
    for col in ['left_knee_flexion', 'right_knee_flexion']:
        if col in df.columns:
            ax.plot(df['timestamp'], df[col], label=col.replace('_', ' '))
    for name, time in events_dict.items():
        ax.axvline(x=time, linestyle='--', alpha=0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Angle (deg)')
    ax.set_title('Knee Flexion')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Elbow angles
    ax = axes[0, 1]
    for col in ['left_elbow_flexion', 'right_elbow_flexion']:
        if col in df.columns:
            ax.plot(df['timestamp'], df[col], label=col.replace('_', ' '))
    for name, time in events_dict.items():
        ax.axvline(x=time, linestyle='--', alpha=0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Angle (deg)')
    ax.set_title('Elbow Flexion')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Torso lean
    ax = axes[1, 0]
    if 'torso_lean' in df.columns:
        ax.plot(df['timestamp'], df['torso_lean'], 'g-', linewidth=2)
    for name, time in events_dict.items():
        ax.axvline(x=time, linestyle='--', alpha=0.5, label=name)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Angle from vertical (deg)')
    ax.set_title('Torso Lean')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Angular velocities
    ax = axes[1, 1]
    velo_cols = [c for c in df.columns if 'velocity' in c and 'knee' in c]
    for col in velo_cols:
        ax.plot(df['timestamp'], df[col], label=col.replace('_', ' '))
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Angular Velocity (deg/s)')
    ax.set_title('Knee Angular Velocities')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle('Swing Biomechanics Summary', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Report saved to {output_path}")


if __name__ == "__main__":
    # Demo with synthetic data
    timestamps = np.linspace(0, 1, 100)
    df = pd.DataFrame({
        'timestamp': timestamps,
        'left_knee_flexion': 30 + 20 * np.sin(2 * np.pi * timestamps),
        'right_knee_flexion': 35 + 15 * np.sin(2 * np.pi * timestamps + 0.5),
        'left_elbow_flexion': 90 + 30 * np.sin(4 * np.pi * timestamps),
    })

    events = {
        'foot_plant': 0.3,
        'contact': 0.8
    }

    plot_joint_angles(df, events=events, title="Demo: Joint Angles")
