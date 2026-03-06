"""
3D visualization utilities using matplotlib.

For users who don't want to install Dash/Plotly, this provides
basic 3D skeleton visualization and kinematic plots.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Optional, Dict
import pandas as pd


# Skeleton connections (H36M format)
SKELETON_CONNECTIONS = [
    (0, 7), (7, 8), (8, 9), (9, 10),  # Spine
    (8, 11), (11, 12), (12, 13),      # Left arm
    (8, 14), (14, 15), (15, 16),      # Right arm
    (0, 4), (4, 5), (5, 6),           # Left leg
    (0, 1), (1, 2), (2, 3),           # Right leg
]

# Colors for body parts
COLORS = {
    'spine': '#e74c3c',
    'left_arm': '#3498db',
    'right_arm': '#9b59b6',
    'left_leg': '#2ecc71',
    'right_leg': '#f39c12'
}


def plot_skeleton_3d(joints_3d: np.ndarray,
                     ax: Optional[plt.Axes] = None,
                     title: str = "3D Pose",
                     elev: float = 20,
                     azim: float = 45) -> plt.Figure:
    """
    Plot a single 3D skeleton.

    Args:
        joints_3d: Array of shape (17, 3)
        ax: Existing axes to plot on
        title: Plot title
        elev: Elevation angle for view
        azim: Azimuth angle for view

    Returns:
        Figure object
    """
    if ax is None:
        fig = plt.figure(figsize=(8, 8), facecolor='#1a1a2e')
        ax = fig.add_subplot(111, projection='3d', facecolor='#1a1a2e')
    else:
        fig = ax.get_figure()

    # Plot bones
    for i, (start, end) in enumerate(SKELETON_CONNECTIONS):
        if i < 4:
            color = COLORS['spine']
        elif i < 7:
            color = COLORS['left_arm']
        elif i < 10:
            color = COLORS['right_arm']
        elif i < 13:
            color = COLORS['left_leg']
        else:
            color = COLORS['right_leg']

        ax.plot3D(
            [joints_3d[start, 0], joints_3d[end, 0]],
            [joints_3d[start, 1], joints_3d[end, 1]],
            [joints_3d[start, 2], joints_3d[end, 2]],
            color=color, linewidth=3
        )

    # Plot joints
    ax.scatter3D(
        joints_3d[:, 0],
        joints_3d[:, 1],
        joints_3d[:, 2],
        c='white', s=30, edgecolors='gray'
    )

    # Set axis properties
    max_range = np.max(np.abs(joints_3d)) * 1.2
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([0, max_range * 2])

    ax.set_xlabel('X', color='white')
    ax.set_ylabel('Y', color='white')
    ax.set_zlabel('Z', color='white')
    ax.set_title(title, color='white', fontsize=14)

    # Style
    ax.tick_params(colors='gray')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('gray')
    ax.yaxis.pane.set_edgecolor('gray')
    ax.zaxis.pane.set_edgecolor('gray')
    ax.grid(True, alpha=0.3)

    ax.view_init(elev=elev, azim=azim)

    return fig


def create_skeleton_animation(poses_3d: List,
                              fps: float = 30,
                              output_path: Optional[str] = None) -> FuncAnimation:
    """
    Create animated 3D skeleton visualization.

    Args:
        poses_3d: List of Pose3D objects
        fps: Frames per second
        output_path: If provided, save animation to file

    Returns:
        FuncAnimation object
    """
    fig = plt.figure(figsize=(10, 8), facecolor='#1a1a2e')
    ax = fig.add_subplot(111, projection='3d', facecolor='#1a1a2e')

    # Initialize plot elements
    lines = []
    for i, (start, end) in enumerate(SKELETON_CONNECTIONS):
        if i < 4:
            color = COLORS['spine']
        elif i < 7:
            color = COLORS['left_arm']
        elif i < 10:
            color = COLORS['right_arm']
        elif i < 13:
            color = COLORS['left_leg']
        else:
            color = COLORS['right_leg']
        line, = ax.plot3D([], [], [], color=color, linewidth=3)
        lines.append(line)

    scatter = ax.scatter3D([], [], [], c='white', s=30, edgecolors='gray')
    title = ax.set_title('', color='white', fontsize=14)

    # Calculate axis limits from all poses
    all_joints = np.array([p.joints_3d for p in poses_3d])
    max_range = np.max(np.abs(all_joints)) * 1.2

    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([0, max_range * 2])
    ax.set_xlabel('X', color='white')
    ax.set_ylabel('Y', color='white')
    ax.set_zlabel('Z', color='white')
    ax.tick_params(colors='gray')
    ax.grid(True, alpha=0.3)

    def init():
        for line in lines:
            line.set_data_3d([], [], [])
        return lines + [scatter]

    def update(frame):
        joints = poses_3d[frame].joints_3d
        timestamp = poses_3d[frame].timestamp

        # Update bones
        for i, (start, end) in enumerate(SKELETON_CONNECTIONS):
            lines[i].set_data_3d(
                [joints[start, 0], joints[end, 0]],
                [joints[start, 1], joints[end, 1]],
                [joints[start, 2], joints[end, 2]]
            )

        # Update joints
        scatter._offsets3d = (joints[:, 0], joints[:, 1], joints[:, 2])

        title.set_text(f't = {timestamp:.3f}s')

        return lines + [scatter, title]

    anim = FuncAnimation(
        fig, update, frames=len(poses_3d),
        init_func=init, blit=False,
        interval=1000/fps
    )

    if output_path:
        print(f"Saving animation to {output_path}...")
        anim.save(output_path, fps=fps, dpi=100,
                  savefig_kwargs={'facecolor': '#1a1a2e'})
        print("Done!")

    return anim


def plot_kinematic_sequence(df: pd.DataFrame,
                            events: Dict = None,
                            save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot kinematic sequence (pelvis → trunk → arm velocity).

    Args:
        df: DataFrame with angular velocity columns
        events: Dictionary of event timestamps
        save_path: Path to save figure

    Returns:
        Figure object
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), facecolor='#1a1a2e')

    colors = {
        'pelvis': '#ef4444',
        'torso': '#22c55e',
        'arm': '#f97316'
    }

    # Top plot: Kinematic sequence
    ax = axes[0]
    ax.set_facecolor('#1a1a2e')

    if 'pelvis_rotation_velocity' in df.columns:
        ax.plot(df['timestamp'], df['pelvis_rotation_velocity'],
                color=colors['pelvis'], linewidth=2, label='Pelvis')

    if 'torso_rotation_velocity' in df.columns:
        ax.plot(df['timestamp'], df['torso_rotation_velocity'],
                color=colors['torso'], linewidth=2, label='Trunk')

    # Find arm velocity column
    for col in ['right_shoulder_rotation_velocity', 'left_shoulder_rotation_velocity',
                'right_elbow_flexion_velocity', 'left_elbow_flexion_velocity']:
        if col in df.columns:
            ax.plot(df['timestamp'], df[col],
                    color=colors['arm'], linewidth=2, label='Arm')
            break

    # Add events
    if events:
        for name, time in events.items():
            if time is not None:
                ax.axvline(x=time, color='gray', linestyle='--', alpha=0.7)
                ax.text(time, ax.get_ylim()[1], name, rotation=90,
                       va='top', ha='right', color='gray', fontsize=8)

    ax.set_xlabel('Time (s)', color='white')
    ax.set_ylabel('Angular Velocity (°/s)', color='white')
    ax.set_title('Kinematic Sequence (Angular Velocity of Pelvis, Trunk and Arm)',
                color='white', fontsize=12)
    ax.legend(loc='upper right', facecolor='#2d2d44', edgecolor='gray',
             labelcolor='white')
    ax.tick_params(colors='white')
    ax.grid(True, alpha=0.3)
    ax.spines['bottom'].set_color('gray')
    ax.spines['left'].set_color('gray')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Bottom plot: X-factor
    ax = axes[1]
    ax.set_facecolor('#1a1a2e')

    if 'hip_shoulder_separation' in df.columns:
        ax.fill_between(df['timestamp'], 0, df['hip_shoulder_separation'],
                       color='#3b82f6', alpha=0.3)
        ax.plot(df['timestamp'], df['hip_shoulder_separation'],
                color='#3b82f6', linewidth=2)

    if events:
        for name, time in events.items():
            if time is not None:
                ax.axvline(x=time, color='gray', linestyle='--', alpha=0.7)

    ax.set_xlabel('Time (s)', color='white')
    ax.set_ylabel('Separation (°)', color='white')
    ax.set_title('X-Factor (Pelvis-Shoulder Separation Angle in Degrees)',
                color='white', fontsize=12)
    ax.tick_params(colors='white')
    ax.grid(True, alpha=0.3)
    ax.spines['bottom'].set_color('gray')
    ax.spines['left'].set_color('gray')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, facecolor='#1a1a2e', dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")

    return fig


def create_analysis_report(results: Dict,
                           output_dir: str,
                           prefix: str = 'swing') -> List[str]:
    """
    Create a full analysis report with multiple figures.

    Args:
        results: Pipeline results
        output_dir: Directory to save figures
        prefix: Filename prefix

    Returns:
        List of saved file paths
    """
    from pathlib import Path
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    saved_files = []
    df = results['timeseries_df']

    # Get events
    events = {}
    if 'events' in results:
        e = results['events']
        events = {
            'First Move': getattr(e, 'first_move', None),
            'Foot Plant': getattr(e, 'foot_plant', None),
            'Contact': getattr(e, 'contact', None)
        }

    # 1. Kinematic sequence
    fig = plot_kinematic_sequence(df, events)
    path = output_path / f'{prefix}_kinematic_sequence.png'
    fig.savefig(path, facecolor='#1a1a2e', dpi=150, bbox_inches='tight')
    saved_files.append(str(path))
    plt.close(fig)

    # 2. 3D skeleton at key frames
    if results.get('poses_3d'):
        # Skeleton at foot plant
        if events.get('Foot Plant'):
            fp_time = events['Foot Plant']
            fp_idx = np.argmin([abs(p.timestamp - fp_time) for p in results['poses_3d']])
            fig = plot_skeleton_3d(results['poses_3d'][fp_idx].joints_3d,
                                  title=f"Pose at Foot Plant (t={fp_time:.3f}s)")
            path = output_path / f'{prefix}_skeleton_foot_plant.png'
            fig.savefig(path, facecolor='#1a1a2e', dpi=150, bbox_inches='tight')
            saved_files.append(str(path))
            plt.close(fig)

        # Skeleton at contact
        if events.get('Contact'):
            c_time = events['Contact']
            c_idx = np.argmin([abs(p.timestamp - c_time) for p in results['poses_3d']])
            fig = plot_skeleton_3d(results['poses_3d'][c_idx].joints_3d,
                                  title=f"Pose at Contact (t={c_time:.3f}s)")
            path = output_path / f'{prefix}_skeleton_contact.png'
            fig.savefig(path, facecolor='#1a1a2e', dpi=150, bbox_inches='tight')
            saved_files.append(str(path))
            plt.close(fig)

    # 3. Joint angles summary
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor='#1a1a2e')

    angle_groups = [
        (['left_knee_flexion', 'right_knee_flexion'], 'Knee Flexion'),
        (['left_elbow_flexion', 'right_elbow_flexion'], 'Elbow Flexion'),
        (['left_hip_flexion', 'right_hip_flexion'], 'Hip Flexion'),
        (['left_shoulder_abduction', 'right_shoulder_abduction'], 'Shoulder Abduction')
    ]

    for ax, (cols, title) in zip(axes.flat, angle_groups):
        ax.set_facecolor('#1a1a2e')
        for col in cols:
            if col in df.columns:
                ax.plot(df['timestamp'], df[col], linewidth=2,
                       label=col.replace('_', ' ').title())

        for name, time in events.items():
            if time is not None:
                ax.axvline(x=time, color='gray', linestyle='--', alpha=0.5)

        ax.set_xlabel('Time (s)', color='white')
        ax.set_ylabel('Angle (°)', color='white')
        ax.set_title(title, color='white')
        ax.legend(loc='best', facecolor='#2d2d44', labelcolor='white', fontsize=8)
        ax.tick_params(colors='white')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = output_path / f'{prefix}_joint_angles.png'
    fig.savefig(path, facecolor='#1a1a2e', dpi=150, bbox_inches='tight')
    saved_files.append(str(path))
    plt.close(fig)

    print(f"Saved {len(saved_files)} figures to {output_dir}")
    return saved_files


if __name__ == "__main__":
    print("3D Visualization module loaded.")
    print("Usage:")
    print("  from visualize_3d import plot_skeleton_3d, create_analysis_report")
    print("  create_analysis_report(pipeline_results, './output')")
