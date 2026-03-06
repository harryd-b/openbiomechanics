"""
Interactive dashboard for biomechanics visualization.

Creates a web-based dashboard similar to commercial tools like UPLIFT,
showing:
- 3D skeleton animation
- Kinematic sequence (pelvis → trunk → arm velocity)
- X-factor over time
- Synchronized video playback

Uses Dash (Plotly) for the web interface.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, List
import json

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Plotly not installed. Run: pip install plotly")

try:
    from dash import Dash, html, dcc, callback, Output, Input, State
    import dash_bootstrap_components as dbc
    DASH_AVAILABLE = True
except ImportError:
    DASH_AVAILABLE = False
    print("Dash not installed. Run: pip install dash dash-bootstrap-components")


# Skeleton connections for visualization (H36M format)
SKELETON_CONNECTIONS = [
    # Spine
    (0, 7), (7, 8), (8, 9), (9, 10),
    # Left arm
    (8, 11), (11, 12), (12, 13),
    # Right arm
    (8, 14), (14, 15), (15, 16),
    # Left leg
    (0, 4), (4, 5), (5, 6),
    # Right leg
    (0, 1), (1, 2), (2, 3),
]

JOINT_NAMES = [
    'Hip Center', 'Right Hip', 'Right Knee', 'Right Ankle',
    'Left Hip', 'Left Knee', 'Left Ankle',
    'Spine', 'Neck', 'Head', 'Head Top',
    'Left Shoulder', 'Left Elbow', 'Left Wrist',
    'Right Shoulder', 'Right Elbow', 'Right Wrist'
]


def create_skeleton_figure(joints_3d: np.ndarray,
                           title: str = "3D Pose") -> go.Figure:
    """
    Create a 3D skeleton visualization with UPLIFT-style floor grid.

    Args:
        joints_3d: Array of shape (17, 3) with joint positions
        title: Figure title

    Returns:
        Plotly Figure object
    """
    fig = go.Figure()

    # Create floor grid that fades into distance
    grid_size = 4  # meters
    grid_lines = 20
    grid_spacing = grid_size / grid_lines

    # Floor grid lines (X direction)
    for i in range(grid_lines + 1):
        y_pos = -grid_size/2 + i * grid_spacing
        # Fade opacity based on distance from center
        distance = abs(y_pos) / (grid_size/2)
        opacity = max(0.1, 0.5 * (1 - distance * 0.7))

        fig.add_trace(go.Scatter3d(
            x=[-grid_size/2, grid_size/2],
            y=[y_pos, y_pos],
            z=[0, 0],
            mode='lines',
            line=dict(color=f'rgba(100, 100, 120, {opacity})', width=1),
            showlegend=False,
            hoverinfo='skip'
        ))

    # Floor grid lines (Y direction)
    for i in range(grid_lines + 1):
        x_pos = -grid_size/2 + i * grid_spacing
        distance = abs(x_pos) / (grid_size/2)
        opacity = max(0.1, 0.5 * (1 - distance * 0.7))

        fig.add_trace(go.Scatter3d(
            x=[x_pos, x_pos],
            y=[-grid_size/2, grid_size/2],
            z=[0, 0],
            mode='lines',
            line=dict(color=f'rgba(100, 100, 120, {opacity})', width=1),
            showlegend=False,
            hoverinfo='skip'
        ))

    # Add shadow on the floor (projected skeleton)
    shadow_z = 0.001  # Just above floor
    shadow_color = 'rgba(0, 0, 0, 0.3)'
    for start, end in SKELETON_CONNECTIONS:
        fig.add_trace(go.Scatter3d(
            x=[joints_3d[start, 0], joints_3d[end, 0]],
            y=[joints_3d[start, 1], joints_3d[end, 1]],
            z=[shadow_z, shadow_z],
            mode='lines',
            line=dict(color=shadow_color, width=4),
            showlegend=False,
            hoverinfo='skip'
        ))

    # Add skeleton bones (red/coral color like UPLIFT)
    for start, end in SKELETON_CONNECTIONS:
        fig.add_trace(go.Scatter3d(
            x=[joints_3d[start, 0], joints_3d[end, 0]],
            y=[joints_3d[start, 1], joints_3d[end, 1]],
            z=[joints_3d[start, 2], joints_3d[end, 2]],
            mode='lines',
            line=dict(color='#ff4444', width=8),
            showlegend=False,
            hoverinfo='skip'
        ))

    # Add joint markers
    fig.add_trace(go.Scatter3d(
        x=joints_3d[:, 0],
        y=joints_3d[:, 1],
        z=joints_3d[:, 2],
        mode='markers',
        marker=dict(size=6, color='#ff6666', symbol='diamond'),
        text=JOINT_NAMES,
        hoverinfo='text',
        showlegend=False
    ))

    # Configure layout - infinite floor look
    fig.update_layout(
        title=None,
        scene=dict(
            xaxis=dict(
                visible=False,
                range=[-2, 2],
                showgrid=False,
                showbackground=False,
                zeroline=False
            ),
            yaxis=dict(
                visible=False,
                range=[-2, 2],
                showgrid=False,
                showbackground=False,
                zeroline=False
            ),
            zaxis=dict(
                visible=False,
                range=[-0.1, 2],
                showgrid=False,
                showbackground=False,
                zeroline=False
            ),
            aspectmode='data',
            aspectratio=dict(x=1, y=1, z=1),
            camera=dict(
                eye=dict(x=1.8, y=0.8, z=0.6),
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0.4)
            ),
            bgcolor='#2a2a3e'
        ),
        paper_bgcolor='#2a2a3e',
        plot_bgcolor='#2a2a3e',
        margin=dict(l=0, r=0, t=0, b=0),
        height=600
    )

    return fig


def smooth_signal(data: np.ndarray, window: int = 5) -> np.ndarray:
    """Apply simple moving average smoothing."""
    if len(data) < window:
        return data
    kernel = np.ones(window) / window
    # Pad to handle edges
    padded = np.pad(data, (window//2, window//2), mode='edge')
    smoothed = np.convolve(padded, kernel, mode='valid')
    return smoothed[:len(data)]


def create_kinematic_sequence_figure(df: pd.DataFrame,
                                     events: Dict = None) -> go.Figure:
    """
    Create kinematic sequence plot (pelvis → trunk → arm velocity).

    This is the classic graph showing the proximal-to-distal sequencing
    that's key to efficient swing mechanics.

    Args:
        df: DataFrame with angular velocity columns
        events: Dictionary of event timestamps

    Returns:
        Plotly Figure object
    """
    fig = go.Figure()

    # Colors matching UPLIFT style
    colors = {
        'pelvis': '#ef4444',   # Red
        'torso': '#22c55e',    # Green
        'arm': '#f97316'       # Orange
    }

    # Plot pelvis rotation velocity (smoothed)
    if 'pelvis_rotation_velocity' in df.columns:
        y_data = smooth_signal(df['pelvis_rotation_velocity'].fillna(0).values, window=7)
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=y_data,
            mode='lines',
            name='Pelvis',
            line=dict(color=colors['pelvis'], width=2)
        ))

    # Plot torso rotation velocity (smoothed)
    if 'torso_rotation_velocity' in df.columns:
        y_data = smooth_signal(df['torso_rotation_velocity'].fillna(0).values, window=7)
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=y_data,
            mode='lines',
            name='Trunk',
            line=dict(color=colors['torso'], width=2)
        ))

    # Plot arm/shoulder velocity (use elbow flexion velocity as proxy)
    arm_col = None
    for col in ['right_elbow_flexion_velocity', 'left_elbow_flexion_velocity',
                'right_shoulder_rotation_velocity', 'left_shoulder_rotation_velocity']:
        if col in df.columns:
            arm_col = col
            break

    if arm_col:
        y_data = smooth_signal(df[arm_col].fillna(0).values, window=7)
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=y_data,
            mode='lines',
            name='Arm',
            line=dict(color=colors['arm'], width=2)
        ))

    # Add event markers with staggered annotations
    if events:
        y_positions = [0.95, 0.85, 0.75]  # Stagger vertical positions
        for i, (event_name, event_time) in enumerate(events.items()):
            if event_time is not None:
                fig.add_vline(
                    x=event_time,
                    line_dash="dash",
                    line_color="gray"
                )
                # Add annotation separately with staggered y position
                fig.add_annotation(
                    x=event_time,
                    y=y_positions[i % len(y_positions)],
                    yref="paper",
                    text=event_name,
                    showarrow=False,
                    font=dict(size=10, color="gray"),
                    bgcolor="rgba(30,30,50,0.8)"
                )

    fig.update_layout(
        title="Kinematic Sequence (Angular Velocity of Pelvis, Trunk and Arm)",
        xaxis_title="Time (s)",
        yaxis_title="Angular Velocity (°/s)",
        template='plotly_dark',
        paper_bgcolor='#1a1a2e',
        plot_bgcolor='#1a1a2e',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=50, r=20, t=60, b=50),
        height=250
    )

    return fig


def create_xfactor_figure(df: pd.DataFrame,
                          events: Dict = None) -> go.Figure:
    """
    Create X-Factor (hip-shoulder separation) plot.

    Args:
        df: DataFrame with hip_shoulder_separation column
        events: Dictionary of event timestamps

    Returns:
        Plotly Figure object
    """
    fig = go.Figure()

    # Plot X-factor (smoothed)
    if 'hip_shoulder_separation' in df.columns:
        y_data = smooth_signal(df['hip_shoulder_separation'].fillna(0).values, window=7)
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=y_data,
            mode='lines',
            name='X-Factor',
            line=dict(color='#3b82f6', width=2),  # Blue
            fill='tozeroy',
            fillcolor='rgba(59, 130, 246, 0.2)'
        ))

    # Add event markers
    if events:
        for event_name, event_time in events.items():
            if event_time is not None:
                fig.add_vline(
                    x=event_time,
                    line_dash="dash",
                    line_color="gray"
                )

    fig.update_layout(
        title="X-Factor (Pelvis-Shoulder Separation Angle in Degrees)",
        xaxis_title="Time (s)",
        yaxis_title="Separation (°)",
        template='plotly_dark',
        paper_bgcolor='#1a1a2e',
        plot_bgcolor='#1a1a2e',
        margin=dict(l=50, r=20, t=60, b=50),
        height=200,
        showlegend=False
    )

    return fig


def create_joint_angle_figure(df: pd.DataFrame,
                              angles: List[str],
                              title: str = "Joint Angles",
                              events: Dict = None) -> go.Figure:
    """
    Create a multi-line plot for specified joint angles.

    Args:
        df: DataFrame with angle columns
        angles: List of column names to plot
        title: Figure title
        events: Dictionary of event timestamps

    Returns:
        Plotly Figure object
    """
    fig = go.Figure()

    colors = ['#ef4444', '#22c55e', '#3b82f6', '#f97316', '#a855f7', '#ec4899']

    for i, angle in enumerate(angles):
        if angle in df.columns:
            fig.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df[angle],
                mode='lines',
                name=angle.replace('_', ' ').title(),
                line=dict(color=colors[i % len(colors)], width=2)
            ))

    if events:
        for event_name, event_time in events.items():
            if event_time is not None:
                fig.add_vline(x=event_time, line_dash="dash", line_color="gray")

    fig.update_layout(
        title=title,
        xaxis_title="Time (s)",
        yaxis_title="Angle (°)",
        template='plotly_dark',
        paper_bgcolor='#1a1a2e',
        plot_bgcolor='#1a1a2e',
        height=250,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=20, t=60, b=50)
    )

    return fig


def create_dashboard_figures(results: Dict) -> Dict[str, go.Figure]:
    """
    Create all dashboard figures from pipeline results.

    Args:
        results: Output from VideoBiomechanicsPipeline.process_video()

    Returns:
        Dictionary of figure name -> Plotly Figure
    """
    figures = {}
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

    # Kinematic sequence
    figures['kinematic_sequence'] = create_kinematic_sequence_figure(df, events)

    # X-factor
    figures['xfactor'] = create_xfactor_figure(df, events)

    # 3D skeleton (middle frame)
    if results.get('poses_3d'):
        mid_idx = len(results['poses_3d']) // 2
        mid_pose = results['poses_3d'][mid_idx]
        figures['skeleton'] = create_skeleton_figure(mid_pose.joints_3d, "3D Pose")

    # Knee angles
    figures['knees'] = create_joint_angle_figure(
        df,
        ['left_knee_flexion', 'right_knee_flexion'],
        "Knee Flexion",
        events
    )

    # Elbow angles
    figures['elbows'] = create_joint_angle_figure(
        df,
        ['left_elbow_flexion', 'right_elbow_flexion'],
        "Elbow Flexion",
        events
    )

    return figures


def export_to_html(results: Dict, output_path: str) -> str:
    """
    Export dashboard to a standalone HTML file.

    Args:
        results: Pipeline results
        output_path: Path to save HTML file

    Returns:
        Path to saved file
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("Plotly required. Run: pip install plotly")

    figures = create_dashboard_figures(results)
    df = results['timeseries_df']

    # Create subplots layout
    fig = make_subplots(
        rows=3, cols=2,
        specs=[
            [{"type": "scene", "rowspan": 2}, {"type": "xy"}],
            [None, {"type": "xy"}],
            [{"type": "xy"}, {"type": "xy"}]
        ],
        subplot_titles=(
            "3D Pose", "Kinematic Sequence",
            "", "X-Factor",
            "Knee Flexion", "Elbow Flexion"
        ),
        vertical_spacing=0.08,
        horizontal_spacing=0.08
    )

    # Add skeleton traces to subplot
    if 'skeleton' in figures:
        for trace in figures['skeleton'].data:
            fig.add_trace(trace, row=1, col=1)

    # Add kinematic sequence
    for trace in figures['kinematic_sequence'].data:
        fig.add_trace(trace, row=1, col=2)

    # Add X-factor
    for trace in figures['xfactor'].data:
        fig.add_trace(trace, row=2, col=2)

    # Add knee angles
    for trace in figures['knees'].data:
        fig.add_trace(trace, row=3, col=1)

    # Add elbow angles
    for trace in figures['elbows'].data:
        fig.add_trace(trace, row=3, col=2)

    # Update layout
    fig.update_layout(
        title="Swing Biomechanics Analysis",
        template='plotly_dark',
        paper_bgcolor='#1a1a2e',
        plot_bgcolor='#1a1a2e',
        height=900,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.1)
    )

    # Scene settings for 3D plot
    if results.get('poses_3d'):
        mid_pose = results['poses_3d'][len(results['poses_3d']) // 2]
        max_range = np.max(np.abs(mid_pose.joints_3d)) * 1.2

        fig.update_scenes(
            xaxis=dict(range=[-max_range, max_range]),
            yaxis=dict(range=[-max_range, max_range]),
            zaxis=dict(range=[0, max_range * 2]),
            aspectmode='cube',
            camera=dict(eye=dict(x=1.5, y=1.5, z=0.5))
        )

    # Save
    fig.write_html(output_path)
    print(f"Dashboard exported to {output_path}")
    return output_path


def run_dash_app(results: Dict, video_path: Optional[str] = None, port: int = 8050):
    """
    Run interactive Dash web app.

    Args:
        results: Pipeline results
        video_path: Optional path to source video
        port: Port to run on
    """
    if not DASH_AVAILABLE:
        raise ImportError("Dash required. Run: pip install dash dash-bootstrap-components")

    app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])

    figures = create_dashboard_figures(results)
    df = results['timeseries_df']
    n_frames = len(df)

    # Build layout
    app.layout = dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H2("Swing Biomechanics Analysis", className="text-center my-4")
            ])
        ]),

        dbc.Row([
            # Left column: 3D skeleton
            dbc.Col([
                dcc.Graph(
                    id='skeleton-graph',
                    figure=figures.get('skeleton', go.Figure()),
                    config={'displayModeBar': False}
                )
            ], width=6),

            # Right column: Video placeholder + kinematic sequence
            dbc.Col([
                html.Div([
                    html.P("Video Source", className="text-muted text-end"),
                    html.Div(
                        "Video playback not implemented yet",
                        style={
                            'height': '200px',
                            'backgroundColor': '#2d2d44',
                            'display': 'flex',
                            'alignItems': 'center',
                            'justifyContent': 'center',
                            'borderRadius': '8px'
                        }
                    )
                ], className="mb-3"),
                dcc.Graph(
                    id='kinematic-sequence',
                    figure=figures['kinematic_sequence'],
                    config={'displayModeBar': False}
                )
            ], width=6)
        ]),

        dbc.Row([
            dbc.Col([
                dcc.Graph(
                    id='xfactor-graph',
                    figure=figures['xfactor'],
                    config={'displayModeBar': False}
                )
            ], width=12)
        ]),

        # Timeline slider
        dbc.Row([
            dbc.Col([
                html.Label("Timeline", className="text-muted"),
                dcc.Slider(
                    id='frame-slider',
                    min=0,
                    max=n_frames - 1,
                    value=n_frames // 2,
                    marks={
                        0: '0:00',
                        n_frames // 4: f'{df["timestamp"].iloc[n_frames//4]:.2f}s',
                        n_frames // 2: f'{df["timestamp"].iloc[n_frames//2]:.2f}s',
                        3 * n_frames // 4: f'{df["timestamp"].iloc[3*n_frames//4]:.2f}s',
                        n_frames - 1: f'{df["timestamp"].iloc[-1]:.2f}s'
                    },
                    tooltip={"placement": "bottom", "always_visible": True}
                )
            ], width=12)
        ], className="my-4"),

        # Bottom row: More graphs
        dbc.Row([
            dbc.Col([
                dcc.Graph(
                    id='knees-graph',
                    figure=figures['knees'],
                    config={'displayModeBar': False}
                )
            ], width=6),
            dbc.Col([
                dcc.Graph(
                    id='elbows-graph',
                    figure=figures['elbows'],
                    config={'displayModeBar': False}
                )
            ], width=6)
        ])
    ], fluid=True, style={'backgroundColor': '#1a1a2e', 'minHeight': '100vh'})

    # Callback to update skeleton on slider change
    @app.callback(
        Output('skeleton-graph', 'figure'),
        Input('frame-slider', 'value')
    )
    def update_skeleton(frame_idx):
        if results.get('poses_3d') and frame_idx < len(results['poses_3d']):
            pose = results['poses_3d'][frame_idx]
            return create_skeleton_figure(
                pose.joints_3d,
                f"3D Pose (t={pose.timestamp:.3f}s)"
            )
        return figures.get('skeleton', go.Figure())

    print(f"\nStarting dashboard at http://localhost:{port}")
    print("Press Ctrl+C to stop\n")
    app.run_server(debug=False, port=port)


if __name__ == "__main__":
    # Demo with synthetic data
    import sys

    print("Dashboard module loaded.")
    print("To use: ")
    print("  from dashboard import run_dash_app, export_to_html")
    print("  run_dash_app(pipeline_results)")
    print("  # or")
    print("  export_to_html(pipeline_results, 'report.html')")
