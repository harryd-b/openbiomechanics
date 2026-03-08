"""
Web application for video biomechanics analysis.

Upload 1-2 videos and get UPLIFT-style biomechanics visualization.

Run with:
    python app.py

Then open http://localhost:8050 in your browser.
"""

import os
import base64
import tempfile
from pathlib import Path
from typing import Optional
import uuid

import numpy as np
import pandas as pd

from dash import Dash, html, dcc, callback, Output, Input, State, ctx, no_update
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import our modules
from multiview import MultiViewPipeline, process_multiview
from dashboard import (
    create_skeleton_figure,
    create_kinematic_sequence_figure,
    create_xfactor_figure,
    create_joint_angle_figure,
    SKELETON_CONNECTIONS
)


# Font Awesome for icons
FA_CDN = "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css"

# Initialize app
app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY, FA_CDN],
    suppress_callback_exceptions=True
)

# Storage for processing results
RESULTS_CACHE = {}
UPLOAD_DIR = tempfile.mkdtemp()


def create_upload_card(camera_num: int):
    """Create an upload card for a camera with video preview."""
    label = "SIDE VIEW" if camera_num == 1 else "BACK VIEW"
    return html.Div([
        html.Div(label, className="section-title mb-2"),
        html.Div([
            # Upload zone (shown when no video)
            html.Div(id=f'upload-zone-{camera_num}', children=[
                dcc.Upload(
                    id=f'upload-video-{camera_num}',
                    children=html.Div([
                        html.I(className="fas fa-cloud-upload-alt",
                               style={'fontSize': '2.5rem', 'color': '#444', 'marginBottom': '12px'}),
                        html.Div("Drop video here", style={'color': '#666', 'fontSize': '0.9rem', 'marginBottom': '4px'}),
                        html.Div([
                            "or ",
                            html.Span("browse", style={'color': '#00d4aa', 'cursor': 'pointer'})
                        ], style={'color': '#555', 'fontSize': '0.85rem'})
                    ], style={'textAlign': 'center'}),
                    style={
                        'width': '100%',
                        'borderWidth': '2px',
                        'borderStyle': 'dashed',
                        'borderRadius': '10px',
                        'borderColor': '#333',
                        'padding': '40px 20px',
                        'cursor': 'pointer',
                        'minHeight': '180px',
                        'display': 'flex',
                        'alignItems': 'center',
                        'justifyContent': 'center',
                        'backgroundColor': 'rgba(30,30,48,0.5)',
                        'transition': 'all 0.2s ease'
                    },
                    multiple=False,
                    accept='video/*'
                ),
            ]),
            # Video preview (shown when video uploaded)
            html.Div(id=f'video-preview-{camera_num}', style={'display': 'none'}, children=[
                html.Div([
                    html.Video(
                        id=f'preview-player-{camera_num}',
                        controls=True,
                        style={'width': '100%', 'borderRadius': '8px', 'backgroundColor': '#000'}
                    ),
                    # Video controls bar
                    html.Div([
                        html.Button([
                            html.I(className="fas fa-expand", id=f'fullscreen-icon-{camera_num}')
                        ], id=f'fullscreen-btn-{camera_num}', className="video-control-btn", title="Fullscreen"),
                        html.Button([
                            html.I(className="fas fa-download")
                        ], id=f'download-btn-{camera_num}', className="video-control-btn", title="Download"),
                        html.Button([
                            html.I(className="fas fa-trash-alt")
                        ], id=f'remove-btn-{camera_num}', className="video-control-btn", title="Remove"),
                    ], className="video-controls-bar"),
                ], className="video-preview-wrapper"),
            ]),
            html.Div(id=f'upload-status-{camera_num}', className="mt-3 text-center",
                     style={'minHeight': '24px'})
        ], className="glass-card", style={'padding': '20px'})
    ])


# Custom CSS for professional UPLIFT-style interface
CUSTOM_CSS = """
/* Global typography */
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }

/* Main container */
.main-container { background: linear-gradient(180deg, #0f0f1a 0%, #1a1a2e 100%); }

/* Cards with glass-morphism effect */
.glass-card {
    background: linear-gradient(145deg, rgba(30,30,48,0.9) 0%, rgba(37,37,64,0.8) 100%);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    backdrop-filter: blur(10px);
    box-shadow: 0 4px 20px rgba(0,0,0,0.4);
}

/* Section titles */
.section-title {
    color: #888;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 12px;
    font-weight: 500;
}

/* Metric values - teal/cyan accent */
.metric-value {
    color: #00d4aa;
    font-size: 1.8rem;
    font-weight: 700;
    font-variant-numeric: tabular-nums;
}
.metric-value-lg {
    color: #00d4aa;
    font-size: 2.2rem;
    font-weight: 700;
}
.metric-label {
    color: #666;
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.metric-unit { color: #00d4aa; font-size: 0.9rem; }

/* Session header */
.session-header {
    background: transparent;
    padding: 16px 0;
    border-bottom: 1px solid rgba(255,255,255,0.05);
    margin-bottom: 24px;
}
.session-title {
    font-size: 1.4rem;
    font-weight: 600;
    color: #fff;
}
.session-subtitle { color: #666; font-size: 0.85rem; }

/* Badge styles */
.badge-analyzed {
    background: rgba(26,71,42,0.8);
    color: #4ade80;
    padding: 6px 16px;
    border-radius: 6px;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 1px;
    border: 1px solid rgba(74,222,128,0.2);
}

/* Visualize button - prominent red */
.btn-visualize {
    background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%);
    border: none;
    padding: 10px 28px;
    font-weight: 600;
    font-size: 0.9rem;
    border-radius: 8px;
    box-shadow: 0 4px 12px rgba(220,38,38,0.3);
    transition: all 0.2s ease;
}
.btn-visualize:hover {
    background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
    transform: translateY(-1px);
    box-shadow: 0 6px 16px rgba(220,38,38,0.4);
}

/* Video containers */
.video-container {
    background: #000;
    border-radius: 8px;
    overflow: hidden;
    position: relative;
    aspect-ratio: 16/9;
    border: 1px solid rgba(255,255,255,0.05);
}
.video-label {
    position: absolute;
    bottom: 8px;
    left: 8px;
    background: rgba(0,0,0,0.7);
    padding: 4px 10px;
    border-radius: 4px;
    font-size: 0.7rem;
    color: #aaa;
}

/* Energy leak indicators */
.leak-row { padding: 6px 0; }
.leak-dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    display: inline-block;
    margin-right: 10px;
    box-shadow: 0 0 8px currentColor;
}
.leak-good { background: #4ade80; color: #4ade80; }
.leak-bad { background: #f87171; color: #f87171; }
.leak-text { color: #aaa; font-size: 0.85rem; }

/* Sidebar sections */
.sidebar-card {
    background: linear-gradient(145deg, rgba(25,25,40,0.95) 0%, rgba(30,30,50,0.9) 100%);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 10px;
    padding: 16px;
    margin-bottom: 16px;
}
.sidebar-label { color: #555; font-size: 0.7rem; text-transform: uppercase; }
.sidebar-value { color: #ccc; font-size: 0.9rem; }

/* Info rows */
.info-row {
    display: flex;
    justify-content: space-between;
    padding: 8px 0;
    border-bottom: 1px solid rgba(255,255,255,0.04);
}
.info-row:last-child { border-bottom: none; }
.info-label { color: #666; font-size: 0.85rem; }
.info-value { color: #fff; font-size: 0.85rem; font-weight: 500; }
.info-value-accent { color: #00d4aa; font-size: 0.85rem; font-weight: 600; }

/* Tab styling */
.nav-tabs { border-bottom: 1px solid rgba(255,255,255,0.1); }
.nav-tabs .nav-link {
    color: #666;
    border: none;
    padding: 12px 20px;
    font-size: 0.85rem;
    font-weight: 500;
}
.nav-tabs .nav-link.active {
    color: #fff;
    background: transparent;
    border-bottom: 2px solid #00d4aa;
}
.nav-tabs .nav-link:hover { color: #aaa; }

/* Accordion styling (UPLIFT-style) */
.accordion { background: transparent !important; }
.accordion-item {
    background: #1e1e32 !important;
    border: none !important;
    border-bottom: 1px solid rgba(255,255,255,0.05) !important;
}
.accordion-button {
    background: #1e1e32 !important;
    color: #ccc !important;
    font-size: 0.9rem !important;
    padding: 12px 16px !important;
    border: none !important;
    box-shadow: none !important;
}
.accordion-button:not(.collapsed) {
    background: #252540 !important;
    color: #fff !important;
}
.accordion-button::after {
    filter: invert(1) brightness(0.7);
}
.accordion-button:focus {
    box-shadow: none !important;
}
.accordion-body {
    background: #1a1a2e !important;
    padding: 0 !important;
}

/* Timeline slider */
.rc-slider-track { background: #00d4aa; }
.rc-slider-handle { border-color: #00d4aa; }

/* Scrollbar styling */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #1a1a2e; }
::-webkit-scrollbar-thumb { background: #333; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #444; }

/* Upload cards */
.upload-card {
    background: linear-gradient(145deg, #1e1e30 0%, #252540 100%);
    border: 2px dashed #333;
    border-radius: 12px;
    transition: all 0.2s ease;
}
.upload-card:hover { border-color: #00d4aa; }

/* Process button */
.btn-process {
    background: linear-gradient(135deg, #00d4aa 0%, #00b894 100%);
    border: none;
    font-weight: 600;
    padding: 12px 32px;
    border-radius: 8px;
    color: #000;
}
.btn-process:hover {
    background: linear-gradient(135deg, #00e4ba 0%, #00c8a4 100%);
    color: #000;
}
.btn-process:disabled {
    background: #333;
    color: #666;
}

/* Video preview */
.video-preview-wrapper {
    position: relative;
    border-radius: 10px;
    overflow: hidden;
    background: #000;
}
.video-controls-bar {
    display: flex;
    gap: 8px;
    padding: 10px;
    background: rgba(0,0,0,0.7);
    justify-content: flex-end;
}
.video-control-btn {
    background: rgba(255,255,255,0.1);
    border: 1px solid rgba(255,255,255,0.2);
    color: #fff;
    padding: 8px 12px;
    border-radius: 6px;
    cursor: pointer;
    transition: all 0.2s ease;
}
.video-control-btn:hover {
    background: rgba(0,212,170,0.3);
    border-color: #00d4aa;
}

/* Loading spinner */
.processing-spinner {
    display: inline-block;
    width: 20px;
    height: 20px;
    border: 2px solid rgba(0,212,170,0.3);
    border-radius: 50%;
    border-top-color: #00d4aa;
    animation: spin 1s linear infinite;
    margin-right: 10px;
}
@keyframes spin {
    to { transform: rotate(360deg); }
}

/* Progress bar */
.progress-container {
    background: rgba(30,30,48,0.9);
    border-radius: 8px;
    padding: 20px;
    margin-top: 20px;
}
.progress-bar-custom {
    height: 6px;
    background: #222;
    border-radius: 3px;
    overflow: hidden;
}
.progress-bar-fill {
    height: 100%;
    background: linear-gradient(90deg, #00d4aa 0%, #00e4ba 100%);
    border-radius: 3px;
    transition: width 0.3s ease;
}

/* Narrow upload container */
.upload-container {
    max-width: 700px;
    margin: 0 auto;
}
"""

# Inject custom CSS via index_string
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>''' + CUSTOM_CSS + '''</style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Layout
app.layout = dbc.Container([

    # ===== UPLOAD PAGE =====
    html.Div(id='upload-page', children=[
        html.Div([  # Centered narrow container
            # Professional header
            html.Div([
                html.H1("Video Biomechanics", style={
                    'fontSize': '2rem', 'fontWeight': '700', 'color': '#fff',
                    'marginBottom': '8px', 'textAlign': 'center'
                }),
                html.P("Upload videos to analyze baseball swing mechanics", style={
                    'color': '#666', 'fontSize': '0.95rem', 'textAlign': 'center', 'marginBottom': '32px'
                })
            ], style={'paddingTop': '24px'}),

            # Training Data Selector
            html.Div([
                html.Div("LOAD TRAINING DATA", className="section-title"),
                html.Div([
                    dbc.Row([
                        dbc.Col([
                            dcc.Dropdown(
                                id='training-session-dropdown',
                                placeholder="Select a training session...",
                                style={'backgroundColor': '#1a1a2e', 'color': '#fff'}
                            ),
                        ], width=8),
                        dbc.Col([
                            dbc.Button("Load Session", id='load-session-btn', color="primary",
                                       outline=True, className="w-100", disabled=True)
                        ], width=4),
                    ]),
                    html.Div(id='load-session-status', className="mt-2",
                             style={'color': '#666', 'fontSize': '0.85rem'})
                ], className="glass-card", style={'padding': '16px'}),
            ], className="mb-4"),

            html.Div([
                html.Hr(style={'borderColor': '#333', 'margin': '20px 0'}),
                html.P("— OR upload new videos —", className="text-center",
                       style={'color': '#555', 'fontSize': '0.85rem'})
            ]),

            # Upload cards
            html.Div("VIDEO UPLOAD", className="section-title"),
            dbc.Row([
                dbc.Col([
                    create_upload_card(1)
                ], width=6, className="mb-4"),
                dbc.Col([
                    create_upload_card(2)
                ], width=6, className="mb-4"),
            ]),

            # UPLIFT Ground Truth upload (optional)
            html.Div(id='uplift-section', style={'display': 'none'}, children=[
                html.Div("GROUND TRUTH (OPTIONAL)", className="section-title mt-4"),
                html.Div([
                    dbc.Row([
                        dbc.Col([
                            dcc.Upload(
                                id='upload-uplift-csv',
                                children=html.Div([
                                    html.I(className="fas fa-file-csv", style={'fontSize': '1.5rem', 'color': '#444', 'marginRight': '12px'}),
                                    html.Span("Drop UPLIFT CSV here or ", style={'color': '#666'}),
                                    html.Span("browse", style={'color': '#00d4aa', 'cursor': 'pointer'})
                                ], style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center'}),
                                style={
                                    'width': '100%',
                                    'borderWidth': '1px',
                                    'borderStyle': 'dashed',
                                    'borderRadius': '8px',
                                    'borderColor': '#333',
                                    'padding': '15px',
                                    'cursor': 'pointer',
                                    'backgroundColor': 'rgba(30,30,48,0.5)',
                                },
                                accept='.csv'
                            ),
                        ], width=8),
                        dbc.Col([
                            html.Div(id='uplift-status', style={'color': '#666', 'fontSize': '0.85rem'})
                        ], width=4, className="d-flex align-items-center"),
                    ]),
                ], className="glass-card", style={'padding': '16px'}),
            ]),

            # Settings section (hidden until video uploaded)
            html.Div(id='settings-section', style={'display': 'none'}, children=[
                html.Div("SETTINGS", className="section-title mt-2"),
                html.Div([
                    dbc.Row([
                        dbc.Col([
                            html.Div("BATTING SIDE", className="metric-label mb-2"),
                            dbc.RadioItems(
                                id='bats-selector',
                                options=[
                                    {'label': ' Right', 'value': 'R'},
                                    {'label': ' Left', 'value': 'L'}
                                ],
                                value='R',
                                inline=True,
                                className="text-light"
                            )
                        ], width=3),
                        dbc.Col([
                            html.Div("CAMERA ANGLE", className="metric-label mb-2"),
                            dbc.Input(
                                id='camera-angle',
                                type='number',
                                value=90,
                                min=30,
                                max=180,
                                step=5,
                                style={'backgroundColor': '#1a1a2e', 'border': '1px solid #333',
                                       'color': '#fff', 'borderRadius': '6px'}
                            ),
                            html.Div("Angle between cameras (°)", style={'color': '#555', 'fontSize': '0.7rem', 'marginTop': '4px'})
                        ], width=3),
                        dbc.Col([
                            html.Div("SIDE DISTANCE", className="metric-label mb-2"),
                            dbc.Input(
                                id='camera-dist-1',
                                type='number',
                                value=2.9,
                                min=1,
                                max=10,
                                step=0.1,
                                style={'backgroundColor': '#1a1a2e', 'border': '1px solid #333',
                                       'color': '#fff', 'borderRadius': '6px'}
                            ),
                            html.Div("Distance in meters", style={'color': '#555', 'fontSize': '0.7rem', 'marginTop': '4px'})
                        ], width=3),
                        dbc.Col([
                            html.Div("BACK DISTANCE", className="metric-label mb-2"),
                            dbc.Input(
                                id='camera-dist-2',
                                type='number',
                                value=2.3,
                                min=1,
                                max=10,
                                step=0.1,
                                style={'backgroundColor': '#1a1a2e', 'border': '1px solid #333',
                                       'color': '#fff', 'borderRadius': '6px'}
                            ),
                            html.Div("Distance in meters", style={'color': '#555', 'fontSize': '0.7rem', 'marginTop': '4px'})
                        ], width=3),
                    ], className="mb-3"),
                    dbc.Row([
                        dbc.Col([
                            html.Div("BAT LENGTH", className="metric-label mb-2"),
                            dbc.Input(
                                id='bat-length',
                                type='number',
                                value=33,
                                min=28,
                                max=36,
                                step=0.5,
                                style={'backgroundColor': '#1a1a2e', 'border': '1px solid #333',
                                       'color': '#fff', 'borderRadius': '6px'}
                            ),
                            html.Div("Length in inches (for scale verification)", style={'color': '#555', 'fontSize': '0.7rem', 'marginTop': '4px'})
                        ], width=3),
                        dbc.Col([
                            html.Div("ATHLETE HEIGHT", className="metric-label mb-2"),
                            dbc.Input(
                                id='athlete-height',
                                type='number',
                                value=72,
                                min=48,
                                max=84,
                                step=1,
                                style={'backgroundColor': '#1a1a2e', 'border': '1px solid #333',
                                       'color': '#fff', 'borderRadius': '6px'}
                            ),
                            html.Div("Height in inches (optional, for scale)", style={'color': '#555', 'fontSize': '0.7rem', 'marginTop': '4px'})
                        ], width=3),
                        dbc.Col([
                            html.Div("SCALE REFERENCE", className="metric-label mb-2"),
                            dbc.RadioItems(
                                id='scale-reference',
                                options=[
                                    {'label': ' Plate', 'value': 'plate'},
                                    {'label': ' Bat', 'value': 'bat'},
                                    {'label': ' Height', 'value': 'height'}
                                ],
                                value='plate',
                                inline=True,
                                className="text-light"
                            ),
                            html.Div("Method to calibrate scale", style={'color': '#555', 'fontSize': '0.7rem', 'marginTop': '4px'})
                        ], width=6),
                    ], className="mb-3"),
                    dbc.Row([
                        dbc.Col([
                            html.Div("POSE METHODS", className="metric-label mb-2"),
                            dbc.Checklist(
                                id='method-selector',
                                options=[
                                    {'label': ' YOLO + Lifting', 'value': 'yolo_lifting'},
                                    {'label': ' MotionBERT', 'value': 'motionbert'},
                                    {'label': ' Triangulation (2 cameras)', 'value': 'triangulation'}
                                ],
                                value=['yolo_lifting', 'motionbert', 'triangulation'],
                                inline=True,
                                className="text-light",
                                style={'fontSize': '0.85rem'}
                            ),
                            html.Div("Select methods for pose estimation",
                                    style={'color': '#555', 'fontSize': '0.7rem', 'marginTop': '4px'})
                        ], width=12)
                    ], className="mb-3"),
                    dbc.Row([
                        dbc.Col([
                            dbc.Button(
                                "Process Videos",
                                id='process-btn',
                                className="btn-process w-100",
                                disabled=True
                            )
                        ], width=12)
                    ])
                ], className="glass-card", style={'padding': '20px'}),
            ]),

            # Processing status (shown during processing)
            html.Div(id='processing-container', style={'display': 'none'}, children=[
                html.Div([
                    html.Div([
                        html.Div(className="processing-spinner"),
                        html.Span("Processing videos...", style={'color': '#00d4aa', 'fontSize': '1rem'})
                    ], style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center', 'marginBottom': '16px'}),
                    html.Div([
                        html.Div(id='progress-bar-fill', className="progress-bar-fill", style={'width': '0%'})
                    ], className="progress-bar-custom"),
                    html.Div(id='processing-status', className="text-center mt-3", style={'color': '#666', 'fontSize': '0.85rem'})
                ], className="progress-container")
            ]),

            # Hidden progress bar for compatibility
            dbc.Progress(id='progress-bar', value=0, style={'display': 'none'}),
        ], className="upload-container"),
    ]),  # End of upload-page

    # Results section (hidden until processing complete)
    html.Div(id='results-section', style={'display': 'none'}, children=[

        # ===== SUMMARY PAGE (UPLIFT-style) =====
        html.Div(id='summary-page', children=[
            # Session header
            html.Div([
                dbc.Row([
                    dbc.Col([
                        html.H4(id='session-title', className="session-title mb-1"),
                        html.Span(id='capture-time', className="session-subtitle")
                    ], width=6),
                    dbc.Col([
                        html.Span("ANALYZED", className="badge-analyzed me-3"),
                        dbc.Button("Visualize", id='visualize-btn', className="btn-visualize")
                    ], width=6, className="text-end d-flex align-items-center justify-content-end")
                ])
            ], className="session-header"),

            dbc.Row([
                # Left column - Main content
                dbc.Col([
                    # Videos section
                    html.Div("VIDEOS", className="section-title"),
                    dbc.Row([
                        dbc.Col([
                            html.Div([
                                html.Video(
                                    id='video-player-1',
                                    controls=True,
                                    style={'width': '100%', 'height': '100%', 'objectFit': 'cover'},
                                ),
                                html.Div("SIDE VIEW", className="video-label")
                            ], className="video-container", id='video-thumb-1'),
                        ], width=6),
                        dbc.Col([
                            html.Div([
                                html.Video(
                                    id='video-player-2',
                                    controls=True,
                                    style={'width': '100%', 'height': '100%', 'objectFit': 'cover'},
                                ),
                                html.Div("BACK VIEW", className="video-label")
                            ], className="video-container", id='video-thumb-2'),
                        ], width=6),
                    ], className="mb-4"),

                    # Movement Metrics section
                    html.Div("MOVEMENT METRICS", className="section-title"),
                    dbc.Row([
                        # Kinematic Sequence card
                        dbc.Col([
                            html.Div([
                                html.Div("KINEMATIC SEQUENCE", className="section-title"),
                                html.Div([
                                    html.Span(id='kin-seq-order', className="metric-value"),
                                ], className="mb-3"),
                                dbc.Row([
                                    dbc.Col([
                                        html.Div("PELVIS", className="metric-label"),
                                        html.Div(id='peak-pelvis-vel', className="metric-value", style={'fontSize': '1.4rem'})
                                    ], width=4, className="text-center"),
                                    dbc.Col([
                                        html.Div("TRUNK", className="metric-label"),
                                        html.Div(id='peak-trunk-vel', className="metric-value", style={'fontSize': '1.4rem'})
                                    ], width=4, className="text-center"),
                                    dbc.Col([
                                        html.Div("ARM", className="metric-label"),
                                        html.Div(id='peak-arm-vel', className="metric-value", style={'fontSize': '1.4rem'})
                                    ], width=4, className="text-center"),
                                ], className="mb-3"),
                                html.Div([
                                    html.Span("Speed Gain (Pelvis→Trunk)", className="info-label"),
                                    html.Span(id='speed-gain', className="info-value-accent")
                                ], className="info-row")
                            ], className="glass-card", style={'padding': '20px'})
                        ], width=6, className="mb-3"),

                        # X-Factor card
                        dbc.Col([
                            html.Div([
                                html.Div("X-FACTOR", className="section-title"),
                                html.Div([
                                    html.Span("Peak", className="info-label"),
                                    html.Span(id='xfactor-peak', className="info-value-accent")
                                ], className="info-row"),
                                html.Div([
                                    html.Span("At Foot Plant", className="info-label"),
                                    html.Span(id='xfactor-fp', className="info-value")
                                ], className="info-row"),
                                html.Div([
                                    html.Span("At Contact", className="info-label"),
                                    html.Span(id='xfactor-contact', className="info-value")
                                ], className="info-row"),
                                html.Div([
                                    html.Span("Trunk Flexion at Contact", className="info-label"),
                                    html.Span(id='trunk-flexion', className="info-value")
                                ], className="info-row")
                            ], className="glass-card", style={'padding': '20px', 'height': '100%'})
                        ], width=6, className="mb-3"),
                    ]),

                    # Energy Leaks card
                    html.Div([
                        html.Div("ENERGY LEAKS", className="section-title"),
                        dbc.Row([
                            dbc.Col([
                                html.Div([
                                    html.Span(className="leak-dot leak-good", id='leak-sway'),
                                    html.Span("Sway", className="leak-text")
                                ], className="leak-row"),
                                html.Div([
                                    html.Span(className="leak-dot leak-good", id='leak-hip-hike'),
                                    html.Span("Hip Hike", className="leak-text")
                                ], className="leak-row"),
                                html.Div([
                                    html.Span(className="leak-dot leak-bad", id='leak-drift'),
                                    html.Span("Drifting Forward", className="leak-text")
                                ], className="leak-row"),
                            ], width=6),
                            dbc.Col([
                                html.Div([
                                    html.Span(className="leak-dot leak-good", id='leak-knee'),
                                    html.Span("Knee Dominance", className="leak-text")
                                ], className="leak-row"),
                                html.Div([
                                    html.Span(className="leak-dot leak-good", id='leak-early-ext'),
                                    html.Span("Coming Out of Swing", className="leak-text")
                                ], className="leak-row"),
                            ], width=6),
                        ])
                    ], className="glass-card", style={'padding': '20px'}),

                ], width=8),

                # Right sidebar
                dbc.Col([
                    # Session Details card
                    html.Div([
                        html.Div("SESSION DETAILS", className="section-title"),
                        html.Div([
                            html.Span("Activity", className="info-label"),
                            html.Span("Baseball", className="info-value")
                        ], className="info-row"),
                        html.Div([
                            html.Span("Movement", className="info-label"),
                            html.Span("Hitting", className="info-value")
                        ], className="info-row"),
                        html.Div([
                            html.Span("Handedness", className="info-label"),
                            html.Span(id='handedness-display', className="info-value")
                        ], className="info-row"),
                    ], className="sidebar-card"),

                    # Capture Info card
                    html.Div([
                        html.Div("CAPTURE INFO", className="section-title"),
                        html.Div([
                            html.Span("Frames", className="info-label"),
                            html.Span(id='frame-count', className="info-value-accent")
                        ], className="info-row"),
                        html.Div([
                            html.Span("Frame Rate", className="info-label"),
                            html.Span(id='fps-display', className="info-value-accent")
                        ], className="info-row"),
                        html.Div([
                            html.Span("Cameras", className="info-label"),
                            html.Span(id='camera-count', className="info-value")
                        ], className="info-row"),
                    ], className="sidebar-card"),

                    # Quality Assurance card
                    html.Div([
                        html.Div("QUALITY", className="section-title"),
                        html.Div([
                            html.Span("Pose Detection", className="info-label"),
                            html.Span("High", className="info-value", style={'color': '#4ade80'})
                        ], className="info-row"),
                        html.Div([
                            html.Span("Calibration", className="info-label"),
                            html.Span("Good", className="info-value", style={'color': '#4ade80'})
                        ], className="info-row"),
                    ], className="sidebar-card"),

                    # Export buttons
                    html.Div([
                        dbc.Button("Export CSV", id='export-csv-btn', color="secondary",
                                   outline=True, className="w-100 mb-2", size="sm"),
                        dbc.Button("Export Report", id='export-report-btn', color="secondary",
                                   outline=True, className="w-100 mb-2", size="sm"),
                        dbc.Button("Save Training Data", id='save-training-btn', color="primary",
                                   outline=True, className="w-100", size="sm"),
                    ]),
                    html.Div(id='save-training-status', className="mt-2 text-center",
                             style={'fontSize': '0.8rem'}),

                ], width=4),
            ]),
        ]),

        # ===== VISUALIZATION PAGE =====
        html.Div(id='visualization-page', style={'display': 'none'}, children=[
            # Back button and header
            dbc.Row([
                dbc.Col([
                    dbc.Button("← Back to Summary", id='back-to-summary-btn', color="link", className="ps-0")
                ], width=4),
                dbc.Col([
                    # Data source toggle
                    html.Div([
                        html.Span("DATA SOURCE:", className="metric-label me-2"),
                        dbc.RadioItems(
                            id='data-source-toggle',
                            options=[
                                {'label': ' Processed', 'value': 'processed'},
                                {'label': ' UPLIFT', 'value': 'uplift', 'disabled': True}
                            ],
                            value='processed',
                            inline=True,
                            className="text-light",
                            style={'display': 'inline-flex'}
                        )
                    ], style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center'})
                ], width=4, className="text-center"),
                dbc.Col([
                    html.H5("Visualization", className="text-end mb-0")
                ], width=4)
            ], className="mb-3"),

            # Tabs for different views (UPLIFT-style)
            dbc.Tabs([
                dbc.Tab(label="Analytics", tab_id="tab-analytics"),
                dbc.Tab(label="Shoulders", tab_id="tab-shoulders"),
                dbc.Tab(label="Pelvis", tab_id="tab-pelvis"),
                dbc.Tab(label="Arms", tab_id="tab-arms"),
                dbc.Tab(label="Legs", tab_id="tab-legs"),
                dbc.Tab(label="Correlation", tab_id="tab-correlation"),
            ], id="analysis-tabs", active_tab="tab-analytics", className="mb-4"),

            # Tab content
            html.Div(id='tab-content'),

            # Timeline with playback controls
            html.Div([
                # Time display
                html.Div([
                    html.Span(id='current-time-display', children="00:00",
                              style={'fontFamily': 'monospace', 'fontSize': '12px', 'color': '#888'})
                ], className="mb-1"),
                # Playback controls and slider
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            dbc.Button(html.I(className="fas fa-play"), id='play-btn',
                                       color="link", size="sm", className="me-2 p-1"),
                            dbc.Button(html.I(className="fas fa-step-backward"), id='prev-frame-btn',
                                       color="link", size="sm", className="me-1 p-1"),
                            dbc.Button(html.I(className="fas fa-step-forward"), id='next-frame-btn',
                                       color="link", size="sm", className="me-2 p-1"),
                        ], className="d-flex align-items-center")
                    ], width="auto"),
                    dbc.Col([
                        dcc.Slider(
                            id='frame-slider',
                            min=0,
                            max=100,
                            value=0,
                            tooltip={"placement": "bottom", "always_visible": False},
                            updatemode='drag'
                        )
                    ])
                ], className="align-items-center"),
                # Store for playback state
                dcc.Interval(id='playback-interval', interval=33, disabled=True),
                dcc.Store(id='is-playing', data=False)
            ], className="mt-3", style={'backgroundColor': 'rgba(30, 30, 50, 0.5)', 'padding': '10px', 'borderRadius': '8px'}),
        ]),

        # Download component
        dcc.Download(id='download-data')
    ]),

    # Store for session data
    dcc.Store(id='session-id'),
    dcc.Store(id='uploaded-videos', data={'video1': None, 'video2': None, 'uplift_csv': None}),
    dcc.Store(id='uplift-poses-data', data=None),  # Parsed UPLIFT 3D poses
    dcc.Store(id='data-source', data='processed'),  # 'processed' or 'uplift'
    dcc.Store(id='results-data'),
    dcc.Store(id='current-page', data='summary'),  # 'summary' or 'visualization'

    # Stores for skeleton viewer (in static layout for clientside callbacks)
    dcc.Store(id='skeleton-poses-data', data=None),
    dcc.Store(id='skeleton-frame-idx', data=0),

    # Dummy div for clientside callback outputs
    html.Div(id='clientside-output-dummy', style={'display': 'none'}),

], fluid=True, style={'backgroundColor': '#1a1a2e', 'minHeight': '100vh', 'padding': '20px'})


# Clientside callback for fullscreen video 1
app.clientside_callback(
    """
    function(n_clicks) {
        if (n_clicks) {
            var video = document.getElementById('preview-player-1');
            if (video && video.requestFullscreen) {
                video.requestFullscreen();
            }
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output('fullscreen-btn-1', 'n_clicks'),
    Input('fullscreen-btn-1', 'n_clicks'),
    prevent_initial_call=True
)

# Clientside callback for fullscreen video 2
app.clientside_callback(
    """
    function(n_clicks) {
        if (n_clicks) {
            var video = document.getElementById('preview-player-2');
            if (video && video.requestFullscreen) {
                video.requestFullscreen();
            }
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output('fullscreen-btn-2', 'n_clicks'),
    Input('fullscreen-btn-2', 'n_clicks'),
    prevent_initial_call=True
)

# Clientside callback for download video 1
app.clientside_callback(
    """
    function(n_clicks) {
        if (n_clicks) {
            var video = document.getElementById('preview-player-1');
            if (video && video.src) {
                var a = document.createElement('a');
                a.href = video.src;
                a.download = 'video_primary.mp4';
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
            }
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output('download-btn-1', 'n_clicks'),
    Input('download-btn-1', 'n_clicks'),
    prevent_initial_call=True
)

# Clientside callback for download video 2
app.clientside_callback(
    """
    function(n_clicks) {
        if (n_clicks) {
            var video = document.getElementById('preview-player-2');
            if (video && video.src) {
                var a = document.createElement('a');
                a.href = video.src;
                a.download = 'video_secondary.mp4';
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
            }
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output('download-btn-2', 'n_clicks'),
    Input('download-btn-2', 'n_clicks'),
    prevent_initial_call=True
)

# Clientside callback to show processing indicator immediately when button clicked
app.clientside_callback(
    """
    function(n_clicks) {
        if (n_clicks) {
            return [
                {'display': 'block'},  // Show processing container
                {'display': 'none'},   // Hide settings
                true                   // Disable button
            ];
        }
        return [window.dash_clientside.no_update, window.dash_clientside.no_update, window.dash_clientside.no_update];
    }
    """,
    [Output('processing-container', 'style', allow_duplicate=True),
     Output('settings-section', 'style', allow_duplicate=True),
     Output('process-btn', 'disabled', allow_duplicate=True)],
    Input('process-btn', 'n_clicks'),
    prevent_initial_call=True
)


# Clientside callback to parse poses from hidden div and set global variable
app.clientside_callback(
    """
    function(posesJsonText, frameIdxText) {
        if (!posesJsonText) return window.dash_clientside.no_update;

        // Parse poses from hidden div content
        try {
            window.skeletonPoses = JSON.parse(posesJsonText);
            window.skeletonFrame = parseInt(frameIdxText) || 0;
            console.log('Set skeletonPoses:', window.skeletonPoses.length, 'frames, frame:', window.skeletonFrame);
        } catch(e) {
            console.error('Error parsing poses:', e);
            return window.dash_clientside.no_update;
        }

        // Try to send to iframe immediately
        var iframe = document.getElementById('threejs-skeleton');
        if (iframe && iframe.contentWindow) {
            iframe.contentWindow.postMessage({type: 'setPoses', poses: window.skeletonPoses}, '*');
            iframe.contentWindow.postMessage({type: 'setFrame', frame: window.skeletonFrame}, '*');
        }

        return window.dash_clientside.no_update;
    }
    """,
    Output('clientside-output-dummy', 'children'),
    [Input('poses-json-holder', 'children'),
     Input('frame-idx-holder', 'children')],
    prevent_initial_call=True
)


# Clientside callback to update skeleton AND video when slider changes
app.clientside_callback(
    """
    function(frameIdx, resultsData) {
        window.skeletonFrame = frameIdx;

        // Update skeleton iframe
        var iframe = document.getElementById('threejs-skeleton');
        if (iframe && iframe.contentWindow) {
            iframe.contentWindow.postMessage({type: 'setFrame', frame: frameIdx}, '*');
        }

        // Sync video player currentTime
        var fps = (resultsData && resultsData.fps) ? resultsData.fps : 30;
        var videoTime = frameIdx / fps;
        var video = document.getElementById('viz-video-player');
        if (video && Math.abs(video.currentTime - videoTime) > 0.05) {
            video.currentTime = videoTime;
        }

        return window.dash_clientside.no_update;
    }
    """,
    Output('clientside-output-dummy', 'title'),
    [Input('frame-slider', 'value'),
     Input('results-data', 'data')],
    prevent_initial_call=True
)


# Set up iframe message listener when visualization page shows
app.clientside_callback(
    """
    function(vizStyle) {
        // Set up listener for iframe ready message
        if (!window.skeletonListenerSet) {
            window.skeletonListenerSet = true;
            window.addEventListener('message', function(event) {
                if (event.data && (event.data.type === 'skeletonReady' || event.data.type === 'requestPoses')) {
                    console.log('Iframe requested poses, have:', window.skeletonPoses ? window.skeletonPoses.length : 0);
                    if (window.skeletonPoses) {
                        var iframe = document.getElementById('threejs-skeleton');
                        if (iframe && iframe.contentWindow) {
                            iframe.contentWindow.postMessage({type: 'setPoses', poses: window.skeletonPoses}, '*');
                            iframe.contentWindow.postMessage({type: 'setFrame', frame: window.skeletonFrame || 0}, '*');
                        }
                    }
                }
            });
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output('visualization-page', 'className'),
    Input('visualization-page', 'style'),
    prevent_initial_call=True
)


# Callback to toggle between summary and visualization pages
@callback(
    [Output('summary-page', 'style'),
     Output('visualization-page', 'style'),
     Output('current-page', 'data')],
    [Input('visualize-btn', 'n_clicks'),
     Input('back-to-summary-btn', 'n_clicks')],
    State('current-page', 'data'),
    prevent_initial_call=True
)
def toggle_page(viz_clicks, back_clicks, current_page):
    triggered = ctx.triggered_id
    if triggered == 'visualize-btn':
        return {'display': 'none'}, {'display': 'block'}, 'visualization'
    elif triggered == 'back-to-summary-btn':
        return {'display': 'block'}, {'display': 'none'}, 'summary'
    return {'display': 'block'}, {'display': 'none'}, 'summary'


# Playback controls callbacks
@callback(
    [Output('playback-interval', 'disabled'),
     Output('is-playing', 'data'),
     Output('play-btn', 'children')],
    Input('play-btn', 'n_clicks'),
    State('is-playing', 'data'),
    prevent_initial_call=True
)
def toggle_playback(n_clicks, is_playing):
    if n_clicks is None:
        raise PreventUpdate
    new_state = not is_playing
    icon = html.I(className="fas fa-pause" if new_state else "fas fa-play")
    return not new_state, new_state, icon


@callback(
    Output('frame-slider', 'value', allow_duplicate=True),
    Input('playback-interval', 'n_intervals'),
    [State('frame-slider', 'value'),
     State('frame-slider', 'max'),
     State('is-playing', 'data')],
    prevent_initial_call=True
)
def auto_advance_frame(n_intervals, current_value, max_value, is_playing):
    if not is_playing or current_value is None:
        raise PreventUpdate
    new_value = current_value + 1
    if new_value > max_value:
        new_value = 0
    return new_value


@callback(
    Output('frame-slider', 'value', allow_duplicate=True),
    [Input('prev-frame-btn', 'n_clicks'),
     Input('next-frame-btn', 'n_clicks')],
    [State('frame-slider', 'value'),
     State('frame-slider', 'max')],
    prevent_initial_call=True
)
def step_frame(prev_clicks, next_clicks, current_value, max_value):
    triggered = ctx.triggered_id
    if triggered == 'prev-frame-btn':
        return max(0, current_value - 1)
    elif triggered == 'next-frame-btn':
        return min(max_value, current_value + 1)
    raise PreventUpdate


@callback(
    Output('current-time-display', 'children'),
    Input('frame-slider', 'value'),
    State('results-data', 'data'),
    prevent_initial_call=True
)
def update_time_display(frame_idx, results_data):
    if results_data is None:
        return "00:00"
    session_id = results_data.get('session_id', '')
    if session_id not in RESULTS_CACHE:
        return "00:00"
    results = RESULTS_CACHE[session_id]
    fps = results.get('fps', 30)
    time_sec = frame_idx / fps if fps else 0
    mins = int(time_sec // 60)
    secs = int(time_sec % 60)
    ms = int((time_sec % 1) * 100)
    return f"{mins:02d}:{secs:02d}.{ms:02d}"


@callback(
    Output('viz-video-player', 'src'),
    Input('visualization-page', 'style'),
    State('uploaded-videos', 'data'),
    prevent_initial_call=True
)
def update_viz_video(viz_style, uploaded_videos):
    if viz_style and viz_style.get('display') == 'none':
        raise PreventUpdate
    if not uploaded_videos or not uploaded_videos.get('video1'):
        return ""
    video_path = uploaded_videos['video1']
    try:
        with open(video_path, 'rb') as f:
            video_data = base64.b64encode(f.read()).decode('utf-8')
        return f"data:video/mp4;base64,{video_data}"
    except Exception:
        return ""


# Callback to populate summary page metrics
@callback(
    [Output('session-title', 'children'),
     Output('capture-time', 'children'),
     Output('kin-seq-order', 'children'),
     Output('peak-pelvis-vel', 'children'),
     Output('peak-trunk-vel', 'children'),
     Output('peak-arm-vel', 'children'),
     Output('speed-gain', 'children'),
     Output('xfactor-peak', 'children'),
     Output('xfactor-fp', 'children'),
     Output('xfactor-contact', 'children'),
     Output('trunk-flexion', 'children'),
     Output('leak-sway', 'className'),
     Output('leak-hip-hike', 'className'),
     Output('leak-drift', 'className'),
     Output('leak-knee', 'className'),
     Output('leak-early-ext', 'className'),
     Output('handedness-display', 'children'),
     Output('frame-count', 'children'),
     Output('fps-display', 'children'),
     Output('camera-count', 'children'),
     Output('video-player-1', 'src'),
     Output('video-player-2', 'src')],
    Input('results-data', 'data'),
    State('uploaded-videos', 'data'),
    prevent_initial_call=True
)
def update_summary_page(results_data, uploaded_videos):
    if results_data is None:
        raise PreventUpdate

    session_id = results_data.get('session_id', '')
    if session_id not in RESULTS_CACHE:
        raise PreventUpdate

    results = RESULTS_CACHE[session_id]
    df = results['timeseries_df']
    metrics = results.get('metrics')
    events = results.get('events')

    # Session info
    session_title = f"Session {session_id[:8]}..."
    capture_time = f"Captured on {pd.Timestamp.now().strftime('%b %d, %I:%M %p')}"

    # Calculate peak velocities and their timing
    pelvis_vel = df['pelvis_rotation_velocity'].abs().max() if 'pelvis_rotation_velocity' in df.columns else 0
    trunk_vel = df['torso_rotation_velocity'].abs().max() if 'torso_rotation_velocity' in df.columns else 0
    arm_vel = df['right_arm_rotation_velocity'].abs().max() if 'right_arm_rotation_velocity' in df.columns else 0

    # Kinematic sequence order - based on TIMING of peak (who fires first)
    # Get frame index of peak velocity for each segment
    pelvis_peak_idx = df['pelvis_rotation_velocity'].abs().idxmax() if 'pelvis_rotation_velocity' in df.columns else 0
    trunk_peak_idx = df['torso_rotation_velocity'].abs().idxmax() if 'torso_rotation_velocity' in df.columns else 0
    arm_peak_idx = df['right_arm_rotation_velocity'].abs().idxmax() if 'right_arm_rotation_velocity' in df.columns else 0

    # Sort by timing (earlier peak = fires first)
    timing = [('Pelvis', pelvis_peak_idx), ('Trunk', trunk_peak_idx), ('Arm', arm_peak_idx)]
    sorted_timing = sorted(timing, key=lambda x: x[1])
    kin_order = ' → '.join([t[0] for t in sorted_timing])

    # Speed gain
    speed_gain = f"{trunk_vel / pelvis_vel:.2f}x" if pelvis_vel > 0 else "N/A"

    # X-Factor values
    xf_col = 'hip_shoulder_separation' if 'hip_shoulder_separation' in df.columns else 'trunk_twist_clockwise'
    if xf_col in df.columns:
        xf_peak = f"{df[xf_col].max():.0f}°"
        # At foot plant (if we have event)
        fp_idx = getattr(events, 'foot_plant_idx', len(df)//2) if events else len(df)//2
        xf_fp = f"{df[xf_col].iloc[min(fp_idx, len(df)-1)]:.0f}°" if fp_idx else "N/A"
        # At contact
        contact_idx = getattr(events, 'contact_idx', -1) if events else -1
        xf_contact = f"{df[xf_col].iloc[contact_idx]:.0f}°" if contact_idx > 0 else "N/A"
    else:
        xf_peak = xf_fp = xf_contact = "N/A"

    # Trunk flexion
    trunk_flex = f"{df['trunk_global_flexion'].iloc[-1]:.0f}°" if 'trunk_global_flexion' in df.columns else "N/A"

    # Energy leak indicators (className based - good or bad)
    leak_good = "leak-dot leak-good"
    leak_bad = "leak-dot leak-bad"

    # Dynamic energy leak detection based on actual metrics
    # Sway: excessive lateral pelvis movement during load phase
    if 'pelvis_obliquity' in df.columns:
        sway_range = df['pelvis_obliquity'].max() - df['pelvis_obliquity'].min()
        leak_sway = leak_bad if sway_range > 15 else leak_good
    else:
        leak_sway = leak_good

    # Hip Hike: excessive vertical pelvis tilt variation
    if 'pelvis_tilt' in df.columns:
        hip_hike_range = df['pelvis_tilt'].max() - df['pelvis_tilt'].min()
        leak_hip = leak_bad if hip_hike_range > 20 else leak_good
    else:
        leak_hip = leak_good

    # Drifting Forward: COM movement towards pitcher during swing
    if 'trunk_center_of_mass_x' in df.columns:
        com_drift = df['trunk_center_of_mass_x'].iloc[-1] - df['trunk_center_of_mass_x'].iloc[0]
        leak_drift = leak_bad if abs(com_drift) > 0.15 else leak_good
    else:
        leak_drift = leak_good

    # Knee Dominance: lead knee extending too early
    if 'left_knee_extension' in df.columns and 'right_knee_extension' in df.columns:
        # Check for early extension before contact
        mid_idx = len(df) // 2
        early_ext = max(df['left_knee_extension'].iloc[:mid_idx].max(),
                       df['right_knee_extension'].iloc[:mid_idx].max())
        leak_knee = leak_bad if early_ext > 160 else leak_good
    else:
        leak_knee = leak_good

    # Early Extension: spine extending before contact
    if 'trunk_global_flexion' in df.columns:
        mid_idx = len(df) // 2
        early_trunk_ext = df['trunk_global_flexion'].iloc[:mid_idx].min()
        leak_ext = leak_bad if early_trunk_ext < -10 else leak_good
    else:
        leak_ext = leak_good

    # Session details
    handedness = "Right" if results_data.get('bats', 'R') == 'R' else "Left"
    frame_count = str(results_data.get('n_frames', len(df)))
    fps_val = f"{results.get('fps', 30):.0f} fps"
    camera_count = str(results_data.get('n_cameras', 1))

    # Video sources - read files and create data URLs for browser playback
    video1_src = ""
    video2_src = ""
    if uploaded_videos:
        if uploaded_videos.get('video1'):
            try:
                with open(uploaded_videos['video1'], 'rb') as f:
                    video_data = base64.b64encode(f.read()).decode('utf-8')
                    video1_src = f"data:video/mp4;base64,{video_data}"
            except Exception:
                video1_src = ""
        if uploaded_videos.get('video2'):
            try:
                with open(uploaded_videos['video2'], 'rb') as f:
                    video_data = base64.b64encode(f.read()).decode('utf-8')
                    video2_src = f"data:video/mp4;base64,{video_data}"
            except Exception:
                video2_src = ""

    return (
        session_title, capture_time, kin_order,
        f"{pelvis_vel:.0f}°/s", f"{trunk_vel:.0f}°/s", f"{arm_vel:.0f}°/s",
        speed_gain, xf_peak, xf_fp, xf_contact, trunk_flex,
        leak_sway, leak_hip, leak_drift, leak_knee, leak_ext,
        handedness, frame_count, fps_val, camera_count,
        video1_src, video2_src
    )


# Training session callbacks
@callback(
    Output('training-session-dropdown', 'options'),
    Input('upload-page', 'style'),  # Trigger on page load
)
def populate_training_sessions(_):
    """Populate dropdown with available training sessions."""
    training_dir = Path('training_data')
    if not training_dir.exists():
        return []

    sessions = []
    for session_dir in sorted(training_dir.glob('session_*')):
        if session_dir.is_dir():
            # Check what files exist
            has_videos = (session_dir / 'side.mp4').exists() or (session_dir / 'back.mp4').exists()
            has_uplift = (session_dir / 'uplift.csv').exists()
            has_poses = (session_dir / 'poses_3d.npy').exists()

            label = session_dir.name
            if has_uplift:
                label += " (UPLIFT)"
            if has_poses:
                label += " + processed"

            sessions.append({'label': label, 'value': session_dir.name})

    return sessions


@callback(
    Output('load-session-btn', 'disabled'),
    Input('training-session-dropdown', 'value')
)
def enable_load_button(session):
    return session is None


@callback(
    [Output('load-session-status', 'children'),
     Output('uploaded-videos', 'data', allow_duplicate=True),
     Output('uplift-poses-data', 'data', allow_duplicate=True),
     Output('process-btn', 'disabled', allow_duplicate=True),
     Output('upload-zone-1', 'style', allow_duplicate=True),
     Output('video-preview-1', 'style', allow_duplicate=True),
     Output('preview-player-1', 'src', allow_duplicate=True),
     Output('settings-section', 'style', allow_duplicate=True),
     Output('uplift-section', 'style', allow_duplicate=True),
     Output('data-source-toggle', 'options', allow_duplicate=True)],
    Input('load-session-btn', 'n_clicks'),
    [State('training-session-dropdown', 'value'),
     State('uploaded-videos', 'data')],
    prevent_initial_call=True
)
def load_training_session(n_clicks, session_name, uploaded_videos):
    """Load videos and UPLIFT data from a training session folder."""
    if not n_clicks or not session_name:
        raise PreventUpdate

    session_dir = Path('training_data') / session_name

    if not session_dir.exists():
        return (f"Session folder not found: {session_name}", uploaded_videos, None, True,
                no_update, no_update, no_update, no_update, no_update, no_update)

    # Find video files
    video_files = list(session_dir.glob('*.mp4')) + list(session_dir.glob('*.mov'))
    if not video_files:
        return (f"No video files found in {session_name}", uploaded_videos, None, True,
                no_update, no_update, no_update, no_update, no_update, no_update)

    # Load first video
    video_path = str(video_files[0])
    uploaded_videos['video1'] = video_path

    # Load second video if present
    if len(video_files) > 1:
        uploaded_videos['video2'] = str(video_files[1])

    # Find UPLIFT CSV
    uplift_poses = None
    uplift_files = list(session_dir.glob('*uplift*.csv')) + list(session_dir.glob('*ground_truth*.csv'))
    if uplift_files:
        try:
            import pandas as pd
            uplift_df = pd.read_csv(uplift_files[0])

            # Convert UPLIFT to H36M format
            UPLIFT_TO_H36M = {
                0: 'pelvis', 1: 'right_hip', 2: 'right_knee', 3: 'right_ankle',
                4: 'left_hip', 5: 'left_knee', 6: 'left_ankle', 7: 'spine_mid',
                8: 'neck', 9: 'neck', 10: 'head', 11: 'left_shoulder',
                12: 'left_elbow', 13: 'left_wrist', 14: 'right_shoulder',
                15: 'right_elbow', 16: 'right_wrist'
            }

            poses = []
            n_frames = len(uplift_df)

            for f in range(n_frames):
                pose = np.zeros((17, 3))
                for h36m_idx, uplift_joint in UPLIFT_TO_H36M.items():
                    for coord_idx, coord in enumerate(['x', 'y', 'z']):
                        col = f'{uplift_joint}_{coord}'
                        if col in uplift_df.columns:
                            pose[h36m_idx, coord_idx] = uplift_df[col].iloc[f]
                poses.append(pose.tolist())

            uplift_poses = poses
        except Exception as e:
            print(f"Failed to load UPLIFT data: {e}")

    # Generate video preview URL
    try:
        import base64
        with open(video_path, 'rb') as f:
            video_data = base64.b64encode(f.read()).decode('utf-8')
            video_src = f"data:video/mp4;base64,{video_data}"
    except Exception:
        video_src = ""

    # Update data source toggle options
    if uplift_poses:
        toggle_options = [
            {'label': ' Processed', 'value': 'processed'},
            {'label': ' UPLIFT', 'value': 'uplift'}
        ]
    else:
        toggle_options = [
            {'label': ' Processed', 'value': 'processed'},
            {'label': ' UPLIFT', 'value': 'uplift', 'disabled': True}
        ]

    status = html.Div([
        html.I(className="fas fa-check-circle", style={'color': '#4ade80', 'marginRight': '8px'}),
        html.Span(f"Loaded {session_name} ({len(video_files)} video(s)" +
                  (f", UPLIFT data" if uplift_poses else "") + ")",
                  style={'color': '#4ade80'})
    ])

    return (
        status,
        uploaded_videos,
        uplift_poses,
        False,  # Enable process button
        {'display': 'none'},  # Hide upload zone
        {'display': 'block'},  # Show video preview
        video_src,
        {'display': 'block'},  # Show settings
        {'display': 'block'},  # Show UPLIFT section
        toggle_options
    )


# Callbacks
@callback(
    [Output('upload-status-1', 'children'),
     Output('uploaded-videos', 'data', allow_duplicate=True),
     Output('process-btn', 'disabled', allow_duplicate=True),
     Output('upload-zone-1', 'style'),
     Output('video-preview-1', 'style'),
     Output('preview-player-1', 'src'),
     Output('settings-section', 'style', allow_duplicate=True),
     Output('uplift-section', 'style', allow_duplicate=True)],
    Input('upload-video-1', 'contents'),
    State('upload-video-1', 'filename'),
    State('uploaded-videos', 'data'),
    prevent_initial_call=True
)
def handle_upload_1(contents, filename, uploaded_videos):
    if contents is None:
        raise PreventUpdate

    # Save file
    filepath = save_uploaded_file(contents, filename, 1)
    uploaded_videos['video1'] = filepath

    # Truncate long filenames
    display_name = filename if len(filename) <= 25 else filename[:22] + "..."
    status = html.Div([
        html.I(className="fas fa-check-circle", style={'color': '#4ade80', 'marginRight': '8px'}),
        html.Span(display_name, style={'color': '#4ade80', 'fontSize': '0.85rem'})
    ], style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center'})

    # Enable process button if at least one video uploaded
    btn_disabled = uploaded_videos['video1'] is None

    return (
        status,
        uploaded_videos,
        btn_disabled,
        {'display': 'none'},  # Hide upload zone
        {'display': 'block'},  # Show video preview
        contents,  # Video source (data URL)
        {'display': 'block'},  # Show settings
        {'display': 'block'}  # Show UPLIFT section
    )


@callback(
    [Output('upload-status-2', 'children'),
     Output('uploaded-videos', 'data', allow_duplicate=True),
     Output('process-btn', 'disabled', allow_duplicate=True),
     Output('upload-zone-2', 'style'),
     Output('video-preview-2', 'style'),
     Output('preview-player-2', 'src'),
     Output('settings-section', 'style', allow_duplicate=True),
     Output('uplift-section', 'style', allow_duplicate=True)],
    Input('upload-video-2', 'contents'),
    State('upload-video-2', 'filename'),
    State('uploaded-videos', 'data'),
    prevent_initial_call=True
)
def handle_upload_2(contents, filename, uploaded_videos):
    if contents is None:
        raise PreventUpdate

    filepath = save_uploaded_file(contents, filename, 2)
    uploaded_videos['video2'] = filepath

    # Truncate long filenames
    display_name = filename if len(filename) <= 25 else filename[:22] + "..."
    status = html.Div([
        html.I(className="fas fa-check-circle", style={'color': '#4ade80', 'marginRight': '8px'}),
        html.Span(display_name, style={'color': '#4ade80', 'fontSize': '0.85rem'})
    ], style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center'})

    btn_disabled = uploaded_videos['video1'] is None

    return (
        status,
        uploaded_videos,
        btn_disabled,
        {'display': 'none'},  # Hide upload zone
        {'display': 'block'},  # Show video preview
        contents,  # Video source (data URL)
        {'display': 'block'},  # Show settings
        {'display': 'block'}  # Show UPLIFT section
    )


# Callback to handle UPLIFT CSV upload
@callback(
    [Output('uplift-status', 'children'),
     Output('uploaded-videos', 'data', allow_duplicate=True),
     Output('uplift-poses-data', 'data')],
    Input('upload-uplift-csv', 'contents'),
    State('upload-uplift-csv', 'filename'),
    State('uploaded-videos', 'data'),
    prevent_initial_call=True
)
def handle_uplift_upload(contents, filename, uploaded_videos):
    if contents is None:
        raise PreventUpdate

    import io

    # Decode the CSV content
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))

    # Save the CSV file
    csv_path = os.path.join(UPLOAD_DIR, f"uplift_{filename}")
    df.to_csv(csv_path, index=False)
    uploaded_videos['uplift_csv'] = csv_path

    # Parse UPLIFT data to H36M format (17 joints, 3 coords)
    H36M_JOINT_NAMES = [
        'pelvis', 'right_hip', 'right_knee', 'right_ankle',
        'left_hip', 'left_knee', 'left_ankle',
        'spine', 'neck', 'head', 'head_top',
        'left_shoulder', 'left_elbow', 'left_wrist',
        'right_shoulder', 'right_elbow', 'right_wrist'
    ]

    # UPLIFT column mapping
    joint_mapping = {
        0: ('pelvis_3d_x', 'pelvis_3d_y', 'pelvis_3d_z'),
        1: ('right_hip_jc_3d_x', 'right_hip_jc_3d_y', 'right_hip_jc_3d_z'),
        2: ('right_knee_jc_3d_x', 'right_knee_jc_3d_y', 'right_knee_jc_3d_z'),
        3: ('right_ankle_jc_3d_x', 'right_ankle_jc_3d_y', 'right_ankle_jc_3d_z'),
        4: ('left_hip_jc_3d_x', 'left_hip_jc_3d_y', 'left_hip_jc_3d_z'),
        5: ('left_knee_jc_3d_x', 'left_knee_jc_3d_y', 'left_knee_jc_3d_z'),
        6: ('left_ankle_jc_3d_x', 'left_ankle_jc_3d_y', 'left_ankle_jc_3d_z'),
        8: ('proximal_neck_3d_x', 'proximal_neck_3d_y', 'proximal_neck_3d_z'),
        9: ('mid_head_3d_x', 'mid_head_3d_y', 'mid_head_3d_z'),
        11: ('left_shoulder_jc_3d_x', 'left_shoulder_jc_3d_y', 'left_shoulder_jc_3d_z'),
        12: ('left_elbow_jc_3d_x', 'left_elbow_jc_3d_y', 'left_elbow_jc_3d_z'),
        13: ('left_wrist_jc_3d_x', 'left_wrist_jc_3d_y', 'left_wrist_jc_3d_z'),
        14: ('right_shoulder_jc_3d_x', 'right_shoulder_jc_3d_y', 'right_shoulder_jc_3d_z'),
        15: ('right_elbow_jc_3d_x', 'right_elbow_jc_3d_y', 'right_elbow_jc_3d_z'),
        16: ('right_wrist_jc_3d_x', 'right_wrist_jc_3d_y', 'right_wrist_jc_3d_z'),
    }

    n_frames = len(df)
    poses = np.zeros((n_frames, 17, 3))

    for joint_idx, cols in joint_mapping.items():
        if all(c in df.columns for c in cols):
            poses[:, joint_idx, 0] = df[cols[0]].values
            poses[:, joint_idx, 1] = df[cols[1]].values
            poses[:, joint_idx, 2] = df[cols[2]].values

    # Interpolate missing joints (spine=7, head_top=10)
    poses[:, 7] = (poses[:, 0] + poses[:, 8]) / 2  # spine
    poses[:, 10] = poses[:, 9] + (poses[:, 9] - poses[:, 8]) * 0.3  # head_top

    status = html.Div([
        html.I(className="fas fa-check-circle", style={'color': '#4ade80', 'marginRight': '8px'}),
        html.Span(f"{filename} ({n_frames} frames)", style={'color': '#4ade80', 'fontSize': '0.85rem'})
    ])

    # Convert to list for JSON serialization
    return status, uploaded_videos, poses.tolist()


# Callback to enable UPLIFT toggle when data is available
@callback(
    Output('data-source-toggle', 'options'),
    Input('uplift-poses-data', 'data'),
    prevent_initial_call=True
)
def enable_uplift_toggle(uplift_poses):
    if uplift_poses is not None and len(uplift_poses) > 0:
        return [
            {'label': ' Processed', 'value': 'processed'},
            {'label': ' UPLIFT', 'value': 'uplift'}  # Enable UPLIFT option
        ]
    return [
        {'label': ' Processed', 'value': 'processed'},
        {'label': ' UPLIFT', 'value': 'uplift', 'disabled': True}
    ]


# Callback to remove video 1
@callback(
    [Output('upload-zone-1', 'style', allow_duplicate=True),
     Output('video-preview-1', 'style', allow_duplicate=True),
     Output('preview-player-1', 'src', allow_duplicate=True),
     Output('uploaded-videos', 'data', allow_duplicate=True),
     Output('upload-status-1', 'children', allow_duplicate=True),
     Output('process-btn', 'disabled', allow_duplicate=True),
     Output('settings-section', 'style', allow_duplicate=True)],
    Input('remove-btn-1', 'n_clicks'),
    State('uploaded-videos', 'data'),
    prevent_initial_call=True
)
def remove_video_1(n_clicks, uploaded_videos):
    if n_clicks is None:
        raise PreventUpdate
    uploaded_videos['video1'] = None
    has_any_video = uploaded_videos.get('video2') is not None
    return (
        {'display': 'block'},  # Show upload zone
        {'display': 'none'},  # Hide preview
        '',  # Clear video src
        uploaded_videos,
        '',  # Clear status
        True,  # Disable process button (need video 1)
        {'display': 'block'} if has_any_video else {'display': 'none'}
    )


# Callback to remove video 2
@callback(
    [Output('upload-zone-2', 'style', allow_duplicate=True),
     Output('video-preview-2', 'style', allow_duplicate=True),
     Output('preview-player-2', 'src', allow_duplicate=True),
     Output('uploaded-videos', 'data', allow_duplicate=True),
     Output('upload-status-2', 'children', allow_duplicate=True)],
    Input('remove-btn-2', 'n_clicks'),
    State('uploaded-videos', 'data'),
    prevent_initial_call=True
)
def remove_video_2(n_clicks, uploaded_videos):
    if n_clicks is None:
        raise PreventUpdate
    uploaded_videos['video2'] = None
    return (
        {'display': 'block'},  # Show upload zone
        {'display': 'none'},  # Hide preview
        '',  # Clear video src
        uploaded_videos,
        ''  # Clear status
    )


def save_uploaded_file(contents: str, filename: str, camera_num: int) -> str:
    """Save uploaded file and return path."""
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)

    filepath = os.path.join(UPLOAD_DIR, f"camera_{camera_num}_{filename}")

    with open(filepath, 'wb') as f:
        f.write(decoded)

    return filepath


@callback(
    [Output('upload-page', 'style'),
     Output('results-section', 'style'),
     Output('results-data', 'data'),
     Output('processing-status', 'children'),
     Output('progress-bar', 'style'),
     Output('frame-slider', 'max'),
     Output('frame-slider', 'marks'),
     Output('processing-container', 'style'),
     Output('settings-section', 'style')],
    Input('process-btn', 'n_clicks'),
    State('uploaded-videos', 'data'),
    State('bats-selector', 'value'),
    State('camera-angle', 'value'),
    State('camera-dist-1', 'value'),
    State('camera-dist-2', 'value'),
    State('method-selector', 'value'),
    State('bat-length', 'value'),
    State('athlete-height', 'value'),
    State('scale-reference', 'value'),
    prevent_initial_call=True,
    background=False
)
def process_videos(n_clicks, uploaded_videos, bats, camera_angle, cam_dist_1, cam_dist_2,
                   selected_methods, bat_length_inches, athlete_height_inches, scale_reference):
    if n_clicks is None:
        raise PreventUpdate

    # Collect video paths
    video_paths = []
    if uploaded_videos.get('video1'):
        video_paths.append(uploaded_videos['video1'])
    if uploaded_videos.get('video2'):
        video_paths.append(uploaded_videos['video2'])

    if not video_paths:
        return (
            {'display': 'block'}, {'display': 'none'}, None, "No videos uploaded",
            {'display': 'none'}, 100, {}, {'display': 'none'}, {'display': 'block'}
        )

    # Validate method selection
    if not selected_methods:
        selected_methods = ['yolo_lifting', 'motionbert']  # Default

    # Triangulation requires 2 cameras
    if 'triangulation' in selected_methods and len(video_paths) < 2:
        selected_methods = [m for m in selected_methods if m != 'triangulation']
        print("[WARNING] Triangulation requires 2 cameras, removing from methods")

    try:
        # Process videos with calibration data
        import os
        print(f"[DEBUG] CWD: {os.getcwd()}")
        print(f"[DEBUG] Creating MultiViewPipeline with methods: {selected_methods}")
        pipeline = MultiViewPipeline(bats=bats, methods=selected_methods)
        print(f"[DEBUG] use_ensemble={pipeline.use_ensemble}")

        # Set camera parameters for 2-view setup with calibration
        if len(video_paths) == 2:
            camera_distances = [cam_dist_1 or 2.9, cam_dist_2 or 2.3]
            pipeline.estimate_cameras_from_videos(
                video_paths,
                camera_angle=camera_angle,
                camera_distances=camera_distances
            )
            print(f"Using calibrated cameras: distances={camera_distances}m, angle={camera_angle}°")

        results = pipeline.process_videos(video_paths)

        # Scale reference information
        bat_length_m = (bat_length_inches or 33) * 0.0254  # Convert inches to meters
        athlete_height_m = (athlete_height_inches or 72) * 0.0254  # Convert inches to meters

        # Store scale reference info in results
        results['scale_info'] = {
            'reference': scale_reference or 'plate',
            'bat_length_m': bat_length_m,
            'bat_length_inches': bat_length_inches or 33,
            'athlete_height_m': athlete_height_m,
            'athlete_height_inches': athlete_height_inches or 72,
        }

        # Compute actual measurements from poses for scale verification
        poses_3d = results.get('poses_3d', [])
        if poses_3d and len(poses_3d) > 0:
            poses_array = np.array(poses_3d)
            # Measure ankle-to-head distance (proxy for height)
            measured_height = np.nanmean(np.linalg.norm(
                poses_array[:, 10] - poses_array[:, 6], axis=1))  # head - l_ankle

            results['scale_info']['measured_height_m'] = float(measured_height)

            # Compute scale factor if using bat or height reference
            if scale_reference == 'bat':
                # Would need bat tracking to compute this
                print(f"[SCALE] Bat reference: {bat_length_inches}\" ({bat_length_m:.3f}m)")
            elif scale_reference == 'height':
                scale_factor = athlete_height_m / measured_height if measured_height > 0 else 1.0
                results['scale_info']['scale_factor'] = float(scale_factor)
                print(f"[SCALE] Height reference: target={athlete_height_m:.2f}m, measured={measured_height:.2f}m, factor={scale_factor:.3f}")
            else:
                print(f"[SCALE] Plate reference: measured height={measured_height:.2f}m")

        # Save results to files for analysis
        output_dir = Path('output')
        output_dir.mkdir(exist_ok=True)

        # Save timeseries data
        results['timeseries_df'].to_csv(output_dir / 'timeseries.csv', index=False)
        print(f"\nSaved timeseries to {output_dir / 'timeseries.csv'}")

        # Save 3D poses
        poses_3d = results.get('poses_3d', [])
        if poses_3d:
            poses_array = np.array(poses_3d)
            np.save(output_dir / 'poses_3d.npy', poses_array)

            # Also save as CSV for easy viewing
            n_frames, n_joints, n_coords = poses_array.shape
            pose_rows = []
            for f in range(n_frames):
                row = {'frame': f}
                for j in range(n_joints):
                    row[f'joint{j}_x'] = poses_array[f, j, 0]
                    row[f'joint{j}_y'] = poses_array[f, j, 1]
                    row[f'joint{j}_z'] = poses_array[f, j, 2]
                pose_rows.append(row)
            pd.DataFrame(pose_rows).to_csv(output_dir / 'poses_3d.csv', index=False)
            print(f"Saved 3D poses ({n_frames} frames, {n_joints} joints) to {output_dir / 'poses_3d.csv'}")

        # Store results
        session_id = str(uuid.uuid4())
        RESULTS_CACHE[session_id] = results

        # Prepare serializable data for dcc.Store
        n_frames = len(results['joint_angles'])
        df = results['timeseries_df']

        results_data = {
            'session_id': session_id,
            'n_frames': n_frames,
            'fps': results['fps'],
            'n_cameras': results['n_cameras'],
            'bats': bats,
            'scale_info': results.get('scale_info', {}),
        }

        # Create slider marks
        marks = {}
        if n_frames > 0:
            for i in [0, n_frames // 4, n_frames // 2, 3 * n_frames // 4, n_frames - 1]:
                if i < len(df):
                    marks[i] = f"{df['timestamp'].iloc[i]:.2f}s"

        return (
            {'display': 'none'},  # Hide upload page
            {'display': 'block'},  # Show results section
            results_data,
            f"Processed {n_frames} frames from {results['n_cameras']} camera(s)",
            {'display': 'none'},
            n_frames - 1 if n_frames > 0 else 100,
            marks,
            {'display': 'none'},  # Hide processing container
            {'display': 'none'}  # Hide settings
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        return (
            {'display': 'block'}, {'display': 'none'}, None, f"Error: {str(e)}",
            {'display': 'none'}, 100, {}, {'display': 'none'}, {'display': 'block'}
        )


@callback(
    Output('tab-content', 'children'),
    [Input('analysis-tabs', 'active_tab'),
     Input('frame-slider', 'value'),
     Input('visualization-page', 'style'),
     Input('data-source-toggle', 'value')],  # Data source selection
    [State('results-data', 'data'),
     State('uploaded-videos', 'data'),
     State('uplift-poses-data', 'data')]
)
def render_tab_content(active_tab, frame_idx, viz_style, data_source, results_data, uploaded_videos, uplift_poses):
    if results_data is None:
        raise PreventUpdate

    # Don't render if visualization page is hidden
    if viz_style and viz_style.get('display') == 'none':
        raise PreventUpdate

    session_id = results_data['session_id']
    if session_id not in RESULTS_CACHE:
        raise PreventUpdate

    results = RESULTS_CACHE[session_id]
    df = results['timeseries_df']
    events = results.get('events')

    events_dict = {}
    if events:
        events_dict = {
            'First Move': getattr(events, 'first_move', None),
            'Foot Plant': getattr(events, 'foot_plant', None),
            'Contact': getattr(events, 'contact', None)
        }

    # Default to analytics tab if none selected
    if not active_tab:
        active_tab = "tab-analytics"

    # All tabs use the same layout: graphs on left, skeleton on right
    return render_tab_with_skeleton(active_tab, results, frame_idx, events_dict, uploaded_videos,
                                    data_source, uplift_poses)


def render_tab_with_skeleton(active_tab, results, frame_idx, events_dict, uploaded_videos=None,
                             data_source='processed', uplift_poses=None):
    """Render any tab with UPLIFT-style layout: graphs left, skeleton center, video top-right."""
    import json

    df = results['timeseries_df']

    # Choose pose data source
    using_uplift = data_source == 'uplift' and uplift_poses is not None
    if using_uplift:
        poses_3d = [np.array(p) for p in uplift_poses]
        print(f"[VIEWER] Using UPLIFT poses ({len(poses_3d)} frames)")
    else:
        poses_3d = results.get('poses_3d', [])
        print(f"[VIEWER] Using processed poses ({len(poses_3d)} frames)")

    # Prepare poses for Three.js viewer (fix coordinates)
    fixed_poses = []
    last_good_head = None  # Track last good head position for interpolation

    for pose in poses_3d:
        if pose is None:
            fixed_poses.append([[0, 0, 0]] * 17)
            continue
        pose = pose.copy()
        max_reasonable = 5.0

        # Fix bad joint values - use last good value or interpolate
        for j in range(len(pose)):
            if np.any(np.abs(pose[j]) > max_reasonable):
                if j == 9 or j == 10:
                    # Head/neck - use last good head position or estimate from spine
                    if last_good_head is not None:
                        pose[j] = last_good_head.copy()
                    elif len(pose) > 8 and np.all(np.abs(pose[8]) < max_reasonable):
                        pose[j] = pose[8] + np.array([0, 0, 0.15])
                    else:
                        pose[j] = np.array([0, 0, 1.7])
                else:
                    pose[j] = np.array([0, 0, 0])

        # Track good head position for next frame
        if np.all(np.abs(pose[10]) < max_reasonable):
            last_good_head = pose[10].copy()

        # Fix coordinate system for Three.js viewer (Y-up)
        fixed = np.zeros_like(pose)
        if using_uplift:
            # UPLIFT data is already Y-up, just use as-is with minor adjustments
            fixed[:, 0] = pose[:, 0]   # X as-is
            fixed[:, 1] = pose[:, 1]   # Y is vertical (up)
            fixed[:, 2] = -pose[:, 2]  # Flip Z so skeleton faces camera
        else:
            # Processed data needs coordinate transformation
            fixed[:, 0] = -pose[:, 0]  # Flip X for mirror effect
            fixed[:, 1] = pose[:, 1]   # Y as vertical
            fixed[:, 2] = -pose[:, 2]  # Flip Z for camera direction

        # Floor the skeleton (shift Y so minimum is at ground level)
        fixed[:, 1] = fixed[:, 1] - fixed[:, 1].min() + 0.02
        fixed_poses.append(fixed.tolist())

    # Get graphs for the active tab
    processed_poses_for_correlation = results.get('poses_3d', [])
    graphs_content = get_tab_graphs(active_tab, df, results, frame_idx, events_dict,
                                     processed_poses_for_correlation, uplift_poses)

    # Convert poses to JSON for iframe
    poses_json = json.dumps(fixed_poses)

    # Get video source directly
    video_src = ""
    if uploaded_videos and uploaded_videos.get('video1'):
        try:
            with open(uploaded_videos['video1'], 'rb') as f:
                video_data = base64.b64encode(f.read()).decode('utf-8')
                video_src = f"data:video/mp4;base64,{video_data}"
        except Exception:
            pass

    # UPLIFT-style layout with Three.js iframe and video overlay
    # Note: poses_json is embedded in a hidden div for the clientside callback to find
    return html.Div([
        # Hidden divs for clientside callbacks (FPS comes from results-data store)
        html.Div(id='poses-json-holder', children=poses_json, style={'display': 'none'}),
        html.Div(id='frame-idx-holder', children=str(frame_idx), style={'display': 'none'}),

        dbc.Row([
            # Left side - stacked graphs (narrower)
            dbc.Col([
                html.Div(graphs_content, style={'overflowY': 'auto', 'maxHeight': '550px'})
            ], width=4, style={'paddingRight': '5px'}),

            # Center/Right - Three.js skeleton viewer with video overlay
            dbc.Col([
                html.Div([
                    # Three.js skeleton viewer (full size)
                    html.Iframe(
                        id='threejs-skeleton',
                        src='/assets/skeleton_viewer.html',
                        style={
                            'width': '100%',
                            'height': '550px',
                            'border': 'none',
                            'borderRadius': '8px',
                            'backgroundColor': '#1e1e32'
                        }
                    ),
                    # Video overlay in top-right corner
                    html.Div([
                        html.Div("Video Source", style={'color': '#888', 'fontSize': '11px', 'marginBottom': '2px'}),
                        html.Div("primary camera", style={'color': '#666', 'fontSize': '9px', 'marginBottom': '4px'}),
                        html.Video(
                            id='viz-video-player',
                            src=video_src,
                            controls=False,  # Disabled - use main slider for sync
                            muted=True,
                            style={
                                'width': '100%',
                                'maxHeight': '140px',
                                'borderRadius': '6px',
                                'backgroundColor': '#000',
                                'pointerEvents': 'none'  # Prevent clicking
                            }
                        )
                    ], style={
                        'position': 'absolute',
                        'top': '10px',
                        'right': '10px',
                        'width': '200px',
                        'padding': '8px',
                        'backgroundColor': 'rgba(30, 30, 50, 0.85)',
                        'borderRadius': '8px',
                        'boxShadow': '0 4px 12px rgba(0,0,0,0.4)'
                    })
                ], style={'position': 'relative'})
            ], width=8)
        ], style={'marginTop': '10px'})
    ])


def create_correlation_tab(processed_poses, uplift_poses):
    """Create correlation visualization comparing processed poses to UPLIFT ground truth."""

    if not processed_poses or not uplift_poses:
        return html.Div([
            html.P("Correlation analysis requires both processed and UPLIFT data.",
                   style={'color': '#888', 'padding': '20px'}),
            html.P("Upload UPLIFT CSV data and process videos to compare.",
                   style={'color': '#666', 'fontSize': '12px', 'paddingLeft': '20px'})
        ])

    # Convert to numpy arrays
    proc = np.array(processed_poses)
    uplift = np.array(uplift_poses)

    # Align frame counts
    n_frames = min(len(proc), len(uplift))
    proc = proc[:n_frames]
    uplift = uplift[:n_frames]

    if n_frames == 0:
        return html.Div("No overlapping frames between datasets", style={'color': '#888'})

    # H36M joint names
    joint_names = [
        'Hip', 'RHip', 'RKnee', 'RAnkle', 'LHip', 'LKnee', 'LAnkle',
        'Spine', 'Thorax', 'Neck', 'Head', 'LShoulder', 'LElbow', 'LWrist',
        'RShoulder', 'RElbow', 'RWrist'
    ]

    # Calculate per-joint MPJPE (in centimeters)
    per_joint_error = np.zeros(proc.shape[1])  # 17 joints
    for j in range(proc.shape[1]):
        errors = np.linalg.norm(proc[:, j, :] - uplift[:, j, :], axis=1)
        per_joint_error[j] = np.mean(errors) * 100  # Convert to cm

    overall_mpjpe = np.mean(per_joint_error)

    # Create bar chart of per-joint MPJPE
    fig_mpjpe = go.Figure()

    colors = ['#ff6b6b' if e > 10 else '#ffd93d' if e > 5 else '#6bcb77' for e in per_joint_error]

    fig_mpjpe.add_trace(go.Bar(
        x=joint_names,
        y=per_joint_error,
        marker_color=colors,
        text=[f'{e:.1f}' for e in per_joint_error],
        textposition='outside',
        textfont=dict(size=9, color='#ccc')
    ))

    fig_mpjpe.update_layout(
        title=dict(
            text=f'Per-Joint MPJPE (Overall: {overall_mpjpe:.1f} cm)',
            font=dict(size=14, color='#fff'),
            x=0.5
        ),
        template='plotly_dark',
        paper_bgcolor='rgba(26,26,46,0)',
        plot_bgcolor='rgba(26,26,46,0.5)',
        height=250,
        margin=dict(l=50, r=20, t=40, b=60),
        xaxis=dict(
            tickangle=45,
            tickfont=dict(size=9, color='#888'),
            gridcolor='rgba(80,80,80,0.3)'
        ),
        yaxis=dict(
            title='Error (cm)',
            tickfont=dict(size=9, color='#888'),
            gridcolor='rgba(80,80,80,0.3)'
        )
    )

    # Create temporal error plot
    frame_errors = []
    for f in range(n_frames):
        err = np.mean(np.linalg.norm(proc[f] - uplift[f], axis=1)) * 100
        frame_errors.append(err)

    fig_temporal = go.Figure()
    fig_temporal.add_trace(go.Scatter(
        x=list(range(n_frames)),
        y=frame_errors,
        mode='lines',
        line=dict(color='#00ff00', width=1.5),
        fill='tozeroy',
        fillcolor='rgba(0,255,0,0.1)'
    ))

    fig_temporal.update_layout(
        title=dict(
            text='MPJPE Over Time',
            font=dict(size=14, color='#fff'),
            x=0.5
        ),
        template='plotly_dark',
        paper_bgcolor='rgba(26,26,46,0)',
        plot_bgcolor='rgba(26,26,46,0.5)',
        height=180,
        margin=dict(l=50, r=20, t=40, b=30),
        xaxis=dict(
            title='Frame',
            tickfont=dict(size=9, color='#888'),
            gridcolor='rgba(80,80,80,0.3)'
        ),
        yaxis=dict(
            title='Error (cm)',
            tickfont=dict(size=9, color='#888'),
            gridcolor='rgba(80,80,80,0.3)'
        )
    )

    # Statistics summary
    stats_text = f"""
    **Correlation Statistics**
    - Frames analyzed: {n_frames}
    - Overall MPJPE: {overall_mpjpe:.2f} cm
    - Best joint: {joint_names[np.argmin(per_joint_error)]} ({per_joint_error.min():.1f} cm)
    - Worst joint: {joint_names[np.argmax(per_joint_error)]} ({per_joint_error.max():.1f} cm)
    - Median error: {np.median(per_joint_error):.1f} cm
    """

    return html.Div([
        dcc.Markdown(stats_text, style={'color': '#ccc', 'fontSize': '12px', 'padding': '10px'}),
        dcc.Graph(figure=fig_mpjpe, config={'displayModeBar': False}),
        dcc.Graph(figure=fig_temporal, config={'displayModeBar': False})
    ])


def get_tab_graphs(active_tab, df, results, frame_idx, events_dict, processed_poses=None, uplift_poses=None):
    """Get the graph content for a specific tab with UPLIFT-style accordions."""

    if active_tab == "tab-analytics":
        # Kinematic sequence and X-factor
        kin_fig = create_kinematic_sequence_figure(df, events_dict)
        xf_fig = create_xfactor_figure(df, events_dict)
        return html.Div([
            dcc.Graph(figure=kin_fig, config={'displayModeBar': False}),
            dcc.Graph(figure=xf_fig, config={'displayModeBar': False})
        ])

    elif active_tab == "tab-shoulders":
        return create_uplift_accordion_tab(df, frame_idx, [
            {
                'title': 'Shoulder Movement',
                'metrics': [
                    ('shoulder_displacement_x', 'shoulder displacement (side-to-side) [m]'),
                    ('shoulder_displacement_y', 'shoulder displacement (front-to-back) [m]'),
                    ('shoulder_displacement_z', 'shoulder displacement (elevation) [m]'),
                ]
            },
            {
                'title': 'Shoulder Rotations',
                'metrics': [
                    ('torso_lateral_tilt', 'shoulder tilt (side-to-side) [degree]'),
                    ('torso_rotation', 'shoulder rotation (internal) [degree]'),
                    ('torso_rotation_velocity', 'shoulder rotational velocity [degree/s]'),
                ]
            }
        ])

    elif active_tab == "tab-pelvis":
        return create_uplift_accordion_tab(df, frame_idx, [
            {
                'title': 'Pelvis Movement',
                'metrics': [
                    ('pelvis_displacement_x', 'pelvis displacement (side-to-side) [m]'),
                    ('pelvis_displacement_y', 'pelvis displacement (front-to-back) [m]'),
                    ('pelvis_displacement_z', 'pelvis displacement (elevation) [m]'),
                ]
            },
            {
                'title': 'Pelvis Rotations',
                'metrics': [
                    ('pelvis_tilt', 'pelvis tilt (side-to-side) [degree]'),
                    ('pelvis_rotation', 'pelvis rotation (internal) [degree]'),
                    ('pelvis_rotation_velocity', 'pelvis rotational velocity [degree/s]'),
                ]
            }
        ])

    elif active_tab == "tab-arms":
        return create_uplift_accordion_tab(df, frame_idx, [
            {
                'title': 'Elbows',
                'metrics': [
                    ('right_elbow_flexion', 'right elbow extension [degree]'),
                    ('left_elbow_flexion', 'left elbow extension [degree]'),
                ]
            },
            {
                'title': 'Elbow Velocities',
                'metrics': [
                    ('left_elbow_flexion_velocity', 'left elbow angular velocity [degree/s]'),
                    ('right_elbow_flexion_velocity', 'right elbow angular velocity [degree/s]'),
                ]
            },
            {
                'title': 'Arm Rotations',
                'metrics': [
                    ('left_shoulder_rotation', 'left arm rotation (external) [degree]'),
                    ('right_shoulder_rotation', 'right arm rotation (external) [degree]'),
                ]
            }
        ])

    elif active_tab == "tab-legs":
        return create_uplift_accordion_tab(df, frame_idx, [
            {
                'title': 'Legs',
                'metrics': [
                    ('right_knee_extension', 'right knee extension [degree]'),
                    ('left_knee_extension', 'left knee extension [degree]'),
                ]
            }
        ])

    elif active_tab == "tab-correlation":
        return create_correlation_tab(processed_poses, uplift_poses)

    return html.Div("Select a tab")


def create_uplift_accordion_tab(df, frame_idx, sections):
    """Create UPLIFT-style accordion sections with graphs."""
    current_time = df['timestamp'].iloc[frame_idx] if frame_idx < len(df) else 0

    accordion_items = []
    for section in sections:
        graphs = []
        for col, label in section['metrics']:
            # Try to find the column or a similar one
            actual_col = find_column(df, col)
            if actual_col is None:
                continue

            fig = create_uplift_graph(df, actual_col, label, current_time, frame_idx)
            graphs.append(dcc.Graph(figure=fig, config={'displayModeBar': False},
                                   style={'marginBottom': '0px'}))

        if graphs:
            accordion_items.append(
                dbc.AccordionItem(
                    html.Div(graphs),
                    title=section['title'],
                    style={'backgroundColor': '#1a1a2e', 'border': 'none'}
                )
            )

    if not accordion_items:
        return html.Div("No data available for this tab", style={'color': '#666', 'padding': '20px'})

    return dbc.Accordion(
        accordion_items,
        start_collapsed=False,
        always_open=True,
        style={'backgroundColor': '#1a1a2e'}
    )


def find_column(df, col_name):
    """Find a column by name or partial match."""
    if col_name in df.columns:
        return col_name

    # Try common variations
    variations = [
        col_name,
        col_name.replace('_', ''),
        col_name.replace('displacement', 'position'),
        col_name.replace('shoulder_displacement', 'trunk_center_of_mass'),
        col_name.replace('pelvis_displacement', 'pelvis_center_of_mass'),
    ]

    for var in variations:
        if var in df.columns:
            return var

    # Try partial match
    for c in df.columns:
        if col_name.replace('_', '') in c.replace('_', ''):
            return c

    return None


def create_uplift_graph(df, col, label, current_time, frame_idx, color='#00ff00'):
    """Create a single UPLIFT-style graph with time marker."""
    fig = go.Figure()

    # Main trace
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df[col],
        mode='lines',
        line=dict(color=color, width=2),
        name=label,
        hovertemplate='%{y:.2f}<extra></extra>'
    ))

    # Calculate Y range
    y_min = df[col].min()
    y_max = df[col].max()
    y_range = y_max - y_min if y_max > y_min else 1
    y_padding = y_range * 0.1

    # Vertical time marker line
    fig.add_shape(
        type="line",
        x0=current_time, x1=current_time,
        y0=y_min - y_padding, y1=y_max + y_padding,
        line=dict(color="rgba(255,255,255,0.7)", width=1)
    )

    # Current position marker (circle on the line)
    current_val = df[col].iloc[frame_idx] if frame_idx < len(df) else 0
    fig.add_trace(go.Scatter(
        x=[current_time],
        y=[current_val],
        mode='markers',
        marker=dict(
            color='white',
            size=10,
            line=dict(color='#00ff00', width=2)
        ),
        showlegend=False,
        hoverinfo='skip'
    ))

    fig.update_layout(
        title=dict(
            text=label,
            font=dict(size=11, color='#ccc'),
            x=0.01,
            xanchor='left'
        ),
        template='plotly_dark',
        paper_bgcolor='rgba(26,26,46,0)',
        plot_bgcolor='rgba(26,26,46,0.5)',
        height=120,
        margin=dict(l=45, r=10, t=25, b=20),
        showlegend=False,
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(80,80,80,0.3)',
            showticklabels=False,
            zeroline=False
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(80,80,80,0.3)',
            tickfont=dict(size=9, color='#888'),
            zeroline=True,
            zerolinecolor='rgba(100,100,100,0.5)'
        )
    )

    return fig


def create_stacked_angle_graphs(df, frame_idx, columns, titles):
    """Create stacked individual graphs with time marker, UPLIFT-style."""
    graphs = []
    current_time = df['timestamp'].iloc[frame_idx] if frame_idx < len(df) else 0

    for col, title in zip(columns, titles):
        if col not in df.columns:
            continue

        fig = go.Figure()

        # Add the angle trace
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df[col],
            mode='lines',
            line=dict(color='#00ff00', width=2),
            name=title
        ))

        # Add vertical time marker
        y_min = df[col].min()
        y_max = df[col].max()
        y_range = y_max - y_min if y_max > y_min else 1

        fig.add_shape(
            type="line",
            x0=current_time, x1=current_time,
            y0=y_min - 0.1 * y_range, y1=y_max + 0.1 * y_range,
            line=dict(color="white", width=1)
        )

        # Add marker point at current time
        current_val = df[col].iloc[frame_idx] if frame_idx < len(df) else 0
        fig.add_trace(go.Scatter(
            x=[current_time],
            y=[current_val],
            mode='markers',
            marker=dict(color='white', size=8),
            showlegend=False
        ))

        fig.update_layout(
            title=dict(text=f"{title} [degree]", font=dict(size=12, color='white')),
            template='plotly_dark',
            paper_bgcolor='#1a1a2e',
            plot_bgcolor='#1a1a2e',
            height=150,
            margin=dict(l=50, r=20, t=30, b=30),
            showlegend=False,
            xaxis=dict(showgrid=True, gridcolor='#333', title=''),
            yaxis=dict(showgrid=True, gridcolor='#333', title='')
        )

        graphs.append(dcc.Graph(figure=fig, config={'displayModeBar': False}))

    return html.Div(graphs)


def render_pelvis_tab(df, events_dict):
    """Render pelvis analysis."""
    angles = ['pelvis_tilt', 'pelvis_obliquity', 'pelvis_rotation']
    available = [a for a in angles if a in df.columns]

    fig = create_joint_angle_figure(df, available, "Pelvis Angles", events_dict)

    # Add velocity plot
    velo_cols = [f'{a}_velocity' for a in available if f'{a}_velocity' in df.columns]
    if velo_cols:
        velo_fig = create_joint_angle_figure(df, velo_cols, "Pelvis Angular Velocities", events_dict)
        return html.Div([
            dcc.Graph(figure=fig, config={'displayModeBar': False}),
            dcc.Graph(figure=velo_fig, config={'displayModeBar': False})
        ])

    return dcc.Graph(figure=fig, config={'displayModeBar': False})


def render_torso_tab(df, events_dict):
    """Render torso analysis."""
    angles = ['torso_flexion', 'torso_lateral_tilt', 'torso_rotation', 'hip_shoulder_separation']
    available = [a for a in angles if a in df.columns]

    fig = create_joint_angle_figure(df, available, "Torso Angles", events_dict)

    return dcc.Graph(figure=fig, config={'displayModeBar': False})


def render_arms_tab(df, events_dict):
    """Render arms analysis."""
    fig = make_subplots(rows=2, cols=2, subplot_titles=(
        'Elbow Flexion', 'Shoulder Abduction',
        'Shoulder Rotation', 'Elbow Velocity'
    ))

    # Elbow flexion
    for col in ['left_elbow_flexion', 'right_elbow_flexion']:
        if col in df.columns:
            fig.add_trace(go.Scatter(x=df['timestamp'], y=df[col], name=col.replace('_', ' ')),
                         row=1, col=1)

    # Shoulder abduction
    for col in ['left_shoulder_abduction', 'right_shoulder_abduction']:
        if col in df.columns:
            fig.add_trace(go.Scatter(x=df['timestamp'], y=df[col], name=col.replace('_', ' ')),
                         row=1, col=2)

    # Shoulder rotation
    for col in ['left_shoulder_rotation', 'right_shoulder_rotation']:
        if col in df.columns:
            fig.add_trace(go.Scatter(x=df['timestamp'], y=df[col], name=col.replace('_', ' ')),
                         row=2, col=1)

    # Elbow velocity
    for col in ['left_elbow_flexion_velocity', 'right_elbow_flexion_velocity']:
        if col in df.columns:
            fig.add_trace(go.Scatter(x=df['timestamp'], y=df[col], name=col.replace('_', ' ')),
                         row=2, col=2)

    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='#1a1a2e',
        plot_bgcolor='#1a1a2e',
        height=600,
        showlegend=True
    )

    return dcc.Graph(figure=fig, config={'displayModeBar': False})


def render_legs_tab(df, events_dict):
    """Render legs analysis."""
    fig = make_subplots(rows=2, cols=2, subplot_titles=(
        'Knee Extension', 'Hip Flexion',
        'Ankle Dorsiflexion', 'Knee Varus'
    ))

    # Knee extension
    for col in ['left_knee_extension', 'right_knee_extension']:
        if col in df.columns:
            fig.add_trace(go.Scatter(x=df['timestamp'], y=df[col], name=col.replace('_', ' ')),
                         row=1, col=1)

    # Hip flexion
    for col in ['left_hip_flexion_with_respect_to_trunk', 'right_hip_flexion_with_respect_to_trunk',
                'left_hip_flexion', 'right_hip_flexion']:
        if col in df.columns:
            fig.add_trace(go.Scatter(x=df['timestamp'], y=df[col], name=col.replace('_', ' ')),
                         row=1, col=2)
            break  # Only add one set

    # Ankle dorsiflexion
    for col in ['left_ankle_dorsiflexion', 'right_ankle_dorsiflexion']:
        if col in df.columns:
            fig.add_trace(go.Scatter(x=df['timestamp'], y=df[col], name=col.replace('_', ' ')),
                         row=2, col=1)

    # Knee varus
    for col in ['left_knee_varus', 'right_knee_varus']:
        if col in df.columns:
            fig.add_trace(go.Scatter(x=df['timestamp'], y=df[col], name=col.replace('_', ' ')),
                         row=2, col=2)

    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='#1a1a2e',
        plot_bgcolor='#1a1a2e',
        height=600,
        showlegend=True
    )

    return dcc.Graph(figure=fig, config={'displayModeBar': False})


def render_advanced_tab(df, events_dict):
    """Render advanced metrics (center of mass, etc.)."""
    fig = make_subplots(rows=2, cols=2, subplot_titles=(
        'Center of Mass X', 'Center of Mass Y',
        'Center of Mass Z', 'Arm Rotation (Kinematic Seq)'
    ))

    # COM X
    for col in ['trunk_center_of_mass_x', 'whole_body_center_of_mass_x']:
        if col in df.columns:
            fig.add_trace(go.Scatter(x=df['timestamp'], y=df[col], name=col.replace('_', ' ')),
                         row=1, col=1)

    # COM Y
    for col in ['trunk_center_of_mass_y', 'whole_body_center_of_mass_y']:
        if col in df.columns:
            fig.add_trace(go.Scatter(x=df['timestamp'], y=df[col], name=col.replace('_', ' ')),
                         row=1, col=2)

    # COM Z
    for col in ['trunk_center_of_mass_z', 'whole_body_center_of_mass_z']:
        if col in df.columns:
            fig.add_trace(go.Scatter(x=df['timestamp'], y=df[col], name=col.replace('_', ' ')),
                         row=2, col=1)

    # Arm rotation
    for col in ['left_arm_rotation', 'right_arm_rotation']:
        if col in df.columns:
            fig.add_trace(go.Scatter(x=df['timestamp'], y=df[col], name=col.replace('_', ' ')),
                         row=2, col=2)

    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='#1a1a2e',
        plot_bgcolor='#1a1a2e',
        height=600,
        showlegend=True
    )

    return dcc.Graph(figure=fig, config={'displayModeBar': False})


@callback(
    Output('download-data', 'data'),
    [Input('export-csv-btn', 'n_clicks'),
     Input('export-report-btn', 'n_clicks')],
    State('results-data', 'data'),
    prevent_initial_call=True
)
def export_data(csv_clicks, report_clicks, results_data):
    if results_data is None:
        raise PreventUpdate

    session_id = results_data['session_id']
    if session_id not in RESULTS_CACHE:
        raise PreventUpdate

    results = RESULTS_CACHE[session_id]
    triggered = ctx.triggered_id

    if triggered == 'export-csv-btn':
        df = results['timeseries_df']
        return dcc.send_data_frame(df.to_csv, "biomechanics_data.csv", index=False)

    elif triggered == 'export-report-btn':
        # Export metrics
        metrics = results['metrics']
        metrics_dict = {k: v for k, v in metrics.__dict__.items() if v is not None}
        metrics_df = pd.DataFrame([metrics_dict])
        return dcc.send_data_frame(metrics_df.to_csv, "poi_metrics.csv", index=False)

    raise PreventUpdate


# Callback to save training data
@callback(
    Output('save-training-status', 'children'),
    Input('save-training-btn', 'n_clicks'),
    [State('results-data', 'data'),
     State('uploaded-videos', 'data'),
     State('uplift-poses-data', 'data')],
    prevent_initial_call=True
)
def save_training_data(n_clicks, results_data, uploaded_videos, uplift_poses):
    if n_clicks is None or results_data is None:
        raise PreventUpdate

    import shutil
    from datetime import datetime

    session_id = results_data['session_id']
    if session_id not in RESULTS_CACHE:
        return html.Span("Error: No results to save", style={'color': '#f87171'})

    results = RESULTS_CACHE[session_id]

    # Create new session folder
    training_dir = Path('training_data')
    existing = list(training_dir.glob('session_*'))
    next_num = len(existing) + 1
    session_dir = training_dir / f'session_{next_num:03d}'
    session_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Copy video files
        if uploaded_videos.get('video1'):
            shutil.copy(uploaded_videos['video1'], session_dir / 'side.mp4')
        if uploaded_videos.get('video2'):
            shutil.copy(uploaded_videos['video2'], session_dir / 'back.mp4')

        # Copy UPLIFT CSV if available
        if uploaded_videos.get('uplift_csv'):
            shutil.copy(uploaded_videos['uplift_csv'], session_dir / 'uplift.csv')
        elif uplift_poses is not None:
            # Save UPLIFT poses as CSV if we have them
            uplift_df = pd.DataFrame()
            joint_names = ['pelvis', 'right_hip', 'right_knee', 'right_ankle',
                          'left_hip', 'left_knee', 'left_ankle', 'spine', 'neck',
                          'head', 'head_top', 'left_shoulder', 'left_elbow', 'left_wrist',
                          'right_shoulder', 'right_elbow', 'right_wrist']
            for j, name in enumerate(joint_names):
                for c, coord in enumerate(['x', 'y', 'z']):
                    col_name = f'{name}_3d_{coord}'
                    uplift_df[col_name] = [frame[j][c] for frame in uplift_poses]
            uplift_df.to_csv(session_dir / 'uplift.csv', index=False)

        # Save processed poses
        poses_3d = results.get('poses_3d', [])
        if poses_3d:
            np.save(session_dir / 'poses_3d.npy', np.array(poses_3d))

        # Save timeseries
        results['timeseries_df'].to_csv(session_dir / 'timeseries.csv', index=False)

        # Save metadata
        metadata = {
            'fps': results['fps'],
            'n_frames': results_data['n_frames'],
            'n_cameras': results['n_cameras'],
            'created': datetime.now().isoformat(),
            'has_uplift': uploaded_videos.get('uplift_csv') is not None or uplift_poses is not None
        }
        import json
        with open(session_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        return html.Span(f"Saved to {session_dir.name}", style={'color': '#4ade80'})

    except Exception as e:
        return html.Span(f"Error: {str(e)}", style={'color': '#f87171'})


def run_app(debug: bool = False, port: int = 8050):
    """Run the web application."""
    print("\n" + "=" * 60)
    print("VIDEO BIOMECHANICS ANALYZER")
    print("=" * 60)
    print(f"\nOpen your browser to: http://localhost:{port}")
    print("\nUpload 1-2 videos to analyze swing biomechanics")
    print("Press Ctrl+C to stop\n")

    app.run(debug=debug, port=port)


if __name__ == "__main__":
    run_app(debug=False)
