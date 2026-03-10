"""Extract diverse frames from videos for bat annotation.

Unlike plate detection (static), bat moves throughout the swing.
This script extracts frames at key phases of the swing:
- Stance/setup
- Load/stride
- Swing initiation
- Contact zone
- Follow-through

Uses motion analysis to identify these phases automatically.
"""

import cv2
import numpy as np
from pathlib import Path
import json
from typing import List, Tuple, Dict
import argparse


def compute_frame_motion(cap: cv2.VideoCapture) -> np.ndarray:
    """Compute motion magnitude for each frame in video.

    Returns:
        Array of motion magnitudes per frame
    """
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    motions = np.zeros(frame_count - 1)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, prev_frame = cap.read()
    if not ret:
        return motions

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    for i in range(frame_count - 1):
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Compute absolute difference
        diff = cv2.absdiff(prev_gray, gray)
        motions[i] = diff.mean()

        prev_gray = gray

    return motions


def find_swing_phases(motions: np.ndarray, fps: float = 240.0) -> Dict[str, int]:
    """Identify key swing phases from motion profile.

    Typical baseball swing phases:
    1. Stance: Low motion at start
    2. Load/stride: Increasing motion
    3. Swing: Peak motion
    4. Contact: During peak
    5. Follow-through: Decreasing motion

    Returns:
        Dictionary mapping phase name to frame index
    """
    n = len(motions)

    # Smooth motion signal
    kernel_size = int(fps * 0.02)  # ~5 frames at 240fps
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel_size = max(3, kernel_size)

    smoothed = np.convolve(motions, np.ones(kernel_size)/kernel_size, mode='same')

    # Find peak motion (swing/contact)
    peak_idx = np.argmax(smoothed)

    # Find swing start (motion increases above threshold before peak)
    threshold = smoothed.mean() + 0.5 * smoothed.std()
    swing_start = 0
    for i in range(peak_idx - 1, -1, -1):
        if smoothed[i] < threshold:
            swing_start = i
            break

    # Find swing end (motion decreases below threshold after peak)
    swing_end = n - 1
    for i in range(peak_idx + 1, n):
        if smoothed[i] < threshold:
            swing_end = i
            break

    # Calculate phase indices (convert to int for JSON serialization)
    phases = {
        'stance': int(max(0, swing_start - int(fps * 0.3))),      # 300ms before swing
        'load': int(max(0, swing_start - int(fps * 0.1))),        # 100ms before swing
        'swing_start': int(swing_start),
        'mid_swing': int((swing_start + peak_idx) // 2),
        'contact': int(peak_idx),
        'follow_through_early': int(min(n-1, peak_idx + int(fps * 0.05))),  # 50ms after
        'follow_through_late': int(min(n-1, swing_end)),
    }

    return phases


def extract_diverse_frames(
    video_path: Path,
    num_frames: int = 10,
    output_dir: Path = None
) -> List[Tuple[int, np.ndarray]]:
    """Extract diverse frames spanning the swing.

    Args:
        video_path: Path to video file
        num_frames: Target number of frames to extract
        output_dir: Optional directory to save frame images

    Returns:
        List of (frame_idx, frame_array) tuples
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Analyzing {video_path.name}: {frame_count} frames at {fps:.1f}fps")

    # Compute motion profile
    motions = compute_frame_motion(cap)

    # Find swing phases
    phases = find_swing_phases(motions, fps)
    print(f"  Detected phases: {phases}")

    # Select frames from different phases
    frame_indices = set()

    # Always include key phases
    for phase_name, idx in phases.items():
        if 0 <= idx < frame_count:
            frame_indices.add(idx)

    # Add intermediate frames if we need more
    if len(frame_indices) < num_frames:
        # Focus on the swing region (more variation there)
        swing_start = phases.get('swing_start', 0)
        swing_end = phases.get('follow_through_late', frame_count - 1)
        swing_duration = swing_end - swing_start

        # Add evenly spaced frames in swing region
        additional_needed = num_frames - len(frame_indices)
        for i in range(additional_needed):
            frac = (i + 1) / (additional_needed + 1)
            idx = int(swing_start + frac * swing_duration)
            if 0 <= idx < frame_count:
                frame_indices.add(idx)

    # Also add some random frames for diversity
    remaining = num_frames - len(frame_indices)
    if remaining > 0:
        candidates = [i for i in range(frame_count) if i not in frame_indices]
        if candidates:
            np.random.shuffle(candidates)
            frame_indices.update(candidates[:remaining])

    # Extract frames (convert numpy types to int for JSON serialization)
    frame_indices = sorted(int(i) for i in frame_indices)[:num_frames]
    frames = []

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append((idx, frame))

            if output_dir:
                output_dir.mkdir(parents=True, exist_ok=True)
                out_path = output_dir / f"{video_path.stem}_frame{idx:04d}.jpg"
                cv2.imwrite(str(out_path), frame)

    cap.release()

    print(f"  Extracted {len(frames)} frames: {[f[0] for f in frames]}")
    return frames


def scan_videos_for_annotation(
    data_dir: Path,
    frames_per_video: int = 10
) -> Dict[str, List[int]]:
    """Scan all videos and select frames for annotation.

    Args:
        data_dir: Directory containing session folders
        frames_per_video: Number of frames to select per video

    Returns:
        Dictionary mapping video path to list of frame indices
    """
    selection = {}

    # Find all video files
    video_patterns = ['**/*primary*.mp4', '**/*secondary*.mp4']
    videos = []
    for pattern in video_patterns:
        videos.extend(data_dir.glob(pattern))

    # Filter to session directories
    videos = [v for v in videos if 'session_' in str(v)]

    print(f"Found {len(videos)} videos to process")

    for video_path in sorted(videos):
        rel_path = str(video_path.relative_to(data_dir))

        try:
            frames = extract_diverse_frames(
                video_path,
                num_frames=frames_per_video
            )
            selection[rel_path] = [idx for idx, _ in frames]
        except Exception as e:
            print(f"  Error processing {video_path}: {e}")
            continue

    return selection


def main():
    parser = argparse.ArgumentParser(description="Extract frames for bat annotation")
    parser.add_argument("--data-dir", type=Path,
                        default=Path(__file__).parent.parent.parent / "training_data",
                        help="Training data directory")
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).parent / "frame_selection.json",
                        help="Output JSON file with frame selections")
    parser.add_argument("--frames-per-video", type=int, default=10,
                        help="Number of frames to select per video")
    parser.add_argument("--extract-images", action="store_true",
                        help="Also save frame images")
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()

    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        return

    print(f"Scanning videos in: {data_dir}")

    # Select frames
    selection = scan_videos_for_annotation(
        data_dir,
        frames_per_video=args.frames_per_video
    )

    # Save selection
    with open(args.output, 'w') as f:
        json.dump({
            "version": "1.0",
            "data_dir": str(data_dir),
            "frames_per_video": args.frames_per_video,
            "selections": selection
        }, f, indent=2)

    total_frames = sum(len(frames) for frames in selection.values())
    print(f"\nSaved selection to {args.output}")
    print(f"Total frames selected: {total_frames} across {len(selection)} videos")

    # Optionally extract images
    if args.extract_images:
        images_dir = args.output.parent / "extracted_frames"
        print(f"\nExtracting frame images to {images_dir}...")

        for video_rel, frame_indices in selection.items():
            video_path = data_dir / video_rel
            video_out_dir = images_dir / video_path.stem

            cap = cv2.VideoCapture(str(video_path))
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    video_out_dir.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(video_out_dir / f"frame{idx:04d}.jpg"), frame)
            cap.release()


if __name__ == "__main__":
    main()
