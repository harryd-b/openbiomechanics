"""GUI tool for annotating plate corners in video frames.

Usage:
    python annotate.py --data-dir <path_to_training_data>

Controls:
    Left Click: Add corner point (5 points needed)
    Right Click: Remove last point
    N / Right Arrow: Next frame (skip 30)
    P / Left Arrow: Previous frame (skip 30)
    ] : Next video
    [ : Previous video
    S: Save annotations
    R: Reset current frame
    Z: Zoom toggle (2x at mouse position)
    Q / Escape: Quit

Corner Order:
    Click corners in this order:
    1. Apex (point toward pitcher)
    2. Back Left (catcher's right side)
    3. Front Left
    4. Front Right
    5. Back Right (catcher's left side)
"""

import cv2
import numpy as np
from pathlib import Path
import argparse
from typing import List, Tuple, Optional

from annotations import AnnotationStore


class PlateAnnotator:
    """Interactive GUI for annotating plate corners."""

    CORNER_NAMES = ["Apex", "Back-Left", "Front-Left", "Front-Right", "Back-Right"]
    CORNER_COLORS = [
        (0, 0, 255),    # Red - Apex
        (0, 255, 0),    # Green - BL
        (255, 0, 0),    # Blue - FL
        (255, 255, 0),  # Cyan - FR
        (255, 0, 255),  # Magenta - BR
    ]

    def __init__(self, data_dir: Path, annotations_file: Path):
        self.data_dir = data_dir
        self.annotations_file = annotations_file

        # Load or create annotation store
        if annotations_file.exists():
            self.store = AnnotationStore.load(annotations_file)
            print(f"Loaded {len(self.store)} existing annotations")
        else:
            self.store = AnnotationStore()

        # Find all videos
        self.videos = self._find_videos()
        if not self.videos:
            raise ValueError(f"No videos found in {data_dir}")

        print(f"Found {len(self.videos)} videos")

        # State
        self.video_idx = 0
        self.frame_idx = 0
        self.current_points: List[Tuple[int, int]] = []
        self.cap: Optional[cv2.VideoCapture] = None
        self.current_frame: Optional[np.ndarray] = None
        self.total_frames = 0
        self.zoom_mode = False
        self.mouse_pos = (0, 0)

        # Varied mode - cycle through sessions/views
        self.varied_mode = True
        self.sample_queue: List[Tuple[int, int]] = []  # (video_idx, frame_idx)
        self.queue_idx = 0
        self._build_sample_queue()

        # Load first sample
        if self.sample_queue:
            vid_idx, frame_idx = self.sample_queue[0]
            self._load_video(vid_idx)
            self._load_frame(frame_idx)
        else:
            self._load_video(0)

    def _find_videos(self) -> List[Path]:
        """Find all video files in data directory."""
        videos = []
        for session_dir in sorted(self.data_dir.glob("session_*")):
            for view in ["primary", "secondary"]:
                video_path = session_dir / f"{view}.mp4"
                if video_path.exists():
                    videos.append(video_path)
        return videos

    def _build_sample_queue(self):
        """Build sample queue with one frame per video.

        Since cameras are static within each session, the plate position
        is constant - we only need one annotation per video.
        Prioritizes unannotated videos first.
        """
        import random

        # Get already annotated videos
        annotated = set(self.store.list_videos())

        # Separate into unannotated (priority) and already annotated
        unannotated = []
        already_done = []

        for vid_idx, video_path in enumerate(self.videos):
            video_key = str(video_path.relative_to(self.data_dir))
            if video_key in annotated:
                already_done.append((vid_idx, 30))
            else:
                unannotated.append((vid_idx, 30))

        # Shuffle within each group
        random.shuffle(unannotated)
        random.shuffle(already_done)

        # Unannotated first, then already done
        self.sample_queue = unannotated + already_done

        print(f"Built sample queue: {len(unannotated)} unannotated + {len(already_done)} done = {len(self.sample_queue)} total")

    def _load_next_sample(self):
        """Load the next sample from the varied queue."""
        if self.queue_idx >= len(self.sample_queue):
            print("Reached end of sample queue!")
            return

        vid_idx, frame_idx = self.sample_queue[self.queue_idx]
        self._load_video(vid_idx)
        self._load_frame(frame_idx)

    def _prev_sample(self):
        """Go to previous sample in queue."""
        if self.queue_idx > 0:
            self.queue_idx -= 1
            self._load_next_sample()

    def _next_sample(self):
        """Go to next sample in queue."""
        if self.queue_idx < len(self.sample_queue) - 1:
            self.queue_idx += 1
            self._load_next_sample()

    def _load_video(self, idx: int) -> None:
        """Load a video by index."""
        if self.cap is not None:
            self.cap.release()

        self.video_idx = idx
        video_path = self.videos[idx]
        self.cap = cv2.VideoCapture(str(video_path))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Get relative path for annotation key
        self.video_key = str(video_path.relative_to(self.data_dir))

        # Check for existing annotations and go to first annotated frame
        existing_frames = self.store.get_frames(self.video_key)
        if existing_frames:
            self.frame_idx = existing_frames[0]
        else:
            self.frame_idx = 0

        self._load_frame(self.frame_idx)
        print(f"Loaded: {self.video_key} ({self.total_frames} frames)")

    def _load_frame(self, idx: int) -> None:
        """Load a specific frame."""
        idx = max(0, min(idx, self.total_frames - 1))
        self.frame_idx = idx

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self.cap.read()

        if ret:
            self.current_frame = frame
            # Load existing annotation if any
            existing = self.store.get(self.video_key, idx)
            if existing:
                self.current_points = list(existing.corners)
            else:
                self.current_points = []

    def _draw_frame(self) -> np.ndarray:
        """Draw the current frame with annotations."""
        if self.current_frame is None:
            return np.zeros((720, 1280, 3), dtype=np.uint8)

        frame = self.current_frame.copy()
        h, w = frame.shape[:2]

        # Draw existing points
        for i, pt in enumerate(self.current_points):
            color = self.CORNER_COLORS[i]
            cv2.circle(frame, pt, 8, color, -1)
            cv2.circle(frame, pt, 10, (255, 255, 255), 2)
            cv2.putText(frame, str(i + 1), (pt[0] + 12, pt[1] + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Draw lines connecting points
        if len(self.current_points) >= 2:
            pts = np.array(self.current_points, dtype=np.int32)
            if len(pts) == 5:
                # Complete pentagon
                cv2.polylines(frame, [pts], True, (0, 255, 255), 2)
            else:
                # Partial
                cv2.polylines(frame, [pts], False, (0, 255, 255), 2)

        # Draw status bar
        status_h = 60
        cv2.rectangle(frame, (0, 0), (w, status_h), (40, 40, 40), -1)

        # Video info
        cv2.putText(frame, f"{self.video_key}", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"Frame: {self.frame_idx}/{self.total_frames-1}", (10, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Points status
        if len(self.current_points) < 5:
            next_corner = self.CORNER_NAMES[len(self.current_points)]
            status = f"Click: {next_corner} ({len(self.current_points)}/5)"
            color = (0, 200, 255)
        else:
            status = "Complete! Press S to save, N for next frame"
            color = (0, 255, 0)

        cv2.putText(frame, status, (w // 2 - 150, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Annotation count and progress
        total_ann = len(self.store)
        if self.varied_mode and self.sample_queue:
            progress = f"Sample {self.queue_idx + 1}/{len(self.sample_queue)}"
        else:
            progress = ""
        cv2.putText(frame, f"Total: {total_ann} annotations  {progress}", (w - 280, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Help hint
        mode_hint = "(varied)" if self.varied_mode else "(sequential)"
        cv2.putText(frame, f"N/P:next/prev {mode_hint}  S:save  R:reset  V:toggle  Q:quit", (w - 420, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

        # Zoom mode
        if self.zoom_mode:
            zoom_size = 200
            zoom_factor = 2
            mx, my = self.mouse_pos

            # Source region
            src_size = zoom_size // zoom_factor
            x1 = max(0, mx - src_size // 2)
            y1 = max(0, my - src_size // 2)
            x2 = min(w, x1 + src_size)
            y2 = min(h, y1 + src_size)

            if x2 - x1 > 0 and y2 - y1 > 0:
                roi = self.current_frame[y1:y2, x1:x2]
                zoomed = cv2.resize(roi, (zoom_size, zoom_size), interpolation=cv2.INTER_LINEAR)

                # Draw zoom window in corner
                frame[status_h:status_h + zoom_size, w - zoom_size:w] = zoomed
                cv2.rectangle(frame, (w - zoom_size, status_h),
                            (w, status_h + zoom_size), (255, 255, 0), 2)

        return frame

    def _on_mouse(self, event: int, x: int, y: int, flags: int, param) -> None:
        """Handle mouse events."""
        # Convert display coordinates to original frame coordinates
        # Step 1: Undo letterbox padding
        pad_x = getattr(self, 'letterbox_pad_x', 0)
        pad_y = getattr(self, 'letterbox_pad_y', 0)
        letterbox_scale = getattr(self, 'letterbox_scale', 1.0)

        # Remove padding offset and scale
        frame_x = (x - pad_x) / letterbox_scale
        frame_y = (y - pad_y) / letterbox_scale

        # Step 2: Undo zoom scaling and add offset
        zoom_x = getattr(self, 'actual_zoom_x', 1.0)
        zoom_y = getattr(self, 'actual_zoom_y', 1.0)
        zoom_offset = getattr(self, 'zoom_offset', (0, 0))

        orig_x = int(frame_x / zoom_x) + zoom_offset[0]
        orig_y = int(frame_y / zoom_y) + zoom_offset[1]

        self.mouse_pos = (orig_x, orig_y)

        if event == cv2.EVENT_LBUTTONDOWN:
            # Add point in original frame coordinates
            if len(self.current_points) < 5:
                self.current_points.append((orig_x, orig_y))
                print(f"  Added {self.CORNER_NAMES[len(self.current_points)-1]}: ({orig_x}, {orig_y})")

        elif event == cv2.EVENT_RBUTTONDOWN:
            # Remove last point
            if self.current_points:
                removed = self.current_points.pop()
                print(f"  Removed point: {removed}")

    def _letterbox(self, frame: np.ndarray, target_w: int, target_h: int) -> Tuple[np.ndarray, int, int, float]:
        """Letterbox frame to target size, preserving aspect ratio.

        Returns:
            letterboxed_frame: Frame with padding
            pad_x: Horizontal padding offset (on left)
            pad_y: Vertical padding offset (on top)
            scale: Scale factor applied to image
        """
        h, w = frame.shape[:2]

        # Calculate scale to fit
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        # Resize frame
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Calculate padding
        pad_x = (target_w - new_w) // 2
        pad_y = (target_h - new_h) // 2

        # Create letterboxed frame with black padding
        letterboxed = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        letterboxed[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

        return letterboxed, pad_x, pad_y, scale

    def run(self) -> None:
        """Run the annotation GUI."""
        window_name = "Plate Annotator"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, self._on_mouse)

        # Set initial window size and track display scaling
        self.display_scale = 1.0
        self.display_size = None  # Will be set based on frame size
        self.zoom_offset = (0, 0)

        # Letterbox padding offsets
        self.letterbox_pad_x = 0
        self.letterbox_pad_y = 0
        self.letterbox_scale = 1.0

        print("\nControls:")
        print("  Left Click: Add corner | Right Click: Remove last")
        print("  N/P or Arrows: Next/Prev frame | [/]: Next/Prev video")
        print("  S: Save | R: Reset | Z: Zoom | Q: Quit")
        print()

        # Initial window size
        initial_width = 1280
        initial_height = 720
        cv2.resizeWindow(window_name, initial_width, initial_height)

        self.zoom_level = 1.0  # 1.0 = normal, 2.0 = 2x zoom, etc.
        self.zoom_center = None  # Center point for zoom (in original coords)
        self.screen_scale = 1.0  # Scale factor for fitting to screen

        while True:
            frame = self._draw_frame()
            h, w = frame.shape[:2]

            # Apply zoom if active
            if self.zoom_level > 1.0 and self.zoom_center is not None:
                # Calculate zoomed region
                zx, zy = self.zoom_center
                half_w = int(w / self.zoom_level / 2)
                half_h = int(h / self.zoom_level / 2)

                # Clamp to image bounds
                x1 = max(0, zx - half_w)
                y1 = max(0, zy - half_h)
                x2 = min(w, zx + half_w)
                y2 = min(h, zy + half_h)

                # Store zoom offset for coordinate conversion
                self.zoom_offset = (x1, y1)

                # Calculate actual zoom based on extracted region
                roi = frame[y1:y2, x1:x2]
                roi_h, roi_w = roi.shape[:2]
                self.actual_zoom_x = w / roi_w if roi_w > 0 else 1.0
                self.actual_zoom_y = h / roi_h if roi_h > 0 else 1.0

                # Scale up the zoomed region
                frame = cv2.resize(roi, (w, h), interpolation=cv2.INTER_LINEAR)
            else:
                self.zoom_offset = (0, 0)
                self.actual_zoom_x = 1.0
                self.actual_zoom_y = 1.0

            # Get current window size
            try:
                rect = cv2.getWindowImageRect(window_name)
                win_w, win_h = rect[2], rect[3]
                if win_w <= 0 or win_h <= 0:
                    win_w, win_h = initial_width, initial_height
            except:
                win_w, win_h = initial_width, initial_height

            # Letterbox to fit window while preserving aspect ratio
            display_frame, self.letterbox_pad_x, self.letterbox_pad_y, self.letterbox_scale = \
                self._letterbox(frame, win_w, win_h)

            # Store screen scale for coordinate conversion
            self.screen_scale = self.letterbox_scale

            cv2.imshow(window_name, display_frame)

            key = cv2.waitKey(30) & 0xFF

            if key == ord('q') or key == 27:  # Q or Escape
                break

            elif key == ord('n') or key == 83:  # N or Right arrow
                if self.varied_mode:
                    self._next_sample()
                else:
                    self._load_frame(self.frame_idx + 30)

            elif key == ord('p') or key == 81:  # P or Left arrow
                if self.varied_mode:
                    self._prev_sample()
                else:
                    self._load_frame(self.frame_idx - 30)

            elif key == ord(']'):  # Next video
                if self.video_idx < len(self.videos) - 1:
                    self._load_video(self.video_idx + 1)

            elif key == ord('['):  # Previous video
                if self.video_idx > 0:
                    self._load_video(self.video_idx - 1)

            elif key == ord('v'):  # Toggle varied mode
                self.varied_mode = not self.varied_mode
                print(f"Varied mode: {'ON' if self.varied_mode else 'OFF'}")

            elif key == ord('s'):  # Save
                if len(self.current_points) == 5:
                    self.store.add(self.video_key, self.frame_idx, self.current_points)
                    self.store.save(self.annotations_file)
                    print(f"Saved annotation for {self.video_key} frame {self.frame_idx}")
                else:
                    print(f"Need 5 points to save, have {len(self.current_points)}")

            elif key == ord('r'):  # Reset
                self.current_points = []
                print("Reset current frame")

            elif key == ord('z'):  # Zoom toggle - 2x zoom centered on mouse
                if self.zoom_level == 1.0:
                    self.zoom_level = 2.5
                    self.zoom_center = self.mouse_pos
                    print(f"Zoom 2.5x at {self.mouse_pos}")
                else:
                    self.zoom_level = 1.0
                    self.zoom_center = None
                    print("Zoom off")

        # Cleanup
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()

        # Final save prompt
        if len(self.store) > 0:
            self.store.save(self.annotations_file)
            print(f"\nSaved {len(self.store)} annotations to {self.annotations_file}")


def main():
    parser = argparse.ArgumentParser(description="Annotate plate corners in videos")
    parser.add_argument("--data-dir", type=Path,
                       default=Path("../../training_data"),
                       help="Path to training data directory")
    parser.add_argument("--output", type=Path,
                       default=Path("plate_annotations.json"),
                       help="Output annotations file")

    args = parser.parse_args()

    # Resolve paths
    script_dir = Path(__file__).parent
    data_dir = args.data_dir
    if not data_dir.is_absolute():
        data_dir = (script_dir / data_dir).resolve()

    output_file = args.output
    if not output_file.is_absolute():
        output_file = (script_dir / output_file).resolve()

    print(f"Data directory: {data_dir}")
    print(f"Annotations file: {output_file}")

    annotator = PlateAnnotator(data_dir, output_file)
    annotator.run()


if __name__ == "__main__":
    main()
