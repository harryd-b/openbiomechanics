"""GUI tool for annotating keypoints in video frames.

Supports two modes:
- plate: 5 corner points of home plate (static, 1 frame per video)
- bat: 2 endpoints of bat (moving, multiple frames per video)

Usage:
    python annotate.py --mode plate --data-dir <path>
    python annotate.py --mode bat --data-dir <path>

Controls:
    Left Click: Add point
    Right Click: Remove last point
    N / Right Arrow: Next frame/sample
    P / Left Arrow: Previous frame/sample
    ] : Next video (plate mode)
    [ : Previous video (plate mode)
    S / Space: Save annotations (Space also advances)
    R: Reset current frame
    Z: Zoom toggle (2x at mouse position)
    Q / Escape: Quit
"""

import cv2
import numpy as np
from pathlib import Path
import argparse
import json
from typing import List, Tuple, Optional, Dict
from abc import ABC, abstractmethod

try:
    from .annotations import AnnotationStore
except ImportError:
    from annotations import AnnotationStore

# Import bat annotations if available
try:
    import sys
    import importlib.util
    bat_ann_path = Path(__file__).parent.parent / "bat_detector" / "annotations.py"
    spec = importlib.util.spec_from_file_location("bat_annotations", bat_ann_path)
    bat_annotations = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bat_annotations)
    BatAnnotationStore = bat_annotations.BatAnnotationStore
except Exception:
    BatAnnotationStore = None


class AnnotationMode(ABC):
    """Abstract base for annotation modes."""

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def num_points(self) -> int:
        pass

    @property
    @abstractmethod
    def point_names(self) -> List[str]:
        pass

    @property
    @abstractmethod
    def point_colors(self) -> List[Tuple[int, int, int]]:
        pass

    @abstractmethod
    def load_store(self, path: Path):
        pass

    @abstractmethod
    def save_annotation(self, store, video_key: str, frame_idx: int, points: List[Tuple[int, int]]):
        pass

    @abstractmethod
    def get_existing(self, store, video_key: str, frame_idx: int) -> Optional[List[Tuple[int, int]]]:
        pass

    @abstractmethod
    def build_sample_queue(self, videos: List[Path], data_dir: Path, store) -> List[Tuple[int, int]]:
        pass

    def draw_shape(self, frame: np.ndarray, points: List[Tuple[int, int]]) -> None:
        """Draw connecting lines between points."""
        pass

    @property
    def supports_skip(self) -> bool:
        """Whether this mode supports skipping frames (e.g., bat not visible)."""
        return False

    def skip_annotation(self, store, video_key: str, frame_idx: int) -> None:
        """Mark frame as skipped/not visible. Override in subclasses that support it."""
        pass


class PlateMode(AnnotationMode):
    """Mode for annotating 5 home plate corners."""

    @property
    def name(self) -> str:
        return "plate"

    @property
    def num_points(self) -> int:
        return 5

    @property
    def point_names(self) -> List[str]:
        return ["Apex", "Back-Left", "Front-Left", "Front-Right", "Back-Right"]

    @property
    def point_colors(self) -> List[Tuple[int, int, int]]:
        return [
            (0, 0, 255),    # Red - Apex
            (0, 255, 0),    # Green - BL
            (255, 0, 0),    # Blue - FL
            (255, 255, 0),  # Cyan - FR
            (255, 0, 255),  # Magenta - BR
        ]

    def load_store(self, path: Path):
        if path.exists():
            return AnnotationStore.load(path)
        return AnnotationStore()

    def save_annotation(self, store, video_key: str, frame_idx: int, points: List[Tuple[int, int]]):
        store.add(video_key, frame_idx, points)

    def get_existing(self, store, video_key: str, frame_idx: int) -> Optional[List[Tuple[int, int]]]:
        existing = store.get(video_key, frame_idx)
        return list(existing.corners) if existing else None

    def build_sample_queue(self, videos: List[Path], data_dir: Path, store) -> List[Tuple[int, int]]:
        """One frame per video (plate is static)."""
        import random

        annotated = set(store.list_videos())
        unannotated = []
        already_done = []

        for vid_idx, video_path in enumerate(videos):
            video_key = str(video_path.relative_to(data_dir))
            if video_key in annotated:
                already_done.append((vid_idx, 30))
            else:
                unannotated.append((vid_idx, 30))

        random.shuffle(unannotated)
        random.shuffle(already_done)

        return unannotated + already_done

    def draw_shape(self, frame: np.ndarray, points: List[Tuple[int, int]]) -> None:
        if len(points) >= 2:
            pts = np.array(points, dtype=np.int32)
            if len(pts) == 5:
                cv2.polylines(frame, [pts], True, (0, 255, 255), 2)
            else:
                cv2.polylines(frame, [pts], False, (0, 255, 255), 2)


class BatMode(AnnotationMode):
    """Mode for annotating 2 bat endpoints (knob and tip)."""

    def __init__(self, frame_selection_file: Optional[Path] = None):
        self.frame_selection_file = frame_selection_file
        self._frame_selection: Dict[str, List[int]] = {}

    @property
    def name(self) -> str:
        return "bat"

    @property
    def num_points(self) -> int:
        return 2

    @property
    def point_names(self) -> List[str]:
        return ["Knob", "Tip"]

    @property
    def point_colors(self) -> List[Tuple[int, int, int]]:
        return [
            (0, 255, 0),    # Green - Knob
            (0, 0, 255),    # Red - Tip
        ]

    def load_store(self, path: Path):
        if BatAnnotationStore is None:
            raise ImportError("BatAnnotationStore not available")
        if path.exists():
            return BatAnnotationStore.load(path)
        return BatAnnotationStore()

    def save_annotation(self, store, video_key: str, frame_idx: int, points: List[Tuple[int, int]]):
        store.add(video_key, frame_idx, points, visible=True)

    def get_existing(self, store, video_key: str, frame_idx: int) -> Optional[List[Tuple[int, int]]]:
        existing = store.get(video_key, frame_idx)
        if existing and existing.visible:
            return list(existing.endpoints)
        return None

    def build_sample_queue(self, videos: List[Path], data_dir: Path, store) -> List[Tuple[int, int]]:
        """Multiple frames per video from frame_selection.json."""
        import random

        # Load frame selection
        if self.frame_selection_file and self.frame_selection_file.exists():
            with open(self.frame_selection_file) as f:
                data = json.load(f)
            self._frame_selection = data.get("selections", {})
        else:
            # Default: sample frames from each video
            self._frame_selection = {}
            for video_path in videos:
                video_key = str(video_path.relative_to(data_dir))
                # Sample frames at 10% intervals
                cap = cv2.VideoCapture(str(video_path))
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                frames = [int(frame_count * i / 10) for i in range(1, 10)]
                self._frame_selection[video_key] = frames

        # Build queue from selections
        annotated = set()
        for ann in store.all_annotations():
            annotated.add((ann.video_path, ann.frame_idx))

        unannotated = []
        already_done = []

        for vid_idx, video_path in enumerate(videos):
            video_key = str(video_path.relative_to(data_dir))
            frames = self._frame_selection.get(video_key, [])

            for frame_idx in frames:
                if (video_key, frame_idx) in annotated:
                    already_done.append((vid_idx, frame_idx))
                else:
                    unannotated.append((vid_idx, frame_idx))

        random.shuffle(unannotated)
        random.shuffle(already_done)

        return unannotated + already_done

    def draw_shape(self, frame: np.ndarray, points: List[Tuple[int, int]]) -> None:
        if len(points) == 2:
            cv2.line(frame, points[0], points[1], (255, 255, 0), 3)

    @property
    def supports_skip(self) -> bool:
        return True

    def skip_annotation(self, store, video_key: str, frame_idx: int) -> None:
        """Mark frame as bat not visible."""
        store.add(video_key, frame_idx, [(0, 0), (0, 0)], visible=False)


class KeypointAnnotator:
    """Interactive GUI for annotating keypoints."""

    def __init__(self, data_dir: Path, annotations_file: Path, mode: AnnotationMode):
        self.data_dir = data_dir
        self.annotations_file = annotations_file
        self.mode = mode

        # Load annotation store
        self.store = mode.load_store(annotations_file)
        print(f"Loaded {len(self.store)} existing {mode.name} annotations")

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

        # Sample queue
        self.sample_queue: List[Tuple[int, int]] = []
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
        """Build sample queue using mode-specific logic."""
        self.sample_queue = self.mode.build_sample_queue(self.videos, self.data_dir, self.store)
        unannotated = sum(1 for _ in self.sample_queue if self.sample_queue)
        print(f"Built sample queue: {len(self.sample_queue)} samples")

    def _load_video(self, idx: int) -> None:
        """Load a video by index."""
        if self.cap is not None:
            self.cap.release()

        self.video_idx = idx
        video_path = self.videos[idx]
        self.cap = cv2.VideoCapture(str(video_path))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_key = str(video_path.relative_to(self.data_dir))

        print(f"Loaded: {self.video_key} ({self.total_frames} frames)")

    def _load_frame(self, idx: int) -> None:
        """Load a specific frame."""
        idx = max(0, min(idx, self.total_frames - 1))
        self.frame_idx = idx

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self.cap.read()

        if ret:
            self.current_frame = frame
            existing = self.mode.get_existing(self.store, self.video_key, idx)
            self.current_points = list(existing) if existing else []

    def _load_next_sample(self):
        """Load the next sample from queue."""
        if self.queue_idx >= len(self.sample_queue):
            print("Reached end of sample queue!")
            return

        vid_idx, frame_idx = self.sample_queue[self.queue_idx]
        self._load_video(vid_idx)
        self._load_frame(frame_idx)

    def _next_sample(self):
        """Go to next sample in queue."""
        if self.queue_idx < len(self.sample_queue) - 1:
            self.queue_idx += 1
            self._load_next_sample()

    def _prev_sample(self):
        """Go to previous sample in queue."""
        if self.queue_idx > 0:
            self.queue_idx -= 1
            self._load_next_sample()

    def _draw_frame(self) -> np.ndarray:
        """Draw the current frame with annotations."""
        if self.current_frame is None:
            return np.zeros((720, 1280, 3), dtype=np.uint8)

        frame = self.current_frame.copy()
        h, w = frame.shape[:2]

        # Draw existing points
        for i, pt in enumerate(self.current_points):
            color = self.mode.point_colors[i] if i < len(self.mode.point_colors) else (255, 255, 255)
            cv2.circle(frame, pt, 8, color, -1)
            cv2.circle(frame, pt, 10, (255, 255, 255), 2)
            cv2.putText(frame, str(i + 1), (pt[0] + 12, pt[1] + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Draw connecting shape
        self.mode.draw_shape(frame, self.current_points)

        # Status bar
        status_h = 60
        cv2.rectangle(frame, (0, 0), (w, status_h), (40, 40, 40), -1)

        # Video info
        cv2.putText(frame, f"{self.video_key}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"Frame: {self.frame_idx}/{self.total_frames-1}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Points status
        num_points = self.mode.num_points
        if len(self.current_points) < num_points:
            next_pt = self.mode.point_names[len(self.current_points)]
            status = f"Click: {next_pt} ({len(self.current_points)}/{num_points})"
            color = (0, 200, 255)
        else:
            status = "Complete! Press S/Space to save"
            color = (0, 255, 0)

        cv2.putText(frame, status, (w // 2 - 150, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Annotation count and progress
        total_ann = len(self.store)
        progress = f"Sample {self.queue_idx + 1}/{len(self.sample_queue)}"
        cv2.putText(frame, f"Total: {total_ann} | {progress}", (w - 250, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Mode indicator
        cv2.putText(frame, f"Mode: {self.mode.name.upper()}", (w - 150, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 200, 255), 1)

        # Help
        help_text = "N/P:nav  S/Space:save  R:reset  Z:zoom  Q:quit"
        if self.mode.supports_skip:
            help_text = "X:skip  " + help_text
        cv2.putText(frame, help_text, (w - 420, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)

        return frame

    def _on_mouse(self, event: int, x: int, y: int, flags: int, param) -> None:
        """Handle mouse events."""
        pad_x = getattr(self, 'letterbox_pad_x', 0)
        pad_y = getattr(self, 'letterbox_pad_y', 0)
        letterbox_scale = getattr(self, 'letterbox_scale', 1.0)

        frame_x = (x - pad_x) / letterbox_scale
        frame_y = (y - pad_y) / letterbox_scale

        zoom_x = getattr(self, 'actual_zoom_x', 1.0)
        zoom_y = getattr(self, 'actual_zoom_y', 1.0)
        zoom_offset = getattr(self, 'zoom_offset', (0, 0))

        orig_x = int(frame_x / zoom_x) + zoom_offset[0]
        orig_y = int(frame_y / zoom_y) + zoom_offset[1]

        self.mouse_pos = (orig_x, orig_y)

        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.current_points) < self.mode.num_points:
                self.current_points.append((orig_x, orig_y))
                print(f"  Added {self.mode.point_names[len(self.current_points)-1]}: ({orig_x}, {orig_y})")

        elif event == cv2.EVENT_RBUTTONDOWN:
            if self.current_points:
                removed = self.current_points.pop()
                print(f"  Removed point: {removed}")

    def _letterbox(self, frame: np.ndarray, target_w: int, target_h: int) -> Tuple[np.ndarray, int, int, float]:
        """Letterbox frame to target size."""
        h, w = frame.shape[:2]
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        pad_x = (target_w - new_w) // 2
        pad_y = (target_h - new_h) // 2

        letterboxed = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        letterboxed[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

        return letterboxed, pad_x, pad_y, scale

    def run(self) -> None:
        """Run the annotation GUI."""
        window_name = f"{self.mode.name.title()} Annotator"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, self._on_mouse)

        self.letterbox_pad_x = 0
        self.letterbox_pad_y = 0
        self.letterbox_scale = 1.0
        self.zoom_offset = (0, 0)
        self.actual_zoom_x = 1.0
        self.actual_zoom_y = 1.0

        print(f"\nControls ({self.mode.name} mode):")
        print("  Left Click: Add point | Right Click: Remove last")
        print("  N/P or Arrows: Next/Prev sample")
        print("  S or Space: Save | R: Reset | Z: Zoom | Q: Quit")
        if self.mode.supports_skip:
            print("  X: Skip (mark as not visible)")
        print()

        initial_width, initial_height = 1280, 720
        cv2.resizeWindow(window_name, initial_width, initial_height)

        self.zoom_level = 1.0
        self.zoom_center = None

        while True:
            frame = self._draw_frame()
            h, w = frame.shape[:2]

            # Apply zoom
            if self.zoom_level > 1.0 and self.zoom_center is not None:
                zx, zy = self.zoom_center
                half_w = int(w / self.zoom_level / 2)
                half_h = int(h / self.zoom_level / 2)

                x1 = max(0, zx - half_w)
                y1 = max(0, zy - half_h)
                x2 = min(w, zx + half_w)
                y2 = min(h, zy + half_h)

                self.zoom_offset = (x1, y1)
                roi = frame[y1:y2, x1:x2]
                roi_h, roi_w = roi.shape[:2]
                self.actual_zoom_x = w / roi_w if roi_w > 0 else 1.0
                self.actual_zoom_y = h / roi_h if roi_h > 0 else 1.0
                frame = cv2.resize(roi, (w, h), interpolation=cv2.INTER_LINEAR)
            else:
                self.zoom_offset = (0, 0)
                self.actual_zoom_x = 1.0
                self.actual_zoom_y = 1.0

            # Get window size
            try:
                rect = cv2.getWindowImageRect(window_name)
                win_w, win_h = rect[2], rect[3]
                if win_w <= 0 or win_h <= 0:
                    win_w, win_h = initial_width, initial_height
            except:
                win_w, win_h = initial_width, initial_height

            display_frame, self.letterbox_pad_x, self.letterbox_pad_y, self.letterbox_scale = \
                self._letterbox(frame, win_w, win_h)

            cv2.imshow(window_name, display_frame)

            key = cv2.waitKey(30) & 0xFF

            if key == ord('q') or key == 27:
                break

            elif key == ord('n') or key == 83:  # N or Right
                self._next_sample()

            elif key == ord('p') or key == 81:  # P or Left
                self._prev_sample()

            elif key == ord('s') or key == ord(' '):  # Save (Space also advances)
                if len(self.current_points) == self.mode.num_points:
                    self.mode.save_annotation(self.store, self.video_key, self.frame_idx, self.current_points)
                    self.store.save(self.annotations_file)
                    print(f"Saved {self.mode.name} annotation for {self.video_key} frame {self.frame_idx}")
                    if key == ord(' '):
                        self._next_sample()
                else:
                    print(f"Need {self.mode.num_points} points to save, have {len(self.current_points)}")

            elif key == ord('r'):
                self.current_points = []
                print("Reset current frame")

            elif key == ord('z'):
                if self.zoom_level == 1.0:
                    self.zoom_level = 2.5
                    self.zoom_center = self.mouse_pos
                    print(f"Zoom 2.5x at {self.mouse_pos}")
                else:
                    self.zoom_level = 1.0
                    self.zoom_center = None
                    print("Zoom off")

            elif key == ord('x'):
                if self.mode.supports_skip:
                    self.mode.skip_annotation(self.store, self.video_key, self.frame_idx)
                    self.store.save(self.annotations_file)
                    print(f"Skipped {self.video_key} frame {self.frame_idx} (marked not visible)")
                    self._next_sample()
                else:
                    print("Skip not supported in this mode")

        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()

        if len(self.store) > 0:
            self.store.save(self.annotations_file)
            print(f"\nSaved {len(self.store)} annotations to {self.annotations_file}")


def main():
    parser = argparse.ArgumentParser(description="Annotate keypoints in videos")
    parser.add_argument("--mode", type=str, choices=["plate", "bat"], default="plate",
                        help="Annotation mode: 'plate' (5 corners) or 'bat' (2 endpoints)")
    parser.add_argument("--data-dir", type=Path,
                        default=Path("../../training_data"),
                        help="Path to training data directory")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output annotations file (default: <mode>_annotations.json)")
    parser.add_argument("--frame-selection", type=Path, default=None,
                        help="Frame selection JSON (bat mode only)")

    args = parser.parse_args()

    script_dir = Path(__file__).parent
    data_dir = args.data_dir
    if not data_dir.is_absolute():
        data_dir = (script_dir / data_dir).resolve()

    # Set default output based on mode
    if args.output is None:
        args.output = Path(f"{args.mode}_annotations.json")

    output_file = args.output
    if not output_file.is_absolute():
        if args.mode == "bat":
            output_file = (script_dir.parent / "bat_detector" / output_file).resolve()
        else:
            output_file = (script_dir / output_file).resolve()

    # Create mode
    if args.mode == "plate":
        mode = PlateMode()
    else:
        frame_selection = args.frame_selection
        if frame_selection is None:
            frame_selection = script_dir.parent / "bat_detector" / "frame_selection.json"
        if not frame_selection.is_absolute():
            frame_selection = (script_dir / frame_selection).resolve()
        mode = BatMode(frame_selection_file=frame_selection)

    print(f"Mode: {args.mode}")
    print(f"Data directory: {data_dir}")
    print(f"Annotations file: {output_file}")

    annotator = KeypointAnnotator(data_dir, output_file, mode)
    annotator.run()


if __name__ == "__main__":
    main()
