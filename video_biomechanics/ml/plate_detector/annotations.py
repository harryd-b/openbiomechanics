"""Plate corner annotation data structures."""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
from pathlib import Path
import json


@dataclass
class PlateAnnotation:
    """Annotation for home plate corners in a video frame.

    Corners should be ordered: [Apex, BL, FL, FR, BR]
    - Apex: The pointed tip toward the pitcher
    - BL: Back left (catcher's right)
    - FL: Front left
    - FR: Front right
    - BR: Back right (catcher's left)
    """
    video_path: str
    frame_idx: int
    corners: List[Tuple[int, int]]

    def __post_init__(self):
        if len(self.corners) != 5:
            raise ValueError(f"Plate must have exactly 5 corners, got {len(self.corners)}")

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "video_path": self.video_path,
            "frame_idx": self.frame_idx,
            "corners": self.corners
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PlateAnnotation":
        """Create from dict."""
        return cls(
            video_path=d["video_path"],
            frame_idx=d["frame_idx"],
            corners=[tuple(c) for c in d["corners"]]
        )


class AnnotationStore:
    """Store for plate corner annotations across multiple videos."""

    def __init__(self):
        self._annotations: Dict[str, Dict[int, PlateAnnotation]] = {}

    def __len__(self) -> int:
        return sum(len(frames) for frames in self._annotations.values())

    def add(self, video_path: str, frame_idx: int, corners: List[Tuple[int, int]]) -> None:
        """Add or update an annotation."""
        ann = PlateAnnotation(video_path=video_path, frame_idx=frame_idx, corners=corners)

        if video_path not in self._annotations:
            self._annotations[video_path] = {}

        self._annotations[video_path][frame_idx] = ann

    def get(self, video_path: str, frame_idx: int) -> Optional[PlateAnnotation]:
        """Get annotation for a specific video and frame."""
        if video_path not in self._annotations:
            return None
        return self._annotations[video_path].get(frame_idx)

    def list_videos(self) -> List[str]:
        """List all videos with annotations."""
        return list(self._annotations.keys())

    def get_frames(self, video_path: str) -> List[int]:
        """Get all annotated frame indices for a video."""
        if video_path not in self._annotations:
            return []
        return sorted(self._annotations[video_path].keys())

    def save(self, path: Path) -> None:
        """Save annotations to JSON file."""
        data = {
            "version": "1.0",
            "annotations": []
        }

        for video_frames in self._annotations.values():
            for ann in video_frames.values():
                data["annotations"].append(ann.to_dict())

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "AnnotationStore":
        """Load annotations from JSON file."""
        store = cls()

        with open(path, 'r') as f:
            data = json.load(f)

        for ann_dict in data.get("annotations", []):
            ann = PlateAnnotation.from_dict(ann_dict)
            if ann.video_path not in store._annotations:
                store._annotations[ann.video_path] = {}
            store._annotations[ann.video_path][ann.frame_idx] = ann

        return store

    def all_annotations(self) -> List[PlateAnnotation]:
        """Get all annotations as a flat list."""
        result = []
        for video_frames in self._annotations.values():
            result.extend(video_frames.values())
        return result
