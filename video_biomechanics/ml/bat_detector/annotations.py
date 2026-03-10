"""Bat endpoint annotation data structures."""

from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from pathlib import Path
import json


@dataclass
class BatAnnotation:
    """Annotation for bat endpoints in a video frame.

    Endpoints: [knob, tip]
    - knob: The handle end (held by hands)
    - tip: The barrel end (hits the ball)
    """
    video_path: str
    frame_idx: int
    endpoints: List[Tuple[int, int]]  # [(knob_x, knob_y), (tip_x, tip_y)]
    visible: bool = True  # Whether bat is visible in frame

    def __post_init__(self):
        if len(self.endpoints) != 2:
            raise ValueError(f"Bat must have exactly 2 endpoints, got {len(self.endpoints)}")

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "video_path": self.video_path,
            "frame_idx": self.frame_idx,
            "endpoints": self.endpoints,
            "visible": self.visible
        }

    @classmethod
    def from_dict(cls, d: dict) -> "BatAnnotation":
        """Create from dict."""
        return cls(
            video_path=d["video_path"],
            frame_idx=d["frame_idx"],
            endpoints=[tuple(c) for c in d["endpoints"]],
            visible=d.get("visible", True)
        )

    @property
    def knob(self) -> Tuple[int, int]:
        return self.endpoints[0]

    @property
    def tip(self) -> Tuple[int, int]:
        return self.endpoints[1]

    @property
    def length_pixels(self) -> float:
        """Calculate bat length in pixels."""
        import numpy as np
        return float(np.linalg.norm(
            np.array(self.tip) - np.array(self.knob)
        ))


class BatAnnotationStore:
    """Store for bat endpoint annotations across multiple videos."""

    def __init__(self):
        self._annotations: Dict[str, Dict[int, BatAnnotation]] = {}

    def __len__(self) -> int:
        return sum(len(frames) for frames in self._annotations.values())

    def add(self, video_path: str, frame_idx: int, endpoints: List[Tuple[int, int]],
            visible: bool = True) -> None:
        """Add or update an annotation."""
        ann = BatAnnotation(
            video_path=video_path, frame_idx=frame_idx,
            endpoints=endpoints, visible=visible
        )

        if video_path not in self._annotations:
            self._annotations[video_path] = {}

        self._annotations[video_path][frame_idx] = ann

    def get(self, video_path: str, frame_idx: int) -> Optional[BatAnnotation]:
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
            "type": "bat_annotations",
            "annotations": []
        }

        for video_frames in self._annotations.values():
            for ann in video_frames.values():
                data["annotations"].append(ann.to_dict())

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "BatAnnotationStore":
        """Load annotations from JSON file."""
        store = cls()

        with open(path, 'r') as f:
            data = json.load(f)

        for ann_dict in data.get("annotations", []):
            ann = BatAnnotation.from_dict(ann_dict)
            if ann.video_path not in store._annotations:
                store._annotations[ann.video_path] = {}
            store._annotations[ann.video_path][ann.frame_idx] = ann

        return store

    def all_annotations(self) -> List[BatAnnotation]:
        """Get all annotations as a flat list."""
        result = []
        for video_frames in self._annotations.values():
            result.extend(video_frames.values())
        return result

    def visible_annotations(self) -> List[BatAnnotation]:
        """Get only annotations where bat is visible."""
        return [a for a in self.all_annotations() if a.visible]
