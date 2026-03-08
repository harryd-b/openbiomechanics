"""Tests for plate corner annotation data structures."""

import pytest
import json
import tempfile
from pathlib import Path
from dataclasses import asdict

from annotations import PlateAnnotation, AnnotationStore


class TestPlateAnnotation:
    """Tests for PlateAnnotation dataclass."""

    def test_create_annotation(self):
        """Test creating a plate annotation."""
        corners = [(100, 200), (150, 180), (200, 200), (180, 250), (120, 250)]
        ann = PlateAnnotation(
            video_path="session_001/primary.mp4",
            frame_idx=15,
            corners=corners
        )
        assert ann.video_path == "session_001/primary.mp4"
        assert ann.frame_idx == 15
        assert len(ann.corners) == 5

    def test_annotation_requires_5_corners(self):
        """Test that annotation validates corner count."""
        with pytest.raises(ValueError):
            PlateAnnotation(
                video_path="test.mp4",
                frame_idx=0,
                corners=[(0, 0), (1, 1)]  # Only 2 corners
            )

    def test_annotation_to_dict(self):
        """Test converting annotation to dict for JSON."""
        corners = [(100, 200), (150, 180), (200, 200), (180, 250), (120, 250)]
        ann = PlateAnnotation(
            video_path="test.mp4",
            frame_idx=10,
            corners=corners
        )
        d = ann.to_dict()
        assert d["video_path"] == "test.mp4"
        assert d["frame_idx"] == 10
        assert d["corners"] == corners

    def test_annotation_from_dict(self):
        """Test creating annotation from dict."""
        d = {
            "video_path": "test.mp4",
            "frame_idx": 5,
            "corners": [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10)]
        }
        ann = PlateAnnotation.from_dict(d)
        assert ann.video_path == "test.mp4"
        assert ann.frame_idx == 5


class TestAnnotationStore:
    """Tests for AnnotationStore."""

    def test_create_empty_store(self):
        """Test creating empty annotation store."""
        store = AnnotationStore()
        assert len(store) == 0

    def test_add_annotation(self):
        """Test adding annotation to store."""
        store = AnnotationStore()
        corners = [(100, 200), (150, 180), (200, 200), (180, 250), (120, 250)]
        store.add("test.mp4", 0, corners)
        assert len(store) == 1

    def test_get_annotation(self):
        """Test getting annotation by video and frame."""
        store = AnnotationStore()
        corners = [(100, 200), (150, 180), (200, 200), (180, 250), (120, 250)]
        store.add("test.mp4", 5, corners)

        ann = store.get("test.mp4", 5)
        assert ann is not None
        assert ann.frame_idx == 5

    def test_get_missing_annotation(self):
        """Test getting non-existent annotation returns None."""
        store = AnnotationStore()
        assert store.get("test.mp4", 0) is None

    def test_save_and_load(self):
        """Test saving and loading annotations."""
        store = AnnotationStore()
        corners = [(100, 200), (150, 180), (200, 200), (180, 250), (120, 250)]
        store.add("video1.mp4", 0, corners)
        store.add("video2.mp4", 10, corners)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)

        try:
            store.save(path)

            loaded = AnnotationStore.load(path)
            assert len(loaded) == 2
            assert loaded.get("video1.mp4", 0) is not None
            assert loaded.get("video2.mp4", 10) is not None
        finally:
            path.unlink()

    def test_update_annotation(self):
        """Test updating existing annotation."""
        store = AnnotationStore()
        corners1 = [(100, 200), (150, 180), (200, 200), (180, 250), (120, 250)]
        corners2 = [(110, 210), (160, 190), (210, 210), (190, 260), (130, 260)]

        store.add("test.mp4", 0, corners1)
        store.add("test.mp4", 0, corners2)  # Update same frame

        assert len(store) == 1  # Still only 1 annotation
        ann = store.get("test.mp4", 0)
        assert ann.corners == corners2  # Updated corners

    def test_list_videos(self):
        """Test listing annotated videos."""
        store = AnnotationStore()
        corners = [(100, 200), (150, 180), (200, 200), (180, 250), (120, 250)]
        store.add("video1.mp4", 0, corners)
        store.add("video1.mp4", 10, corners)
        store.add("video2.mp4", 5, corners)

        videos = store.list_videos()
        assert len(videos) == 2
        assert "video1.mp4" in videos
        assert "video2.mp4" in videos

    def test_get_frames_for_video(self):
        """Test getting all annotated frames for a video."""
        store = AnnotationStore()
        corners = [(100, 200), (150, 180), (200, 200), (180, 250), (120, 250)]
        store.add("video1.mp4", 0, corners)
        store.add("video1.mp4", 10, corners)
        store.add("video1.mp4", 20, corners)

        frames = store.get_frames("video1.mp4")
        assert frames == [0, 10, 20]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
