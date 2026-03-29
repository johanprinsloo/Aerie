"""Tests for annotate_frame."""

from __future__ import annotations

import numpy as np

from aerie_vision.annotate import annotate_frame
from aerie_vision.detection.types import RawDetection


def _frame(value: int = 128) -> np.ndarray:
    f = np.zeros((480, 640, 3), dtype=np.uint8)
    f[:] = value
    return f


def _det(
    label: str = "fire",
    conf: float = 0.9,
    bbox: tuple[int, int, int, int] = (50, 50, 200, 200),
) -> RawDetection:
    return RawDetection(
        label=label, confidence=conf, bbox=bbox,
        model_name="test", frame_number=0, timestamp=0.0,
    )


class TestAnnotateFrame:
    def test_returns_copy(self) -> None:
        frame = _frame()
        result = annotate_frame(frame, [_det()])
        # Modifying result should not affect original
        result[:] = 0
        assert frame.mean() > 0

    def test_dimensions_unchanged(self) -> None:
        frame = _frame()
        result = annotate_frame(frame, [_det()])
        assert result.shape == frame.shape

    def test_pixels_changed_with_detections(self) -> None:
        frame = _frame(128)
        result = annotate_frame(frame, [_det()])
        assert not np.array_equal(frame, result)

    def test_no_detections_returns_identical_copy(self) -> None:
        frame = _frame(100)
        result = annotate_frame(frame, [])
        np.testing.assert_array_equal(frame, result)
        assert frame is not result

    def test_multiple_detections(self) -> None:
        frame = _frame()
        dets = [
            _det("fire", 0.95, (10, 10, 100, 100)),
            _det("smoke", 0.80, (200, 200, 400, 400)),
            _det("person", 0.70, (300, 50, 500, 300)),
        ]
        result = annotate_frame(frame, dets)
        assert result.shape == frame.shape
        assert not np.array_equal(frame, result)

    def test_consistent_colors_per_label(self) -> None:
        frame = _frame()
        result1 = annotate_frame(frame, [_det("fire", 0.9, (50, 50, 100, 100))])
        result2 = annotate_frame(frame, [_det("fire", 0.8, (50, 50, 100, 100))])
        # Same label, same bbox location -> same pixels drawn (except confidence text)
        # At minimum both should have modified the same region
        diff1 = (frame != result1).any(axis=2)
        diff2 = (frame != result2).any(axis=2)
        overlap = (diff1 & diff2).sum()
        assert overlap > 0
