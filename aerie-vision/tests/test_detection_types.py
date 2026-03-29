"""Tests for RawDetection and DetectionResult data types."""

from __future__ import annotations

import numpy as np
import pytest

from aerie_vision.detection.types import DetectionResult, RawDetection


def _det(label: str = "fire", conf: float = 0.9, bbox: tuple[int, int, int, int] = (10, 20, 100, 200)) -> RawDetection:
    return RawDetection(
        label=label, confidence=conf, bbox=bbox,
        model_name="test", frame_number=0, timestamp=0.0,
    )


class TestRawDetection:
    def test_construction(self) -> None:
        d = _det()
        assert d.label == "fire"
        assert d.confidence == 0.9
        assert d.bbox == (10, 20, 100, 200)
        assert d.model_name == "test"

    def test_frozen(self) -> None:
        d = _det()
        with pytest.raises(AttributeError):
            d.label = "smoke"  # type: ignore[misc]

    def test_different_labels(self) -> None:
        a = _det(label="fire")
        b = _det(label="smoke")
        assert a.label != b.label


class TestDetectionResult:
    def test_construction(self) -> None:
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        dets = (_det(), _det(label="smoke", conf=0.7))
        result = DetectionResult(
            frame_number=5, timestamp=1.0,
            detections=dets, frame=frame, inference_ms=12.3,
        )
        assert result.frame_number == 5
        assert len(result.detections) == 2
        assert result.inference_ms == 12.3

    def test_empty_detections(self) -> None:
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        result = DetectionResult(
            frame_number=0, timestamp=0.0,
            detections=(), frame=frame, inference_ms=5.0,
        )
        assert len(result.detections) == 0

    def test_frozen(self) -> None:
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        result = DetectionResult(
            frame_number=0, timestamp=0.0,
            detections=(), frame=frame, inference_ms=5.0,
        )
        with pytest.raises(AttributeError):
            result.frame_number = 99  # type: ignore[misc]
