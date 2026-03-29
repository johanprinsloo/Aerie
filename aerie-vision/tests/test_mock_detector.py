"""Tests for MockDetector."""

from __future__ import annotations

import numpy as np

from aerie_vision.detection.mock_detector import MockDetector
from aerie_vision.detection.types import RawDetection


def _frame() -> np.ndarray:
    return np.zeros((240, 320, 3), dtype=np.uint8)


def _det(label: str = "fire", frame_number: int = 0) -> RawDetection:
    return RawDetection(
        label=label, confidence=0.95, bbox=(10, 20, 100, 200),
        model_name="mock", frame_number=frame_number, timestamp=0.0,
    )


class TestMockDetector:
    def test_scripted_frame_returns_detections(self) -> None:
        scripted = {5: [_det("fire", 5), _det("smoke", 5)]}
        det = MockDetector(scripted=scripted)
        results = det.detect(_frame(), frame_number=5)
        assert len(results) == 2
        assert results[0].label == "fire"
        assert results[1].label == "smoke"

    def test_unscripted_frame_returns_empty(self) -> None:
        scripted = {5: [_det("fire", 5)]}
        det = MockDetector(scripted=scripted)
        results = det.detect(_frame(), frame_number=99)
        assert results == []

    def test_no_script_returns_empty(self) -> None:
        det = MockDetector()
        results = det.detect(_frame(), frame_number=0)
        assert results == []

    def test_name(self) -> None:
        det = MockDetector(name="my-mock")
        assert det.name == "my-mock"

    def test_warm_up_is_noop(self) -> None:
        det = MockDetector()
        det.warm_up()  # should not raise

    def test_returns_copy(self) -> None:
        original = [_det("fire", 5)]
        scripted = {5: original}
        det = MockDetector(scripted=scripted)
        results = det.detect(_frame(), frame_number=5)
        results.clear()
        # Original script should be unaffected
        assert len(det.detect(_frame(), frame_number=5)) == 1
