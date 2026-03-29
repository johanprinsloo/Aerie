"""Tests for DetectionRouter: single, parallel, and escalation modes."""

from __future__ import annotations

import numpy as np

from aerie_vision.detection.mock_detector import MockDetector
from aerie_vision.detection.router import DetectionRouter
from aerie_vision.detection.types import RawDetection


def _frame() -> np.ndarray:
    return np.zeros((240, 320, 3), dtype=np.uint8)


def _det(
    label: str = "fire",
    conf: float = 0.9,
    bbox: tuple[int, int, int, int] = (10, 20, 100, 200),
    model: str = "primary",
    frame_number: int = 0,
) -> RawDetection:
    return RawDetection(
        label=label, confidence=conf, bbox=bbox,
        model_name=model, frame_number=frame_number, timestamp=0.0,
    )


class TestSingleModel:
    def test_primary_only(self) -> None:
        primary = MockDetector(
            scripted={0: [_det("fire", 0.9)]},
            name="primary",
        )
        router = DetectionRouter(primary=primary)
        result = router.detect(_frame(), frame_number=0, timestamp=0.0)

        assert len(result.detections) == 1
        assert result.detections[0].label == "fire"
        assert result.inference_ms > 0

    def test_no_detections(self) -> None:
        primary = MockDetector(name="primary")
        router = DetectionRouter(primary=primary)
        result = router.detect(_frame(), frame_number=0)
        assert len(result.detections) == 0

    def test_warm_up(self) -> None:
        primary = MockDetector(name="primary")
        router = DetectionRouter(primary=primary)
        router.warm_up()  # should not raise


class TestParallelSecondaries:
    def test_merges_primary_and_secondary(self) -> None:
        primary = MockDetector(
            scripted={0: [_det("fire", 0.9, model="primary")]},
            name="primary",
        )
        secondary = MockDetector(
            scripted={0: [_det("person", 0.8, model="secondary", bbox=(200, 200, 300, 300))]},
            name="secondary",
        )
        router = DetectionRouter(primary=primary, secondaries=[secondary])
        result = router.detect(_frame(), frame_number=0)

        labels = {d.label for d in result.detections}
        assert "fire" in labels
        assert "person" in labels

    def test_deduplicates_overlapping(self) -> None:
        # Both models detect fire at the same location
        primary = MockDetector(
            scripted={0: [_det("fire", 0.9, bbox=(10, 20, 100, 200), model="primary")]},
            name="primary",
        )
        secondary = MockDetector(
            scripted={0: [_det("fire", 0.7, bbox=(10, 20, 100, 200), model="secondary")]},
            name="secondary",
        )
        router = DetectionRouter(primary=primary, secondaries=[secondary])
        result = router.detect(_frame(), frame_number=0)

        assert len(result.detections) == 1
        assert result.detections[0].confidence == 0.9

    def test_multiple_secondaries(self) -> None:
        primary = MockDetector(scripted={0: [_det("fire", 0.9, model="p")]}, name="p")
        sec_a = MockDetector(scripted={0: [_det("smoke", 0.8, model="a", bbox=(200, 200, 300, 300))]}, name="a")
        sec_b = MockDetector(scripted={0: [_det("vehicle", 0.7, model="b", bbox=(50, 50, 60, 60))]}, name="b")
        router = DetectionRouter(primary=primary, secondaries=[sec_a, sec_b])
        result = router.detect(_frame(), frame_number=0)

        labels = {d.label for d in result.detections}
        assert labels == {"fire", "smoke", "vehicle"}


class TestEscalation:
    def test_escalation_triggered_on_low_confidence(self) -> None:
        primary = MockDetector(
            scripted={0: [_det("fire", 0.3, model="primary")]},
            name="primary",
        )
        escalation = MockDetector(
            scripted={0: [_det("fire", 0.85, bbox=(10, 20, 100, 200), model="escalation")]},
            name="escalation",
        )
        router = DetectionRouter(
            primary=primary,
            escalation=escalation,
            escalation_threshold=0.5,
        )
        result = router.detect(_frame(), frame_number=0)

        # The escalation model's higher-confidence detection should win
        assert len(result.detections) == 1
        assert result.detections[0].confidence == 0.85

    def test_no_escalation_when_confident(self) -> None:
        primary = MockDetector(
            scripted={0: [_det("fire", 0.9, model="primary")]},
            name="primary",
        )
        escalation = MockDetector(
            scripted={0: [_det("fire", 0.95, bbox=(10, 20, 100, 200), model="escalation")]},
            name="escalation",
        )
        router = DetectionRouter(
            primary=primary,
            escalation=escalation,
            escalation_threshold=0.5,
        )
        result = router.detect(_frame(), frame_number=0)

        # Primary was confident enough; escalation should not run.
        # The result should contain only the primary's detection.
        assert len(result.detections) == 1
        assert result.detections[0].model_name == "primary"

    def test_escalation_not_triggered_on_empty(self) -> None:
        primary = MockDetector(name="primary")
        escalation = MockDetector(
            scripted={0: [_det("fire", 0.9, model="escalation")]},
            name="escalation",
        )
        router = DetectionRouter(
            primary=primary,
            escalation=escalation,
            escalation_threshold=0.5,
        )
        result = router.detect(_frame(), frame_number=0)

        # No primary detections means no low-confidence trigger
        assert len(result.detections) == 0
