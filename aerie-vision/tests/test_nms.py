"""Tests for IoU computation and cross-model NMS merge."""

from __future__ import annotations

from aerie_vision.detection.nms import iou, merge_detections
from aerie_vision.detection.types import RawDetection


def _det(
    label: str = "fire",
    conf: float = 0.9,
    bbox: tuple[int, int, int, int] = (0, 0, 100, 100),
    model: str = "a",
) -> RawDetection:
    return RawDetection(
        label=label, confidence=conf, bbox=bbox,
        model_name=model, frame_number=0, timestamp=0.0,
    )


class TestIoU:
    def test_identical_boxes(self) -> None:
        box = (0, 0, 100, 100)
        assert iou(box, box) == 1.0

    def test_disjoint_boxes(self) -> None:
        assert iou((0, 0, 50, 50), (100, 100, 200, 200)) == 0.0

    def test_partial_overlap(self) -> None:
        # box A: 0,0 -> 100,100 = area 10000
        # box B: 50,50 -> 150,150 = area 10000
        # intersection: 50,50 -> 100,100 = area 2500
        # union: 10000 + 10000 - 2500 = 17500
        result = iou((0, 0, 100, 100), (50, 50, 150, 150))
        assert abs(result - 2500 / 17500) < 1e-6

    def test_contained_box(self) -> None:
        # small box fully inside big box
        # small: 25,25 -> 75,75 = area 2500
        # big: 0,0 -> 100,100 = area 10000
        # intersection = 2500, union = 10000
        result = iou((0, 0, 100, 100), (25, 25, 75, 75))
        assert abs(result - 2500 / 10000) < 1e-6

    def test_zero_area_box(self) -> None:
        assert iou((0, 0, 0, 0), (0, 0, 100, 100)) == 0.0

    def test_touching_boxes_no_overlap(self) -> None:
        assert iou((0, 0, 50, 50), (50, 0, 100, 50)) == 0.0


class TestMergeDetections:
    def test_single_detection(self) -> None:
        dets = [_det()]
        merged = merge_detections(dets)
        assert len(merged) == 1

    def test_empty(self) -> None:
        assert merge_detections([]) == []

    def test_suppresses_overlapping_same_label(self) -> None:
        # Two fire detections from different models, same box, different confidence
        d_high = _det(label="fire", conf=0.9, bbox=(0, 0, 100, 100), model="a")
        d_low = _det(label="fire", conf=0.6, bbox=(0, 0, 100, 100), model="b")
        merged = merge_detections([d_low, d_high], iou_threshold=0.5)
        assert len(merged) == 1
        assert merged[0].confidence == 0.9

    def test_keeps_different_labels(self) -> None:
        d_fire = _det(label="fire", conf=0.9, bbox=(0, 0, 100, 100), model="a")
        d_person = _det(label="person", conf=0.8, bbox=(0, 0, 100, 100), model="b")
        merged = merge_detections([d_fire, d_person], iou_threshold=0.5)
        assert len(merged) == 2

    def test_keeps_non_overlapping_same_label(self) -> None:
        d1 = _det(label="fire", conf=0.9, bbox=(0, 0, 50, 50), model="a")
        d2 = _det(label="fire", conf=0.8, bbox=(200, 200, 300, 300), model="b")
        merged = merge_detections([d1, d2], iou_threshold=0.5)
        assert len(merged) == 2

    def test_iou_below_threshold_not_suppressed(self) -> None:
        d1 = _det(label="fire", conf=0.9, bbox=(0, 0, 100, 100), model="a")
        d2 = _det(label="fire", conf=0.8, bbox=(80, 80, 180, 180), model="b")
        # IoU of these is quite small (~4%)
        merged = merge_detections([d1, d2], iou_threshold=0.5)
        assert len(merged) == 2

    def test_three_overlapping_keeps_best(self) -> None:
        d1 = _det(label="fire", conf=0.5, bbox=(0, 0, 100, 100), model="a")
        d2 = _det(label="fire", conf=0.9, bbox=(0, 0, 100, 100), model="b")
        d3 = _det(label="fire", conf=0.7, bbox=(0, 0, 100, 100), model="c")
        merged = merge_detections([d1, d2, d3], iou_threshold=0.5)
        assert len(merged) == 1
        assert merged[0].confidence == 0.9
