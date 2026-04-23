"""Tests for annotate_segmentation."""

from __future__ import annotations

import numpy as np

from aerie_vision.annotate_segmentation import annotate_segmentation
from aerie_vision.segmentation.types import InstanceMask


def _frame(value: int = 100) -> np.ndarray:
    f = np.zeros((240, 320, 3), dtype=np.uint8)
    f[:] = value
    return f


def _mask(h: int = 240, w: int = 320, x1: int = 50, y1: int = 50, x2: int = 200, y2: int = 200) -> np.ndarray:
    m = np.zeros((h, w), dtype=bool)
    m[y1:y2, x1:x2] = True
    return m


def _inst(
    instance_id: int = 1,
    label: str = "fire",
    bbox: tuple[int, int, int, int] = (50, 50, 200, 200),
) -> InstanceMask:
    return InstanceMask(
        instance_id=instance_id, label=label, confidence=0.9,
        bbox=bbox, centroid=((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2),
        area_px=(bbox[2] - bbox[0]) * (bbox[3] - bbox[1]),
        mask=_mask(x1=bbox[0], y1=bbox[1], x2=bbox[2], y2=bbox[3]),
    )


class TestAnnotateSegmentation:
    def test_returns_copy(self) -> None:
        frame = _frame()
        result = annotate_segmentation(frame, [_inst()])
        result[:] = 0
        assert frame.mean() > 0

    def test_dimensions_unchanged(self) -> None:
        frame = _frame()
        result = annotate_segmentation(frame, [_inst()])
        assert result.shape == frame.shape

    def test_pixels_changed_with_instances(self) -> None:
        frame = _frame(100)
        result = annotate_segmentation(frame, [_inst()])
        assert not np.array_equal(frame, result)

    def test_no_instances_returns_identical_copy(self) -> None:
        frame = _frame(100)
        result = annotate_segmentation(frame, [])
        np.testing.assert_array_equal(frame, result)
        assert frame is not result

    def test_masked_region_changes_color(self) -> None:
        frame = _frame(100)
        inst = _inst(bbox=(50, 50, 100, 100))
        result = annotate_segmentation(frame, [inst])
        # Inside the mask, pixels should differ from the original solid color
        inside = result[60:90, 60:90]
        assert not np.all(inside == 100)

    def test_mask_shape_mismatch_skipped(self) -> None:
        frame = _frame()
        # Mask of wrong shape -- should be silently skipped, no crash
        wrong = InstanceMask(
            instance_id=1, label="x", confidence=0.5,
            bbox=(0, 0, 10, 10), centroid=(5, 5), area_px=100,
            mask=np.ones((50, 50), dtype=bool),
        )
        result = annotate_segmentation(frame, [wrong])
        assert result.shape == frame.shape

    def test_multiple_instances(self) -> None:
        frame = _frame()
        instances = [
            _inst(1, "fire", (10, 10, 100, 100)),
            _inst(2, "smoke", (150, 150, 300, 230)),
        ]
        result = annotate_segmentation(frame, instances)
        assert result.shape == frame.shape
        assert not np.array_equal(frame, result)
