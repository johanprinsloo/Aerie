"""Tests for the Ultralytics-Result -> InstanceMask converter.

The Sam3SourceRunner thread itself is not unit-tested here because it depends
on a real Ultralytics SAM 3 install + downloaded weights. The pure converter
function is tested against duck-typed mock objects that mirror the shape of
``ultralytics.engine.results.Results``.
"""

from __future__ import annotations

import numpy as np

from aerie_vision.segmentation.sam3_source_runner import (
    ultralytics_result_to_instances,
)


class _FakeTensor:
    """Stand-in for a torch tensor: supports .cpu().numpy() and len()."""

    def __init__(self, arr: np.ndarray) -> None:
        self._arr = arr

    def cpu(self) -> "_FakeTensor":
        return self

    def numpy(self) -> np.ndarray:
        return self._arr

    def __len__(self) -> int:
        return len(self._arr)


class _FakeMasks:
    def __init__(self, data: np.ndarray) -> None:
        self.data = _FakeTensor(data)


class _FakeBoxes:
    def __init__(
        self,
        conf: np.ndarray,
        cls: np.ndarray,
        ids: np.ndarray | None,
    ) -> None:
        self.conf = _FakeTensor(conf)
        self.cls = _FakeTensor(cls)
        self.id = _FakeTensor(ids) if ids is not None else None


class _FakeResult:
    def __init__(
        self,
        masks_data: np.ndarray | None,
        boxes: _FakeBoxes | None,
        names: dict[int, str] | None = None,
    ) -> None:
        self.masks = _FakeMasks(masks_data) if masks_data is not None else None
        self.boxes = boxes
        self.names = names or {0: "fire", 1: "smoke"}


def _mask(h: int = 100, w: int = 100, x1: int = 10, y1: int = 20, x2: int = 60, y2: int = 80) -> np.ndarray:
    m = np.zeros((h, w), dtype=bool)
    m[y1:y2, x1:x2] = True
    return m


class TestUltralyticsResultConverter:
    def test_basic_two_instances_with_ids(self) -> None:
        masks = np.stack([_mask(x1=10, y1=20, x2=60, y2=80),
                          _mask(x1=70, y1=10, x2=95, y2=50)])
        boxes = _FakeBoxes(
            conf=np.array([0.9, 0.7]),
            cls=np.array([0, 1]),
            ids=np.array([42, 7]),
        )
        result = _FakeResult(masks_data=masks, boxes=boxes)

        out = ultralytics_result_to_instances(result)

        assert len(out) == 2
        assert {i.instance_id for i in out} == {42, 7}
        assert {i.label for i in out} == {"fire", "smoke"}
        assert all(i.area_px > 0 for i in out)
        assert all(i.bbox[0] < i.bbox[2] and i.bbox[1] < i.bbox[3] for i in out)

    def test_missing_ids_fallback_to_index(self) -> None:
        masks = np.stack([_mask()])
        boxes = _FakeBoxes(
            conf=np.array([0.8]),
            cls=np.array([0]),
            ids=None,
        )
        result = _FakeResult(masks_data=masks, boxes=boxes)

        out = ultralytics_result_to_instances(result)
        assert len(out) == 1
        assert out[0].instance_id == 0

    def test_no_masks_returns_empty(self) -> None:
        result = _FakeResult(masks_data=None, boxes=None)
        assert ultralytics_result_to_instances(result) == []

    def test_no_boxes_returns_empty(self) -> None:
        masks = np.stack([_mask()])
        result = _FakeResult(masks_data=masks, boxes=None)
        assert ultralytics_result_to_instances(result) == []

    def test_zero_masks_returns_empty(self) -> None:
        empty = np.zeros((0, 100, 100), dtype=bool)
        boxes = _FakeBoxes(
            conf=np.array([]),
            cls=np.array([], dtype=int),
            ids=None,
        )
        result = _FakeResult(masks_data=empty, boxes=boxes)
        assert ultralytics_result_to_instances(result) == []

    def test_empty_mask_pixels_skipped(self) -> None:
        # Two masks: one with content, one all-zeros -> only one instance returned
        masks = np.stack([_mask(), np.zeros((100, 100), dtype=bool)])
        boxes = _FakeBoxes(
            conf=np.array([0.9, 0.5]),
            cls=np.array([0, 1]),
            ids=np.array([1, 2]),
        )
        result = _FakeResult(masks_data=masks, boxes=boxes)
        out = ultralytics_result_to_instances(result)
        assert len(out) == 1
        assert out[0].instance_id == 1

    def test_unknown_class_id_falls_back_to_str(self) -> None:
        masks = np.stack([_mask()])
        boxes = _FakeBoxes(
            conf=np.array([0.9]),
            cls=np.array([99]),  # not in names dict
            ids=np.array([1]),
        )
        result = _FakeResult(masks_data=masks, boxes=boxes, names={0: "fire"})
        out = ultralytics_result_to_instances(result)
        assert out[0].label == "99"

    def test_centroid_inside_bbox(self) -> None:
        masks = np.stack([_mask(x1=10, y1=20, x2=60, y2=80)])
        boxes = _FakeBoxes(
            conf=np.array([0.9]),
            cls=np.array([0]),
            ids=np.array([1]),
        )
        result = _FakeResult(masks_data=masks, boxes=boxes)
        out = ultralytics_result_to_instances(result)
        cx, cy = out[0].centroid
        x1, y1, x2, y2 = out[0].bbox
        assert x1 <= cx <= x2
        assert y1 <= cy <= y2
