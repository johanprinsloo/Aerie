"""Tests for MockSegmenter."""

from __future__ import annotations

import numpy as np

from aerie_vision.segmentation.mock_segmenter import MockSegmenter
from aerie_vision.segmentation.types import InstanceMask, TextPrompts


def _frame() -> np.ndarray:
    return np.zeros((240, 320, 3), dtype=np.uint8)


def _mask() -> np.ndarray:
    m = np.zeros((240, 320), dtype=bool)
    m[50:100, 80:160] = True
    return m


def _inst(instance_id: int = 1, label: str = "fire") -> InstanceMask:
    return InstanceMask(
        instance_id=instance_id,
        label=label,
        confidence=0.95,
        bbox=(80, 50, 160, 100),
        centroid=(120, 75),
        area_px=4000,
        mask=_mask(),
    )


class TestMockSegmenter:
    def test_scripted_frame_returns_instances(self) -> None:
        scripted = {3: [_inst(1, "fire"), _inst(2, "smoke")]}
        seg = MockSegmenter(scripted=scripted)
        results = seg.segment(_frame(), TextPrompts(labels=("fire", "smoke")), frame_number=3)
        assert len(results) == 2
        assert {r.label for r in results} == {"fire", "smoke"}

    def test_unscripted_frame_returns_empty(self) -> None:
        seg = MockSegmenter(scripted={3: [_inst()]})
        assert seg.segment(_frame(), TextPrompts(labels=("fire",)), frame_number=99) == []

    def test_no_script_returns_empty(self) -> None:
        seg = MockSegmenter()
        assert seg.segment(_frame(), TextPrompts(labels=("fire",)), frame_number=0) == []

    def test_name(self) -> None:
        assert MockSegmenter(name="custom").name == "custom"

    def test_warm_up_is_noop(self) -> None:
        MockSegmenter().warm_up()

    def test_reset_increments_count(self) -> None:
        seg = MockSegmenter()
        assert seg.reset_count == 0
        seg.reset()
        seg.reset()
        assert seg.reset_count == 2

    def test_returns_copy(self) -> None:
        original = [_inst()]
        seg = MockSegmenter(scripted={5: original})
        results = seg.segment(_frame(), TextPrompts(labels=("fire",)), frame_number=5)
        results.clear()
        assert len(seg.segment(_frame(), TextPrompts(labels=("fire",)), frame_number=5)) == 1
