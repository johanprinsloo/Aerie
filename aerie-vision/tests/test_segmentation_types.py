"""Tests for segmentation data types."""

from __future__ import annotations

import numpy as np
import pytest

from aerie_vision.segmentation.types import (
    InstanceMask,
    SegmentationResult,
    TextPrompts,
)


def _mask(h: int = 10, w: int = 10) -> np.ndarray:
    m = np.zeros((h, w), dtype=bool)
    m[2:6, 3:7] = True
    return m


class TestInstanceMask:
    def test_construct(self) -> None:
        inst = InstanceMask(
            instance_id=1,
            label="fire",
            confidence=0.9,
            bbox=(3, 2, 7, 6),
            centroid=(5, 4),
            area_px=16,
            mask=_mask(),
        )
        assert inst.instance_id == 1
        assert inst.label == "fire"
        assert inst.bbox == (3, 2, 7, 6)
        assert inst.area_px == 16

    def test_is_frozen(self) -> None:
        inst = InstanceMask(
            instance_id=1, label="fire", confidence=0.9,
            bbox=(0, 0, 1, 1), centroid=(0, 0), area_px=1, mask=_mask(),
        )
        with pytest.raises((AttributeError, TypeError)):
            inst.label = "smoke"  # type: ignore[misc]


class TestSegmentationResult:
    def test_construct(self) -> None:
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        inst = InstanceMask(
            instance_id=0, label="x", confidence=1.0,
            bbox=(0, 0, 1, 1), centroid=(0, 0), area_px=1, mask=_mask(),
        )
        result = SegmentationResult(
            frame_number=42,
            timestamp=1.5,
            instances=(inst,),
            frame=frame,
            inference_ms=12.3,
        )
        assert result.frame_number == 42
        assert len(result.instances) == 1
        assert result.inference_ms == 12.3


class TestTextPrompts:
    def test_construct(self) -> None:
        p = TextPrompts(labels=("fire", "smoke"))
        assert p.labels == ("fire", "smoke")

    def test_empty(self) -> None:
        p = TextPrompts(labels=())
        assert p.labels == ()
