"""Core data types for the segmentation layer."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class InstanceMask:
    """A single segmented instance produced by a segmenter on a single frame.

    The ``mask`` field is a per-frame boolean array; it is intentionally
    excluded from JSONL serialization (see :class:`text_output.JsonlOutputStream`).
    """

    instance_id: int
    label: str
    confidence: float
    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2)
    centroid: tuple[int, int]  # (cx, cy)
    area_px: int
    mask: np.ndarray  # bool, shape (H, W)


@dataclass(frozen=True, slots=True)
class SegmentationResult:
    """All instances segmented for one frame."""

    frame_number: int
    timestamp: float
    instances: tuple[InstanceMask, ...]
    frame: np.ndarray
    inference_ms: float


@dataclass(frozen=True, slots=True)
class TextPrompts:
    """Open-vocabulary text prompts (e.g. ``("fire", "smoke")``)."""

    labels: tuple[str, ...]


# Future extension point: BoxPrompts, ClickPrompts, etc.
Prompts = TextPrompts
