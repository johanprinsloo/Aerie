"""Core data types for the detection layer."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class RawDetection:
    """A single detection produced by a single model on a single frame."""

    label: str
    confidence: float
    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2) pixel coords
    model_name: str
    frame_number: int
    timestamp: float  # frame capture time (from FrameMeta)


@dataclass(frozen=True, slots=True)
class DetectionResult:
    """All detections for one frame, possibly merged from multiple models."""

    frame_number: int
    timestamp: float
    detections: tuple[RawDetection, ...]
    frame: np.ndarray
    inference_ms: float
