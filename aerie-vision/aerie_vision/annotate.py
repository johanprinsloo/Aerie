"""Frame annotation: draw bounding boxes and labels onto video frames."""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np

from .detection.types import RawDetection

# Distinct colours per label, derived from a stable hash so the same label
# always gets the same colour across frames.
_PALETTE = [
    (0, 255, 0),
    (0, 0, 255),
    (255, 0, 0),
    (0, 255, 255),
    (255, 0, 255),
    (255, 255, 0),
    (128, 0, 255),
    (0, 128, 255),
    (255, 128, 0),
    (0, 255, 128),
]


def _color_for_label(label: str) -> tuple[int, int, int]:
    return _PALETTE[hash(label) % len(_PALETTE)]


def annotate_frame(
    frame: np.ndarray,
    detections: Sequence[RawDetection],
    *,
    font_scale: float = 0.6,
    thickness: int = 2,
) -> np.ndarray:
    """Return a copy of *frame* with bounding boxes and labels drawn."""
    if not detections:
        return frame.copy()

    out = frame.copy()
    for det in detections:
        color = _color_for_label(det.label)
        x1, y1, x2, y2 = det.bbox
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)

        tag = f"{det.label} {det.confidence:.0%}"
        (tw, th), baseline = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        # Background rectangle behind the text
        cv2.rectangle(out, (x1, y1 - th - baseline - 4), (x1 + tw + 4, y1), color, cv2.FILLED)
        cv2.putText(
            out, tag, (x1 + 2, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness,
        )

    return out
