"""Segmentation annotation: overlay translucent masks + labels onto frames."""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np

from .segmentation.types import InstanceMask

# Stable per-instance palette, indexed by instance_id.
_PALETTE: list[tuple[int, int, int]] = [
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


def _color_for_instance(instance_id: int) -> tuple[int, int, int]:
    return _PALETTE[instance_id % len(_PALETTE)]


def annotate_segmentation(
    frame: np.ndarray,
    instances: Sequence[InstanceMask],
    *,
    mask_alpha: float = 0.45,
    font_scale: float = 0.6,
    thickness: int = 2,
) -> np.ndarray:
    """Return a copy of *frame* with mask overlays and per-instance labels."""
    if not instances:
        return frame.copy()

    out = frame.copy()
    h, w = out.shape[:2]

    overlay = out.copy()
    for inst in instances:
        if inst.mask.shape[:2] != (h, w):
            continue
        color = _color_for_instance(inst.instance_id)
        overlay[inst.mask] = color

    cv2.addWeighted(overlay, mask_alpha, out, 1.0 - mask_alpha, 0.0, out)

    for inst in instances:
        color = _color_for_instance(inst.instance_id)
        x1, y1, x2, y2 = inst.bbox
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)

        tag = f"#{inst.instance_id} {inst.label} {inst.confidence:.0%}"
        (tw, th), baseline = cv2.getTextSize(
            tag, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        cv2.rectangle(
            out, (x1, y1 - th - baseline - 4), (x1 + tw + 4, y1), color, cv2.FILLED
        )
        cv2.putText(
            out,
            tag,
            (x1 + 2, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            thickness,
        )

    return out
