"""Cross-model IoU-based NMS for merging detections from multiple models."""

from __future__ import annotations

from .types import RawDetection


def iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    """Compute Intersection-over-Union for two ``(x1, y1, x2, y2)`` boxes."""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0

    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def merge_detections(
    detections: list[RawDetection],
    iou_threshold: float = 0.5,
) -> list[RawDetection]:
    """Merge detections from potentially multiple models.

    When two detections of the **same label** overlap above *iou_threshold*,
    keep only the one with higher confidence.  Detections of different labels
    are never suppressed against each other.
    """
    if len(detections) <= 1:
        return list(detections)

    # Sort by confidence descending so we greedily keep the strongest first.
    ranked = sorted(detections, key=lambda d: d.confidence, reverse=True)
    kept: list[RawDetection] = []

    for det in ranked:
        suppressed = False
        for existing in kept:
            if det.label == existing.label and iou(det.bbox, existing.bbox) >= iou_threshold:
                suppressed = True
                break
        if not suppressed:
            kept.append(det)

    return kept
