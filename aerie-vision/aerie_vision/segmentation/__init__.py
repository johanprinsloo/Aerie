"""Segmentation layer: SAM-style mask + tracking models.

Two execution shapes coexist:

- Per-frame :class:`Segmenter` protocol consumed by :class:`SegmentationRunner`
  (used by :class:`MockSegmenter` for tests).
- Source-driven :class:`Sam3SourceRunner` for real Ultralytics SAM 3, which
  owns ingest itself via ``model.track(source=..., stream=True)``.
"""

from .protocol import Segmenter
from .types import InstanceMask, Prompts, SegmentationResult, TextPrompts

__all__ = [
    "InstanceMask",
    "Prompts",
    "SegmentationResult",
    "Segmenter",
    "TextPrompts",
]
