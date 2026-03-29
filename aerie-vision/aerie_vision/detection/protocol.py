"""Detector protocol -- the contract every model backend implements."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from .types import RawDetection


class Detector(Protocol):
    """Interface that all detection backends must satisfy."""

    @property
    def name(self) -> str:
        """Human-readable identifier for this detector instance."""
        ...

    def detect(self, frame: np.ndarray, frame_number: int = 0, timestamp: float = 0.0) -> list[RawDetection]:
        """Run inference on *frame* and return detections."""
        ...

    def warm_up(self) -> None:
        """Run a dummy inference to pre-load weights / JIT-compile / allocate memory."""
        ...
