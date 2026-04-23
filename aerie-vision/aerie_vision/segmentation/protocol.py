"""Segmenter protocol -- the contract every segmentation backend implements."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from .types import InstanceMask, Prompts


class Segmenter(Protocol):
    """Interface that all segmentation backends must satisfy."""

    @property
    def name(self) -> str:
        """Human-readable identifier for this segmenter instance."""
        ...

    def segment(
        self,
        frame: np.ndarray,
        prompts: Prompts,
        frame_number: int = 0,
        timestamp: float = 0.0,
    ) -> list[InstanceMask]:
        """Run inference on *frame* and return per-instance masks."""
        ...

    def warm_up(self) -> None:
        """Run a dummy inference to pre-load weights and caches."""
        ...

    def reset(self) -> None:
        """Clear any tracker state so the next call starts fresh."""
        ...
