"""MockSegmenter: deterministic, scripted segmentation results for testing."""

from __future__ import annotations

import numpy as np

from .types import InstanceMask, Prompts


class MockSegmenter:
    """Returns pre-scripted instances keyed by frame number.

    Parameters
    ----------
    scripted:
        Mapping of ``frame_number -> list[InstanceMask]``. Frames not in the
        mapping produce an empty list.
    name:
        Identifier for this segmenter instance.
    """

    def __init__(
        self,
        scripted: dict[int, list[InstanceMask]] | None = None,
        name: str = "mock-seg",
    ) -> None:
        self._scripted = scripted or {}
        self._name = name
        self._reset_count = 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def reset_count(self) -> int:
        return self._reset_count

    def segment(
        self,
        frame: np.ndarray,
        prompts: Prompts,
        frame_number: int = 0,
        timestamp: float = 0.0,
    ) -> list[InstanceMask]:
        return list(self._scripted.get(frame_number, []))

    def warm_up(self) -> None:
        pass

    def reset(self) -> None:
        self._reset_count += 1
