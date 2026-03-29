"""MockDetector: deterministic, scripted detections for testing."""

from __future__ import annotations

import numpy as np

from .types import RawDetection


class MockDetector:
    """Returns pre-scripted detections keyed by frame number.

    Parameters
    ----------
    scripted:
        Mapping of ``frame_number -> list[RawDetection]``.  Frames not in
        the mapping produce an empty detection list.
    name:
        Identifier for this detector instance.
    """

    def __init__(
        self,
        scripted: dict[int, list[RawDetection]] | None = None,
        name: str = "mock",
    ) -> None:
        self._scripted = scripted or {}
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def detect(
        self, frame: np.ndarray, frame_number: int = 0, timestamp: float = 0.0
    ) -> list[RawDetection]:
        return list(self._scripted.get(frame_number, []))

    def warm_up(self) -> None:
        pass
