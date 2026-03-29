"""ModelSink: rate-limited frame consumer for the detection model."""

from __future__ import annotations

import time

import numpy as np

from .frame_bus import FrameMeta, FrameSlot


class ModelSink:
    """Delivers frames to the detection model at a bounded rate.

    Each call to :meth:`get_frame` blocks until enough time has elapsed
    since the previous call (honouring *max_fps*), then returns the
    latest frame from the underlying :class:`FrameSlot`.

    Parameters
    ----------
    slot:
        The :class:`FrameSlot` to read from.
    max_fps:
        Upper bound on the consumption rate.  The actual rate will be
        ``min(max_fps, source_fps, 1/inference_time)`` since the caller
        also takes time between calls.
    """

    def __init__(self, slot: FrameSlot, max_fps: float = 5.0) -> None:
        self._slot = slot
        self._interval = 1.0 / max_fps
        self._max_fps = max_fps
        self._last_read = 0.0

    @property
    def max_fps(self) -> float:
        return self._max_fps

    @max_fps.setter
    def max_fps(self, value: float) -> None:
        if value <= 0:
            raise ValueError("max_fps must be positive")
        self._max_fps = value
        self._interval = 1.0 / value

    def get_frame(self, timeout: float = 5.0) -> tuple[np.ndarray, FrameMeta]:
        """Return ``(frame, meta)`` respecting the configured rate limit.

        Blocks until both:
        1. At least ``1 / max_fps`` seconds have elapsed since the last call.
        2. A frame is available in the slot.

        Raises ``TimeoutError`` if no frame arrives within *timeout* seconds.
        """
        now = time.monotonic()
        wait = self._interval - (now - self._last_read)
        if wait > 0:
            time.sleep(wait)

        frame, meta = self._slot.get(timeout=timeout)
        self._last_read = time.monotonic()
        return frame, meta
