"""Thread-safe frame distribution: FrameMeta, FrameSlot, and FrameBus.

The FrameBus publishes captured frames to one or more FrameSlots.  Each slot
holds only the *latest* frame so that slow consumers never cause backpressure
or unbounded memory growth.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class FrameMeta:
    """Metadata attached to every captured frame."""

    frame_number: int
    timestamp: float  # time.monotonic() at capture
    source_fps: float
    width: int
    height: int


class FrameSlot:
    """A single-item, overwrite-on-put slot for the most recent frame.

    Writers call :meth:`put` (non-blocking).  Readers call :meth:`get`
    (blocking) or :meth:`get_latest_nonblocking`.
    """

    def __init__(self, name: str = "") -> None:
        self.name = name
        self._lock = threading.Lock()
        self._event = threading.Event()
        self._frame: np.ndarray | None = None
        self._meta: FrameMeta | None = None

    def put(self, frame: np.ndarray, meta: FrameMeta) -> None:
        """Store *frame* and *meta*, waking any blocked reader."""
        with self._lock:
            self._frame = frame
            self._meta = meta
        self._event.set()

    def get(self, timeout: float | None = None) -> tuple[np.ndarray, FrameMeta]:
        """Block until a frame is available, then return ``(frame, meta)``.

        Raises ``TimeoutError`` if *timeout* seconds elapse with no frame.
        """
        if not self._event.wait(timeout=timeout):
            raise TimeoutError(f"FrameSlot({self.name!r}): no frame within {timeout}s")
        with self._lock:
            self._event.clear()
            assert self._frame is not None and self._meta is not None
            return self._frame.copy(), self._meta

    def get_latest_nonblocking(self) -> tuple[np.ndarray | None, FrameMeta | None]:
        """Return the most recent ``(frame, meta)`` without blocking.

        Returns ``(None, None)`` if no frame has been published yet.
        """
        with self._lock:
            if self._frame is None:
                return None, None
            return self._frame.copy(), self._meta


class FrameBus:
    """Fan-out distributor: publishes each frame to every registered slot."""

    def __init__(self) -> None:
        self._slots: list[FrameSlot] = []
        self._lock = threading.Lock()

    def create_slot(self, name: str = "") -> FrameSlot:
        """Create and register a new :class:`FrameSlot`."""
        slot = FrameSlot(name=name)
        with self._lock:
            self._slots.append(slot)
        return slot

    def publish(self, frame: np.ndarray, meta: FrameMeta) -> None:
        """Push *frame* and *meta* into every registered slot."""
        with self._lock:
            slots = list(self._slots)
        for slot in slots:
            slot.put(frame, meta)
