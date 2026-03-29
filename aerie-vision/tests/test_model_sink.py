"""Tests for ModelSink."""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from aerie_vision.frame_bus import FrameMeta, FrameSlot
from aerie_vision.model_sink import ModelSink


def _make_meta(n: int = 0) -> FrameMeta:
    return FrameMeta(frame_number=n, timestamp=time.monotonic(), source_fps=30.0, width=320, height=240)


def _make_frame() -> np.ndarray:
    return np.zeros((240, 320, 3), dtype=np.uint8)


class TestModelSink:
    def test_basic_get_frame(self) -> None:
        slot = FrameSlot("model")
        sink = ModelSink(slot, max_fps=100.0)

        slot.put(_make_frame(), _make_meta(0))
        frame, meta = sink.get_frame(timeout=2.0)
        assert frame.shape == (240, 320, 3)
        assert meta.frame_number == 0

    def test_rate_limiting(self) -> None:
        slot = FrameSlot("model")
        sink = ModelSink(slot, max_fps=10.0)

        # Feed frames fast
        stop = threading.Event()
        counter = 0

        def producer() -> None:
            nonlocal counter
            while not stop.is_set():
                slot.put(_make_frame(), _make_meta(counter))
                counter += 1
                time.sleep(0.005)

        t = threading.Thread(target=producer, daemon=True)
        t.start()

        consumed = 0
        t0 = time.monotonic()
        while time.monotonic() - t0 < 1.0:
            sink.get_frame(timeout=2.0)
            consumed += 1

        stop.set()
        t.join(timeout=2.0)

        # At max_fps=10, we expect ~10 frames in 1 second.  Allow generous
        # tolerance for CI jitter.
        assert 5 <= consumed <= 18, f"Expected ~10 frames/s, got {consumed}"

    def test_always_latest_frame(self) -> None:
        slot = FrameSlot("model")
        sink = ModelSink(slot, max_fps=100.0)

        for i in range(20):
            slot.put(_make_frame(), _make_meta(i))

        frame, meta = sink.get_frame(timeout=2.0)
        assert meta.frame_number == 19

    def test_max_fps_setter(self) -> None:
        slot = FrameSlot("model")
        sink = ModelSink(slot, max_fps=5.0)
        assert sink.max_fps == 5.0

        sink.max_fps = 10.0
        assert sink.max_fps == 10.0

    def test_max_fps_setter_rejects_zero(self) -> None:
        slot = FrameSlot("model")
        sink = ModelSink(slot, max_fps=5.0)
        with pytest.raises(ValueError):
            sink.max_fps = 0

    def test_timeout_when_no_frames(self) -> None:
        slot = FrameSlot("model")
        sink = ModelSink(slot, max_fps=5.0)
        with pytest.raises(TimeoutError):
            sink.get_frame(timeout=0.1)
