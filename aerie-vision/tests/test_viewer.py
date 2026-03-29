"""Tests for ViewerSink (MJPEG-over-HTTP)."""

from __future__ import annotations

import threading
import time
import urllib.request

import cv2
import numpy as np
import pytest

from aerie_vision.frame_bus import FrameMeta, FrameSlot
from aerie_vision.viewer import ViewerSink


def _make_meta(n: int = 0) -> FrameMeta:
    return FrameMeta(frame_number=n, timestamp=time.monotonic(), source_fps=30.0, width=320, height=240)


def _make_frame(value: int = 128) -> np.ndarray:
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    frame[:] = value
    return frame


class TestViewerSink:
    def test_index_page(self) -> None:
        slot = FrameSlot("viewer")
        viewer = ViewerSink(slot, host="127.0.0.1", port=0)

        # Port 0 → OS picks a free port; we need the real port after bind.
        # ViewerSink currently requires a fixed port, so pick a high one.
        viewer = ViewerSink(slot, host="127.0.0.1", port=18091)
        viewer.start()
        try:
            resp = urllib.request.urlopen("http://127.0.0.1:18091/", timeout=3)
            body = resp.read()
            assert b"<img" in body
            assert b"/stream" in body
            assert resp.status == 200
        finally:
            viewer.stop()

    def test_mjpeg_stream_returns_jpeg_frames(self) -> None:
        slot = FrameSlot("viewer")
        viewer = ViewerSink(slot, host="127.0.0.1", port=18092)
        viewer.start()

        # Pump frames in a background thread
        stop = threading.Event()

        def producer() -> None:
            n = 0
            while not stop.is_set():
                slot.put(_make_frame(n % 256), _make_meta(n))
                n += 1
                time.sleep(0.03)

        t = threading.Thread(target=producer, daemon=True)
        t.start()

        try:
            resp = urllib.request.urlopen("http://127.0.0.1:18092/stream", timeout=5)
            content_type = resp.headers.get("Content-Type", "")
            assert "multipart/x-mixed-replace" in content_type

            # Read enough bytes to capture at least one JPEG frame
            data = resp.read(64 * 1024)
            assert b"Content-Type: image/jpeg" in data
            # JPEG magic bytes
            assert b"\xff\xd8" in data
        finally:
            stop.set()
            t.join(timeout=2.0)
            viewer.stop()

    def test_404_for_unknown_path(self) -> None:
        slot = FrameSlot("viewer")
        viewer = ViewerSink(slot, host="127.0.0.1", port=18093)
        viewer.start()
        try:
            with pytest.raises(urllib.error.HTTPError) as exc_info:
                urllib.request.urlopen("http://127.0.0.1:18093/nope", timeout=3)
            assert exc_info.value.code == 404
        finally:
            viewer.stop()
