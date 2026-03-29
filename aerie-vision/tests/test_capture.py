"""Tests for FrameGrabber."""

from __future__ import annotations

import pathlib
import time

import pytest

from aerie_vision.capture import FrameGrabber
from aerie_vision.frame_bus import FrameBus


class TestFrameGrabber:
    def test_reads_video_file(self, test_video_path: pathlib.Path) -> None:
        bus = FrameBus()
        slot = bus.create_slot("test")
        grabber = FrameGrabber(str(test_video_path), bus, loop=False)

        grabber.start()
        frame, meta = slot.get(timeout=5.0)

        assert frame.shape == (240, 320, 3)
        assert meta.frame_number == 0
        assert meta.source_fps > 0

        grabber.stop()

    def test_reports_source_fps(self, test_video_path: pathlib.Path) -> None:
        bus = FrameBus()
        grabber = FrameGrabber(str(test_video_path), bus, loop=False)
        grabber.start()
        time.sleep(0.3)
        assert grabber.source_fps > 0
        grabber.stop()

    def test_stops_cleanly(self, test_video_path: pathlib.Path) -> None:
        bus = FrameBus()
        grabber = FrameGrabber(str(test_video_path), bus, loop=True)
        grabber.start()
        time.sleep(0.3)
        assert grabber.is_running
        grabber.stop(timeout=3.0)
        assert not grabber.is_running

    def test_nonexistent_source(self, tmp_path: pathlib.Path) -> None:
        bus = FrameBus()
        slot = bus.create_slot("test")
        grabber = FrameGrabber(str(tmp_path / "nope.mp4"), bus, loop=False, reconnect_delay=0.1)
        grabber.start()
        time.sleep(0.5)
        # Should not have produced frames
        frame, meta = slot.get_latest_nonblocking()
        assert frame is None
        grabber.stop()

    def test_loop_replays(self, test_video_path: pathlib.Path) -> None:
        bus = FrameBus()
        slot = bus.create_slot("test")
        grabber = FrameGrabber(str(test_video_path), bus, loop=True)
        grabber.start()

        # The test video has 90 frames.  Wait long enough for at least one
        # full loop (90 frames at ~30fps = 3s) plus some extra.
        time.sleep(4.5)
        grabber.stop()

        _, meta = slot.get_latest_nonblocking()
        assert meta is not None
        assert meta.frame_number > 90, "Expected more frames than the file contains (looping)"

    def test_no_loop_stops(self, test_video_path: pathlib.Path) -> None:
        bus = FrameBus()
        slot = bus.create_slot("test")
        grabber = FrameGrabber(str(test_video_path), bus, loop=False)
        grabber.start()
        time.sleep(5.0)

        _, meta = slot.get_latest_nonblocking()
        assert meta is not None
        # Without looping, frame count should be roughly the file's frame count.
        assert meta.frame_number < 100
        assert not grabber.is_running

    def test_string_device_index(self) -> None:
        """Verify that a string like "0" is interpreted as a device index."""
        bus = FrameBus()
        grabber = FrameGrabber("0", bus, loop=False, reconnect_delay=0.05)
        # We just verify it doesn't crash; the device may or may not exist in CI.
        grabber.start()
        time.sleep(0.2)
        grabber.stop()
