"""Tests for FrameMeta, FrameSlot, and FrameBus."""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from aerie_vision.frame_bus import FrameBus, FrameMeta, FrameSlot


def _make_meta(n: int = 0) -> FrameMeta:
    return FrameMeta(frame_number=n, timestamp=time.monotonic(), source_fps=30.0, width=320, height=240)


def _make_frame(value: int = 0) -> np.ndarray:
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    frame[:, :, 0] = value
    return frame


class TestFrameSlot:
    def test_put_and_get(self) -> None:
        slot = FrameSlot("test")
        frame = _make_frame(42)
        meta = _make_meta()
        slot.put(frame, meta)

        out_frame, out_meta = slot.get(timeout=1.0)
        assert out_meta.frame_number == 0
        assert out_frame[0, 0, 0] == 42

    def test_get_returns_copy(self) -> None:
        slot = FrameSlot()
        frame = _make_frame(10)
        slot.put(frame, _make_meta())

        out1, _ = slot.get(timeout=1.0)
        frame[:, :, 0] = 99
        assert out1[0, 0, 0] == 10, "get() must return a copy, not a reference"

    def test_get_timeout(self) -> None:
        slot = FrameSlot("empty")
        with pytest.raises(TimeoutError):
            slot.get(timeout=0.05)

    def test_get_latest_nonblocking_empty(self) -> None:
        slot = FrameSlot()
        frame, meta = slot.get_latest_nonblocking()
        assert frame is None
        assert meta is None

    def test_get_latest_nonblocking_has_frame(self) -> None:
        slot = FrameSlot()
        slot.put(_make_frame(77), _make_meta(5))

        frame, meta = slot.get_latest_nonblocking()
        assert frame is not None
        assert meta is not None
        assert meta.frame_number == 5
        assert frame[0, 0, 0] == 77

    def test_overwrite_semantics(self) -> None:
        slot = FrameSlot()
        slot.put(_make_frame(1), _make_meta(1))
        slot.put(_make_frame(2), _make_meta(2))

        frame, meta = slot.get(timeout=1.0)
        assert meta.frame_number == 2
        assert frame[0, 0, 0] == 2

    def test_concurrent_put_get(self) -> None:
        slot = FrameSlot()
        results: list[int] = []

        def writer() -> None:
            for i in range(50):
                slot.put(_make_frame(i), _make_meta(i))
                time.sleep(0.001)

        def reader() -> None:
            for _ in range(20):
                frame, meta = slot.get(timeout=2.0)
                results.append(meta.frame_number)

        t_w = threading.Thread(target=writer)
        t_r = threading.Thread(target=reader)
        t_w.start()
        t_r.start()
        t_w.join()
        t_r.join()

        assert len(results) == 20
        # frame numbers should be non-decreasing (latest-frame semantics)
        for a, b in zip(results, results[1:]):
            assert b >= a


class TestFrameBus:
    def test_single_slot_receives(self) -> None:
        bus = FrameBus()
        slot = bus.create_slot("a")
        bus.publish(_make_frame(11), _make_meta(0))

        frame, meta = slot.get(timeout=1.0)
        assert frame[0, 0, 0] == 11

    def test_multiple_slots_receive_same_frame(self) -> None:
        bus = FrameBus()
        s1 = bus.create_slot("s1")
        s2 = bus.create_slot("s2")
        bus.publish(_make_frame(22), _make_meta(7))

        f1, m1 = s1.get(timeout=1.0)
        f2, m2 = s2.get(timeout=1.0)

        assert m1.frame_number == m2.frame_number == 7
        np.testing.assert_array_equal(f1, f2)

    def test_slow_consumer_gets_latest(self) -> None:
        bus = FrameBus()
        slot = bus.create_slot("slow")

        for i in range(10):
            bus.publish(_make_frame(i), _make_meta(i))

        frame, meta = slot.get(timeout=1.0)
        assert meta.frame_number == 9
