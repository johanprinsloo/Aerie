"""Tests for SegmentationRunner integration with ModelSink."""

from __future__ import annotations

import time

import numpy as np

from aerie_vision.frame_bus import FrameBus, FrameMeta
from aerie_vision.model_sink import ModelSink
from aerie_vision.segmentation.mock_segmenter import MockSegmenter
from aerie_vision.segmentation.runner import SegmentationRunner
from aerie_vision.segmentation.types import (
    InstanceMask,
    SegmentationResult,
    TextPrompts,
)


def _meta(n: int) -> FrameMeta:
    return FrameMeta(frame_number=n, timestamp=time.monotonic(), source_fps=30.0, width=320, height=240)


def _frame() -> np.ndarray:
    return np.zeros((240, 320, 3), dtype=np.uint8)


def _mask() -> np.ndarray:
    m = np.zeros((240, 320), dtype=bool)
    m[20:40, 30:60] = True
    return m


def _inst(instance_id: int = 1, label: str = "fire") -> InstanceMask:
    return InstanceMask(
        instance_id=instance_id, label=label, confidence=0.9,
        bbox=(30, 20, 60, 40), centroid=(45, 30), area_px=600, mask=_mask(),
    )


class TestSegmentationRunner:
    def test_processes_frames_and_calls_back(self) -> None:
        bus = FrameBus()
        slot = bus.create_slot("seg")
        sink = ModelSink(slot, max_fps=100.0)

        scripted = {0: [_inst(1, "fire")], 2: [_inst(2, "smoke")]}
        seg = MockSegmenter(scripted=scripted)

        results: list[SegmentationResult] = []
        runner = SegmentationRunner(
            model_sink=sink,
            segmenter=seg,
            prompts=TextPrompts(labels=("fire", "smoke")),
            on_result=results.append,
        )
        runner.start()

        for i in range(5):
            bus.publish(_frame(), _meta(i))
            time.sleep(0.05)

        time.sleep(0.3)
        runner.stop()

        assert runner.frames_processed >= 3
        labels = {inst.label for r in results for inst in r.instances}
        assert "fire" in labels

    def test_start_stop_lifecycle(self) -> None:
        bus = FrameBus()
        slot = bus.create_slot("seg")
        sink = ModelSink(slot, max_fps=100.0)
        seg = MockSegmenter()
        runner = SegmentationRunner(
            model_sink=sink, segmenter=seg, prompts=TextPrompts(labels=("x",)),
        )
        runner.start()
        assert runner.is_running
        time.sleep(0.1)
        runner.stop()
        assert not runner.is_running

    def test_stop_resets_segmenter(self) -> None:
        bus = FrameBus()
        slot = bus.create_slot("seg")
        sink = ModelSink(slot, max_fps=100.0)
        seg = MockSegmenter()
        runner = SegmentationRunner(
            model_sink=sink, segmenter=seg, prompts=TextPrompts(labels=("x",)),
        )
        runner.start()
        time.sleep(0.05)
        runner.stop()
        assert seg.reset_count == 1

    def test_stop_without_start(self) -> None:
        bus = FrameBus()
        slot = bus.create_slot("seg")
        sink = ModelSink(slot, max_fps=100.0)
        seg = MockSegmenter()
        runner = SegmentationRunner(
            model_sink=sink, segmenter=seg, prompts=TextPrompts(labels=("x",)),
        )
        runner.stop()  # should not raise
        # Reset is still called even if never started; that's fine.
        assert seg.reset_count == 1

    def test_callback_exception_does_not_crash(self) -> None:
        bus = FrameBus()
        slot = bus.create_slot("seg")
        sink = ModelSink(slot, max_fps=100.0)
        seg = MockSegmenter(scripted={0: [_inst()]})

        def bad(result: SegmentationResult) -> None:
            raise RuntimeError("boom")

        runner = SegmentationRunner(
            model_sink=sink, segmenter=seg,
            prompts=TextPrompts(labels=("fire",)), on_result=bad,
        )
        runner.start()
        bus.publish(_frame(), _meta(0))
        time.sleep(0.3)
        runner.stop()
        assert runner.frames_processed >= 1

    def test_avg_inference_ms(self) -> None:
        bus = FrameBus()
        slot = bus.create_slot("seg")
        sink = ModelSink(slot, max_fps=100.0)
        seg = MockSegmenter()
        runner = SegmentationRunner(
            model_sink=sink, segmenter=seg, prompts=TextPrompts(labels=("x",)),
        )
        runner.start()
        for i in range(5):
            bus.publish(_frame(), _meta(i))
            time.sleep(0.02)
        time.sleep(0.3)
        runner.stop()
        assert runner.frames_processed > 0
        assert runner.avg_inference_ms >= 0
