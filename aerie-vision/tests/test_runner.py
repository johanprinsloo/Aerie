"""Tests for DetectionRunner integration with ModelSink."""

from __future__ import annotations

import threading
import time

import numpy as np

from aerie_vision.detection.mock_detector import MockDetector
from aerie_vision.detection.router import DetectionRouter
from aerie_vision.detection.runner import DetectionRunner
from aerie_vision.detection.types import DetectionResult, RawDetection
from aerie_vision.frame_bus import FrameBus, FrameMeta
from aerie_vision.model_sink import ModelSink


def _make_meta(n: int) -> FrameMeta:
    return FrameMeta(frame_number=n, timestamp=time.monotonic(), source_fps=30.0, width=320, height=240)


def _make_frame() -> np.ndarray:
    return np.zeros((240, 320, 3), dtype=np.uint8)


def _det(label: str = "fire", frame_number: int = 0) -> RawDetection:
    return RawDetection(
        label=label, confidence=0.9, bbox=(10, 20, 100, 200),
        model_name="mock", frame_number=frame_number, timestamp=0.0,
    )


class TestDetectionRunner:
    def test_processes_frames_and_calls_back(self) -> None:
        bus = FrameBus()
        slot = bus.create_slot("model")
        sink = ModelSink(slot, max_fps=100.0)

        scripted = {0: [_det("fire", 0)], 2: [_det("smoke", 2)]}
        detector = MockDetector(scripted=scripted, name="mock")
        router = DetectionRouter(primary=detector)

        results: list[DetectionResult] = []
        runner = DetectionRunner(model_sink=sink, router=router, on_result=results.append)
        runner.start()

        for i in range(5):
            bus.publish(_make_frame(), _make_meta(i))
            time.sleep(0.05)

        time.sleep(0.3)
        runner.stop()

        assert runner.frames_processed >= 3
        detected_labels = {d.label for r in results for d in r.detections}
        assert "fire" in detected_labels

    def test_start_stop_lifecycle(self) -> None:
        bus = FrameBus()
        slot = bus.create_slot("model")
        sink = ModelSink(slot, max_fps=100.0)
        detector = MockDetector(name="mock")
        router = DetectionRouter(primary=detector)
        runner = DetectionRunner(model_sink=sink, router=router)

        runner.start()
        assert runner.is_running
        time.sleep(0.1)
        runner.stop()
        assert not runner.is_running

    def test_stop_without_start(self) -> None:
        bus = FrameBus()
        slot = bus.create_slot("model")
        sink = ModelSink(slot, max_fps=100.0)
        detector = MockDetector(name="mock")
        router = DetectionRouter(primary=detector)
        runner = DetectionRunner(model_sink=sink, router=router)
        runner.stop()  # should not raise

    def test_callback_exception_does_not_crash(self) -> None:
        bus = FrameBus()
        slot = bus.create_slot("model")
        sink = ModelSink(slot, max_fps=100.0)
        detector = MockDetector(scripted={0: [_det()]}, name="mock")
        router = DetectionRouter(primary=detector)

        def bad_callback(result: DetectionResult) -> None:
            raise RuntimeError("boom")

        runner = DetectionRunner(model_sink=sink, router=router, on_result=bad_callback)
        runner.start()

        bus.publish(_make_frame(), _make_meta(0))
        time.sleep(0.3)
        runner.stop()
        assert runner.frames_processed >= 1

    def test_avg_inference_ms(self) -> None:
        bus = FrameBus()
        slot = bus.create_slot("model")
        sink = ModelSink(slot, max_fps=100.0)
        detector = MockDetector(name="mock")
        router = DetectionRouter(primary=detector)
        runner = DetectionRunner(model_sink=sink, router=router)
        runner.start()

        for i in range(5):
            bus.publish(_make_frame(), _make_meta(i))
            time.sleep(0.02)

        time.sleep(0.3)
        runner.stop()

        assert runner.avg_inference_ms >= 0
        assert runner.frames_processed > 0
