"""Integration tests for VisionPipeline orchestrator."""

from __future__ import annotations

import json
import pathlib
import time
import urllib.request

import pytest

from aerie_vision.config import PipelineConfig
from aerie_vision.detection.mock_detector import MockDetector
from aerie_vision.detection.types import RawDetection
from aerie_vision.orchestrator import VisionPipeline


def _det(label: str = "fire", frame_number: int = 0) -> RawDetection:
    return RawDetection(
        label=label, confidence=0.92, bbox=(50, 50, 200, 200),
        model_name="mock", frame_number=frame_number, timestamp=0.0,
    )


class TestVisionPipelineIngestOnly:
    """When detector_model is empty, the orchestrator runs ingest-only."""

    def test_starts_and_stops(self, test_video_path: pathlib.Path) -> None:
        config = PipelineConfig(
            source=str(test_video_path),
            viewer_enabled=False,
            detector_model="",
        )
        vp = VisionPipeline(config)
        vp.start()
        time.sleep(0.5)
        vp.stop()
        assert vp.runner is None

    def test_raw_viewer_works(self, test_video_path: pathlib.Path) -> None:
        config = PipelineConfig(
            source=str(test_video_path),
            viewer_enabled=True,
            viewer_host="127.0.0.1",
            viewer_port=18095,
            detector_model="",
        )
        vp = VisionPipeline(config)
        vp.start()
        try:
            resp = urllib.request.urlopen("http://127.0.0.1:18095/", timeout=3)
            assert resp.status == 200
        finally:
            vp.stop()


class TestVisionPipelineWithDetection:
    """Detection enabled via mock detector injected manually."""

    def _make_pipeline(
        self,
        test_video_path: pathlib.Path,
        *,
        jsonl_path: str = "",
        record_path: str = "",
        annotated_port: int = 18096,
        console: bool = False,
    ) -> VisionPipeline:
        config = PipelineConfig(
            source=str(test_video_path),
            viewer_enabled=False,
            detector_model="mock",
            model_max_fps=20.0,
            annotated_viewer_enabled=True,
            annotated_viewer_port=annotated_port,
            viewer_host="127.0.0.1",
            jsonl_output=jsonl_path,
            console_output=console,
            record_path=record_path,
            loop_video_file=True,
        )
        vp = VisionPipeline(config)
        return vp

    def test_annotated_viewer_reachable(self, test_video_path: pathlib.Path) -> None:
        vp = self._make_pipeline(test_video_path, annotated_port=18096)
        vp.start()
        try:
            time.sleep(1.5)
            resp = urllib.request.urlopen("http://127.0.0.1:18096/", timeout=3)
            assert resp.status == 200
        finally:
            vp.stop()

    def test_jsonl_output_written(self, test_video_path: pathlib.Path, tmp_path) -> None:
        jsonl_path = str(tmp_path / "detections.jsonl")
        vp = self._make_pipeline(test_video_path, jsonl_path=jsonl_path, annotated_port=18097)
        vp.start()
        time.sleep(2.0)
        vp.stop()

        assert pathlib.Path(jsonl_path).exists()
        with open(jsonl_path) as f:
            lines = f.readlines()
        assert len(lines) > 0
        record = json.loads(lines[0])
        assert "frame_number" in record
        assert "detections" in record
        assert "inference_ms" in record

    def test_video_recording(self, test_video_path: pathlib.Path, tmp_path) -> None:
        import cv2

        record_path = str(tmp_path / "recorded.mp4")
        vp = self._make_pipeline(test_video_path, record_path=record_path, annotated_port=18098)
        vp.start()
        time.sleep(2.0)
        vp.stop()

        cap = cv2.VideoCapture(record_path)
        assert cap.isOpened()
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        assert frame_count > 0
        cap.release()

    def test_stop_is_clean(self, test_video_path: pathlib.Path) -> None:
        vp = self._make_pipeline(test_video_path, annotated_port=18099)
        vp.start()
        time.sleep(0.5)
        vp.stop()
        # Second stop should not raise
        vp.stop()
