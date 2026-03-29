"""End-to-end tests for the Pipeline orchestrator."""

from __future__ import annotations

import pathlib
import time
import urllib.request

import pytest

from aerie_vision.config import PipelineConfig
from aerie_vision.pipeline import Pipeline


class TestPipeline:
    def test_full_pipeline_with_video(self, test_video_path: pathlib.Path) -> None:
        config = PipelineConfig(
            source=str(test_video_path),
            viewer_enabled=True,
            viewer_host="127.0.0.1",
            viewer_port=18094,
            model_max_fps=10.0,
            loop_video_file=True,
        )
        pipeline = Pipeline(config)
        pipeline.start()

        try:
            # Model sink should deliver frames
            frame, meta = pipeline.model_sink.get_frame(timeout=5.0)
            assert frame.shape[2] == 3
            assert meta.frame_number >= 0

            # Viewer should be reachable
            resp = urllib.request.urlopen("http://127.0.0.1:18094/", timeout=3)
            assert resp.status == 200
        finally:
            pipeline.stop()

    def test_pipeline_without_viewer(self, test_video_path: pathlib.Path) -> None:
        config = PipelineConfig(
            source=str(test_video_path),
            viewer_enabled=False,
            model_max_fps=10.0,
            loop_video_file=True,
        )
        pipeline = Pipeline(config)
        pipeline.start()

        try:
            frame, meta = pipeline.model_sink.get_frame(timeout=5.0)
            assert frame is not None
        finally:
            pipeline.stop()

    def test_pipeline_start_stop_idempotent(self, test_video_path: pathlib.Path) -> None:
        config = PipelineConfig(
            source=str(test_video_path),
            viewer_enabled=False,
            loop_video_file=False,
        )
        pipeline = Pipeline(config)
        pipeline.start()
        time.sleep(0.5)
        pipeline.stop()
        # second stop should not raise
        pipeline.stop()

    def test_model_sink_rate(self, test_video_path: pathlib.Path) -> None:
        """Model sink should deliver frames at approximately the configured rate."""
        config = PipelineConfig(
            source=str(test_video_path),
            viewer_enabled=False,
            model_max_fps=10.0,
            loop_video_file=True,
        )
        pipeline = Pipeline(config)
        pipeline.start()

        try:
            consumed = 0
            t0 = time.monotonic()
            while time.monotonic() - t0 < 1.0:
                pipeline.model_sink.get_frame(timeout=3.0)
                consumed += 1

            assert 5 <= consumed <= 18, f"Expected ~10 fps, got {consumed}"
        finally:
            pipeline.stop()
