"""Integration tests for VisionPipeline in segmentation mode."""

from __future__ import annotations

import json
import logging
import pathlib
import time
import urllib.request

import numpy as np

from aerie_vision.config import PipelineConfig
from aerie_vision.orchestrator import VisionPipeline
from aerie_vision.segmentation.mock_segmenter import MockSegmenter
from aerie_vision.segmentation.types import InstanceMask


def _mask() -> np.ndarray:
    m = np.zeros((240, 320), dtype=bool)
    m[40:100, 60:160] = True
    return m


def _inst(instance_id: int = 1, label: str = "fire") -> InstanceMask:
    return InstanceMask(
        instance_id=instance_id, label=label, confidence=0.9,
        bbox=(60, 40, 160, 100), centroid=(110, 70), area_px=6000, mask=_mask(),
    )


def _seg_pipeline(
    test_video_path: pathlib.Path,
    *,
    seg_jsonl: str = "",
    annotated_port: int = 18190,
    scripted_instances: list[InstanceMask] | None = None,
) -> VisionPipeline:
    config = PipelineConfig(
        source=str(test_video_path),
        viewer_enabled=False,
        detector_model="",
        segmenter_model="mock",
        segmenter_classes=["fire"],
        segmenter_jsonl_output=seg_jsonl,
        model_max_fps=20.0,
        annotated_viewer_enabled=True,
        annotated_viewer_port=annotated_port,
        viewer_host="127.0.0.1",
        loop_video_file=True,
    )
    vp = VisionPipeline(config)
    if scripted_instances is not None:
        # Replace the auto-built MockSegmenter with one that emits scripted
        # instances on every frame (key 0 is unused by the runner; we use
        # an "everything" mock by registering all frame numbers we expect).
        scripted = {n: list(scripted_instances) for n in range(0, 200)}
        new_seg = MockSegmenter(scripted=scripted)
        # Internal handle: replace the segmenter on the runner.
        assert vp.seg_runner is not None
        vp.seg_runner._segmenter = new_seg  # type: ignore[attr-defined]
    return vp


class TestVisionPipelineSegmentation:
    def test_seg_runner_constructed_when_segmenter_set(
        self, test_video_path: pathlib.Path
    ) -> None:
        vp = _seg_pipeline(test_video_path, annotated_port=18190)
        try:
            assert vp.seg_runner is not None
            assert vp.runner is None
        finally:
            vp.stop()

    def test_segmentation_wins_over_detection(
        self, test_video_path: pathlib.Path, caplog
    ) -> None:
        config = PipelineConfig(
            source=str(test_video_path),
            viewer_enabled=False,
            detector_model="mock",
            segmenter_model="mock",
            segmenter_classes=["fire"],
            annotated_viewer_enabled=False,
            loop_video_file=True,
        )
        with caplog.at_level(logging.WARNING, logger="aerie_vision.orchestrator"):
            vp = VisionPipeline(config)
        try:
            assert vp.seg_runner is not None
            assert vp.runner is None
            assert any("segmentation wins" in r.message for r in caplog.records)
        finally:
            vp.stop()

    def test_annotated_viewer_reachable(self, test_video_path: pathlib.Path) -> None:
        vp = _seg_pipeline(test_video_path, annotated_port=18191,
                           scripted_instances=[_inst()])
        vp.start()
        try:
            time.sleep(1.5)
            resp = urllib.request.urlopen("http://127.0.0.1:18191/", timeout=3)
            assert resp.status == 200
        finally:
            vp.stop()

    def test_seg_jsonl_output_written(
        self, test_video_path: pathlib.Path, tmp_path
    ) -> None:
        seg_jsonl = str(tmp_path / "seg.jsonl")
        vp = _seg_pipeline(
            test_video_path, seg_jsonl=seg_jsonl, annotated_port=18192,
            scripted_instances=[_inst(1, "fire")],
        )
        vp.start()
        time.sleep(2.0)
        vp.stop()

        assert pathlib.Path(seg_jsonl).exists()
        with open(seg_jsonl) as f:
            lines = f.readlines()
        assert len(lines) > 0
        record = json.loads(lines[0])
        assert "frame_number" in record
        assert "instances" in record
        assert "inference_ms" in record
        # Centroid-only schema: no mask pixels in output
        if record["instances"]:
            inst = record["instances"][0]
            assert set(inst.keys()) == {
                "instance_id", "label", "confidence",
                "bbox", "centroid", "area_px",
            }
            assert "mask" not in inst

    def test_stop_is_clean(self, test_video_path: pathlib.Path) -> None:
        vp = _seg_pipeline(test_video_path, annotated_port=18193)
        vp.start()
        time.sleep(0.5)
        vp.stop()
        vp.stop()  # second stop should not raise

    def test_disabled_when_segmenter_empty(
        self, test_video_path: pathlib.Path
    ) -> None:
        config = PipelineConfig(
            source=str(test_video_path),
            viewer_enabled=False,
            detector_model="",
            segmenter_model="",
            annotated_viewer_enabled=False,
        )
        vp = VisionPipeline(config)
        try:
            assert vp.seg_runner is None
            assert vp.runner is None
        finally:
            vp.stop()
