"""VisionPipeline: top-level orchestrator wiring ingest + detection + outputs."""

from __future__ import annotations

import logging

from .annotate import annotate_frame
from .config import PipelineConfig
from .detection.mock_detector import MockDetector
from .detection.router import DetectionRouter
from .detection.runner import DetectionRunner
from .detection.types import DetectionResult
from .frame_bus import FrameMeta, FrameSlot
from .pipeline import Pipeline
from .text_output import ConsoleOutputStream, JsonlOutputStream
from .video_recorder import VideoRecorder
from .viewer import ViewerSink

logger = logging.getLogger(__name__)


class VisionPipeline:
    """Orchestrate video ingest, detection, and all output sinks.

    When ``config.detector_model`` is empty, runs in ingest-only mode
    (backward-compatible with the bare :class:`Pipeline`).
    """

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self._config = config or PipelineConfig()
        self._pipeline = Pipeline(self._config)

        self._detection_enabled = bool(self._config.detector_model)

        self._runner: DetectionRunner | None = None
        self._annotated_slot: FrameSlot | None = None
        self._annotated_viewer: ViewerSink | None = None
        self._recorder: VideoRecorder | None = None
        self._jsonl: JsonlOutputStream | None = None
        self._console: ConsoleOutputStream | None = None

        if self._detection_enabled:
            self._init_detection()

    # -- lifecycle ------------------------------------------------------------

    def start(self) -> None:
        self._pipeline.start()
        if self._annotated_viewer is not None:
            self._annotated_viewer.start()
        if self._runner is not None:
            self._runner.start()
        logger.info("VisionPipeline running  detection=%s", self._detection_enabled)

    def stop(self) -> None:
        if self._runner is not None:
            self._runner.stop()
        self._pipeline.stop()
        if self._annotated_viewer is not None:
            self._annotated_viewer.stop()
        if self._recorder is not None:
            self._recorder.close()
        if self._jsonl is not None:
            self._jsonl.close()
        logger.info("VisionPipeline stopped")

    # -- accessors for testing ------------------------------------------------

    @property
    def pipeline(self) -> Pipeline:
        return self._pipeline

    @property
    def config(self) -> PipelineConfig:
        return self._config

    @property
    def runner(self) -> DetectionRunner | None:
        return self._runner

    # -- internal setup -------------------------------------------------------

    def _init_detection(self) -> None:
        detector = self._build_detector()
        router = DetectionRouter(primary=detector)

        self._runner = DetectionRunner(
            model_sink=self._pipeline.model_sink,
            router=router,
            on_result=self._on_result,
        )

        if self._config.annotated_viewer_enabled:
            self._annotated_slot = FrameSlot("annotated")
            self._annotated_viewer = ViewerSink(
                slot=self._annotated_slot,
                host=self._config.viewer_host,
                port=self._config.annotated_viewer_port,
            )

        if self._config.record_path:
            self._recorder = VideoRecorder(
                path=self._config.record_path,
                fps=self._config.record_fps,
            )

        if self._config.jsonl_output:
            self._jsonl = JsonlOutputStream(path=self._config.jsonl_output)

        if self._config.console_output:
            self._console = ConsoleOutputStream()

    def _build_detector(self) -> object:
        model = self._config.detector_model

        if model == "mock":
            return MockDetector(name="mock")

        try:
            from .detection.ultralytics_detector import UltralyticsDetector

            return UltralyticsDetector(
                model_path=model,
                confidence=self._config.detector_confidence,
                device=self._config.detector_device,
                classes=self._config.detector_classes or None,
                name=model,
            )
        except ImportError:
            logger.warning(
                "ultralytics not installed — install with: uv sync --extra yolo"
            )
            raise

    def _on_result(self, result: DetectionResult) -> None:
        annotated = annotate_frame(result.frame, result.detections)

        if self._annotated_slot is not None:
            meta = FrameMeta(
                frame_number=result.frame_number,
                timestamp=result.timestamp,
                source_fps=0.0,
                width=annotated.shape[1],
                height=annotated.shape[0],
            )
            self._annotated_slot.put(annotated, meta)

        if self._recorder is not None:
            self._recorder.write(annotated)

        if self._jsonl is not None:
            self._jsonl.write(result)

        if self._console is not None:
            self._console.write(result)
