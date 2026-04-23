"""VisionPipeline: top-level orchestrator wiring ingest + detection/segmentation + outputs."""

from __future__ import annotations

import logging

from .annotate import annotate_frame
from .annotate_segmentation import annotate_segmentation
from .config import PipelineConfig
from .detection.mock_detector import MockDetector
from .detection.router import DetectionRouter
from .detection.runner import DetectionRunner
from .detection.types import DetectionResult
from .frame_bus import FrameMeta, FrameSlot
from .pipeline import Pipeline
from .segmentation.mock_segmenter import MockSegmenter
from .segmentation.runner import SegmentationRunner
from .segmentation.sam3_source_runner import Sam3SourceRunner
from .segmentation.types import SegmentationResult, TextPrompts
from .text_output import (
    ConsoleOutputStream,
    JsonlOutputStream,
    SegmentationJsonlOutputStream,
)
from .video_recorder import VideoRecorder
from .viewer import ViewerSink

logger = logging.getLogger(__name__)


class VisionPipeline:
    """Orchestrate video ingest, detection or segmentation, and all output sinks.

    The default model is ``rfdetr-medium`` (RF-DETR).  Model names starting with
    ``rfdetr-`` are routed to :class:`~detection.rfdetr_detector.RFDETRDetector`;
    all other paths are handled by :class:`~detection.ultralytics_detector.UltralyticsDetector`.
    Set ``config.detector_model`` to ``""`` to disable detection entirely.

    When ``config.segmenter_model`` is set, segmentation runs and the detector
    is disabled (mutually exclusive in MVP).
    """

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self._config = config or PipelineConfig()

        self._segmentation_enabled = bool(self._config.segmenter_model)
        if self._segmentation_enabled and self._config.detector_model:
            logger.warning(
                "Both segmenter_model=%r and detector_model=%r set; "
                "segmentation wins, detection disabled.",
                self._config.segmenter_model,
                self._config.detector_model,
            )
        self._detection_enabled = (
            bool(self._config.detector_model) and not self._segmentation_enabled
        )

        # Real SAM 3 (Ultralytics) owns ingest itself via model.track(stream=True);
        # the Pipeline must skip FrameGrabber to avoid two cv2 captures of the same
        # source. The mock segmenter still uses the per-frame runner and needs the
        # grabber, so external_source is only set for the real backend.
        external_source = self._uses_external_source()
        self._pipeline = Pipeline(self._config, external_source=external_source)

        self._runner: DetectionRunner | None = None
        self._seg_runner: SegmentationRunner | None = None
        self._sam3_runner: Sam3SourceRunner | None = None
        self._annotated_slot: FrameSlot | None = None
        self._annotated_viewer: ViewerSink | None = None
        self._recorder: VideoRecorder | None = None
        self._jsonl: JsonlOutputStream | None = None
        self._seg_jsonl: SegmentationJsonlOutputStream | None = None
        self._console: ConsoleOutputStream | None = None

        if self._detection_enabled:
            self._init_detection()
        elif self._segmentation_enabled:
            self._init_segmentation()

    def _uses_external_source(self) -> bool:
        return (
            self._segmentation_enabled
            and self._config.segmenter_model.lower() == "sam3"
        )

    # -- lifecycle ------------------------------------------------------------

    def start(self) -> None:
        self._pipeline.start()
        if self._annotated_viewer is not None:
            self._annotated_viewer.start()
        if self._runner is not None:
            self._runner.start()
        if self._seg_runner is not None:
            self._seg_runner.start()
        if self._sam3_runner is not None:
            self._sam3_runner.start()
        logger.info(
            "VisionPipeline running  detection=%s  segmentation=%s",
            self._detection_enabled,
            self._segmentation_enabled,
        )

    def stop(self) -> None:
        if self._runner is not None:
            self._runner.stop()
        if self._seg_runner is not None:
            self._seg_runner.stop()
        if self._sam3_runner is not None:
            self._sam3_runner.stop()
        self._pipeline.stop()
        if self._annotated_viewer is not None:
            self._annotated_viewer.stop()
        if self._recorder is not None:
            self._recorder.close()
        if self._jsonl is not None:
            self._jsonl.close()
        if self._seg_jsonl is not None:
            self._seg_jsonl.close()
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

    @property
    def seg_runner(self) -> SegmentationRunner | None:
        return self._seg_runner

    @property
    def sam3_runner(self) -> Sam3SourceRunner | None:
        return self._sam3_runner

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

    def _init_segmentation(self) -> None:
        model = self._config.segmenter_model.lower()

        if model == "sam3":
            self._sam3_runner = Sam3SourceRunner(
                source=self._config.source,
                prompts=list(self._config.segmenter_classes),
                confidence=self._config.segmenter_confidence,
                device=self._config.segmenter_device,
                bus=self._pipeline.bus,
                on_result=self._on_seg_result,
                model_path=self._config.segmenter_weights,
            )
        elif model == "mock":
            segmenter = MockSegmenter(name="mock-seg")
            prompts = TextPrompts(labels=tuple(self._config.segmenter_classes))
            self._seg_runner = SegmentationRunner(
                model_sink=self._pipeline.segmentation_sink,
                segmenter=segmenter,
                prompts=prompts,
                on_result=self._on_seg_result,
            )
        else:
            raise ValueError(
                f"Unknown segmenter_model {self._config.segmenter_model!r}. "
                "Valid: 'sam3', 'mock', or '' to disable."
            )

        if self._config.annotated_viewer_enabled:
            self._annotated_slot = FrameSlot("annotated")
            self._annotated_viewer = ViewerSink(
                slot=self._annotated_slot,
                host=self._config.viewer_host,
                port=self._config.annotated_viewer_port,
            )

        if self._config.segmenter_jsonl_output:
            self._seg_jsonl = SegmentationJsonlOutputStream(
                path=self._config.segmenter_jsonl_output
            )

    def _build_detector(self) -> object:
        model = self._config.detector_model

        if model == "mock":
            return MockDetector(name="mock")

        if model.lower().startswith("rfdetr-"):
            try:
                from .detection.rfdetr_detector import RFDETRDetector

                return RFDETRDetector(
                    model_name=model,
                    confidence=self._config.detector_confidence,
                    device=self._config.detector_device,
                    name=model,
                )
            except ImportError:
                logger.warning(
                    "rfdetr not installed — reinstall aerie-vision: uv sync"
                )
                raise

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

    def _on_seg_result(self, result: SegmentationResult) -> None:
        annotated = annotate_segmentation(result.frame, result.instances)

        if self._annotated_slot is not None:
            meta = FrameMeta(
                frame_number=result.frame_number,
                timestamp=result.timestamp,
                source_fps=0.0,
                width=annotated.shape[1],
                height=annotated.shape[0],
            )
            self._annotated_slot.put(annotated, meta)

        if self._seg_jsonl is not None:
            self._seg_jsonl.write(result)
