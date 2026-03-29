"""Pipeline: wires FrameGrabber -> FrameBus -> ViewerSink + ModelSink."""

from __future__ import annotations

import logging

from .capture import FrameGrabber
from .config import PipelineConfig
from .frame_bus import FrameBus
from .model_sink import ModelSink
from .viewer import ViewerSink

logger = logging.getLogger(__name__)


class Pipeline:
    """Top-level orchestrator for the video ingest pipeline.

    Creates and manages the lifecycle of every component:
    FrameGrabber -> FrameBus -> ViewerSink + ModelSink.
    """

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self._config = config or PipelineConfig()
        self._bus = FrameBus()

        viewer_slot = self._bus.create_slot("viewer")
        model_slot = self._bus.create_slot("model")

        self._grabber = FrameGrabber(
            source=self._config.source,
            bus=self._bus,
            loop=self._config.loop_video_file,
            reconnect_delay=self._config.reconnect_delay,
        )

        self._viewer: ViewerSink | None = None
        if self._config.viewer_enabled:
            self._viewer = ViewerSink(
                slot=viewer_slot,
                host=self._config.viewer_host,
                port=self._config.viewer_port,
            )

        self._model_sink = ModelSink(
            slot=model_slot,
            max_fps=self._config.model_max_fps,
        )

    # -- public API -----------------------------------------------------------

    def start(self) -> None:
        """Start all components (viewer first, then capture)."""
        if self._viewer is not None:
            self._viewer.start()
        self._grabber.start()
        logger.info("Pipeline running  source=%s", self._config.source)

    def stop(self) -> None:
        """Stop all components cleanly."""
        self._grabber.stop()
        if self._viewer is not None:
            self._viewer.stop()
        logger.info("Pipeline stopped")

    @property
    def model_sink(self) -> ModelSink:
        """The rate-limited consumer that Phase 4.2 (Detection) plugs into."""
        return self._model_sink

    @property
    def grabber(self) -> FrameGrabber:
        return self._grabber

    @property
    def bus(self) -> FrameBus:
        return self._bus

    @property
    def config(self) -> PipelineConfig:
        return self._config
