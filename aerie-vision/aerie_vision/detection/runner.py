"""DetectionRunner: background thread that pulls frames and runs detection."""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable

from ..model_sink import ModelSink
from .router import DetectionRouter
from .types import DetectionResult

logger = logging.getLogger(__name__)


class DetectionRunner:
    """Consume frames from a :class:`ModelSink`, run them through a
    :class:`DetectionRouter`, and deliver results via callback.

    Parameters
    ----------
    model_sink:
        Rate-limited frame source (from Phase 4.1).
    router:
        The detection router (single or multi-model).
    on_result:
        Called with each :class:`DetectionResult`.  This is where
        geocoding and event publishing (Phases 4.3/4.4) plug in.
    """

    def __init__(
        self,
        model_sink: ModelSink,
        router: DetectionRouter,
        on_result: Callable[[DetectionResult], None] | None = None,
    ) -> None:
        self._sink = model_sink
        self._router = router
        self._on_result = on_result
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        self._frames_processed = 0
        self._total_inference_ms = 0.0

    # -- public API -----------------------------------------------------------

    def start(self) -> None:
        """Start the detection loop in a background daemon thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._router.warm_up()
        self._thread = threading.Thread(target=self._run, daemon=True, name="detection-runner")
        self._thread.start()
        logger.info("DetectionRunner started")

    def stop(self, timeout: float = 5.0) -> None:
        """Signal the detection thread to stop and wait for it."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        logger.info(
            "DetectionRunner stopped  frames=%d  avg_inference=%.1fms",
            self._frames_processed,
            self.avg_inference_ms,
        )

    @property
    def frames_processed(self) -> int:
        return self._frames_processed

    @property
    def avg_inference_ms(self) -> float:
        if self._frames_processed == 0:
            return 0.0
        return self._total_inference_ms / self._frames_processed

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # -- internals ------------------------------------------------------------

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                frame, meta = self._sink.get_frame(timeout=1.0)
            except TimeoutError:
                continue

            result = self._router.detect(frame, meta.frame_number, meta.timestamp)

            self._frames_processed += 1
            self._total_inference_ms += result.inference_ms

            if self._on_result is not None:
                try:
                    self._on_result(result)
                except Exception:
                    logger.exception("on_result callback failed for frame %d", meta.frame_number)
