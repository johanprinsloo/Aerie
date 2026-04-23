"""SegmentationRunner: background thread that pulls frames and runs segmentation."""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable

from ..model_sink import ModelSink
from .protocol import Segmenter
from .types import Prompts, SegmentationResult

logger = logging.getLogger(__name__)


class SegmentationRunner:
    """Consume frames from a :class:`ModelSink`, run a :class:`Segmenter`,
    and deliver :class:`SegmentationResult`s via callback.

    The segmenter is stateful (video tracker); ``stop()`` resets it so the
    next ``start()`` begins with a clean tracker.

    Parameters
    ----------
    model_sink:
        Rate-limited frame source.
    segmenter:
        The segmenter instance (single model, stateful tracker).
    prompts:
        Prompts passed on every ``segment()`` call.
    on_result:
        Called with each :class:`SegmentationResult`.
    """

    def __init__(
        self,
        model_sink: ModelSink,
        segmenter: Segmenter,
        prompts: Prompts,
        on_result: Callable[[SegmentationResult], None] | None = None,
    ) -> None:
        self._sink = model_sink
        self._segmenter = segmenter
        self._prompts = prompts
        self._on_result = on_result
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        self._frames_processed = 0
        self._total_inference_ms = 0.0

    # -- public API -----------------------------------------------------------

    def start(self) -> None:
        """Start the segmentation loop in a background daemon thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._segmenter.warm_up()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="segmentation-runner"
        )
        self._thread.start()
        logger.info("SegmentationRunner started")

    def stop(self, timeout: float = 5.0) -> None:
        """Signal the segmentation thread to stop, wait, then reset the segmenter."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        self._segmenter.reset()
        logger.info(
            "SegmentationRunner stopped  frames=%d  avg_inference=%.1fms",
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

            t0 = time.monotonic()
            instances = self._segmenter.segment(
                frame, self._prompts, meta.frame_number, meta.timestamp
            )
            elapsed_ms = (time.monotonic() - t0) * 1000.0

            result = SegmentationResult(
                frame_number=meta.frame_number,
                timestamp=meta.timestamp,
                instances=tuple(instances),
                frame=frame,
                inference_ms=elapsed_ms,
            )

            self._frames_processed += 1
            self._total_inference_ms += elapsed_ms

            if self._on_result is not None:
                try:
                    self._on_result(result)
                except Exception:
                    logger.exception(
                        "on_result callback failed for frame %d", meta.frame_number
                    )
