"""DetectionRouter: multi-model orchestration with parallel and escalation support."""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .nms import merge_detections
from .protocol import Detector
from .types import DetectionResult, RawDetection

logger = logging.getLogger(__name__)


class DetectionRouter:
    """Route frames through one or more detectors and merge results.

    Parameters
    ----------
    primary:
        Always-on detector.  Every frame passes through this model.
    secondaries:
        Optional additional detectors that run **in parallel** with the
        primary on every frame.  Useful for running specialised models
        alongside a general-purpose one.
    escalation:
        Optional heavier detector invoked only when the primary (and
        secondaries) return detections below *escalation_threshold*.
    escalation_threshold:
        Confidence below which a detection triggers escalation.
    iou_merge_threshold:
        IoU threshold for cross-model NMS when merging results.
    """

    def __init__(
        self,
        primary: Detector,
        secondaries: list[Detector] | None = None,
        escalation: Detector | None = None,
        escalation_threshold: float = 0.5,
        iou_merge_threshold: float = 0.5,
    ) -> None:
        self._primary = primary
        self._secondaries = secondaries or []
        self._escalation = escalation
        self._escalation_threshold = escalation_threshold
        self._iou_merge_threshold = iou_merge_threshold

    def detect(
        self, frame: np.ndarray, frame_number: int = 0, timestamp: float = 0.0
    ) -> DetectionResult:
        """Run the frame through the configured detector(s) and return merged results."""
        t0 = time.monotonic()

        all_detections: list[RawDetection] = []

        if self._secondaries:
            all_detections = self._run_parallel(frame, frame_number, timestamp)
        else:
            all_detections = self._primary.detect(frame, frame_number, timestamp)

        needs_escalation = (
            self._escalation is not None
            and any(d.confidence < self._escalation_threshold for d in all_detections)
        )
        if needs_escalation:
            assert self._escalation is not None
            escalation_dets = self._escalation.detect(frame, frame_number, timestamp)
            all_detections.extend(escalation_dets)

        merged = merge_detections(all_detections, iou_threshold=self._iou_merge_threshold)

        elapsed_ms = (time.monotonic() - t0) * 1000.0
        return DetectionResult(
            frame_number=frame_number,
            timestamp=timestamp,
            detections=tuple(merged),
            frame=frame,
            inference_ms=elapsed_ms,
        )

    def warm_up(self) -> None:
        """Warm up all configured detectors."""
        self._primary.warm_up()
        for sec in self._secondaries:
            sec.warm_up()
        if self._escalation is not None:
            self._escalation.warm_up()

    def _run_parallel(
        self, frame: np.ndarray, frame_number: int, timestamp: float
    ) -> list[RawDetection]:
        """Run primary + secondaries concurrently via a thread pool."""
        all_detectors = [self._primary, *self._secondaries]
        results: list[RawDetection] = []

        with ThreadPoolExecutor(max_workers=len(all_detectors)) as pool:
            futures = [
                pool.submit(det.detect, frame, frame_number, timestamp)
                for det in all_detectors
            ]
            for future in futures:
                results.extend(future.result())

        return results
