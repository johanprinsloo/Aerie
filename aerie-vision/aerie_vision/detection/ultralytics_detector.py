"""UltralyticsDetector: wraps any model loadable via the ultralytics YOLO API.

Supports YOLOE (open-vocab with text prompts), YOLO26, YOLOv8, and any other
model that ``ultralytics.YOLO`` can load.  Requires the ``yolo`` extra::

    pip install aerie-vision[yolo]
"""

from __future__ import annotations

import logging

import numpy as np

from .types import RawDetection

logger = logging.getLogger(__name__)


class UltralyticsDetector:
    """Detector backed by the ``ultralytics`` library.

    Parameters
    ----------
    model_path:
        A model name (``"yoloe-11s-seg.pt"``, ``"yolo26n.pt"``) that
        ultralytics will download automatically, or a local file path.
        YOLOE models use the ``-seg`` suffix (e.g. ``yoloe-11s-seg.pt``,
        ``yoloe-26s-seg.pt``).
    confidence:
        Minimum confidence threshold passed to the model.
    device:
        Inference device — ``""`` (auto), ``"cpu"``, ``"cuda:0"``, ``"mps"``.
    classes:
        For YOLOE open-vocabulary mode: list of text prompts describing the
        categories to detect (e.g. ``["fire", "smoke", "person"]``).
        Ignored for closed-set models like YOLO26.
    name:
        Human-readable identifier for this detector instance.
    """

    def __init__(
        self,
        model_path: str = "yoloe-11s-seg.pt",
        confidence: float = 0.25,
        device: str = "",
        classes: list[str] | None = None,
        name: str = "",
    ) -> None:
        self._model_path = model_path
        self._confidence = confidence
        self._device = device
        self._classes = classes
        self._name = name or model_path
        self._model: object | None = None  # lazy-loaded

    @property
    def name(self) -> str:
        return self._name

    def detect(
        self, frame: np.ndarray, frame_number: int = 0, timestamp: float = 0.0
    ) -> list[RawDetection]:
        model = self._ensure_model()
        predict_kwargs: dict[str, object] = {
            "conf": self._confidence,
            "verbose": False,
        }
        if self._device:
            predict_kwargs["device"] = self._device
        results = model.predict(frame, **predict_kwargs)
        detections: list[RawDetection] = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].cpu().numpy().astype(int)
                conf = float(boxes.conf[i].cpu().numpy())
                cls_id = int(boxes.cls[i].cpu().numpy())
                label = result.names.get(cls_id, str(cls_id))
                detections.append(
                    RawDetection(
                        label=label,
                        confidence=conf,
                        bbox=(int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])),
                        model_name=self._name,
                        frame_number=frame_number,
                        timestamp=timestamp,
                    )
                )
        return detections

    def warm_up(self) -> None:
        """Load the model and run a dummy inference to prime caches."""
        model = self._ensure_model()
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        model.predict(dummy, conf=self._confidence, verbose=False)
        logger.info("Warmed up %s", self._name)

    def _ensure_model(self) -> object:
        if self._model is None:
            try:
                from ultralytics import YOLO
            except ImportError as exc:
                raise ImportError(
                    "ultralytics is required for UltralyticsDetector. "
                    "Install with: pip install aerie-vision[yolo]"
                ) from exc

            self._model = YOLO(self._model_path)

            if self._classes:
                self._model.set_classes(self._classes)
                logger.info("YOLOE open-vocab classes: %s", self._classes)

            logger.info("Loaded model %s", self._model_path)
        return self._model
