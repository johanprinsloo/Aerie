"""RFDETRDetector: wraps RF-DETR models from the rfdetr library.

Supports detection model sizes Nano → Large (Apache 2.0).
Plus sizes (XLarge, 2XLarge) require ``pip install rfdetr[plus]``.

Install aerie-vision to get RF-DETR included by default::

    pip install aerie-vision
"""

from __future__ import annotations

import logging

import numpy as np

from .types import RawDetection

logger = logging.getLogger(__name__)

_MODEL_CLASSES: dict[str, str] = {
    "rfdetr-nano": "RFDETRNano",
    "rfdetr-small": "RFDETRSmall",
    "rfdetr-medium": "RFDETRMedium",
    "rfdetr-large": "RFDETRLarge",
    "rfdetr-xlarge": "RFDETRXLarge",
    "rfdetr-2xlarge": "RFDETR2XLarge",
}


class RFDETRDetector:
    """Detector backed by the ``rfdetr`` library (RF-DETR).

    Parameters
    ----------
    model_name:
        One of ``rfdetr-nano``, ``rfdetr-small``, ``rfdetr-medium``,
        ``rfdetr-large``, ``rfdetr-xlarge`` (plus), ``rfdetr-2xlarge`` (plus).
    confidence:
        Minimum confidence threshold.
    device:
        Inference device — ``""`` (auto), ``"cpu"``, ``"cuda:0"``, ``"mps"``.
    name:
        Human-readable identifier for this detector instance.
    """

    def __init__(
        self,
        model_name: str = "rfdetr-medium",
        confidence: float = 0.5,
        device: str = "",
        name: str = "",
    ) -> None:
        self._model_name = model_name.lower()
        self._confidence = confidence
        self._device = device
        self._name = name or model_name
        self._model: object | None = None
        self._class_names: dict[int, str] = {}

    @property
    def name(self) -> str:
        return self._name

    def detect(
        self, frame: np.ndarray, frame_number: int = 0, timestamp: float = 0.0
    ) -> list[RawDetection]:
        model = self._ensure_model()
        sv_detections = model.predict(frame, threshold=self._confidence)
        return self._to_raw(sv_detections, frame_number, timestamp)

    def warm_up(self) -> None:
        """Load the model and run a dummy inference to prime caches."""
        model = self._ensure_model()
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        model.predict(dummy, threshold=self._confidence)
        logger.info("Warmed up %s", self._name)

    # -- internals ------------------------------------------------------------

    def _to_raw(
        self, sv_detections: object, frame_number: int, timestamp: float
    ) -> list[RawDetection]:
        xyxy = getattr(sv_detections, "xyxy", None)
        if xyxy is None or len(xyxy) == 0:
            return []

        confidences = sv_detections.confidence
        class_ids = sv_detections.class_id

        results: list[RawDetection] = []
        for i in range(len(xyxy)):
            cls_id = int(class_ids[i])
            label = self._class_names.get(cls_id, str(cls_id))
            results.append(
                RawDetection(
                    label=label,
                    confidence=float(confidences[i]),
                    bbox=(
                        int(xyxy[i][0]),
                        int(xyxy[i][1]),
                        int(xyxy[i][2]),
                        int(xyxy[i][3]),
                    ),
                    model_name=self._name,
                    frame_number=frame_number,
                    timestamp=timestamp,
                )
            )
        return results

    def _ensure_model(self) -> object:
        if self._model is not None:
            return self._model

        try:
            import rfdetr
            from rfdetr.assets.coco_classes import COCO_CLASSES
        except ImportError as exc:
            raise ImportError(
                "rfdetr is required for RFDETRDetector. "
                "Install with: pip install aerie-vision"
            ) from exc

        class_name = _MODEL_CLASSES.get(self._model_name)
        if class_name is None:
            raise ValueError(
                f"Unknown RF-DETR model '{self._model_name}'. "
                f"Valid options: {', '.join(_MODEL_CLASSES)}"
            )

        model_cls = getattr(rfdetr, class_name, None)
        if model_cls is None:
            raise ImportError(
                f"{class_name} not found in rfdetr. "
                "XL/2XL sizes require: pip install rfdetr[plus]"
            )

        kwargs: dict[str, object] = {}
        if self._device:
            kwargs["device"] = self._device

        self._model = model_cls(**kwargs)

        # COCO_CLASSES may be a dict {id: name} or a sequence; normalise to dict.
        if isinstance(COCO_CLASSES, dict):
            self._class_names = {int(k): str(v) for k, v in COCO_CLASSES.items()}
        else:
            self._class_names = {i: str(v) for i, v in enumerate(COCO_CLASSES)}

        logger.info("Loaded RF-DETR model: %s", self._model_name)
        return self._model
