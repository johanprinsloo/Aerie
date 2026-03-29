"""OnnxDetector: ONNX Runtime inference for production deployment.

No PyTorch dependency -- only ``onnxruntime`` (or a provider-specific
variant like ``onnxruntime-gpu``)::

    pip install aerie-vision[onnx]
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from .types import RawDetection

logger = logging.getLogger(__name__)


class OnnxDetector:
    """Detector backed by ONNX Runtime.

    Parameters
    ----------
    model_path:
        Path to a ``.onnx`` file exported from ultralytics (or any YOLO
        variant).  The model is expected to accept ``(1, 3, H, W)`` float32
        input and produce the standard ultralytics output tensor.
    labels:
        Ordered list of class names matching the model's output indices.
    confidence:
        Minimum confidence threshold.
    input_size:
        ``(width, height)`` the model was exported at.
    name:
        Human-readable identifier.
    """

    def __init__(
        self,
        model_path: str,
        labels: list[str],
        confidence: float = 0.25,
        input_size: tuple[int, int] = (640, 640),
        name: str = "",
    ) -> None:
        self._model_path = model_path
        self._labels = labels
        self._confidence = confidence
        self._input_w, self._input_h = input_size
        self._name = name or model_path
        self._session: object | None = None  # lazy-loaded

    @property
    def name(self) -> str:
        return self._name

    def detect(
        self, frame: np.ndarray, frame_number: int = 0, timestamp: float = 0.0
    ) -> list[RawDetection]:
        session = self._ensure_session()
        orig_h, orig_w = frame.shape[:2]

        blob = self._preprocess(frame)

        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: blob})
        preds = outputs[0]  # shape: (1, num_det, 4+num_classes) or (1, 4+num_classes, num_det)

        return self._postprocess(preds, orig_w, orig_h, frame_number, timestamp)

    def warm_up(self) -> None:
        session = self._ensure_session()
        dummy = np.zeros((1, 3, self._input_h, self._input_w), dtype=np.float32)
        input_name = session.get_inputs()[0].name
        session.run(None, {input_name: dummy})
        logger.info("Warmed up ONNX model %s", self._name)

    # -- preprocessing --------------------------------------------------------

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Resize, normalize, and transpose to NCHW float32."""
        img = cv2.resize(frame, (self._input_w, self._input_h))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)  # HWC -> CHW
        return np.expand_dims(img, axis=0)  # -> NCHW

    # -- postprocessing -------------------------------------------------------

    def _postprocess(
        self,
        preds: np.ndarray,
        orig_w: int,
        orig_h: int,
        frame_number: int,
        timestamp: float,
    ) -> list[RawDetection]:
        """Decode the ultralytics ONNX output tensor into RawDetections."""
        # Ultralytics ONNX output shape: (1, 4+num_classes, num_detections)
        # Transpose to (num_detections, 4+num_classes)
        if preds.ndim == 3:
            preds = preds[0]
        if preds.shape[0] == (4 + len(self._labels)):
            preds = preds.T

        detections: list[RawDetection] = []
        scale_x = orig_w / self._input_w
        scale_y = orig_h / self._input_h

        for row in preds:
            cx, cy, w, h = row[:4]
            class_scores = row[4:]
            cls_id = int(np.argmax(class_scores))
            conf = float(class_scores[cls_id])

            if conf < self._confidence:
                continue

            x1 = int((cx - w / 2) * scale_x)
            y1 = int((cy - h / 2) * scale_y)
            x2 = int((cx + w / 2) * scale_x)
            y2 = int((cy + h / 2) * scale_y)

            label = self._labels[cls_id] if cls_id < len(self._labels) else str(cls_id)
            detections.append(
                RawDetection(
                    label=label,
                    confidence=conf,
                    bbox=(x1, y1, x2, y2),
                    model_name=self._name,
                    frame_number=frame_number,
                    timestamp=timestamp,
                )
            )

        return detections

    def _ensure_session(self) -> object:
        if self._session is None:
            try:
                import onnxruntime as ort
            except ImportError as exc:
                raise ImportError(
                    "onnxruntime is required for OnnxDetector. "
                    "Install with: pip install aerie-vision[onnx]"
                ) from exc

            providers = ort.get_available_providers()
            logger.info("ONNX Runtime providers: %s", providers)
            self._session = ort.InferenceSession(self._model_path, providers=providers)
            logger.info("Loaded ONNX model %s", self._model_path)
        return self._session
