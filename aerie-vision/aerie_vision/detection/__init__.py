"""Detection layer: model-agnostic protocol, multi-model routing, and backends."""

from .types import DetectionResult, RawDetection
from .protocol import Detector

__all__ = ["Detector", "DetectionResult", "RawDetection"]
