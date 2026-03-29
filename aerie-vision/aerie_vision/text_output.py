"""Text output sinks: JSONL (structured) and console (human-readable)."""

from __future__ import annotations

import json
import sys
from typing import IO

from .detection.types import DetectionResult


class JsonlOutputStream:
    """Write one JSON object per :class:`DetectionResult` to a file handle.

    Parameters
    ----------
    path:
        ``"-"`` for stdout, otherwise a filesystem path (opened in append mode).
    """

    def __init__(self, path: str = "-") -> None:
        self._path = path
        self._fh: IO[str] | None = None

    def write(self, result: DetectionResult) -> None:
        fh = self._ensure_open()
        record = {
            "frame_number": result.frame_number,
            "timestamp": result.timestamp,
            "inference_ms": round(result.inference_ms, 2),
            "detections": [
                {
                    "label": d.label,
                    "confidence": round(d.confidence, 4),
                    "bbox": list(d.bbox),
                    "model_name": d.model_name,
                }
                for d in result.detections
            ],
        }
        fh.write(json.dumps(record, separators=(",", ":")) + "\n")
        fh.flush()

    def close(self) -> None:
        if self._fh is not None and self._fh is not sys.stdout:
            self._fh.close()
        self._fh = None

    def _ensure_open(self) -> IO[str]:
        if self._fh is None:
            if self._path == "-":
                self._fh = sys.stdout
            else:
                self._fh = open(self._path, "a")  # noqa: SIM115
        return self._fh


class ConsoleOutputStream:
    """Write compact human-readable detection summaries to stderr.

    Only prints lines when detections are present (quiet when nothing found).
    """

    def write(self, result: DetectionResult) -> None:
        if not result.detections:
            return
        parts = [f"{d.label} {d.confidence:.0%}" for d in result.detections]
        line = (
            f"#{result.frame_number:<6d}  "
            f"{len(result.detections)} detection{'s' if len(result.detections) != 1 else ''}  "
            f"[{', '.join(parts)}]  "
            f"{result.inference_ms:.1f}ms"
        )
        print(line, file=sys.stderr)
