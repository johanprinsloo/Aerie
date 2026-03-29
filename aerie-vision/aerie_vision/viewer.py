"""ViewerSink: MJPEG-over-HTTP live viewer using stdlib http.server."""

from __future__ import annotations

import logging
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import cv2
import numpy as np

from .frame_bus import FrameSlot

logger = logging.getLogger(__name__)

_BOUNDARY = b"--aerie-frame"

_INDEX_HTML = b"""\
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Aerie Vision</title>
  <style>
    body { margin: 0; background: #111; display: flex;
           justify-content: center; align-items: center; height: 100vh; }
    img  { max-width: 100vw; max-height: 100vh; }
  </style>
</head>
<body>
  <img src="/stream" alt="live feed">
</body>
</html>
"""


class _MJPEGHandler(BaseHTTPRequestHandler):
    """Handles ``/`` (index page) and ``/stream`` (MJPEG)."""

    slot: FrameSlot
    jpeg_quality: int
    overlay: bool

    def do_GET(self) -> None:  # noqa: N802 — required by BaseHTTPRequestHandler
        if self.path == "/":
            self._serve_index()
        elif self.path == "/stream":
            self._serve_stream()
        else:
            self.send_error(404)

    # suppress per-request log lines
    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        pass

    def _serve_index(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(_INDEX_HTML)))
        self.end_headers()
        self.wfile.write(_INDEX_HTML)

    def _serve_stream(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={_BOUNDARY.decode()}")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        try:
            while True:
                frame, meta = self.slot.get(timeout=2.0)

                if self.overlay and meta is not None:
                    frame = _overlay_info(frame, meta)

                ok, jpeg = cv2.imencode(
                    ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
                )
                if not ok:
                    continue

                payload = jpeg.tobytes()
                self.wfile.write(_BOUNDARY + b"\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(payload)}\r\n".encode())
                self.wfile.write(b"\r\n")
                self.wfile.write(payload)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            pass


def _overlay_info(frame: np.ndarray, meta: "FrameMeta") -> np.ndarray:
    """Burn frame number and timestamp into the top-left corner."""
    from .frame_bus import FrameMeta as _FM  # avoid circular at module level

    if not isinstance(meta, _FM):
        return frame
    out = frame.copy()
    text = f"#{meta.frame_number}  {meta.source_fps:.0f}fps"
    cv2.putText(out, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    return out


class ViewerSink:
    """Serve a live MJPEG stream over HTTP.

    Parameters
    ----------
    slot:
        :class:`FrameSlot` to read frames from.
    host:
        Network interface to bind to (``"0.0.0.0"`` for all).
    port:
        HTTP port.
    jpeg_quality:
        JPEG compression quality (1-100).
    overlay:
        If *True*, burn frame metadata into the image.
    """

    def __init__(
        self,
        slot: FrameSlot,
        host: str = "0.0.0.0",
        port: int = 8090,
        *,
        jpeg_quality: int = 80,
        overlay: bool = False,
    ) -> None:
        self._slot = slot
        self._host = host
        self._port = port
        self._jpeg_quality = jpeg_quality
        self._overlay = overlay
        self._server: HTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start serving in a background daemon thread."""
        handler_cls = _make_handler_class(
            slot=self._slot,
            jpeg_quality=self._jpeg_quality,
            overlay=self._overlay,
        )
        self._server = HTTPServer((self._host, self._port), handler_cls)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True, name="mjpeg-viewer")
        self._thread.start()
        logger.info("Viewer started at http://%s:%d/", self._host, self._port)

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        logger.info("Viewer stopped")

    @property
    def url(self) -> str:
        return f"http://{self._host}:{self._port}/"


def _make_handler_class(
    slot: FrameSlot,
    jpeg_quality: int,
    overlay: bool,
) -> type[_MJPEGHandler]:
    """Return a handler subclass with *slot* and settings baked in as class attrs."""

    class _Handler(_MJPEGHandler):
        pass

    _Handler.slot = slot  # type: ignore[attr-defined]
    _Handler.jpeg_quality = jpeg_quality  # type: ignore[attr-defined]
    _Handler.overlay = overlay  # type: ignore[attr-defined]
    return _Handler
