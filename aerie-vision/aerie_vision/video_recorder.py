"""VideoRecorder: write annotated frames to an MP4 file."""

from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class VideoRecorder:
    """Write frames to a video file using :class:`cv2.VideoWriter`.

    The writer is lazily initialised on the first call to :meth:`write`
    so that the resolution is taken from the actual frame data.

    Parameters
    ----------
    path:
        Output file path (e.g. ``"output.mp4"``).
    fps:
        Recording frame rate.  If ``0`` the *fps_hint* passed to the
        first :meth:`write` call is used.
    """

    def __init__(self, path: str, fps: float = 0.0) -> None:
        self._path = path
        self._fps = fps
        self._writer: cv2.VideoWriter | None = None
        self._frames_written = 0

    def write(self, frame: np.ndarray, fps_hint: float = 30.0) -> None:
        """Append *frame* to the video file."""
        if self._writer is None:
            fps = self._fps if self._fps > 0 else fps_hint
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter.fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(self._path, fourcc, fps, (w, h))
            if not self._writer.isOpened():
                logger.error("Failed to open video writer: %s", self._path)
                return
            logger.info("Recording to %s  %dx%d @ %.1f fps", self._path, w, h, fps)
        self._writer.write(frame)
        self._frames_written += 1

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            logger.info("Recording saved: %s  (%d frames)", self._path, self._frames_written)
        self._writer = None

    @property
    def frames_written(self) -> int:
        return self._frames_written
