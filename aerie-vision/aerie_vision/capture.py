"""FrameGrabber: captures frames from a cv2-compatible video source."""

from __future__ import annotations

import logging
import threading
import time

import cv2

from .frame_bus import FrameBus, FrameMeta

logger = logging.getLogger(__name__)


class FrameGrabber:
    """Decode frames from a video source in a background thread.

    Parameters
    ----------
    source:
        Anything ``cv2.VideoCapture`` accepts — an integer device index,
        a file path, or an RTSP/UDP URL.
    bus:
        Every decoded frame is published here.
    loop:
        If *True* and *source* is a file, seek back to the start on EOF.
    reconnect_delay:
        Seconds to wait before retrying after a source read failure.
    """

    def __init__(
        self,
        source: str | int,
        bus: FrameBus,
        *,
        loop: bool = True,
        reconnect_delay: float = 2.0,
    ) -> None:
        self._source = source
        self._bus = bus
        self._loop = loop
        self._reconnect_delay = reconnect_delay

        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._source_fps: float = 0.0
        self._running = False

    # -- public API -----------------------------------------------------------

    def start(self) -> None:
        """Open the source and begin capturing in a daemon thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="frame-grabber")
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        """Signal the capture thread to stop and wait for it to finish."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        self._running = False

    @property
    def source_fps(self) -> float:
        """Frames-per-second reported by the source (0 if unknown)."""
        return self._source_fps

    @property
    def is_running(self) -> bool:
        return self._running

    # -- internals ------------------------------------------------------------

    def _open_source(self) -> cv2.VideoCapture | None:
        source = self._source
        if isinstance(source, str) and source.isdigit():
            source = int(source)
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            logger.error("Cannot open video source: %s", self._source)
            return None
        fps = cap.get(cv2.CAP_PROP_FPS)
        self._source_fps = fps if fps > 0 else 30.0
        logger.info(
            "Opened source %s  fps=%.1f  size=%dx%d",
            self._source,
            self._source_fps,
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        return cap

    def _run(self) -> None:
        self._running = True
        frame_number = 0
        cap = self._open_source()

        while not self._stop_event.is_set():
            if cap is None or not cap.isOpened():
                logger.warning("Source lost — retrying in %.1fs", self._reconnect_delay)
                if cap is not None:
                    cap.release()
                time.sleep(self._reconnect_delay)
                cap = self._open_source()
                continue

            ok, frame = cap.read()
            if not ok:
                if self._is_file_source() and self._loop:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                if self._is_file_source():
                    logger.info("End of video file — stopping")
                    break
                logger.warning("Frame read failed — retrying in %.1fs", self._reconnect_delay)
                time.sleep(self._reconnect_delay)
                continue

            meta = FrameMeta(
                frame_number=frame_number,
                timestamp=time.monotonic(),
                source_fps=self._source_fps,
                width=frame.shape[1],
                height=frame.shape[0],
            )
            self._bus.publish(frame, meta)
            frame_number += 1

        if cap is not None:
            cap.release()
        self._running = False
        logger.info("FrameGrabber stopped after %d frames", frame_number)

    def _is_file_source(self) -> bool:
        if isinstance(self._source, int):
            return False
        s = self._source.lower()
        return not s.startswith(("rtsp://", "rtsps://", "udp://", "http://", "https://"))
