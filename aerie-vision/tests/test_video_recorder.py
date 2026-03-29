"""Tests for VideoRecorder."""

from __future__ import annotations

import cv2
import numpy as np

from aerie_vision.video_recorder import VideoRecorder


def _frame(value: int = 128) -> np.ndarray:
    f = np.zeros((240, 320, 3), dtype=np.uint8)
    f[:] = value
    return f


class TestVideoRecorder:
    def test_write_and_read_back(self, tmp_path) -> None:
        path = str(tmp_path / "out.mp4")
        rec = VideoRecorder(path=path, fps=10.0)

        for i in range(15):
            rec.write(_frame(i * 10))
        rec.close()

        cap = cv2.VideoCapture(path)
        assert cap.isOpened()
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        assert count == 15
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        assert (w, h) == (320, 240)
        cap.release()

    def test_frames_written_counter(self, tmp_path) -> None:
        path = str(tmp_path / "out.mp4")
        rec = VideoRecorder(path=path, fps=10.0)
        assert rec.frames_written == 0

        rec.write(_frame())
        rec.write(_frame())
        assert rec.frames_written == 2
        rec.close()

    def test_lazy_init_uses_fps_hint(self, tmp_path) -> None:
        path = str(tmp_path / "out.mp4")
        rec = VideoRecorder(path=path, fps=0.0)
        rec.write(_frame(), fps_hint=15.0)
        rec.write(_frame(), fps_hint=15.0)
        rec.close()

        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        assert abs(fps - 15.0) < 1.0
        cap.release()

    def test_close_without_write(self, tmp_path) -> None:
        path = str(tmp_path / "out.mp4")
        rec = VideoRecorder(path=path)
        rec.close()  # should not raise
        assert rec.frames_written == 0
