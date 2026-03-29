"""Shared test fixtures for aerie-vision."""

from __future__ import annotations

import pathlib

import cv2
import numpy as np
import pytest

ASSETS = pathlib.Path(__file__).parent / "assets"
TEST_VIDEO = ASSETS / "test_clip.mp4"

# Synthetic video properties
_WIDTH, _HEIGHT, _FPS, _FRAMES = 320, 240, 30, 90  # 3 seconds


def _generate_test_video() -> None:
    """Create a short MP4 with coloured frames and a frame counter."""
    ASSETS.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(TEST_VIDEO), fourcc, _FPS, (_WIDTH, _HEIGHT))
    for i in range(_FRAMES):
        frame = np.zeros((_HEIGHT, _WIDTH, 3), dtype=np.uint8)
        # Cycle hue so each frame is visually distinct
        hue = int((i / _FRAMES) * 180)
        frame[:] = (hue, 200, 200)
        frame = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)
        cv2.putText(frame, str(i), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        writer.write(frame)
    writer.release()


@pytest.fixture(scope="session", autouse=True)
def ensure_test_video() -> pathlib.Path:
    """Generate the synthetic test video once per session."""
    if not TEST_VIDEO.exists():
        _generate_test_video()
    return TEST_VIDEO


@pytest.fixture()
def test_video_path(ensure_test_video: pathlib.Path) -> pathlib.Path:
    return ensure_test_video


@pytest.fixture()
def synthetic_frame() -> np.ndarray:
    """A single 320x240 BGR frame filled with a solid colour."""
    frame = np.zeros((_HEIGHT, _WIDTH, 3), dtype=np.uint8)
    frame[:, :] = (0, 128, 255)  # orange-ish in BGR
    return frame
