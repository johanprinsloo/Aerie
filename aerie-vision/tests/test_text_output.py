"""Tests for JsonlOutputStream and ConsoleOutputStream."""

from __future__ import annotations

import io
import json

import numpy as np

from aerie_vision.detection.types import DetectionResult, RawDetection
from aerie_vision.text_output import ConsoleOutputStream, JsonlOutputStream


def _det(label: str = "fire", conf: float = 0.9) -> RawDetection:
    return RawDetection(
        label=label, confidence=conf, bbox=(10, 20, 100, 200),
        model_name="test-model", frame_number=42, timestamp=1.5,
    )


def _result(dets: tuple[RawDetection, ...] = ()) -> DetectionResult:
    return DetectionResult(
        frame_number=42, timestamp=1.5,
        detections=dets,
        frame=np.zeros((240, 320, 3), dtype=np.uint8),
        inference_ms=12.34,
    )


class TestJsonlOutputStream:
    def test_writes_valid_json(self, tmp_path) -> None:
        path = str(tmp_path / "out.jsonl")
        stream = JsonlOutputStream(path=path)
        stream.write(_result((_det("fire", 0.95), _det("smoke", 0.80))))
        stream.close()

        with open(path) as f:
            line = f.readline()
        record = json.loads(line)

        assert record["frame_number"] == 42
        assert record["timestamp"] == 1.5
        assert record["inference_ms"] == 12.34
        assert len(record["detections"]) == 2
        assert record["detections"][0]["label"] == "fire"
        assert record["detections"][0]["confidence"] == 0.95
        assert record["detections"][0]["bbox"] == [10, 20, 100, 200]
        assert record["detections"][0]["model_name"] == "test-model"

    def test_multiple_writes(self, tmp_path) -> None:
        path = str(tmp_path / "out.jsonl")
        stream = JsonlOutputStream(path=path)
        stream.write(_result((_det(),)))
        stream.write(_result((_det("smoke"),)))
        stream.close()

        with open(path) as f:
            lines = f.readlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["detections"][0]["label"] == "fire"
        assert json.loads(lines[1])["detections"][0]["label"] == "smoke"

    def test_empty_detections(self, tmp_path) -> None:
        path = str(tmp_path / "out.jsonl")
        stream = JsonlOutputStream(path=path)
        stream.write(_result())
        stream.close()

        with open(path) as f:
            record = json.loads(f.readline())
        assert record["detections"] == []

    def test_close_twice_safe(self, tmp_path) -> None:
        path = str(tmp_path / "out.jsonl")
        stream = JsonlOutputStream(path=path)
        stream.write(_result())
        stream.close()
        stream.close()  # should not raise


class TestConsoleOutputStream:
    def test_prints_when_detections_present(self, capsys) -> None:
        stream = ConsoleOutputStream()
        stream.write(_result((_det("fire", 0.94),)))
        captured = capsys.readouterr()
        assert "fire 94%" in captured.err
        assert "#42" in captured.err

    def test_silent_when_no_detections(self, capsys) -> None:
        stream = ConsoleOutputStream()
        stream.write(_result())
        captured = capsys.readouterr()
        assert captured.err == ""

    def test_multiple_detections_listed(self, capsys) -> None:
        stream = ConsoleOutputStream()
        stream.write(_result((_det("fire", 0.94), _det("smoke", 0.81))))
        captured = capsys.readouterr()
        assert "fire 94%" in captured.err
        assert "smoke 81%" in captured.err
        assert "2 detections" in captured.err
