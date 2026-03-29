# aerie-vision

Video ingest and ML detection pipeline for the Aerie drone supervision system.
Captures frames from a webcam, HDMI frame grabber, RTSP stream, or video file;
distributes them to a live web viewer and a rate-limited detection layer; and
runs pluggable object detection models with support for parallel multi-model
inference and confidence-based escalation.

## Install

Requires [uv](https://docs.astral.sh/uv/).

```bash
cd aerie-vision
uv sync             # creates .venv and installs runtime deps
uv sync --group dev # also installs pytest, ruff
```

ML model backends are optional extras so the core pipeline runs without
heavyweight ML dependencies:

```bash
uv sync --extra yolo   # adds ultralytics (YOLOE, YOLO26, YOLOv8, ...)
uv sync --extra onnx   # adds onnxruntime  (production ONNX inference)
```

## Quick start

**Ingest only** (no detection, no ML download needed):

```bash
uv run python -m aerie_vision --source 0
```

Open http://localhost:8090 in a browser to see the raw webcam feed.

**Webcam + YOLO26 detection** (auto-downloads ~5 MB model on first run):

```bash
uv sync --extra yolo
uv run python -m aerie_vision --source 0 --model yolo26n.pt --model-fps 5
```

Open two browser tabs:
- http://localhost:8090 -- raw video feed
- http://localhost:8091 -- annotated feed with bounding boxes and labels

YOLO26 detects all 80 COCO classes (people, chairs, phones, etc.) out of the
box.

**Webcam + YOLOE zero-shot** (detect custom classes with no training):

```bash
uv sync --extra yolo
uv run python -m aerie_vision \
  --source 0 \
  --model yoloe-11s-seg.pt \
  --classes fire smoke person \
  --model-fps 5
```

Only detects the classes you specify -- no training data needed.  YOLOE model
names always include `-seg` (e.g. `yoloe-11s-seg.pt`, `yoloe-26s-seg.pt`).

**Video file + detection + all outputs:**

```bash
uv run python -m aerie_vision \
  --source /path/to/drone_footage.mp4 \
  --model yolo26n.pt \
  --model-fps 10 \
  --jsonl detections.jsonl \
  --record annotated_output.mp4
```

This runs detection on the video, streams annotated video at :8091, writes
JSONL detection records to `detections.jsonl`, records the annotated video to
`annotated_output.mp4`, and prints detection summaries to the console.

**JSONL to stdout** (for piping to other tools):

```bash
uv run python -m aerie_vision \
  --source 0 \
  --model yolo26n.pt \
  --jsonl - \
  --no-console
```

**Mock detector** (no ML dependencies, for testing the pipeline):

```bash
uv run python -m aerie_vision --source 0 --model mock
```

Both viewers will be live but the mock detector only produces detections on
scripted frame numbers (none by default on live input).

Press Ctrl-C to stop any of the above.

## GPU acceleration

The `--device` flag selects the inference device.  By default ultralytics
picks automatically, but you can force a specific backend:

| Platform | Flag | Notes |
|----------|------|-------|
| Apple Silicon (M1/M2/M3/M4) | `--device mps` | Uses Metal Performance Shaders. PyTorch MPS is included by default. |
| NVIDIA GPU | `--device cuda:0` | Requires CUDA-enabled PyTorch (`pip install torch --index-url ...`). |
| CPU (any platform) | `--device cpu` | Always works, no GPU driver needed. |

**Apple Silicon example** (typically 2-3x faster than CPU):

```bash
uv run python -m aerie_vision \
  --source 0 \
  --model yoloe-11s-seg.pt \
  --classes person coffee_mug laptop \
  --model-fps 10 \
  --device mps
```

Check if MPS is available on your system:

```bash
uv run python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
```

With GPU acceleration you can increase `--model-fps` since inference is
faster (e.g. ~20-40ms on MPS vs ~80-95ms on CPU for `yoloe-11s-seg`).

## Video sources

The `--source` flag (or `AERIE_VISION_SOURCE` env var) accepts three kinds of
input.  OpenCV handles the platform-specific driver layer, so the same command
works on macOS, Linux, and Windows.

### Webcam or USB camera

Pass a device index.  `0` is typically the built-in webcam; `1`, `2`, etc. are
additional USB cameras.

```bash
uv run python -m aerie_vision --source 0
uv run python -m aerie_vision --source 1
```

### HDMI frame grabber

USB HDMI capture cards (Elgato Cam Link, AVerMedia, generic UVC devices) appear
to the OS as regular video devices.  Find the device index and pass it just like
a webcam.

**macOS** — list devices with:

```bash
system_profiler SPCameraDataType
# or
ffmpeg -f avfoundation -list_devices true -i "" 2>&1 | grep -i video
```

**Linux** — list devices with:

```bash
v4l2-ctl --list-devices
# or
ls /dev/video*
```

**Windows** — list devices with:

```powershell
ffmpeg -f dshow -list_devices true -i "" 2>&1 | Select-String "video"
```

Then run:

```bash
# If the frame grabber shows up as device 2:
uv run python -m aerie_vision --source 2
```

### Video file

Pass a path to any file OpenCV can decode (MP4, AVI, MKV, etc.).  By default
the file loops forever; add `--no-loop` to stop at the end.

```bash
uv run python -m aerie_vision --source /path/to/drone_footage.mp4
uv run python -m aerie_vision --source /path/to/clip.avi --no-loop
```

### RTSP / UDP stream

Pass the full URL.  Works with any RTSP or UDP source (IP cameras, GStreamer
pipelines, OBS RTSP output, etc.).

```bash
uv run python -m aerie_vision --source rtsp://192.168.1.100:8554/live
uv run python -m aerie_vision --source udp://0.0.0.0:5600
```

## CLI options

### Video ingest

| Flag | Default | Description |
|------|---------|-------------|
| `--source` | `0` | Device index, file path, or stream URL |
| `--port` | `8090` | HTTP port for the raw MJPEG web viewer |
| `--model-fps` | `5.0` | Max frames/sec delivered to the detection model |
| `--no-viewer` | off | Disable the raw web viewer entirely |
| `--no-loop` | off | Don't loop video files (stop at end) |
| `--overlay` | off | Burn frame number and FPS into the raw viewer image |

### Detection

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | (none) | Model path (`yoloe-11s-seg.pt`, `yolo26n.pt`) or `mock` for testing. Omit to run ingest-only. |
| `--confidence` | `0.25` | Minimum detection confidence threshold |
| `--classes` | (all) | YOLOE open-vocab class prompts (e.g. `--classes fire smoke person`) |
| `--device` | auto | Inference device: `cpu`, `cuda:0`, `mps` |

### Output

| Flag | Default | Description |
|------|---------|-------------|
| `--annotated-port` | `8091` | HTTP port for the annotated MJPEG viewer |
| `--no-annotated-viewer` | off | Disable the annotated web viewer |
| `--jsonl` | (none) | JSONL output: `-` for stdout, or a file path |
| `--no-console` | off | Suppress human-readable detection summaries on stderr |
| `--record` | (none) | Record annotated video to an MP4 file |
| `--record-fps` | auto | Recording FPS (default: match model inference rate) |

### General

| Flag | Default | Description |
|------|---------|-------------|
| `-v, --verbose` | off | Enable debug-level logging |

## Environment variables

Every setting can also be set via an environment variable prefixed with
`AERIE_VISION_`.  CLI flags take precedence.

| Variable | Equivalent flag |
|----------|-----------------|
| `AERIE_VISION_SOURCE` | `--source` |
| `AERIE_VISION_VIEWER_PORT` | `--port` |
| `AERIE_VISION_MODEL_MAX_FPS` | `--model-fps` |
| `AERIE_VISION_VIEWER_ENABLED` | inverse of `--no-viewer` |
| `AERIE_VISION_LOOP_VIDEO_FILE` | inverse of `--no-loop` |
| `AERIE_VISION_RECONNECT_DELAY` | (no flag) seconds between retries |
| `AERIE_VISION_DETECTOR_MODEL` | `--model` |
| `AERIE_VISION_DETECTOR_CONFIDENCE` | `--confidence` |
| `AERIE_VISION_DETECTOR_CLASSES` | `--classes` |
| `AERIE_VISION_DETECTOR_DEVICE` | `--device` |
| `AERIE_VISION_ANNOTATED_VIEWER_PORT` | `--annotated-port` |
| `AERIE_VISION_ANNOTATED_VIEWER_ENABLED` | inverse of `--no-annotated-viewer` |
| `AERIE_VISION_JSONL_OUTPUT` | `--jsonl` |
| `AERIE_VISION_CONSOLE_OUTPUT` | inverse of `--no-console` |
| `AERIE_VISION_RECORD_PATH` | `--record` |
| `AERIE_VISION_RECORD_FPS` | `--record-fps` |

## Architecture

```
                                                           ┌──> ViewerSink (MJPEG @ ~30 fps)
                                                           │
VideoSource ──> FrameGrabber ──> FrameBus ──> FrameSlots ──┤
                                                           │
                                                           └──> ModelSink ──> DetectionRunner
                                                                (rate-limited)       │
                                                                              DetectionRouter
                                                                              ┌──────┼──────┐
                                                                              │      │      │
                                                                          Primary  Secs  Escalation
                                                                              │      │      │
                                                                              └──────┴──────┘
                                                                                     │
                                                                              merged results
                                                                                     │
                                                                              on_result callback
```

### Video ingest layer

- **FrameGrabber** -- decodes in a background thread at the source's native rate.
- **FrameBus** -- fans each frame out to independent consumer slots.
- **FrameSlot** -- holds only the latest frame (overwrite-on-put).  Slow
  consumers never stall the pipeline and always get the freshest frame.
- **ViewerSink** -- serves MJPEG over HTTP on the configured port.
- **ModelSink** -- rate-limited consumer that feeds the detection layer.

### Detection layer

- **Detector** (protocol) -- the interface every model backend implements:
  `detect(frame) -> list[RawDetection]`.  All backends are interchangeable.
- **DetectionRouter** -- orchestrates one or more detectors per frame and
  merges results via IoU-based NMS.  Supports three modes (see below).
- **DetectionRunner** -- daemon thread that pulls frames from `ModelSink`,
  passes them through the router, and delivers `DetectionResult`s via an
  `on_result` callback.

### Design decisions

**Model-agnostic protocol.**  The pipeline never knows what model it is
running.  Every backend (ultralytics, ONNX Runtime, a future TensorRT
wrapper, or a mock for testing) implements the same three-method `Detector`
protocol.

**Dependency isolation.**  Core types, protocol, mock detector, NMS, router,
and runner have zero ML dependencies -- only numpy.  `ultralytics` and
`onnxruntime` are imported lazily inside their respective detector classes and
are opt-in pip extras.  This means the full test suite runs in CI without
downloading model weights or installing PyTorch.

**Latest-frame slots, not queues.**  The ingest layer and detection layer are
decoupled by frame rate.  The video source may produce 30 fps, the viewer
consumes at display rate, and the model consumes at whatever rate inference
allows.  Each consumer always gets the freshest available frame; stale frames
are silently dropped.

## Detection

### Detector backends

#### UltralyticsDetector (development / training)

Wraps any model the `ultralytics` library can load: YOLOE (open-vocabulary),
YOLO26, YOLOv8, and anything they release next.

```python
from aerie_vision.detection.ultralytics_detector import UltralyticsDetector

# Closed-set detection with YOLO26
detector = UltralyticsDetector(model_path="yolo26n.pt", confidence=0.3)

# Zero-shot open-vocabulary with YOLOE
detector = UltralyticsDetector(
    model_path="yoloe-11s-seg.pt",
    confidence=0.25,
    classes=["fire", "smoke", "person on roof"],
)
```

Requires `uv sync --extra yolo`.

#### OnnxDetector (production deployment)

Uses ONNX Runtime directly -- no PyTorch dependency.  Works on CPU, CUDA,
TensorRT, CoreML, and DirectML via ONNX Runtime execution providers.

```python
from aerie_vision.detection.onnx_detector import OnnxDetector

detector = OnnxDetector(
    model_path="yolo26n.onnx",
    labels=["fire", "smoke", "person", "vehicle"],
    confidence=0.3,
)
```

Requires `uv sync --extra onnx`.  Export a model to ONNX with ultralytics:

```python
from ultralytics import YOLO
YOLO("yolo26n.pt").export(format="onnx")
```

#### MockDetector (testing)

Returns pre-scripted detections keyed by frame number.  No ML dependencies.

```python
from aerie_vision.detection.mock_detector import MockDetector
from aerie_vision.detection.types import RawDetection

detector = MockDetector(scripted={
    10: [RawDetection("fire", 0.95, (100, 100, 200, 200), "mock", 10, 0.0)],
    20: [RawDetection("smoke", 0.80, (50, 50, 150, 150), "mock", 20, 0.0)],
})
```

### Multi-model routing

The `DetectionRouter` supports three configurations:

**Single model** -- every frame goes to one detector.  Zero overhead.

```python
from aerie_vision.detection.router import DetectionRouter

router = DetectionRouter(primary=my_detector)
```

**Parallel ensemble** -- primary plus one or more secondary detectors run
concurrently on every frame.  Results are merged with IoU-based NMS (same-label
overlapping boxes are deduplicated, highest confidence wins).

```python
router = DetectionRouter(
    primary=general_detector,
    secondaries=[fire_specialist, structural_damage_model],
)
```

**Confidence-based escalation** -- a fast primary model runs on every frame.
When any detection falls below a confidence threshold, the same frame is
re-sent to a heavier escalation model for a second opinion.  Results are
merged.

```python
router = DetectionRouter(
    primary=UltralyticsDetector("yoloe-11s-seg.pt"),
    escalation=UltralyticsDetector("yoloe-11l-seg.pt"),
    escalation_threshold=0.5,
)
```

These modes compose: you can use parallel secondaries *and* escalation
together.

### Running detection independently

You can run a detector directly on a single image without the video pipeline:

```python
import cv2
from aerie_vision.detection.ultralytics_detector import UltralyticsDetector

detector = UltralyticsDetector("yolo26s.pt", confidence=0.3)
detector.warm_up()

frame = cv2.imread("test_image.jpg")
detections = detector.detect(frame, frame_number=0)

for d in detections:
    print(f"{d.label} ({d.confidence:.0%}) at {d.bbox}")
```

### Running detection as a sink for the video pipeline

Wire the detection layer into the ingest pipeline via `DetectionRunner`:

```python
from aerie_vision.config import PipelineConfig
from aerie_vision.pipeline import Pipeline
from aerie_vision.detection.router import DetectionRouter
from aerie_vision.detection.runner import DetectionRunner
from aerie_vision.detection.ultralytics_detector import UltralyticsDetector

# 1. Start the video ingest pipeline
config = PipelineConfig(source="0", model_max_fps=5.0)
pipeline = Pipeline(config)

# 2. Configure detection
detector = UltralyticsDetector(
    model_path="yoloe-11s-seg.pt",
    classes=["fire", "smoke", "person"],
)
router = DetectionRouter(primary=detector)

# 3. Define what happens with each detection result
def handle_result(result):
    for d in result.detections:
        print(f"Frame {d.frame_number}: {d.label} ({d.confidence:.0%})")

# 4. Wire them together and start
runner = DetectionRunner(
    model_sink=pipeline.model_sink,
    router=router,
    on_result=handle_result,
)

pipeline.start()
runner.start()
# ... Ctrl-C to stop ...
runner.stop()
pipeline.stop()
```

The `on_result` callback is the extension point where geocoding (Phase 4.3)
and event publishing (Phase 4.4) will plug in.

## Project layout

```
aerie_vision/
    __init__.py
    __main__.py             # CLI: python -m aerie_vision
    config.py               # PipelineConfig (Pydantic settings)
    capture.py              # FrameGrabber (cv2 in a thread)
    frame_bus.py            # FrameBus, FrameSlot, FrameMeta
    viewer.py               # MJPEG-over-HTTP ViewerSink
    model_sink.py           # Rate-limited ModelSink
    pipeline.py             # Ingest-layer Pipeline
    annotate.py             # Draw bboxes + labels onto frames
    text_output.py          # JsonlOutputStream + ConsoleOutputStream
    video_recorder.py       # VideoRecorder (cv2.VideoWriter wrapper)
    orchestrator.py         # VisionPipeline (ingest + detection + outputs)
    detection/
        __init__.py
        types.py            # RawDetection, DetectionResult
        protocol.py         # Detector protocol
        mock_detector.py    # Scripted mock (testing)
        ultralytics_detector.py  # YOLOE, YOLO26, YOLOv8, ...
        onnx_detector.py    # ONNX Runtime (production)
        nms.py              # IoU + cross-model merge
        router.py           # DetectionRouter (single / parallel / escalation)
        runner.py           # DetectionRunner (thread consuming ModelSink)
```

## Tests

```bash
uv sync --group dev
uv run pytest tests/ -v
```

All tests use `MockDetector` -- no model weights, no GPU, no `ultralytics`
import required.  Tests generate a short synthetic video on first run and
require no hardware, network, or display server.

For headless CI, swap in the `headless` extra (uses `opencv-python-headless`
instead of `opencv-python`):

```bash
uv sync --group dev --extra headless
```
