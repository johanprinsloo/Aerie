"""Sam3SourceRunner: source-driven SAM 3 segmentation + tracking via Ultralytics.

SAM 3 in Ultralytics (>=8.3.237) is a source-driven, stateful video tracker —
it owns ingest via ``model.track(source=..., stream=True)`` rather than
accepting per-frame numpy arrays. This runner wraps that loop, republishes
each captured frame to the pipeline's :class:`FrameBus` (so the raw viewer
on :8090 still works), and converts each ``Results`` into a
:class:`SegmentationResult` for the orchestrator's ``on_result`` callback.

Install via the optional extra::

    uv sync --extra sam3

The SAM 3 weights (``sam3.pt``, ``sam3n.pt``, etc.) and the BPE vocabulary
(``bpe_simple_vocab_16e6.txt.gz``) must be downloaded manually; they are NOT
auto-fetched by Ultralytics. See https://huggingface.co/facebook/sam3.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable, Iterable

import cv2
import numpy as np

from ..frame_bus import FrameBus, FrameMeta
from .types import InstanceMask, SegmentationResult

logger = logging.getLogger(__name__)


_INSTALL_HINT = (
    "ultralytics>=8.3.237 is required for SAM 3. Install with: "
    "uv sync --extra sam3. SAM 3 weights (sam3.pt) and the BPE vocab must be "
    "downloaded manually from https://huggingface.co/facebook/sam3"
)

_WEIGHTS_HINT = (
    "SAM 3 weights file not found. Ultralytics does NOT auto-download SAM 3 "
    "weights. Download sam3.pt (or sam3n.pt) and bpe_simple_vocab_16e6.txt.gz "
    "from https://huggingface.co/facebook/sam3 (you must accept the gated "
    "model terms first) and place sam3.pt in the working directory."
)


class Sam3SourceRunner:
    """Drive the Ultralytics SAM 3 video tracker against a live source.

    Unlike :class:`~segmentation.runner.SegmentationRunner`, this runner does
    not consume from a :class:`ModelSink` — Ultralytics' ``model.track(source=...,
    stream=True)`` owns ingest itself. The Pipeline must therefore be
    constructed with ``external_source=True`` so :class:`FrameGrabber` is not
    started (two cv2 captures of the same device would conflict).

    Parameters
    ----------
    source:
        Anything ``model.track`` accepts: device index ("0"), file path,
        RTSP URL, etc.
    prompts:
        Open-vocabulary text prompts (e.g. ``["fire", "smoke"]``).
    confidence:
        Minimum mask confidence threshold passed to Ultralytics.
    device:
        Inference device — ``""`` (auto), ``"cuda:0"``, ``"mps"``, ``"cpu"``.
    bus:
        :class:`FrameBus` to republish captured frames to (so the raw MJPEG
        viewer on :8090 still works in SAM 3 mode).
    on_result:
        Called with each :class:`SegmentationResult`.
    model_path:
        SAM 3 checkpoint filename (``"sam3.pt"`` or ``"sam3n.pt"``).
    """

    def __init__(
        self,
        source: str,
        prompts: list[str],
        confidence: float,
        device: str,
        bus: FrameBus,
        on_result: Callable[[SegmentationResult], None] | None = None,
        model_path: str = "sam3.pt",
    ) -> None:
        self._source = source
        self._prompts = list(prompts)
        self._confidence = confidence
        self._device = device
        self._bus = bus
        self._on_result = on_result
        self._model_path = model_path

        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._frames_processed = 0
        self._total_inference_ms = 0.0
        self._frame_counter = 0

    # -- public API -----------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="sam3-source-runner"
        )
        self._thread.start()
        logger.info("Sam3SourceRunner started")

    def stop(self, timeout: float = 5.0) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        logger.info(
            "Sam3SourceRunner stopped  frames=%d  avg_inference=%.1fms",
            self._frames_processed,
            self.avg_inference_ms,
        )

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def frames_processed(self) -> int:
        return self._frames_processed

    @property
    def avg_inference_ms(self) -> float:
        if self._frames_processed == 0:
            return 0.0
        return self._total_inference_ms / self._frames_processed

    # -- internals ------------------------------------------------------------

    def _run(self) -> None:
        # IMPORTANT — what works on live streams in Ultralytics 8.3.237:
        #
        # SAM3VideoSemanticPredictor (text-prompt video tracking) requires
        # ``dataset.mode == "video"`` — it works only for video FILES with a
        # known frame count, NOT for webcam/RTSP. SAM3Predictor (image mode,
        # what SAM("sam3.pt") routes to) DOES work on live streams: when
        # called without prompts its ``inference()`` falls through to
        # ``generate()`` (segment-everything mode), which samples a grid of
        # points across each frame and returns one mask per detected object.
        #
        # Tradeoffs:
        #   * No persistent instance IDs across frames (no tracker).
        #   * No text prompting through this path; ``self._prompts`` is
        #     ignored (logged below if set).
        #   * Slow on CPU/MPS — expect seconds to tens of seconds per frame
        #     because generate() runs SAM forward for every grid point.
        #
        # For text-prompt + tracking on live video, YOLOE-seg via the
        # existing detector path (--model yoloe-11s-seg.pt --classes ...)
        # is the working alternative today.
        try:
            from ultralytics import SAM
        except ImportError as exc:
            logger.exception(_INSTALL_HINT)
            raise ImportError(_INSTALL_HINT) from exc

        if self._prompts:
            logger.warning(
                "SAM 3 segmenter_classes=%s ignored: Ultralytics 8.3.237 has "
                "no live-stream text-prompt path. Running segment-everything.",
                self._prompts,
            )

        try:
            model = SAM(self._model_path)
        except FileNotFoundError as exc:
            logger.error("%s (looking for %r)", _WEIGHTS_HINT, self._model_path)
            raise FileNotFoundError(_WEIGHTS_HINT) from exc

        kwargs: dict[str, Any] = {
            "source": self._source,
            "stream": True,
            "conf": self._confidence,
            "verbose": False,
            "save": False,
            "show": False,
        }
        if self._device:
            kwargs["device"] = self._device

        logger.info(
            "SAM 3 starting (segment-everything mode): model=%s source=%s "
            "[expect slow per-frame inference on CPU/MPS]",
            self._model_path,
            self._source,
        )

        for result in model(**kwargs):
            if self._stop_event.is_set():
                break
            self._handle_result(result)

    def _handle_result(self, result: Any) -> None:
        t0 = time.monotonic()
        frame = result.orig_img
        if frame is None:
            return

        instances = ultralytics_result_to_instances(result)
        elapsed_ms = (time.monotonic() - t0) * 1000.0
        timestamp = time.monotonic()
        self._frame_counter += 1

        meta = FrameMeta(
            frame_number=self._frame_counter,
            timestamp=timestamp,
            source_fps=0.0,
            width=frame.shape[1],
            height=frame.shape[0],
        )
        self._bus.publish(frame, meta)

        seg_result = SegmentationResult(
            frame_number=self._frame_counter,
            timestamp=timestamp,
            instances=tuple(instances),
            frame=frame,
            inference_ms=elapsed_ms,
        )
        self._frames_processed += 1
        self._total_inference_ms += elapsed_ms

        if self._on_result is not None:
            try:
                self._on_result(seg_result)
            except Exception:
                logger.exception(
                    "on_result callback failed for frame %d", self._frame_counter
                )


def ultralytics_result_to_instances(result: Any) -> list[InstanceMask]:
    """Convert one Ultralytics ``Results`` into a list of :class:`InstanceMask`.

    Tolerates missing ``masks``, missing ``boxes.id`` (no tracker active),
    and torch tensors vs plain numpy. Used as a module-level helper so it
    can be unit-tested against mock objects without instantiating the runner.
    """
    masks_obj = getattr(result, "masks", None)
    boxes_obj = getattr(result, "boxes", None)
    if masks_obj is None or boxes_obj is None:
        return []

    masks_data = getattr(masks_obj, "data", None)
    if masks_data is None:
        return []
    masks = _to_numpy(masks_data).astype(bool)
    if masks.ndim != 3 or masks.shape[0] == 0:
        return []

    confs = _to_numpy(boxes_obj.conf)
    clss = _to_numpy(boxes_obj.cls).astype(int)
    ids_attr = getattr(boxes_obj, "id", None)
    if ids_attr is not None:
        ids = _to_numpy(ids_attr).astype(int)
    else:
        ids = np.arange(len(masks), dtype=int)

    names = result.names if isinstance(result.names, dict) else {
        i: str(v) for i, v in enumerate(result.names)
    }

    out: list[InstanceMask] = []
    for i in range(len(masks)):
        inst = _build_instance(
            instance_id=int(ids[i]),
            label=str(names.get(int(clss[i]), str(int(clss[i])))),
            confidence=float(confs[i]),
            mask=masks[i],
        )
        if inst is not None:
            out.append(inst)
    return out


def _to_numpy(x: Any) -> np.ndarray:
    """Coerce a torch tensor or numpy array to numpy without importing torch."""
    cpu_fn = getattr(x, "cpu", None)
    if callable(cpu_fn):
        x = cpu_fn()
    numpy_fn = getattr(x, "numpy", None)
    if callable(numpy_fn):
        return numpy_fn()
    return np.asarray(x)


def _build_instance(
    instance_id: int, label: str, confidence: float, mask: np.ndarray
) -> InstanceMask | None:
    """Compute bbox, centroid, area for a boolean mask. Returns None if empty."""
    if mask.size == 0 or not mask.any():
        return None

    mask_u8 = mask.astype(np.uint8)
    x, y, w, h = cv2.boundingRect(mask_u8)
    moments = cv2.moments(mask_u8, binaryImage=True)
    area = int(moments["m00"])
    if area == 0:
        return None
    cx = int(moments["m10"] / area)
    cy = int(moments["m01"] / area)

    return InstanceMask(
        instance_id=instance_id,
        label=label,
        confidence=confidence,
        bbox=(int(x), int(y), int(x + w), int(y + h)),
        centroid=(cx, cy),
        area_px=area,
        mask=mask,
    )


__all__: Iterable[str] = ("Sam3SourceRunner", "ultralytics_result_to_instances")
