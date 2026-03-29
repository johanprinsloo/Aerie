"""CLI entry point: ``python -m aerie_vision``."""

from __future__ import annotations

import argparse
import logging
import signal
import sys

from .config import PipelineConfig
from .orchestrator import VisionPipeline


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="aerie-vision",
        description="Video ingest + ML detection pipeline for Aerie drone supervision",
    )

    # -- Video ingest ---------------------------------------------------------
    ingest = parser.add_argument_group("video ingest")
    ingest.add_argument(
        "--source", default=None,
        help="Video source: device index, file path, or RTSP URL (default: 0)",
    )
    ingest.add_argument(
        "--port", type=int, default=None,
        help="Raw MJPEG viewer HTTP port (default: 8090)",
    )
    ingest.add_argument(
        "--model-fps", type=float, default=None,
        help="Max FPS for the model sink (default: 5.0)",
    )
    ingest.add_argument(
        "--no-viewer", action="store_true",
        help="Disable the raw MJPEG web viewer",
    )
    ingest.add_argument(
        "--no-loop", action="store_true",
        help="Don't loop video files",
    )

    # -- Detection ------------------------------------------------------------
    det = parser.add_argument_group("detection")
    det.add_argument(
        "--model", default=None,
        help='Detection model path (e.g. "yoloe-11s.pt"), or "mock" for testing',
    )
    det.add_argument(
        "--confidence", type=float, default=None,
        help="Minimum detection confidence (default: 0.25)",
    )
    det.add_argument(
        "--classes", nargs="*", default=None,
        help="YOLOE open-vocab class prompts (e.g. fire smoke person)",
    )
    det.add_argument(
        "--device", default=None,
        help='Inference device: "" = auto, "cpu", "cuda:0", "mps"',
    )

    # -- Output ---------------------------------------------------------------
    out = parser.add_argument_group("output")
    out.add_argument(
        "--annotated-port", type=int, default=None,
        help="Annotated MJPEG viewer port (default: 8091)",
    )
    out.add_argument(
        "--no-annotated-viewer", action="store_true",
        help="Disable the annotated web viewer",
    )
    out.add_argument(
        "--jsonl", default=None,
        help='JSONL output: "-" for stdout, or a file path',
    )
    out.add_argument(
        "--no-console", action="store_true",
        help="Suppress human-readable console output",
    )
    out.add_argument(
        "--record", default=None,
        help="Record annotated video to an MP4 file",
    )
    out.add_argument(
        "--record-fps", type=float, default=None,
        help="Recording FPS (default: match model rate)",
    )

    # -- General --------------------------------------------------------------
    parser.add_argument(
        "--overlay", action="store_true",
        help="Overlay frame metadata on the raw viewer",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    overrides: dict[str, object] = {}
    if args.source is not None:
        overrides["source"] = args.source
    if args.port is not None:
        overrides["viewer_port"] = args.port
    if args.model_fps is not None:
        overrides["model_max_fps"] = args.model_fps
    if args.no_viewer:
        overrides["viewer_enabled"] = False
    if args.no_loop:
        overrides["loop_video_file"] = False
    if args.model is not None:
        overrides["detector_model"] = args.model
    if args.confidence is not None:
        overrides["detector_confidence"] = args.confidence
    if args.classes is not None:
        overrides["detector_classes"] = args.classes
    if args.device is not None:
        overrides["detector_device"] = args.device
    if args.annotated_port is not None:
        overrides["annotated_viewer_port"] = args.annotated_port
    if args.no_annotated_viewer:
        overrides["annotated_viewer_enabled"] = False
    if args.jsonl is not None:
        overrides["jsonl_output"] = args.jsonl
    if args.no_console:
        overrides["console_output"] = False
    if args.record is not None:
        overrides["record_path"] = args.record
    if args.record_fps is not None:
        overrides["record_fps"] = args.record_fps

    config = PipelineConfig(**overrides)  # type: ignore[arg-type]
    vision = VisionPipeline(config)

    shutdown = False

    def _signal_handler(sig: int, frame: object) -> None:
        nonlocal shutdown
        if shutdown:
            sys.exit(1)
        shutdown = True
        logging.info("Shutting down…")
        vision.stop()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    vision.start()

    logging.info("VisionPipeline running. Press Ctrl-C to stop.")
    if config.viewer_enabled:
        logging.info("Raw viewer:       http://localhost:%d/", config.viewer_port)
    if config.detector_model and config.annotated_viewer_enabled:
        logging.info("Annotated viewer: http://localhost:%d/", config.annotated_viewer_port)

    try:
        signal.pause()
    except AttributeError:
        import time
        while not shutdown:
            time.sleep(0.5)


if __name__ == "__main__":
    main()
