from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings


class PipelineConfig(BaseSettings):
    """Configuration for the aerie-vision pipeline.

    Values are read from environment variables prefixed with ``AERIE_VISION_``
    (e.g. ``AERIE_VISION_SOURCE=0``).
    """

    model_config = {"env_prefix": "AERIE_VISION_", "protected_namespaces": ()}

    # -- Video ingest ---------------------------------------------------------

    source: str = Field(
        default="0",
        description="Video source: device index, file path, or RTSP URL",
    )
    viewer_enabled: bool = Field(default=True, description="Enable the raw MJPEG web viewer")
    viewer_host: str = Field(default="0.0.0.0", description="Viewer HTTP bind address")
    viewer_port: int = Field(default=8090, ge=1, le=65535, description="Raw viewer HTTP port")
    model_max_fps: float = Field(
        default=5.0,
        gt=0,
        description="Maximum frame rate delivered to the detection model",
    )
    loop_video_file: bool = Field(
        default=True,
        description="Loop video files instead of stopping at end",
    )
    reconnect_delay: float = Field(
        default=2.0,
        gt=0,
        description="Seconds between reconnection attempts on source loss",
    )

    # -- Detection ------------------------------------------------------------

    detector_model: str = Field(
        default="rfdetr-medium",
        description=(
            'Detection model: "rfdetr-medium" (default), other RF-DETR sizes '
            '("rfdetr-nano/small/large/xlarge/2xlarge"), '
            'ultralytics model path ("yolo26n.pt", "yoloe-11s-seg.pt"), '
            'or "" to disable detection'
        ),
    )
    detector_confidence: float = Field(
        default=0.25,
        ge=0,
        le=1,
        description="Minimum detection confidence threshold",
    )
    detector_classes: list[str] = Field(
        default_factory=list,
        description="YOLOE open-vocab text prompts (e.g. fire, smoke)",
    )
    detector_device: str = Field(
        default="",
        description='Inference device: "" = auto, "cpu", "cuda:0", "mps"',
    )

    # -- Segmentation ---------------------------------------------------------

    segmenter_model: str = Field(
        default="",
        description=(
            'Segmentation model: "" = disabled, "sam3" = Meta SAM 3 via '
            'Ultralytics (requires `uv sync --extra sam3` and a manually-'
            'downloaded sam3.pt), "mock" = scripted segmenter for tests. '
            'When set, the detector is disabled (mutually exclusive).'
        ),
    )
    segmenter_classes: list[str] = Field(
        default_factory=list,
        description="Open-vocab text prompts for SAM3 (e.g. fire smoke person)",
    )
    segmenter_confidence: float = Field(
        default=0.3,
        ge=0,
        le=1,
        description="Minimum mask confidence threshold",
    )
    segmenter_device: str = Field(
        default="",
        description='Inference device: "" = auto, "cuda:0", "mps", "cpu"',
    )
    segmenter_jsonl_output: str = Field(
        default="",
        description=(
            '"" = disabled, "-" = stdout, or a file path for JSONL '
            'centroid-only segmentation records'
        ),
    )
    segmenter_weights: str = Field(
        default="sam3.pt",
        description=(
            'Path to the SAM 3 weights file. Anything containing "sam3" in '
            'the stem is routed to the SAM 3 builder by Ultralytics. Examples: '
            '"sam3.pt" (default), "sam3n.pt", or a path like '
            '"sam3_1/sam3.1_multiplex.pt" (note: SAM 3.1 multiplex is not '
            'fully supported by Ultralytics 8.3.237 — tracker weights load '
            'with strict=False and may be partially zero-initialized).'
        ),
    )

    # -- Annotated viewer -----------------------------------------------------

    annotated_viewer_enabled: bool = Field(
        default=True,
        description="Enable the annotated MJPEG web viewer (requires detection)",
    )
    annotated_viewer_port: int = Field(
        default=8091,
        ge=1,
        le=65535,
        description="Annotated viewer HTTP port",
    )

    # -- Text output ----------------------------------------------------------

    jsonl_output: str = Field(
        default="",
        description='"" = disabled, "-" = stdout, or a file path for JSONL detection records',
    )
    console_output: bool = Field(
        default=True,
        description="Print human-readable detection summaries to stderr",
    )

    # -- Video recording ------------------------------------------------------

    record_path: str = Field(
        default="",
        description='"" = disabled, otherwise output MP4 file path',
    )
    record_fps: float = Field(
        default=0.0,
        ge=0,
        description="Recording FPS (0 = match model inference rate)",
    )
