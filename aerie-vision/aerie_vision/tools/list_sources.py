"""TUI utility: list active video sources and optionally capture snapshots."""

from __future__ import annotations

import argparse
import os
import platform
import subprocess
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


def _suppress_cv_stderr() -> tuple[int, int]:
    """Redirect fd 2 to /dev/null, return (saved_fd, devnull_fd)."""
    saved = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 2)
    return saved, devnull


def _restore_cv_stderr(saved: int, devnull: int) -> None:
    os.dup2(saved, 2)
    os.close(saved)
    os.close(devnull)


def _probe_device(index: int) -> dict | None:
    """Open an OpenCV device by index and read its properties."""
    saved, devnull = _suppress_cv_stderr()
    try:
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            cap.release()
            return None

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        backend = cap.getBackendName()
        fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
        fourcc = "".join(chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)) if fourcc_int else ""

        ok, frame = cap.read()
        cap.release()

        if not ok and w == 0 and h == 0:
            return None

        return {
            "index": index,
            "width": w,
            "height": h,
            "fps": fps if fps > 0 else 0.0,
            "backend": backend,
            "fourcc": fourcc.strip(),
            "readable": ok,
            "frame": frame if ok else None,
        }
    finally:
        _restore_cv_stderr(saved, devnull)


def _capture_snapshot(index: int, label: str = "") -> np.ndarray | None:
    """Capture a single frame and burn the device index into it."""
    saved, devnull = _suppress_cv_stderr()
    try:
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            cap.release()
            return None
        # Discard a few frames to let auto-exposure settle
        for _ in range(5):
            cap.read()
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            return None

        tag = f"--source {index}"
        if label:
            tag += f"  ({label})"
        cv2.putText(frame, tag, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 255), 4)
        cv2.putText(frame, tag, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 255), 2)
        return frame
    finally:
        _restore_cv_stderr(saved, devnull)


# ---------------------------------------------------------------------------
# Snapshot web page
# ---------------------------------------------------------------------------

def _generate_snapshot_page(snapshot_dir: Path, devices: list[dict]) -> Path:
    """Save snapshots as JPEGs and create an HTML index page."""
    parts: list[str] = []
    for dev in devices:
        idx = dev["index"]
        frame = dev.get("snapshot")
        if frame is None:
            parts.append(f'<div class="card"><h2>--source {idx}</h2><p>No frame captured</p></div>')
            continue

        jpg_name = f"source_{idx}.jpg"
        cv2.imwrite(str(snapshot_dir / jpg_name), frame)
        res = f"{dev['width']}x{dev['height']}"
        fps = f"{dev['fps']:.0f}" if dev["fps"] > 0 else "?"
        parts.append(
            f'<div class="card">'
            f'<h2>--source {idx}</h2>'
            f'<img src="{jpg_name}" alt="source {idx}">'
            f'<p>{res} @ {fps}fps &mdash; {dev["backend"]}</p>'
            f'</div>'
        )

    html = f"""\
<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Aerie Source Snapshots</title>
<style>
body {{ font-family: system-ui; background: #1a1a2e; color: #eee; margin: 2rem; }}
h1 {{ text-align: center; }}
.grid {{ display: flex; flex-wrap: wrap; gap: 1.5rem; justify-content: center; }}
.card {{ background: #16213e; border-radius: 12px; padding: 1rem; max-width: 640px; }}
.card h2 {{ color: #0ff; margin: 0 0 0.5rem; font-family: monospace; }}
.card img {{ width: 100%; border-radius: 8px; }}
.card p {{ margin: 0.5rem 0 0; color: #aaa; text-align: center; }}
</style></head>
<body><h1>Aerie Video Source Snapshots</h1>
<p style="text-align:center;color:#888;">Each image shows what <code>--source N</code> captures right now.</p>
<div class="grid">{''.join(parts)}</div>
</body></html>"""

    page = snapshot_dir / "index.html"
    page.write_text(html)
    return page


def _run_snapshot_mode(max_index: int, console: Console) -> None:
    """Capture a snapshot from each device, save as JPEG, serve as HTML."""
    console.print("[bold]Capturing snapshots from each device...[/bold]\n")

    devices: list[dict] = []
    for i in range(max_index):
        info = _probe_device(i)
        if info is not None:
            res = f"{info['width']}x{info['height']}"
            console.print(f"  Device {i}: {res} @ {info['fps']:.0f}fps — capturing snapshot...")
            snapshot = _capture_snapshot(i, f"{res} {info['fps']:.0f}fps")
            info["snapshot"] = snapshot
            devices.append(info)

    if not devices:
        console.print("\n[red]No video devices found.[/red]")
        return

    snapshot_dir = Path(tempfile.mkdtemp(prefix="aerie_snapshots_"))
    page = _generate_snapshot_page(snapshot_dir, devices)

    console.print(f"\n[bold green]Saved {len(devices)} snapshots to:[/bold green]  {snapshot_dir}")
    console.print(f"[bold]Open in browser:[/bold]  file://{page}")

    # Try to open automatically
    import webbrowser
    webbrowser.open(f"file://{page}")


# ---------------------------------------------------------------------------
# TUI table rendering
# ---------------------------------------------------------------------------

def _build_devices_table(devices: list[dict], scan_range: int) -> Table:
    table = Table(
        title="Active Video Devices  (use Index as --source N)",
        show_lines=True,
        expand=True,
    )
    table.add_column("Index", style="bold cyan", width=6, justify="center")
    table.add_column("Resolution", justify="center", width=14)
    table.add_column("FPS", justify="center", width=8)
    table.add_column("FourCC", justify="center", width=10)
    table.add_column("Backend", justify="center", width=14)
    table.add_column("Status", justify="center", width=10)
    table.add_column("Pipeline command", style="dim", min_width=20)

    if not devices:
        table.add_row(
            "--", Text("No video devices found", style="dim italic"),
            "--", "--", "--", Text("--", style="dim"), "",
        )
        return table

    for dev in devices:
        res = f"{dev['width']}x{dev['height']}"
        fps = f"{dev['fps']:.0f}" if dev["fps"] > 0 else "?"
        status = Text("Active", style="bold green") if dev["readable"] else Text("Busy", style="yellow")
        cli = f"--source {dev['index']}"
        table.add_row(str(dev["index"]), res, fps, dev["fourcc"], dev["backend"], status, cli)

    return table


def _build_display(devices: list[dict], scan_range: int) -> object:
    from rich.console import Group
    parts: list[object] = [_build_devices_table(devices, scan_range)]
    parts.append(Text(
        "\nRun with --snapshot to visually identify which camera is at each index",
        style="dim italic",
    ))
    parts.append(Text(f"Scanning devices 0-{scan_range - 1}  |  Press Ctrl-C to exit", style="dim"))
    return Group(*parts)


def _run_scan_mode(max_index: int, interval: float, console: Console) -> None:
    with Live(console=console, refresh_per_second=1) as live:
        try:
            while True:
                devices = []
                for i in range(max_index):
                    info = _probe_device(i)
                    if info is not None:
                        devices.append(info)

                display = _build_display(devices, max_index)
                live.update(display)
                time.sleep(interval)
        except KeyboardInterrupt:
            pass
    console.print("\n[dim]Scanner stopped.[/dim]")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="aerie-list-sources",
        description="Scan video sources. Use --snapshot to visually identify each device.",
    )
    parser.add_argument(
        "--max-index", type=int, default=10,
        help="Highest device index to probe (default: 10)",
    )
    parser.add_argument(
        "--interval", type=float, default=3.0,
        help="Seconds between scans in TUI mode (default: 3.0)",
    )
    parser.add_argument(
        "--snapshot", action="store_true",
        help="Capture a frame from each device and open a web page showing all snapshots",
    )
    args = parser.parse_args(argv)

    console = Console()
    console.print("[bold]Aerie Video Source Scanner[/bold]\n")

    if args.snapshot:
        _run_snapshot_mode(args.max_index, console)
    else:
        _run_scan_mode(args.max_index, args.interval, console)


if __name__ == "__main__":
    main()
