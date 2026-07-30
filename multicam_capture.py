"""Sequential RGB and depth capture from multiple RealSense cameras.

This module can be:

1. Run directly from the command line.
2. Imported into another Python program.

Only one camera pipeline is active at a time. Each camera is started, allowed to
warm up, used to capture one corresponding RGB/depth frameset, and then stopped
before the next camera is started.
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable

_MISSING_CAPTURE_DEPENDENCIES: list[str] = []

try:
    import cv2
except ModuleNotFoundError:
    cv2 = None  # type: ignore[assignment]
    _MISSING_CAPTURE_DEPENDENCIES.append("opencv-python")

try:
    import numpy as np
except ModuleNotFoundError:
    np = None  # type: ignore[assignment]
    _MISSING_CAPTURE_DEPENDENCIES.append("numpy")

try:
    import pyrealsense2 as rs
except ModuleNotFoundError:
    rs = None  # type: ignore[assignment]
    _MISSING_CAPTURE_DEPENDENCIES.append("pyrealsense2")

LOGGER = logging.getLogger(__name__)


def _require_capture_dependencies() -> None:
    if _MISSING_CAPTURE_DEPENDENCIES:
        missing = ", ".join(_MISSING_CAPTURE_DEPENDENCIES)
        raise RuntimeError(
            "Missing camera-capture dependency/dependencies: " + missing
        )


@dataclass(frozen=True)
class CameraDefinition:
    """One camera entry loaded from the configuration CSV file."""

    name: str
    serial_number: str

    @property
    def folder_name(self) -> str:
        """Return a filesystem-safe folder name that remains unique."""
        safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", self.name).strip("._-")
        if not safe_name:
            safe_name = "camera"
        return f"{safe_name}_{self.serial_number}"


@dataclass(frozen=True)
class CaptureResult:
    """Information returned after successfully saving one RGB/depth pair."""

    camera_name: str
    serial_number: str
    timestamp: str
    rgb_path: Path
    depth_path: Path
    color_frame_number: int
    depth_frame_number: int


@dataclass(frozen=True)
class CaptureFailure:
    """Information returned when a camera fails during a sweep."""

    camera_name: str
    serial_number: str
    error: str


@dataclass(frozen=True)
class SweepResult:
    """Results from one complete pass through the configured cameras."""

    captures: tuple[CaptureResult, ...]
    failures: tuple[CaptureFailure, ...]

    @property
    def succeeded(self) -> bool:
        return not self.failures


@dataclass(frozen=True)
class LoopResult:
    """Summary returned when repeated capture stops."""

    cycles_completed: int
    last_sweep: SweepResult | None


@dataclass(frozen=True)
class CaptureSettings:
    """Stream and capture settings shared by every configured camera."""

    output_root: Path = Path("captures")
    width: int = 1280
    height: int = 720
    fps: int = 5
    warmup_frames: int = 15
    frame_timeout_ms: int = 15_000
    device_wait_timeout_s: float = 20.0
    inter_camera_delay_s: float = 1.0
    align_depth_to_color: bool = True

    def validate(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("width and height must be positive")
        if self.fps <= 0:
            raise ValueError("fps must be positive")
        if self.warmup_frames < 0:
            raise ValueError("warmup_frames cannot be negative")
        if self.frame_timeout_ms <= 0:
            raise ValueError("frame_timeout_ms must be positive")
        if self.device_wait_timeout_s <= 0:
            raise ValueError("device_wait_timeout_s must be positive")
        if self.inter_camera_delay_s < 0:
            raise ValueError("inter_camera_delay_s cannot be negative")


def load_camera_definitions(csv_path: str | Path) -> list[CameraDefinition]:
    """Load and validate camera names and serial numbers from a CSV file.

    Required CSV header::

        camera_name,serial_number

    Blank rows are ignored. Camera names and serial numbers must be unique.
    """
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"Camera CSV file not found: {path}")

    cameras: list[CameraDefinition] = []

    with path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)

        if reader.fieldnames is None:
            raise ValueError(f"Camera CSV has no header: {path}")

        normalised_headers = {
            header.strip().lower(): header for header in reader.fieldnames if header
        }
        required_headers = {"camera_name", "serial_number"}
        missing = required_headers - normalised_headers.keys()
        if missing:
            missing_text = ", ".join(sorted(missing))
            raise ValueError(
                f"Camera CSV is missing required column(s): {missing_text}. "
                "Expected header: camera_name,serial_number"
            )

        name_column = normalised_headers["camera_name"]
        serial_column = normalised_headers["serial_number"]

        for row_number, row in enumerate(reader, start=2):
            camera_name = (row.get(name_column) or "").strip()
            serial_number = (row.get(serial_column) or "").strip()

            if not camera_name and not serial_number:
                continue

            if not camera_name or not serial_number:
                raise ValueError(
                    f"Incomplete camera entry on CSV row {row_number}: "
                    "both camera_name and serial_number are required"
                )

            cameras.append(
                CameraDefinition(name=camera_name, serial_number=serial_number)
            )

    if not cameras:
        raise ValueError(f"No camera entries were found in {path}")

    duplicate_names = _find_duplicates(camera.name for camera in cameras)
    if duplicate_names:
        raise ValueError(
            "Duplicate camera_name value(s): " + ", ".join(duplicate_names)
        )

    duplicate_serials = _find_duplicates(
        camera.serial_number for camera in cameras
    )
    if duplicate_serials:
        raise ValueError(
            "Duplicate serial_number value(s): " + ", ".join(duplicate_serials)
        )

    return cameras


def _find_duplicates(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return sorted(duplicates)


class SequentialCameraCapture:
    """Capture one RGB/depth pair per camera, one camera at a time."""

    def __init__(
        self,
        cameras: list[CameraDefinition],
        settings: CaptureSettings | None = None,
    ) -> None:
        if not cameras:
            raise ValueError("At least one camera must be configured")

        _require_capture_dependencies()

        self.cameras = list(cameras)
        self.settings = settings or CaptureSettings()
        self.settings.validate()

        self._context = rs.context()
        self._active_serial: str | None = None
        self._active_pipeline: rs.pipeline | None = None
        self._active_align: rs.align | None = None

    @classmethod
    def from_csv(
        cls,
        csv_path: str | Path,
        settings: CaptureSettings | None = None,
    ) -> "SequentialCameraCapture":
        """Construct a capture controller directly from a camera CSV file."""
        return cls(load_camera_definitions(csv_path), settings=settings)

    def __enter__(self) -> "SequentialCameraCapture":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.stop_active_camera()

    def _available_serial_numbers(self) -> set[str]:
        serials: set[str] = set()
        for device in self._context.query_devices():
            try:
                serials.add(device.get_info(rs.camera_info.serial_number))
            except RuntimeError:
                continue
        return serials

    def wait_for_camera(self, camera: CameraDefinition) -> None:
        """Wait until the selected network camera is visible to librealsense."""
        deadline = time.monotonic() + self.settings.device_wait_timeout_s

        while time.monotonic() < deadline:
            if camera.serial_number in self._available_serial_numbers():
                return
            time.sleep(0.25)

        visible = sorted(self._available_serial_numbers())
        visible_text = ", ".join(visible) if visible else "none"
        raise TimeoutError(
            f"Camera '{camera.name}' ({camera.serial_number}) did not become "
            f"available within {self.settings.device_wait_timeout_s:.1f} seconds. "
            f"Visible serial numbers: {visible_text}"
        )

    def start_camera(self, camera: CameraDefinition) -> None:
        """Start one camera; another camera must not already be streaming."""
        if self._active_pipeline is not None:
            raise RuntimeError(
                f"Camera {self._active_serial} is already active. "
                "Stop it before starting another camera."
            )

        self.wait_for_camera(camera)

        pipeline = rs.pipeline(self._context)
        config = rs.config()
        config.enable_device(camera.serial_number)
        config.enable_stream(
            rs.stream.color,
            self.settings.width,
            self.settings.height,
            rs.format.bgr8,
            self.settings.fps,
        )
        config.enable_stream(
            rs.stream.depth,
            self.settings.width,
            self.settings.height,
            rs.format.z16,
            self.settings.fps,
        )

        try:
            pipeline.start(config)
        except Exception:
            try:
                pipeline.stop()
            except Exception:
                pass
            raise

        self._active_serial = camera.serial_number
        self._active_pipeline = pipeline
        self._active_align = rs.align(rs.stream.color)
        LOGGER.info(
            "Started camera '%s' (%s)", camera.name, camera.serial_number
        )

    def stop_active_camera(self) -> None:
        """Stop the current stream while leaving the camera powered over PoE."""
        pipeline = self._active_pipeline
        serial = self._active_serial

        self._active_pipeline = None
        self._active_align = None
        self._active_serial = None

        if pipeline is None:
            return

        try:
            pipeline.stop()
            LOGGER.info("Stopped camera %s", serial)
        except Exception as exc:
            LOGGER.warning("Failed to stop camera %s cleanly: %s", serial, exc)

    def capture_camera(self, camera: CameraDefinition) -> CaptureResult:
        """Start one camera, save one pair, and stop it in all circumstances."""
        self.start_camera(camera)
        try:
            return self._capture_from_active_camera(camera)
        finally:
            self.stop_active_camera()

    def _capture_from_active_camera(
        self, camera: CameraDefinition
    ) -> CaptureResult:
        if (
            self._active_pipeline is None
            or self._active_serial != camera.serial_number
        ):
            raise RuntimeError(
                f"Camera '{camera.name}' ({camera.serial_number}) is not active"
            )

        pipeline = self._active_pipeline

        LOGGER.info(
            "Warming up camera '%s' with %d frame(s)",
            camera.name,
            self.settings.warmup_frames,
        )
        for _ in range(self.settings.warmup_frames):
            pipeline.wait_for_frames(self.settings.frame_timeout_ms)

        frames = pipeline.wait_for_frames(self.settings.frame_timeout_ms)
        if self.settings.align_depth_to_color:
            if self._active_align is None:
                raise RuntimeError("Depth alignment object is unavailable")
            frames = self._active_align.process(frames)

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            raise RuntimeError(
                f"Camera '{camera.name}' did not return both RGB and depth"
            )

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        if color_image.dtype != np.uint8:
            raise RuntimeError(
                f"Unexpected RGB dtype from '{camera.name}': {color_image.dtype}"
            )
        if depth_image.dtype != np.uint16:
            raise RuntimeError(
                f"Unexpected depth dtype from '{camera.name}': "
                f"{depth_image.dtype}"
            )

        timestamp = datetime.now().astimezone().strftime(
            "%Y%m%d_%H%M%S_%f"
        )[:-3]

        camera_root = self.settings.output_root / camera.folder_name
        rgb_dir = camera_root / "rgb"
        depth_dir = camera_root / "depth"
        rgb_dir.mkdir(parents=True, exist_ok=True)
        depth_dir.mkdir(parents=True, exist_ok=True)

        rgb_path = rgb_dir / f"{timestamp}.png"
        depth_path = depth_dir / f"{timestamp}.png"

        if not cv2.imwrite(str(rgb_path), color_image):
            raise OSError(f"Failed to save RGB image: {rgb_path}")

        if not cv2.imwrite(str(depth_path), depth_image):
            rgb_path.unlink(missing_ok=True)
            raise OSError(f"Failed to save depth image: {depth_path}")

        result = CaptureResult(
            camera_name=camera.name,
            serial_number=camera.serial_number,
            timestamp=timestamp,
            rgb_path=rgb_path,
            depth_path=depth_path,
            color_frame_number=color_frame.get_frame_number(),
            depth_frame_number=depth_frame.get_frame_number(),
        )

        LOGGER.info(
            "Saved '%s': RGB=%s, depth=%s",
            camera.name,
            rgb_path,
            depth_path,
        )
        return result

    def run_once(self, *, continue_on_error: bool = True) -> SweepResult:
        """Capture every configured camera once and then return.

        Each camera is stopped before the next camera starts. This keeps only one
        high-bandwidth RGB/depth stream active on the network at a time.
        """
        captures: list[CaptureResult] = []
        failures: list[CaptureFailure] = []

        try:
            for index, camera in enumerate(self.cameras):
                LOGGER.info(
                    "Capturing camera %d/%d: '%s' (%s)",
                    index + 1,
                    len(self.cameras),
                    camera.name,
                    camera.serial_number,
                )

                try:
                    captures.append(self.capture_camera(camera))
                except Exception as exc:
                    failure = CaptureFailure(
                        camera_name=camera.name,
                        serial_number=camera.serial_number,
                        error=str(exc),
                    )
                    failures.append(failure)
                    LOGGER.exception(
                        "Capture failed for '%s' (%s)",
                        camera.name,
                        camera.serial_number,
                    )
                    if not continue_on_error:
                        raise
                finally:
                    self.stop_active_camera()

                if (
                    index < len(self.cameras) - 1
                    and self.settings.inter_camera_delay_s > 0
                ):
                    time.sleep(self.settings.inter_camera_delay_s)
        finally:
            self.stop_active_camera()

        return SweepResult(
            captures=tuple(captures),
            failures=tuple(failures),
        )

    def run_loop(
        self,
        *,
        cycle_delay_s: float = 0.0,
        max_cycles: int | None = None,
        stop_event: threading.Event | None = None,
        continue_on_error: bool = True,
        on_sweep: Callable[[int, SweepResult], None] | None = None,
    ) -> LoopResult:
        """Repeatedly perform complete sequential camera sweeps.

        The loop continues until one of the following occurs:

        - ``max_cycles`` sweeps have completed.
        - ``stop_event`` is set by the calling application.
        - The user presses Ctrl+C when running from the command line.

        ``on_sweep`` is called after each completed sweep and can be used by an
        importing application to update a database, UI, or status log. Only the
        latest sweep is retained, so an indefinite loop does not consume steadily
        increasing memory.
        """
        if cycle_delay_s < 0:
            raise ValueError("cycle_delay_s cannot be negative")
        if max_cycles is not None and max_cycles <= 0:
            raise ValueError("max_cycles must be positive or None")

        event = stop_event or threading.Event()
        cycle_number = 0
        last_sweep: SweepResult | None = None

        try:
            while not event.is_set():
                if max_cycles is not None and cycle_number >= max_cycles:
                    break

                cycle_number += 1
                LOGGER.info("Starting camera sweep %d", cycle_number)
                last_sweep = self.run_once(
                    continue_on_error=continue_on_error
                )
                LOGGER.info(
                    "Completed sweep %d: %d capture(s), %d failure(s)",
                    cycle_number,
                    len(last_sweep.captures),
                    len(last_sweep.failures),
                )

                if on_sweep is not None:
                    on_sweep(cycle_number, last_sweep)

                if max_cycles is not None and cycle_number >= max_cycles:
                    break

                if cycle_delay_s > 0 and event.wait(cycle_delay_s):
                    break
        finally:
            self.stop_active_camera()

        return LoopResult(
            cycles_completed=cycle_number,
            last_sweep=last_sweep,
        )


def capture_all_once(
    camera_csv: str | Path,
    *,
    settings: CaptureSettings | None = None,
    continue_on_error: bool = True,
) -> SweepResult:
    """Library convenience function: capture every CSV camera exactly once."""
    with SequentialCameraCapture.from_csv(
        camera_csv, settings=settings
    ) as capture_controller:
        return capture_controller.run_once(
            continue_on_error=continue_on_error
        )


def capture_all_loop(
    camera_csv: str | Path,
    *,
    settings: CaptureSettings | None = None,
    cycle_delay_s: float = 0.0,
    max_cycles: int | None = None,
    stop_event: threading.Event | None = None,
    continue_on_error: bool = True,
    on_sweep: Callable[[int, SweepResult], None] | None = None,
) -> LoopResult:
    """Library convenience function: repeatedly capture every CSV camera."""
    with SequentialCameraCapture.from_csv(
        camera_csv, settings=settings
    ) as capture_controller:
        return capture_controller.run_loop(
            cycle_delay_s=cycle_delay_s,
            max_cycles=max_cycles,
            stop_event=stop_event,
            continue_on_error=continue_on_error,
            on_sweep=on_sweep,
        )


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Capture matching RGB/depth image pairs sequentially from "
            "RealSense cameras listed in a CSV file."
        )
    )
    parser.add_argument(
        "--csv",
        default="cameras.csv",
        help="Camera configuration CSV (default: cameras.csv)",
    )
    parser.add_argument(
        "--mode",
        choices=("once", "loop"),
        default="once",
        help="Run one sweep or repeat sweeps until stopped (default: once)",
    )
    parser.add_argument(
        "--output",
        default="captures",
        help="Root output directory (default: captures)",
    )
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--warmup-frames", type=int, default=15)
    parser.add_argument("--frame-timeout-ms", type=int, default=15_000)
    parser.add_argument("--device-wait-timeout", type=float, default=20.0)
    parser.add_argument("--inter-camera-delay", type=float, default=1.0)
    parser.add_argument(
        "--cycle-delay",
        type=float,
        default=0.0,
        help="Seconds between complete sweeps in loop mode",
    )
    parser.add_argument(
        "--max-cycles",
        type=int,
        default=None,
        help="Optional number of sweeps in loop mode; omit to run indefinitely",
    )
    parser.add_argument(
        "--no-align",
        action="store_true",
        help="Save native depth without aligning it to the RGB image",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop a sweep immediately if one camera fails",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point; importing this module does not execute it."""
    args = _build_argument_parser().parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    settings = CaptureSettings(
        output_root=Path(args.output),
        width=args.width,
        height=args.height,
        fps=args.fps,
        warmup_frames=args.warmup_frames,
        frame_timeout_ms=args.frame_timeout_ms,
        device_wait_timeout_s=args.device_wait_timeout,
        inter_camera_delay_s=args.inter_camera_delay,
        align_depth_to_color=not args.no_align,
    )

    try:
        if args.mode == "once":
            result = capture_all_once(
                args.csv,
                settings=settings,
                continue_on_error=not args.fail_fast,
            )
            LOGGER.info(
                "Finished: %d capture(s), %d failure(s)",
                len(result.captures),
                len(result.failures),
            )
            return 0 if result.succeeded else 1

        capture_all_loop(
            args.csv,
            settings=settings,
            cycle_delay_s=args.cycle_delay,
            max_cycles=args.max_cycles,
            continue_on_error=not args.fail_fast,
        )
        return 0

    except KeyboardInterrupt:
        LOGGER.info("Capture stopped by user")
        return 130
    except Exception:
        LOGGER.exception("Fatal capture error")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
