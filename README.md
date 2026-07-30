# RS D555 Multi-Camera Manager

Capture matching RGB and depth images from multiple Intel RealSense cameras
without streaming from every camera at the same time.

The manager processes cameras sequentially: it starts one camera, waits for the
stream to stabilise, saves an RGB/depth pair, stops that camera, and then moves
to the next one. This is useful when network or USB bandwidth cannot support all
configured streams concurrently.

## Features

- One-shot capture or continuous capture loops
- Camera selection by serial number from a CSV file
- Matching RGB and 16-bit depth PNG files
- Optional depth-to-colour alignment
- Configurable resolution, frame rate, timeouts, and delays
- Per-camera error reporting, with optional fail-fast behaviour
- Command-line and Python APIs
- Guaranteed pipeline cleanup when capture finishes or fails

## Requirements

- Python 3.10 or newer
- Intel RealSense cameras visible to `librealsense`
- Intel RealSense SDK 2.0 / `librealsense`
- Python packages:
  - `numpy`
  - `opencv-python`
  - `pyrealsense2`

Create a virtual environment and install the Python dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy opencv-python pyrealsense2
```

> `pyrealsense2` availability depends on your platform and Python version. If a
> compatible wheel is unavailable, install it using the instructions for the
> Intel RealSense SDK on your system.

## Camera configuration

Edit `cameras.csv` and add one row per camera:

```csv
camera_name,serial_number
Camera_1,419222301842
Camera_2,409122302580
```

Both values are required and must be unique. The serial number must match the
value reported by `librealsense` for that device.

## Usage

Capture one RGB/depth pair from every configured camera:

```bash
python multicam_capture.py
```

Choose a different configuration file and output directory:

```bash
python multicam_capture.py \
  --csv cameras.csv \
  --output captures
```

Run continuous sweeps until you press <kbd>Ctrl</kbd>+<kbd>C</kbd>:

```bash
python multicam_capture.py --mode loop --cycle-delay 10
```

Run a fixed number of sweeps:

```bash
python multicam_capture.py --mode loop --max-cycles 5
```

Use different stream settings:

```bash
python multicam_capture.py \
  --width 1280 \
  --height 720 \
  --fps 5 \
  --warmup-frames 15
```

See every available option:

```bash
python multicam_capture.py --help
```

### Useful options

| Option | Default | Description |
| --- | ---: | --- |
| `--mode` | `once` | Run one sweep or loop continuously |
| `--output` | `captures` | Root directory for saved images |
| `--width`, `--height` | `1280`, `720` | RGB and depth stream dimensions |
| `--fps` | `5` | Stream frame rate |
| `--warmup-frames` | `15` | Frames discarded after starting a camera |
| `--frame-timeout-ms` | `15000` | Timeout while waiting for a frame |
| `--device-wait-timeout` | `20` | Seconds to wait for a camera to appear |
| `--inter-camera-delay` | `1` | Delay between cameras in a sweep |
| `--cycle-delay` | `0` | Delay between sweeps in loop mode |
| `--max-cycles` | unlimited | Stop loop mode after this many sweeps |
| `--no-align` | off | Keep depth in its native coordinate system |
| `--fail-fast` | off | Stop immediately when a camera fails |
| `--log-level` | `INFO` | Logging verbosity |

By default, a failed camera is recorded and the sweep continues with the next
camera. The command exits with status `1` if a one-shot sweep has any failures.

## Output

Images are grouped by a filesystem-safe camera name and serial number:

```text
captures/
└── Camera_1_419222301842/
    ├── rgb/
    │   └── 20260730_142530_123.png
    └── depth/
        └── 20260730_142530_123.png
```

The matching RGB and depth images use the same local timestamp. RGB images are
8-bit BGR PNGs, and depth images are lossless 16-bit PNGs containing the raw
RealSense depth values. Unless `--no-align` is used, depth is aligned to the
colour frame before saving.

## Python API

The module can be imported without starting a capture:

```python
from pathlib import Path

from multicam_capture import CaptureSettings, capture_all_once

settings = CaptureSettings(
    output_root=Path("captures"),
    width=1280,
    height=720,
    fps=5,
)

result = capture_all_once("cameras.csv", settings=settings)

for capture in result.captures:
    print(capture.camera_name, capture.rgb_path, capture.depth_path)

for failure in result.failures:
    print(failure.camera_name, failure.error)
```

For repeated capture, use `capture_all_loop`. It accepts a `stop_event` for
coordinated shutdown and an `on_sweep` callback for processing each completed
sweep:

```python
import threading

from multicam_capture import capture_all_loop

stop_event = threading.Event()

summary = capture_all_loop(
    "cameras.csv",
    cycle_delay_s=10,
    stop_event=stop_event,
    on_sweep=lambda cycle, result: print(
        cycle, len(result.captures), len(result.failures)
    ),
)
```

Only one camera pipeline is active at any point in both modes.

## Troubleshooting

- **Camera timeout:** confirm the configured serial number and verify that the
  camera appears in a RealSense SDK tool such as `rs-enumerate-devices`.
- **Unsupported stream profile:** choose a width, height, and frame-rate
  combination supported by every configured camera.
- **Missing dependency:** activate the intended virtual environment and install
  the package named in the error.
- **Incomplete sweep:** review the log for the affected camera. Use
  `--fail-fast` when you want the first failure to stop the process.
- **Depth looks black in an image viewer:** the file contains 16-bit raw depth
  values and normally needs scaling or colourisation for display.

## Licence

This project is licensed under the terms in [LICENSE](LICENSE).
