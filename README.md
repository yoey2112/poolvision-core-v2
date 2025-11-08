# poolvision-core-v2

Real-time billiards ball detection and tracking.

Features
- Classical CV detection (Hough circles + LAB color + stripe scoring)
- Table rectification via homography
- Kalman tracking + assignment
- Simple pocket event detection
- Frame JSON state output to stdout (newline-delimited)

Build

Windows (using vcpkg and MSVC):

1. Install vcpkg and libraries: follow vcpkg docs.
2. From repository root:

```powershell
.
# Example: bootstrap vcpkg, then
vcpkg install opencv eigen3
cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE=C:/path/to/vcpkg/scripts/buildsystems/vcpkg.cmake
cmake --build build --config Release
```

Ubuntu (apt):

```bash
sudo apt update
sudo apt install build-essential cmake libopencv-dev libeigen3-dev
cmake -S . -B build
cmake --build build -j
```

Run

Windows example:

```powershell
.
build\Release\table_daemon.exe --config config/table.yaml --camera config/camera.yaml --colors config/colors.yaml --source 0
```

Notes
- Optional DL engine can be enabled with -DBUILD_DL_ENGINE=ON and provisioning ONNX Runtime; code contains a stub.
- Config files are simple YAML-like and parsed with a minimal parser included in core/util/Config.
