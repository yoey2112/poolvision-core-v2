# Pool Vision Core v2 🎱

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux-lightgrey.svg)]()
[![OpenCV](https://img.shields.io/badge/OpenCV-4.11.0-green.svg)](https://opencv.org/)

Real-time computer vision system for billiards/pool table monitoring with ball detection, tracking, physics simulation, and game state management.

## 🚀 Features

### Core Vision System
- **Ball Detection**: Advanced Hough circle detection with adaptive parameters
- **Color Classification**: LAB color space with Mahalanobis distance matching
- **Stripe Detection**: Automatic solid vs. stripe ball classification
- **Multi-Camera Support**: Select between multiple cameras with `--list-cameras`
- **Table Rectification**: Homography-based perspective correction
- **Real-time Processing**: GPU-accelerated video capture (D3D11)

### Tracking & Physics
- **Kalman Filter Tracking**: 6-state filter (position, velocity, acceleration)
- **Ball-to-Ball Collision**: Physics-based collision detection and response
- **Cushion Reflection**: Realistic table boundary interactions with damping
- **Velocity Estimation**: Accurate motion prediction and trajectory forecasting
- **ID Persistence**: Robust track assignment using Hungarian algorithm

### Game Intelligence
- **Game State Management**: Full 8-ball and 9-ball pool rule implementation
- **Turn Tracking**: Automatic player turn switching
- **Scoring System**: Real-time score calculation
- **Foul Detection**: Scratch and illegal shot recognition
- **Pocket Events**: Automatic detection of pocketed balls
- **Win Conditions**: Game-over detection with winner determination

### User Interface
- **Enhanced Visual Overlay**: Real-time game status display
- **Ball Visualization**: Color-coded balls with labels and trajectories
- **Velocity Vectors**: Arrow indicators showing ball movement
- **Scoreboard**: Live player scores and remaining balls
- **Turn Indicator**: Visual player turn notification
- **Fullscreen Mode**: Immersive viewing experience

### Output & Integration
- **JSON Streaming**: Real-time newline-delimited JSON output
- **Event System**: Comprehensive event detection and logging
- **Configuration Validation**: Built-in config file validation with error reporting
- **Test Suite**: Unit and integration tests with Google Test

## 🛠️ Build

### Windows (using vcpkg and MSVC)

### Windows (using vcpkg and MSVC)

**Prerequisites:**
- Visual Studio 2022 (with C++ Desktop Development)
- CMake 3.15+
- Git

**Build Steps:**

1. Clone the repository:
```powershell
git clone https://github.com/yoey2112/poolvision-core-v2.git
cd poolvision-core-v2
```

2. Bootstrap vcpkg (if not already installed):
```powershell
git clone https://github.com/Microsoft/vcpkg.git
.\vcpkg\bootstrap-vcpkg.bat
```

3. Configure and build:
```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 -DCMAKE_TOOLCHAIN_FILE="$pwd\vcpkg\scripts\buildsystems\vcpkg.cmake"
cmake --build build --config Debug
```

### Linux (Ubuntu/Debian)

### Linux (Ubuntu/Debian)

```bash
# Install dependencies
sudo apt update
sudo apt install build-essential cmake libopencv-dev libeigen3-dev

# Build
cmake -S . -B build
cmake --build build -j$(nproc)
```

## 📷 Usage

### List Available Cameras
```powershell
.\build\Debug\table_daemon.exe --list-cameras
```

### Run with Specific Camera
```powershell
# Use first camera (default)
.\build\Debug\table_daemon.exe --source 0

# Use second camera
.\build\Debug\table_daemon.exe --source 1

# Use video file
.\build\Debug\table_daemon.exe --source video.mp4
```

### Full Command-Line Options
```powershell
.\build\Debug\table_daemon.exe [options]

Options:
  --config <file>      Table configuration file (default: config/table.yaml)
  --camera <file>      Camera configuration file (default: config/camera.yaml)
  --colors <file>      Colors configuration file (default: config/colors.yaml)
  --source <source>    Video source: camera index (0,1,2...) or file path (default: 0)
  --engine <type>      Detection engine: 'classical' or 'dl' (default: classical)
  --fpscap <fps>       Cap frame rate (0 = unlimited, default: 0)
  --list-cameras       List available cameras and exit
  --help               Show help message
```

### Calibration Tool
```powershell
.\build\Debug\calibrate.exe
```

Interactive calibration tool for setting up homography transformation.

### Controls
- **ESC** or **Q** - Quit application
- Click on video window to interact (calibration mode)

## 📊 Output Format

The system outputs newline-delimited JSON to stdout:

```json
{
  "timestamp": 1.5,
  "balls": [
    {
      "id": 1,
      "x": 320.5,
      "y": 240.2,
      "r": 12.5,
      "stripe": 0.95,
      "label": 8
    }
  ],
  "tracks": [
    {
      "id": 1,
      "x": 320.5,
      "y": 240.2,
      "vx": 15.3,
      "vy": -8.7
    }
  ],
  "events": [
    {
      "type": 0,
      "ball_id": 8,
      "t": 1.5
    }
  ]
}
```

## ⚙️ Configuration

### Table Configuration (`config/table.yaml`)
```yaml
table_width: 2540      # Table width in mm
table_height: 1270     # Table height in mm
ball_radius_px: 20     # Ball radius in pixels
homography: [1, 0, 0, 0, 1, 0, 0, 0, 1]  # 3x3 homography matrix
pockets: [[0, 0], [1270, 0], ...]  # Pocket positions
```

### Camera Configuration (`config/camera.yaml`)
```yaml
width: 1920
height: 1080
fps: 30
```

### Colors Configuration (`config/colors.yaml`)
```yaml
# LAB color space values (L: 0-100, a/b: -128-127)
1: [65, 25, 45]   # Ball 1 color
2: [60, -20, 35]  # Ball 2 color
cue: [95, 0, 0]   # Cue ball (white)
```

## 🧪 Testing

Run unit and integration tests:

```powershell
.\build\Debug\unit_tests.exe
```

## 🏗️ Project Structure

```
poolvision-core-v2/
├── apps/
│   ├── calibrate/          # Calibration tool
│   └── table_daemon/       # Main application
├── core/
│   ├── calib/             # Calibration & homography
│   ├── detect/            # Ball detection (classical & DL)
│   ├── events/            # Event detection engine
│   ├── game/              # Game state management
│   ├── io/                # Video I/O and JSON output
│   ├── track/             # Tracking & physics
│   └── util/              # Utilities (config, UI, types)
├── config/                # Configuration files
├── tests/                 # Unit and integration tests
└── scripts/              # Build and setup scripts
```

## 🔬 Technical Details

### Ball Detection Pipeline
1. **Preprocessing**: Grayscale conversion and Gaussian blur
2. **Circle Detection**: Hough Circle Transform with adaptive parameters
3. **Color Classification**: LAB color space matching with Mahalanobis distance
4. **Stripe Analysis**: Radial gradient analysis for solid/stripe classification
5. **Validation**: Edge consistency and circularity checks

### Tracking Algorithm
- **State Vector**: `[x, y, vx, vy, ax, ay]` (6 dimensions)
- **Prediction**: Kalman filter with constant acceleration model
- **Update**: Measurement correction with configurable noise parameters
- **Association**: Hungarian algorithm for optimal track-detection matching
- **Lifecycle**: Automatic track creation, confirmation, and deletion

### Physics Engine
- **Collision Detection**: Spatial partitioning for O(n²) detection
- **Response**: Impulse-based collision with configurable restitution (0.95)
- **Friction**: Rolling resistance with μ = 0.01
- **Boundaries**: Elastic cushion collisions with damping

## 🎮 Game Rules Support

### 8-Ball Pool
- Break shot detection
- Group assignment (solids/stripes)
- Legal shot validation
- Scratch detection
- 8-ball win/loss conditions

### 9-Ball Pool
- Sequential ball requirements
- Push-out rules
- 9-ball win condition

## 🚧 Future Enhancements

- [ ] Deep learning detection engine (ONNX Runtime integration)
- [ ] Cue stick tracking
- [ ] Shot power estimation
- [ ] Advanced statistics dashboard
- [ ] Web interface for remote monitoring
- [ ] Multi-table support
- [ ] Tournament mode
- [ ] Replay system

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

- GitHub: [@yoey2112](https://github.com/yoey2112)
- Repository: [poolvision-core-v2](https://github.com/yoey2112/poolvision-core-v2)

## 🙏 Acknowledgments

- OpenCV team for the excellent computer vision library
- Eigen library for linear algebra operations
- vcpkg for simplified dependency management

---

**Built with ❤️ for the billiards community**
