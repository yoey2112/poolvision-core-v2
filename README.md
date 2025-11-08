# Pool Vision Core v2 🎱

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux-lightgrey.svg)]()
[![OpenCV](https://img.shields.io/badge/OpenCV-4.11.0-green.svg)](https://opencv.org/)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)]()
[![Phase](https://img.shields.io/badge/phase-2%20(100%25)-brightgreen.svg)](ROADMAP.md)

Real-time computer vision system for billiards/pool table monitoring with ball detection, tracking, physics simulation, and game state management.

**Latest Update (Nov 8, 2025)**: Phase 2 Main Menu & Settings Complete! ✅

## 🚀 Features

### Main Menu System ✅ COMPLETE (NEW!)
- **Modern Startup Screen**: Clean, stylish UI with animated background
- **Interactive Navigation**: Large buttons with hover effects and visual feedback
- **Quick Access**: Launch games, drills, profiles, analytics, settings, and calibration
- **Keyboard Shortcuts**: Number keys (1-7) for instant navigation
- **Professional Design**: Neon-accented dark theme with pool table aesthetics

### Settings Interface ✅ COMPLETE (NEW!)
- **Tabbed Organization**: General, Camera, Game, and Display settings
- **Real-time Updates**: Changes persist immediately to YAML config
- **Visual Controls**: Toggles, sliders, and dropdowns for intuitive adjustments
- **General Settings**: Language, theme, sound effects, notifications
- **Camera Settings**: Device selection, resolution, FPS, brightness/contrast
- **Game Settings**: Default game type, rule variants, auto-detection, shot timer
- **Display Settings**: Fullscreen, UI scale, overlay options, color schemes

### Setup Wizard ✅ COMPLETE
- **Graphical Setup Assistant**: Interactive wizard for initial system configuration
- **Camera Selection**: Visual preview and selection of available cameras
- **Camera Orientation**: Rotation (0°/90°/180°/270°) and flip controls with live preview
- **Table Calibration**: Interactive corner selection with bird's-eye view transformation
- **Table Dimensions**: Standard presets (7ft/8ft/9ft) or custom dimensions (metric/imperial)
- **Configuration Persistence**: Automatic saving to YAML files with validation
- **Configuration Summary**: Review and confirm all settings before completion

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

### Main Menu Application (Recommended)
```powershell
.\build\Debug\pool_vision.exe
```

The main menu provides access to all Pool Vision features:
- **New Game** - Start a new game session (Coming in Phase 3)
- **Drills & Practice** - Access practice drills (Coming in Phase 6)
- **Player Profiles** - Manage player profiles (Coming in Phase 3)
- **Analytics** - View statistics and performance data (Coming in Phase 4)
- **Settings** - Configure all application settings
- **Calibration** - Re-run the setup wizard
- **Exit** - Close the application

### Setup Wizard (First-Time Setup)
```powershell
.\build\Debug\setup_wizard.exe
```

The interactive setup wizard will guide you through:
1. **Camera Selection** - Choose from available cameras with live preview
2. **Camera Orientation** - Adjust rotation (0°/90°/180°/270°) and flip settings
3. **Table Calibration** - Mark table corners for perspective correction
4. **Table Dimensions** - Select standard size (7ft/8ft/9ft) or custom dimensions
5. **Configuration Save** - Automatically generates config/camera.yaml, config/table.yaml, config/colors.yaml

### Command Line (Advanced Users)

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
│   ├── pool_vision/       # Main menu application (NEW!)
│   ├── setup_wizard/      # Setup wizard
│   ├── calibrate/         # Calibration tool
│   └── table_daemon/      # Vision daemon
├── core/
│   ├── calib/             # Calibration & homography
│   ├── detect/            # Ball detection (classical & DL)
│   ├── events/            # Event detection engine
│   ├── game/              # Game state management
│   ├── io/                # Video I/O and JSON output
│   ├── track/             # Tracking & physics
│   ├── ui/                # UI components (NEW!)
│   │   ├── menu/          # Menu pages (MainMenu, Settings)
│   │   ├── wizard/        # Setup wizard pages
│   │   ├── UITheme.*      # Design system
│   │   └── WizardManager.*# Wizard controller
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

## 🚧 Development Roadmap

See [ROADMAP.md](ROADMAP.md) for detailed feature planning.

### Current Phase: Phase 2 - Main Menu & Settings ✅ COMPLETE
- ✅ Modern main menu with animated background
- ✅ Navigation system with 7 menu options
- ✅ Settings interface with 4 tabbed sections
- ✅ General, Camera, Game, and Display settings
- ✅ YAML persistence for all settings
- ✅ Professional neon-accented dark theme

### Completed Phases
- **Phase 1**: Setup Wizard & Calibration System ✅
  - Camera selection and orientation
  - Interactive table calibration
  - Table dimensions configuration
  - YAML configuration saving

### Upcoming Phases
- **Phase 3**: Player profile management and database
- **Phase 4**: Real-time overlay and shot prediction
- **Phase 5**: Analytics dashboard and game recording
- **Phase 6**: Practice drills system
- **Phase 7**: Advanced features (AI, multiplayer, tournaments)

### Future Enhancements
- [ ] Deep learning detection engine (ONNX Runtime integration)
- [ ] Cue stick tracking
- [ ] Shot power estimation
- [ ] Web interface for remote monitoring
- [ ] Multi-table support
- [ ] Replay system and video highlights
- [ ] Mobile companion app

## 🤝 Contributing

Contributions are welcome! Please check the [ROADMAP.md](ROADMAP.md) for current development priorities and open tasks.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'feat: Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📊 Project Statistics

- **Lines of Code**: ~8,500+ (including menu system and settings)
- **Files**: 75+ source files
- **Build Time**: ~45 seconds (incremental)
- **Executables**: 5 (pool_vision, table_daemon, setup_wizard, calibrate, unit_tests)
- **Test Coverage**: Unit and integration tests with Google Test

## 🔗 Repository

GitHub: [https://github.com/yoey2112/poolvision-core-v2](https://github.com/yoey2112/poolvision-core-v2)

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
