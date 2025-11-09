# Pool Vision Core v2 🎱

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux-lightgrey.svg)]()
[![OpenCV](https://img.shields.io/badge/OpenCV-4.11.0-green.svg)](https://opencv.org/)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)]()
[![Phase](https://img.shields.io/badge/phase-7%20(100%25)-brightgreen.svg)](ROADMAP.md)

Real-time computer vision system for billiards/pool table monitoring with ball detection, tracking, physics simulation, and game state management.

**Latest Update (November 2024)**: Complete User Configuration System & Installation Flow ✅

## 🚀 Features

### User Configuration System ✅ COMPLETE (LATEST!)
- **First-Run Setup**: Automatic detection of first-time installation with guided configuration
- **User Directory Management**: Platform-specific configuration storage (`%APPDATA%\PoolVision` on Windows, `~/.config/poolvision` on Linux)
- **Setup Wizard Integration**: Automatic launch of setup wizard on first run with persistent user settings
- **Configuration Persistence**: All settings saved to user directories without requiring admin privileges
- **Installation Scripts**: Complete deployment package with `install.bat` and `install.sh` for professional installation
- **Config Launcher**: Smart startup flow that validates configuration and manages setup process
- **Cross-Platform Support**: Windows and Linux directory detection with proper path management
- **Zero-Config Installation**: Works out of the box with automatic first-run configuration

### Drill System & Match System ✅ COMPLETE
- **Drill System**: Comprehensive practice system with 50+ predefined drills across 10 categories (Break, Safety, Position Play, etc.)
- **Drill Execution**: Real-time drill tracking with difficulty levels (1-5), attempt counting, and performance evaluation
- **Custom Drills**: Create custom practice drills with flexible ball positioning and success criteria
- **Drill Library**: Organized drill collection with search, filtering, and difficulty progression
- **Performance Tracking**: Session-based statistics with success rates, improvement trends, and achievement unlocks
- **Match System**: Professional competitive match management with multiple formats (Race to N, Best of N, Time Limit)
- **Tournament Support**: Single-elimination and round-robin tournament brackets with participant management
- **Live Statistics**: Real-time shot tracking, success rates, and player comparison during matches
- **Head-to-Head Records**: Complete match history and statistics between any two players
- **Match UI**: Professional interface with docked panels, glass effects, and real-time visualization
- **Shot Clock**: Configurable shot timer with warnings and automatic forfeit
- **Database Integration**: Extended schema for drill sessions, match records, and tournament data

### Historical Analysis & Training 🟡 MOSTLY COMPLETE
- **Game Recording**: Session metadata capture with ball positions, trajectories, and events
- **Session Playback**: Timeline-based replay system with speed control and seeking (frame storage pending)
- **Training Mode**: Interactive practice system with shot evaluation and multiple exercise types
- **Shot Library**: Comprehensive shot collection with search, categorization, and management
- **Analytics Dashboard**: Statistics visualization framework with charts and metrics (UI complete, data integration pending)
- **Database Integration**: Extended database schema for game sessions and training records
- **Training Exercises**: Target practice, position play, speed control, pattern recognition exercises
- **Shot Evaluation**: Real-time feedback system with accuracy scoring and improvement suggestions
- **Progress Tracking**: Historical performance framework and achievement system

### Real-time Overlays & Shot Prediction ✅ COMPLETE
- **Ball Highlighting**: Color-coded legal/illegal ball indicators with number overlays
- **Shot Line**: Aiming line with power meter and direction arrows
- **Trajectory Prediction**: Physics-based ball path prediction with bounce points
- **Ghost Ball**: Visual indicator showing ideal cue ball contact position
- **Game State HUD**: Real-time display of player turn, scores, and remaining balls
- **Position Aids**: Heatmap showing position quality for strategic play
- **Shot Statistics**: Real-time shot difficulty evaluation and suggestions
- **Keyboard Controls**: Toggle overlay features (t/g/p/s/o keys)
- **Mouse Interaction**: Click and drag to aim shots and see predictions
- **Semi-transparent Overlays**: Non-intrusive visualization preserving table view

### Player Profile Management ✅ COMPLETE
- **SQLite Database**: Persistent player data with game history and statistics
- **Profile CRUD**: Create, view, edit, and delete player profiles
- **Player Statistics**: Track games played, win rate, shot success rate
- **Skill Levels**: 5-tier system (Beginner to Professional)
- **Handedness**: Track right/left/ambidextrous players
- **Game Preferences**: Store preferred game types and settings per player
- **Search & Filter**: Quick player search with real-time results
- **Visual UI**: Card-based layout with statistics dashboard
- **Game Session Tracking**: Record player matchups, winners, scores, duration
- **Shot Recording**: Log individual shots with position, success, and speed

### Main Menu System ✅ COMPLETE
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

**First-Time Installation Experience:**
1. **Automatic First-Run Detection** - System detects no user configuration exists
2. **User Directory Creation** - Creates `%APPDATA%\PoolVision` (Windows) or `~/.config/poolvision` (Linux)
3. **Setup Wizard Launch** - Automatically launches setup wizard for configuration
4. **Guided Configuration** - Step-by-step camera, table, and preferences setup
5. **Persistent Settings** - All configuration saved to user directories
6. **Normal Startup** - Subsequent runs load existing settings and start immediately

The main menu provides access to all Pool Vision features:
- **New Game** - Start a new game session with match system
- **Drills & Practice** - Access comprehensive drill system with 50+ exercises
- **Player Profiles** - Manage player profiles with detailed statistics
- **Analytics** - View statistics and performance data with advanced charts
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
- **T** - Show shot trajectory overlay
- **G** - Show ghost ball overlay
- **P** - Show position aids overlay
- **S** - Show all overlays including statistics
- **O** - Hide all overlays
- Click and drag on video to aim shots (shows trajectory prediction)

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
rotation: 0  # 0, 90, 180, or 270 degrees
flip: false  # Horizontal flip
brightness: 0.5
contrast: 0.5
```

### Colors Configuration (`config/colors.yaml`)
```yaml
# LAB color space values (L: 0-100, a/b: -128-127)
1: [65, 25, 45]   # Ball 1 color
2: [60, -20, 35]  # Ball 2 color
cue: [95, 0, 0]   # Cue ball (white)
```

### Settings Configuration (`config/settings.yaml`)
```yaml
general:
  language: "en"
  theme: "dark"
  sound_enabled: true
  
camera:
  device_id: 0
  resolution: "1920x1080"
  fps: 30
  
game:
  default_type: "8-Ball"
  auto_detect: true
  shot_timer_enabled: false
  
display:
  fullscreen: false
  ui_scale: 1.0
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
│   ├── pool_vision/       # Main menu application with user config system
│   ├── setup_wizard/      # Setup wizard with user directory saving
│   ├── calibrate/         # Calibration tool
│   └── table_daemon/      # Vision daemon
├── core/
│   ├── calib/             # Calibration & homography
│   ├── db/                # Database layer (PlayerProfile, Database)
│   ├── detect/            # Ball detection (classical & DL)
│   ├── events/            # Event detection engine
│   ├── game/              # Game state management & drill/match systems
│   │   ├── GameRecorder.* # Session recording system
│   │   ├── SessionPlayback.* # Replay and analysis system
│   │   ├── TrainingMode.* # Interactive training system
│   │   ├── ShotLibrary.*  # Shot collection management
│   │   ├── DrillSystem.*  # Comprehensive drill system
│   │   ├── DrillLibrary.* # Drill templates and library
│   │   └── MatchSystem.*  # Professional match management
│   ├── io/                # Video I/O and JSON output
│   ├── track/             # Tracking & physics
│   ├── ui/                # UI components
│   │   ├── menu/          # Menu pages (MainMenu, Settings, PlayerProfiles, AnalyticsPage, DrillsPage)
│   │   ├── wizard/        # Setup wizard pages
│   │   ├── OverlayRenderer.* # Real-time overlay system
│   │   ├── UITheme.*      # Design system
│   │   ├── MatchUI.*      # Professional match interface
│   │   └── WizardManager.*# Wizard controller
│   └── util/              # Utilities (config, UI, types)
│       ├── UserConfig.*   # User configuration system (NEW!)
│       └── ConfigLauncher.* # Installation flow management (NEW!)
├── config/                # Default configuration templates
├── data/                  # Database files
├── tests/                 # Unit and integration tests
├── scripts/              # Build and setup scripts
├── install.bat           # Windows installation script (NEW!)
├── install.sh            # Linux installation script (NEW!)
└── USER_CONFIG_SYSTEM.md # User configuration documentation (NEW!)
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

### Current Phase: Phase 7 - Match System & Enhanced UI ✅ COMPLETE
- ✅ Professional match management system with multiple formats
- ✅ Tournament support with bracket management
- ✅ Enhanced UI with docked panels and glass effects
- ✅ Live statistics and shot clock integration
- ✅ Head-to-head records and match history
- ✅ Database integration for match and tournament data

### Completed Phases
- **Phase 7**: Match System & Enhanced UI ✅ 
  - MatchSystem with tournament support and live statistics
  - MatchUI with professional docked panel interface
  - Shot clock, head-to-head records, and match history
  - Extended database schema for competitive play
- **Phase 6**: Drill System ✅ 
  - DrillSystem with 50+ predefined drills across 10 categories
  - DrillLibrary with custom drill creation and templates
  - DrillsPage UI with execution tracking and progress visualization
  - Performance analytics and achievement system
- **Phase 5**: Historical Analysis & Training ✅ 
  - GameRecorder, SessionPlayback, TrainingMode, ShotLibrary, AnalyticsPage
  - Extended database schema and analytics visualization
  - Training exercises and shot evaluation systems
- **Phase 4**: Real-time Overlays ✅
  - SQLite database integration
  - Player CRUD operations with UI
  - Statistics tracking and game session recording
- **Phase 2**: Main Menu & Settings ✅
  - Modern main menu with animated background
  - Settings interface with 4 tabbed sections
  - YAML persistence and theme system
  
- **Phase 1**: Setup Wizard & Calibration System ✅
  - Camera selection and orientation
  - Interactive table calibration
  - Table dimensions configuration

### Upcoming Phases
- **Phase 8**: AI Opponents and Advanced Analytics
- **Phase 9**: Tournament Management and Streaming

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

- **Lines of Code**: ~22,000+ (including user configuration system)
- **Files**: 130+ source files
- **Build Time**: ~90 seconds (incremental)
- **Executables**: 5 (pool_vision, table_daemon, setup_wizard, calibrate, unit_tests)
- **Test Coverage**: Unit and integration tests with Google Test
- **Database**: SQLite3 with 8 tables (players, game_sessions, shot_records, training_exercises, shot_library, drill_sessions, match_records, tournaments)
- **Configuration**: Complete user directory management with first-run detection
- **Installation**: Professional deployment scripts for Windows and Linux

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
- SQLite for the lightweight database engine
- vcpkg for simplified dependency management

---

**Built with ❤️ for the billiards community**
