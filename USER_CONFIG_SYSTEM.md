# Pool Vision - User Configuration System

## Overview

Pool Vision v2 implements a comprehensive user configuration system that handles first-run setup and persistent configuration storage.

## Installation Flow

### 1. **Post-Installation Setup**
When Pool Vision is installed and run for the first time, the system automatically:

- Detects that no user configuration exists
- Creates user configuration directories:
  - **Windows**: `%APPDATA%\PoolVision`
  - **Linux**: `~/.config/poolvision`
- Launches the **Setup Wizard** automatically

### 2. **Setup Wizard** 
The setup wizard guides users through:

1. **Camera Selection** - Choose and configure camera device
2. **Camera Orientation** - Set rotation and flip settings  
3. **Table Calibration** - Define table boundaries and homography
4. **Table Dimensions** - Set physical table measurements

### 3. **Configuration Storage**
Setup wizard saves configuration to user directories:

```
Windows: %APPDATA%\PoolVision\
├── camera.yaml      # Camera settings
├── table.yaml       # Table calibration & dimensions
├── colors.yaml      # Ball color definitions
├── settings.yaml    # User preferences
└── data/
    └── poolvision.db # Player data & statistics
```

```
Linux: ~/.config/poolvision/
├── camera.yaml      # Camera settings  
├── table.yaml       # Table calibration & dimensions
├── colors.yaml      # Ball color definitions
├── settings.yaml    # User preferences
└── data/
    └── poolvision.db # Player data & statistics
```

## Configuration Files

### camera.yaml
```yaml
# Camera Configuration
width: 1920
height: 1080
fps: 30
device_id: 0
rotation: 0              # 0, 90, 180, 270 degrees
flip_horizontal: false
flip_vertical: false
```

### table.yaml  
```yaml
# Table Configuration
table_width: 2540        # mm (8 feet)
table_height: 4570       # mm (9 feet)  
ball_radius_px: 25       # pixels

# Homography matrix (3x3, row-major)
homography: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]

# Pocket positions (6 pockets)
pockets: [100, 50, 960, 50, 1820, 50, 100, 1030, 960, 1030, 1820, 1030]
```

### settings.yaml
```yaml
# Pool Vision User Settings
window_width: 1280
window_height: 720
fullscreen: false
theme: dark

# Performance Settings  
max_fps: 60
detection_quality: medium
tracking_smoothing: true

# Audio Settings
sound_enabled: true
volume: 0.8

# Database Settings
database_path: /path/to/user/data/poolvision.db
auto_backup: true
backup_interval_days: 7
```

## Usage

### First Run
1. **Install Pool Vision** using provided installer
2. **Run pool_vision.exe** - System detects first run
3. **Setup Wizard launches automatically**
4. **Complete wizard steps** to configure camera and table
5. **Start using Pool Vision** with saved configuration

### Subsequent Runs  
1. **Run pool_vision.exe** - System loads existing configuration
2. **Application starts normally** with user settings

### Reconfiguration
To reconfigure the system:
1. **Run setup_wizard.exe** manually to reconfigure settings
2. **Or delete config files** to trigger first-run setup again

## Installation Scripts

### Windows (install.bat)
```batch
@echo off
echo Installing Pool Vision...

REM Install to Program Files
set INSTALL_DIR=%ProgramFiles%\PoolVision
mkdir "%INSTALL_DIR%"
copy /Y "build\Debug\*.exe" "%INSTALL_DIR%\"
copy /Y "build\Debug\*.dll" "%INSTALL_DIR%\"

REM Run initial setup
echo Starting first-time setup...
"%INSTALL_DIR%\pool_vision.exe"
```

### Linux (install.sh)
```bash
#!/bin/bash
echo "Installing Pool Vision..."

# Install to /opt
sudo mkdir -p /opt/poolvision  
sudo cp build/Debug/* /opt/poolvision/
sudo ln -sf /opt/poolvision/pool_vision /usr/local/bin/

# Run initial setup
echo "Starting first-time setup..."
pool_vision
```

## Technical Architecture

### UserConfig Class
- **Singleton pattern** for global access
- **Cross-platform** directory detection
- **Automatic directory creation**
- **First-run detection** via marker file

### ConfigLauncher Class
- **Setup flow management**
- **Wizard process launching**
- **Configuration validation**  
- **Error handling and recovery**

### Code Example
```cpp
// Application startup
ConfigLauncher::LaunchResult result = ConfigLauncher::checkAndPrepareConfig();

switch (result) {
    case ConfigLauncher::LaunchResult::SetupRequired:
        // First run - launch setup wizard
        ConfigLauncher::runSetupWizard();
        break;
        
    case ConfigLauncher::LaunchResult::ReadyToRun:
        // Configuration exists - start application
        PoolVisionApp app;
        app.run();
        break;
}
```

## Benefits

✅ **Zero-configuration installation** - Works out of the box  
✅ **User-specific settings** - No admin rights required  
✅ **Cross-platform compatibility** - Windows & Linux support  
✅ **Guided setup process** - User-friendly wizard interface  
✅ **Persistent storage** - Settings saved between sessions  
✅ **Easy reconfiguration** - Setup wizard can be re-run  
✅ **Robust error handling** - Graceful failure recovery  

## Conclusion

This configuration system provides a professional, user-friendly installation and setup experience that automatically handles first-run configuration while maintaining user settings persistence across platforms.