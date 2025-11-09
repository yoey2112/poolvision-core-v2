@echo off
echo ========================================
echo   Pool Vision Core v2 - Installation
echo ========================================
echo.
echo This script will install Pool Vision to your system
echo and run the initial configuration.
echo.

REM Check if running as administrator (optional, but recommended)
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo Warning: Not running as administrator.
    echo Some features may not work correctly.
    echo.
)

REM Create installation directory
set INSTALL_DIR=%ProgramFiles%\PoolVision
echo Creating installation directory: %INSTALL_DIR%
if not exist "%INSTALL_DIR%" (
    mkdir "%INSTALL_DIR%"
)

REM Copy application files
echo Copying application files...
copy /Y "build\Debug\*.exe" "%INSTALL_DIR%\"
copy /Y "build\Debug\*.dll" "%INSTALL_DIR%\"
copy /Y "build\Debug\*.lib" "%INSTALL_DIR%\"

REM Copy default configuration templates
if not exist "%INSTALL_DIR%\config" (
    mkdir "%INSTALL_DIR%\config"
)
copy /Y "config\*.*" "%INSTALL_DIR%\config\"

echo.
echo ========================================
echo   Installation complete!
echo ========================================
echo.
echo Starting initial configuration wizard...
echo This will set up your camera and table settings.
echo.

REM Run setup wizard for first-time configuration
"%INSTALL_DIR%\setup_wizard.exe"

if %errorLevel% equ 0 (
    echo.
    echo ========================================
    echo   Setup completed successfully!
    echo ========================================
    echo.
    echo Pool Vision has been installed and configured.
    echo.
    echo To start Pool Vision:
    echo   1. Run: "%INSTALL_DIR%\pool_vision.exe"
    echo   2. Or use the desktop shortcut if created
    echo.
    echo To reconfigure later:
    echo   Run: "%INSTALL_DIR%\setup_wizard.exe"
    echo.
) else (
    echo.
    echo ========================================
    echo   Setup was cancelled or failed
    echo ========================================
    echo.
    echo You can run the setup wizard later by executing:
    echo   "%INSTALL_DIR%\setup_wizard.exe"
    echo.
)

pause