#!/bin/bash

echo "========================================"
echo "  Pool Vision Core v2 - Installation"
echo "========================================"
echo ""
echo "This script will install Pool Vision to your system"
echo "and run the initial configuration."
echo ""

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    echo "Warning: Running as root."
    echo "Consider running as a regular user for better security."
    echo ""
fi

# Create installation directory
INSTALL_DIR="/opt/poolvision"
echo "Creating installation directory: $INSTALL_DIR"
sudo mkdir -p "$INSTALL_DIR"

# Copy application files
echo "Copying application files..."
sudo cp build/Debug/pool_vision "$INSTALL_DIR/"
sudo cp build/Debug/setup_wizard "$INSTALL_DIR/"
sudo cp build/Debug/table_daemon "$INSTALL_DIR/"

# Copy default configuration templates
sudo mkdir -p "$INSTALL_DIR/config"
sudo cp config/*.yaml "$INSTALL_DIR/config/"

# Set permissions
sudo chmod +x "$INSTALL_DIR/pool_vision"
sudo chmod +x "$INSTALL_DIR/setup_wizard"
sudo chmod +x "$INSTALL_DIR/table_daemon"

# Create symlinks for easy access
sudo ln -sf "$INSTALL_DIR/pool_vision" "/usr/local/bin/pool_vision"
sudo ln -sf "$INSTALL_DIR/setup_wizard" "/usr/local/bin/pool_vision_setup"

echo ""
echo "========================================"
echo "   Installation complete!"
echo "========================================"
echo ""
echo "Starting initial configuration wizard..."
echo "This will set up your camera and table settings."
echo ""

# Run setup wizard for first-time configuration
"$INSTALL_DIR/setup_wizard"

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "   Setup completed successfully!"
    echo "========================================"
    echo ""
    echo "Pool Vision has been installed and configured."
    echo ""
    echo "To start Pool Vision:"
    echo "  1. Run: pool_vision"
    echo "  2. Or run: $INSTALL_DIR/pool_vision"
    echo ""
    echo "To reconfigure later:"
    echo "  Run: pool_vision_setup"
    echo ""
else
    echo ""
    echo "========================================"
    echo "   Setup was cancelled or failed"
    echo "========================================"
    echo ""
    echo "You can run the setup wizard later by executing:"
    echo "  pool_vision_setup"
    echo "  or: $INSTALL_DIR/setup_wizard"
    echo ""
fi