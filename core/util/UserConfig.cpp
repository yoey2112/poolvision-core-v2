#include "UserConfig.hpp"
#include <iostream>
#include <fstream>
#include <filesystem>

#ifdef _WIN32
#include <windows.h>
#include <shlobj.h>
#include <process.h>
#else
#include <unistd.h>
#include <sys/wait.h>
#include <pwd.h>
#endif

namespace fs = std::filesystem;
using namespace pv;

// Static constants
const std::string UserConfig::CAMERA_CONFIG_FILE = "camera.yaml";
const std::string UserConfig::TABLE_CONFIG_FILE = "table.yaml";
const std::string UserConfig::COLORS_CONFIG_FILE = "colors.yaml";
const std::string UserConfig::SETTINGS_CONFIG_FILE = "settings.yaml";
const std::string UserConfig::DATABASE_FILE = "poolvision.db";
const std::string UserConfig::SETUP_MARKER_FILE = ".setup_complete";

UserConfig& UserConfig::instance() {
    static UserConfig instance;
    return instance;
}

std::string UserConfig::getApplicationDataPath() const {
#ifdef _WIN32
    char path[MAX_PATH];
    if (SUCCEEDED(SHGetFolderPathA(NULL, CSIDL_APPDATA, NULL, 0, path))) {
        return std::string(path) + "\\PoolVision";
    }
    return "";
#else
    const char* home = getenv("HOME");
    if (!home) {
        struct passwd* pw = getpwuid(getuid());
        home = pw->pw_dir;
    }
    return std::string(home) + "/.config/poolvision";
#endif
}

bool UserConfig::ensureDirectoryExists(const std::string& path) const {
    try {
        if (!fs::exists(path)) {
            return fs::create_directories(path);
        }
        return fs::is_directory(path);
    } catch (const std::exception& e) {
        std::cerr << "Error creating directory " << path << ": " << e.what() << std::endl;
        return false;
    }
}

bool UserConfig::initializeUserDirectories() {
    userConfigDir_ = getApplicationDataPath();
    if (userConfigDir_.empty()) {
        std::cerr << "Could not determine user config directory" << std::endl;
        return false;
    }
    
    userDataDir_ = userConfigDir_ + "/data";
    
    // Create directories
    if (!ensureDirectoryExists(userConfigDir_)) {
        std::cerr << "Failed to create config directory: " << userConfigDir_ << std::endl;
        return false;
    }
    
    if (!ensureDirectoryExists(userDataDir_)) {
        std::cerr << "Failed to create data directory: " << userDataDir_ << std::endl;
        return false;
    }
    
    std::cout << "User directories initialized:" << std::endl;
    std::cout << "  Config: " << userConfigDir_ << std::endl;
    std::cout << "  Data: " << userDataDir_ << std::endl;
    
    return true;
}

std::string UserConfig::getUserConfigDir() const {
    if (userConfigDir_.empty()) {
        const_cast<UserConfig*>(this)->initializeUserDirectories();
    }
    return userConfigDir_;
}

std::string UserConfig::getUserDataDir() const {
    if (userDataDir_.empty()) {
        const_cast<UserConfig*>(this)->initializeUserDirectories();
    }
    return userDataDir_;
}

bool UserConfig::isFirstRun() const {
    std::string markerPath = getUserConfigDir() + "/" + SETUP_MARKER_FILE;
    return !fs::exists(markerPath);
}

void UserConfig::markConfigured() {
    std::string markerPath = getUserConfigDir() + "/" + SETUP_MARKER_FILE;
    std::ofstream marker(markerPath);
    if (marker.is_open()) {
        marker << "Setup completed on: " << std::time(nullptr) << std::endl;
        marker.close();
    }
}

std::string UserConfig::getCameraConfigPath() const {
    return getUserConfigDir() + "/" + CAMERA_CONFIG_FILE;
}

std::string UserConfig::getTableConfigPath() const {
    return getUserConfigDir() + "/" + TABLE_CONFIG_FILE;
}

std::string UserConfig::getColorsConfigPath() const {
    return getUserConfigDir() + "/" + COLORS_CONFIG_FILE;
}

std::string UserConfig::getSettingsConfigPath() const {
    return getUserConfigDir() + "/" + SETTINGS_CONFIG_FILE;
}

std::string UserConfig::getDatabasePath() const {
    return getUserDataDir() + "/" + DATABASE_FILE;
}

bool UserConfig::hasValidConfiguration() const {
    std::vector<std::string> required = {
        getCameraConfigPath(),
        getTableConfigPath(),
        getColorsConfigPath(),
        getSettingsConfigPath()
    };
    
    for (const auto& path : required) {
        if (!fs::exists(path) || fs::file_size(path) == 0) {
            return false;
        }
    }
    
    return true;
}

std::vector<std::string> UserConfig::getMissingConfigFiles() const {
    std::vector<std::string> missing;
    
    struct ConfigFile {
        std::string name;
        std::string path;
    };
    
    std::vector<ConfigFile> configs = {
        {"Camera", getCameraConfigPath()},
        {"Table", getTableConfigPath()},
        {"Colors", getColorsConfigPath()},
        {"Settings", getSettingsConfigPath()}
    };
    
    for (const auto& config : configs) {
        if (!fs::exists(config.path) || fs::file_size(config.path) == 0) {
            missing.push_back(config.name);
        }
    }
    
    return missing;
}

bool UserConfig::copyDefaultConfigs() {
    // Copy default config files from installation directory to user directory
    
    struct ConfigMapping {
        std::string sourceName;
        std::string targetPath;
    };
    
    std::vector<ConfigMapping> mappings = {
        {"config/camera.yaml", getCameraConfigPath()},
        {"config/table.yaml", getTableConfigPath()},
        {"config/colors.yaml", getColorsConfigPath()}
    };
    
    bool success = true;
    
    for (const auto& mapping : mappings) {
        try {
            if (fs::exists(mapping.sourceName)) {
                fs::copy_file(mapping.sourceName, mapping.targetPath, 
                             fs::copy_options::overwrite_existing);
                std::cout << "Copied " << mapping.sourceName << " to " << mapping.targetPath << std::endl;
            } else {
                std::cerr << "Warning: Default config " << mapping.sourceName << " not found" << std::endl;
                success = false;
            }
        } catch (const std::exception& e) {
            std::cerr << "Error copying " << mapping.sourceName << ": " << e.what() << std::endl;
            success = false;
        }
    }
    
    return success;
}

bool UserConfig::createDefaultSettings() {
    std::string settingsPath = getSettingsConfigPath();
    
    try {
        std::ofstream settings(settingsPath);
        if (!settings.is_open()) {
            return false;
        }
        
        settings << "# Pool Vision User Settings" << std::endl;
        settings << "# Generated on first run" << std::endl;
        settings << std::endl;
        settings << "# UI Settings" << std::endl;
        settings << "window_width: 1280" << std::endl;
        settings << "window_height: 720" << std::endl;
        settings << "fullscreen: false" << std::endl;
        settings << "theme: dark" << std::endl;
        settings << std::endl;
        settings << "# Performance Settings" << std::endl;
        settings << "max_fps: 60" << std::endl;
        settings << "detection_quality: medium" << std::endl;
        settings << "tracking_smoothing: true" << std::endl;
        settings << std::endl;
        settings << "# Audio Settings" << std::endl;
        settings << "sound_enabled: true" << std::endl;
        settings << "volume: 0.8" << std::endl;
        settings << std::endl;
        settings << "# Database Settings" << std::endl;
        settings << "database_path: " << getDatabasePath() << std::endl;
        settings << "auto_backup: true" << std::endl;
        settings << "backup_interval_days: 7" << std::endl;
        
        settings.close();
        
        std::cout << "Created default settings: " << settingsPath << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Error creating default settings: " << e.what() << std::endl;
        return false;
    }
}

// ConfigLauncher implementation

ConfigLauncher::LaunchResult ConfigLauncher::checkAndPrepareConfig() {
    UserConfig& config = UserConfig::instance();
    
    // Initialize user directories
    if (!config.initializeUserDirectories()) {
        std::cerr << "Failed to initialize user directories" << std::endl;
        return LaunchResult::Error;
    }
    
    // Check if this is first run
    if (config.isFirstRun()) {
        std::cout << "First run detected, setup required" << std::endl;
        return LaunchResult::SetupRequired;
    }
    
    // Check if configuration is valid
    if (!config.hasValidConfiguration()) {
        std::vector<std::string> missing = config.getMissingConfigFiles();
        std::cout << "Invalid configuration detected. Missing: ";
        for (size_t i = 0; i < missing.size(); ++i) {
            std::cout << missing[i];
            if (i < missing.size() - 1) std::cout << ", ";
        }
        std::cout << std::endl;
        
        return LaunchResult::SetupRequired;
    }
    
    std::cout << "Configuration valid, ready to run" << std::endl;
    return LaunchResult::ReadyToRun;
}

bool ConfigLauncher::runSetupWizard() {
    std::cout << "Starting setup wizard..." << std::endl;
    
    if (!launchSetupWizardProcess()) {
        std::cerr << "Failed to launch setup wizard" << std::endl;
        return false;
    }
    
    if (!waitForSetupCompletion()) {
        std::cerr << "Setup wizard did not complete successfully" << std::endl;
        return false;
    }
    
    // Validate the configuration was created
    if (!validateAndFixConfig()) {
        std::cerr << "Setup completed but configuration is invalid" << std::endl;
        return false;
    }
    
    // Mark as configured
    UserConfig::instance().markConfigured();
    
    std::cout << "Setup completed successfully" << std::endl;
    return true;
}

bool ConfigLauncher::launchSetupWizardProcess() {
#ifdef _WIN32
    // Launch setup wizard and wait for completion
    STARTUPINFOA si = { sizeof(si) };
    PROCESS_INFORMATION pi = { 0 };
    
    std::string command = "build\\Debug\\setup_wizard.exe";
    
    if (!CreateProcessA(
        NULL,
        const_cast<char*>(command.c_str()),
        NULL,
        NULL,
        FALSE,
        0,
        NULL,
        NULL,
        &si,
        &pi)) {
        std::cerr << "Failed to create setup wizard process" << std::endl;
        return false;
    }
    
    // Wait for the process to complete
    WaitForSingleObject(pi.hProcess, INFINITE);
    
    DWORD exitCode;
    GetExitCodeProcess(pi.hProcess, &exitCode);
    
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);
    
    return exitCode == 0;
    
#else
    // Unix/Linux implementation
    pid_t pid = fork();
    
    if (pid == 0) {
        // Child process
        execl("./build/Debug/setup_wizard", "setup_wizard", NULL);
        exit(1); // If execl fails
    } else if (pid > 0) {
        // Parent process
        int status;
        waitpid(pid, &status, 0);
        return WEXITSTATUS(status) == 0;
    }
    
    return false;
#endif
}

bool ConfigLauncher::waitForSetupCompletion() {
    // Additional validation that setup completed successfully
    // This could involve checking for specific files or markers
    return UserConfig::instance().hasValidConfiguration();
}

bool ConfigLauncher::validateAndFixConfig() {
    UserConfig& config = UserConfig::instance();
    
    // Check if all required files exist
    if (config.hasValidConfiguration()) {
        return true;
    }
    
    // Try to fix common issues
    std::vector<std::string> missing = config.getMissingConfigFiles();
    
    if (!missing.empty()) {
        std::cout << "Attempting to fix configuration..." << std::endl;
        
        // Try copying default configs
        if (!config.copyDefaultConfigs()) {
            std::cerr << "Failed to copy default configurations" << std::endl;
        }
        
        // Create default settings if missing
        if (!fs::exists(config.getSettingsConfigPath())) {
            if (!config.createDefaultSettings()) {
                std::cerr << "Failed to create default settings" << std::endl;
                return false;
            }
        }
    }
    
    return config.hasValidConfiguration();
}