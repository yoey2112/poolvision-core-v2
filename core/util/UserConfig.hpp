#pragma once

#include <string>
#include <filesystem>
#include <map>

namespace pv {

/**
 * @brief Manages user configuration directories and first-run setup
 * 
 * This class handles:
 * - Creating user config directories
 * - Detecting first run vs configured systems
 * - Managing config file paths for user installations
 */
class UserConfig {
public:
    static UserConfig& instance();
    
    // Directory management
    bool initializeUserDirectories();
    std::string getUserConfigDir() const;
    std::string getUserDataDir() const;
    
    // First run detection
    bool isFirstRun() const;
    void markConfigured();
    
    // Config file paths
    std::string getCameraConfigPath() const;
    std::string getTableConfigPath() const;
    std::string getColorsConfigPath() const;
    std::string getSettingsConfigPath() const;
    std::string getDatabasePath() const;
    
    // Config validation
    bool hasValidConfiguration() const;
    std::vector<std::string> getMissingConfigFiles() const;
    
    // Installation helpers
    bool copyDefaultConfigs();
    bool createDefaultSettings();
    
private:
    UserConfig() = default;
    
    mutable std::string userConfigDir_;
    mutable std::string userDataDir_;
    
    bool ensureDirectoryExists(const std::string& path) const;
    std::string getApplicationDataPath() const;
    
    // Config file names
    static const std::string CAMERA_CONFIG_FILE;
    static const std::string TABLE_CONFIG_FILE;
    static const std::string COLORS_CONFIG_FILE;
    static const std::string SETTINGS_CONFIG_FILE;
    static const std::string DATABASE_FILE;
    static const std::string SETUP_MARKER_FILE;
};

/**
 * @brief Configuration launcher that handles first-run setup
 * 
 * This class manages the application startup flow:
 * - Checks if configuration exists
 * - Launches setup wizard if needed
 * - Ensures main app has valid config before starting
 */
class ConfigLauncher {
public:
    enum class LaunchResult {
        ReadyToRun,      // Config exists and valid, proceed with main app
        SetupRequired,   // Need to run setup wizard
        SetupCompleted,  // Setup just completed, ready to run
        SetupCancelled,  // User cancelled setup
        Error           // Error occurred
    };
    
    static LaunchResult checkAndPrepareConfig();
    static bool runSetupWizard();
    static bool validateAndFixConfig();
    
private:
    static bool launchSetupWizardProcess();
    static bool waitForSetupCompletion();
};

}