#pragma once
#include "../UITheme.hpp"
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <map>

namespace pv {

/**
 * @brief Settings configuration structure
 */
struct AppSettings {
    // General
    std::string language = "English";
    std::string theme = "Dark";
    bool soundEffects = true;
    bool notifications = true;
    
    // Camera
    int cameraIndex = 0;
    int resolutionWidth = 1920;
    int resolutionHeight = 1080;
    int fps = 60;
    float brightness = 0.5f;
    float contrast = 0.5f;
    
    // Game
    std::string defaultGameType = "8-Ball";
    std::string ruleVariant = "Standard";
    bool autoDetection = true;
    int shotTimer = 0;  // 0 = disabled
    
    // Display
    bool fullscreen = false;
    int windowWidth = 1280;
    int windowHeight = 720;
    float uiScale = 1.0f;
    bool showOverlay = true;
    bool showVelocityVectors = true;
    bool showTrajectories = true;
    std::string colorScheme = "Neon";
    
    // File paths
    std::string settingsPath = "config/settings.yaml";
    
    /**
     * @brief Load settings from file
     */
    bool load();
    
    /**
     * @brief Save settings to file
     */
    bool save() const;
};

/**
 * @brief Settings page with tabbed interface
 */
class SettingsPage {
public:
    SettingsPage();
    ~SettingsPage() = default;
    
    /**
     * @brief Initialize the page
     */
    void init();
    
    /**
     * @brief Render the settings page
     */
    cv::Mat render();
    
    /**
     * @brief Handle mouse events
     */
    void onMouse(int event, int x, int y, int flags);
    
    /**
     * @brief Handle keyboard events
     */
    void onKey(int key);
    
    /**
     * @brief Get the settings
     */
    const AppSettings& getSettings() const { return settings_; }
    
    /**
     * @brief Check if user wants to go back
     */
    bool shouldGoBack() const { return goBack_; }
    
    /**
     * @brief Reset go back flag
     */
    void resetGoBack() { goBack_ = false; }
    
    /**
     * @brief Set window size
     */
    void setWindowSize(int width, int height);
    
private:
    enum class Tab {
        General = 0,
        Camera = 1,
        Game = 2,
        Display = 3
    };
    
    struct SettingControl {
        std::string label;
        cv::Rect labelRect;
        cv::Rect controlRect;
        bool isHovered = false;
    };
    
    void updateLayout();
    void drawTabs(cv::Mat& img);
    void drawGeneralSettings(cv::Mat& img);
    void drawCameraSettings(cv::Mat& img);
    void drawGameSettings(cv::Mat& img);
    void drawDisplaySettings(cv::Mat& img);
    void drawButtons(cv::Mat& img);
    
    void handleTabClick(int x, int y);
    void handleControlClick(int x, int y);
    
    AppSettings settings_;
    Tab currentTab_;
    cv::Point mousePos_;
    int windowWidth_;
    int windowHeight_;
    bool goBack_;
    
    cv::Rect saveButtonRect_;
    cv::Rect backButtonRect_;
    cv::Rect tabBarRect_;
    
    std::vector<SettingControl> currentControls_;
};

} // namespace pv
