#include "SettingsPage.hpp"
#include <iostream>

namespace pv {

bool AppSettings::load() {
    try {
        cv::FileStorage fs(settingsPath, cv::FileStorage::READ);
        if (!fs.isOpened()) {
            std::cout << "Settings file not found, using defaults" << std::endl;
            return false;
        }
        
        // General
        fs["language"] >> language;
        fs["theme"] >> theme;
        fs["soundEffects"] >> soundEffects;
        fs["notifications"] >> notifications;
        
        // Camera
        fs["cameraIndex"] >> cameraIndex;
        fs["resolutionWidth"] >> resolutionWidth;
        fs["resolutionHeight"] >> resolutionHeight;
        fs["fps"] >> fps;
        fs["brightness"] >> brightness;
        fs["contrast"] >> contrast;
        
        // Game
        fs["defaultGameType"] >> defaultGameType;
        fs["ruleVariant"] >> ruleVariant;
        fs["autoDetection"] >> autoDetection;
        fs["shotTimer"] >> shotTimer;
        
        // Display
        fs["fullscreen"] >> fullscreen;
        fs["windowWidth"] >> windowWidth;
        fs["windowHeight"] >> windowHeight;
        fs["uiScale"] >> uiScale;
        fs["showOverlay"] >> showOverlay;
        fs["showVelocityVectors"] >> showVelocityVectors;
        fs["showTrajectories"] >> showTrajectories;
        fs["colorScheme"] >> colorScheme;
        
        fs.release();
        return true;
    }
    catch (const cv::Exception& e) {
        std::cerr << "Error loading settings: " << e.what() << std::endl;
        return false;
    }
}

bool AppSettings::save() const {
    try {
        // Create config directory if it doesn't exist
#ifdef _WIN32
        system("if not exist config mkdir config");
#else
        system("mkdir -p config");
#endif
        
        cv::FileStorage fs(settingsPath, cv::FileStorage::WRITE);
        if (!fs.isOpened()) {
            std::cerr << "Failed to open settings file for writing" << std::endl;
            return false;
        }
        
        // General
        fs << "language" << language;
        fs << "theme" << theme;
        fs << "soundEffects" << soundEffects;
        fs << "notifications" << notifications;
        
        // Camera
        fs << "cameraIndex" << cameraIndex;
        fs << "resolutionWidth" << resolutionWidth;
        fs << "resolutionHeight" << resolutionHeight;
        fs << "fps" << fps;
        fs << "brightness" << brightness;
        fs << "contrast" << contrast;
        
        // Game
        fs << "defaultGameType" << defaultGameType;
        fs << "ruleVariant" << ruleVariant;
        fs << "autoDetection" << autoDetection;
        fs << "shotTimer" << shotTimer;
        
        // Display
        fs << "fullscreen" << fullscreen;
        fs << "windowWidth" << windowWidth;
        fs << "windowHeight" << windowHeight;
        fs << "uiScale" << uiScale;
        fs << "showOverlay" << showOverlay;
        fs << "showVelocityVectors" << showVelocityVectors;
        fs << "showTrajectories" << showTrajectories;
        fs << "colorScheme" << colorScheme;
        
        fs.release();
        std::cout << "✓ Settings saved to " << settingsPath << std::endl;
        return true;
    }
    catch (const cv::Exception& e) {
        std::cerr << "Error saving settings: " << e.what() << std::endl;
        return false;
    }
}

SettingsPage::SettingsPage()
    : currentTab_(Tab::General)
    , windowWidth_(1280)
    , windowHeight_(720)
    , goBack_(false) {
}

void SettingsPage::init() {
    settings_.load();
    currentTab_ = Tab::General;
    goBack_ = false;
    updateLayout();
}

void SettingsPage::setWindowSize(int width, int height) {
    windowWidth_ = width;
    windowHeight_ = height;
    updateLayout();
}

void SettingsPage::updateLayout() {
    // Tab bar at top
    tabBarRect_ = cv::Rect(0, 80, windowWidth_, 60);
    
    // Buttons at bottom
    int buttonWidth = 150;
    int buttonHeight = 50;
    int margin = 20;
    
    backButtonRect_ = cv::Rect(margin, windowHeight_ - margin - buttonHeight,
                               buttonWidth, buttonHeight);
    saveButtonRect_ = cv::Rect(windowWidth_ - margin - buttonWidth,
                               windowHeight_ - margin - buttonHeight,
                               buttonWidth, buttonHeight);
}

cv::Mat SettingsPage::render() {
    cv::Mat img(windowHeight_, windowWidth_, CV_8UC3, UITheme::Colors::DarkBg);
    
    // Draw title
    UITheme::drawTitleBar(img, "Settings", 80);
    
    // Draw tabs
    drawTabs(img);
    
    // Draw current tab content
    switch (currentTab_) {
        case Tab::General:
            drawGeneralSettings(img);
            break;
        case Tab::Camera:
            drawCameraSettings(img);
            break;
        case Tab::Game:
            drawGameSettings(img);
            break;
        case Tab::Display:
            drawDisplaySettings(img);
            break;
    }
    
    // Draw action buttons
    drawButtons(img);
    
    return img;
}

void SettingsPage::drawTabs(cv::Mat& img) {
    std::vector<std::string> tabs = {"General", "Camera", "Game", "Display"};
    UITheme::drawTabBar(img, tabs, static_cast<int>(currentTab_), tabBarRect_);
}

void SettingsPage::drawGeneralSettings(cv::Mat& img) {
    currentControls_.clear();
    
    int startY = 180;
    int labelWidth = 250;
    int controlWidth = 300;
    int rowHeight = 70;
    int leftMargin = (windowWidth_ - labelWidth - controlWidth - 40) / 2;
    
    // Language
    SettingControl langControl;
    langControl.label = "Language";
    langControl.labelRect = cv::Rect(leftMargin, startY, labelWidth, 40);
    langControl.controlRect = cv::Rect(leftMargin + labelWidth + 40, startY, controlWidth, 40);
    currentControls_.push_back(langControl);
    
    cv::putText(img, langControl.label, 
               cv::Point(langControl.labelRect.x, langControl.labelRect.y + 30),
               UITheme::Fonts::FontFace, UITheme::Fonts::BodySize,
               UITheme::Colors::TextPrimary, UITheme::Fonts::BodyThickness);
    UITheme::drawDropdown(img, settings_.language, langControl.controlRect);
    
    // Theme
    startY += rowHeight;
    SettingControl themeControl;
    themeControl.label = "Theme";
    themeControl.labelRect = cv::Rect(leftMargin, startY, labelWidth, 40);
    themeControl.controlRect = cv::Rect(leftMargin + labelWidth + 40, startY, controlWidth, 40);
    currentControls_.push_back(themeControl);
    
    cv::putText(img, themeControl.label,
               cv::Point(themeControl.labelRect.x, themeControl.labelRect.y + 30),
               UITheme::Fonts::FontFace, UITheme::Fonts::BodySize,
               UITheme::Colors::TextPrimary, UITheme::Fonts::BodyThickness);
    UITheme::drawDropdown(img, settings_.theme, themeControl.controlRect);
    
    // Sound Effects Toggle
    startY += rowHeight;
    SettingControl soundControl;
    soundControl.label = "Sound Effects";
    soundControl.labelRect = cv::Rect(leftMargin, startY, labelWidth, 40);
    soundControl.controlRect = cv::Rect(leftMargin + labelWidth + 40, startY, 80, 40);
    currentControls_.push_back(soundControl);
    
    cv::putText(img, soundControl.label,
               cv::Point(soundControl.labelRect.x, soundControl.labelRect.y + 30),
               UITheme::Fonts::FontFace, UITheme::Fonts::BodySize,
               UITheme::Colors::TextPrimary, UITheme::Fonts::BodyThickness);
    UITheme::drawToggle(img, settings_.soundEffects, soundControl.controlRect);
    
    // Notifications Toggle
    startY += rowHeight;
    SettingControl notifControl;
    notifControl.label = "Notifications";
    notifControl.labelRect = cv::Rect(leftMargin, startY, labelWidth, 40);
    notifControl.controlRect = cv::Rect(leftMargin + labelWidth + 40, startY, 80, 40);
    currentControls_.push_back(notifControl);
    
    cv::putText(img, notifControl.label,
               cv::Point(notifControl.labelRect.x, notifControl.labelRect.y + 30),
               UITheme::Fonts::FontFace, UITheme::Fonts::BodySize,
               UITheme::Colors::TextPrimary, UITheme::Fonts::BodyThickness);
    UITheme::drawToggle(img, settings_.notifications, notifControl.controlRect);
}

void SettingsPage::drawCameraSettings(cv::Mat& img) {
    currentControls_.clear();
    
    int startY = 180;
    int labelWidth = 250;
    int controlWidth = 300;
    int rowHeight = 70;
    int leftMargin = (windowWidth_ - labelWidth - controlWidth - 40) / 2;
    
    // Camera Selection
    SettingControl camControl;
    camControl.label = "Camera Device";
    camControl.labelRect = cv::Rect(leftMargin, startY, labelWidth, 40);
    camControl.controlRect = cv::Rect(leftMargin + labelWidth + 40, startY, controlWidth, 40);
    currentControls_.push_back(camControl);
    
    cv::putText(img, camControl.label,
               cv::Point(camControl.labelRect.x, camControl.labelRect.y + 30),
               UITheme::Fonts::FontFace, UITheme::Fonts::BodySize,
               UITheme::Colors::TextPrimary, UITheme::Fonts::BodyThickness);
    UITheme::drawDropdown(img, "Camera " + std::to_string(settings_.cameraIndex), 
                         camControl.controlRect);
    
    // Resolution
    startY += rowHeight;
    SettingControl resControl;
    resControl.label = "Resolution";
    resControl.labelRect = cv::Rect(leftMargin, startY, labelWidth, 40);
    resControl.controlRect = cv::Rect(leftMargin + labelWidth + 40, startY, controlWidth, 40);
    currentControls_.push_back(resControl);
    
    std::string resText = std::to_string(settings_.resolutionWidth) + " x " + 
                         std::to_string(settings_.resolutionHeight);
    cv::putText(img, resControl.label,
               cv::Point(resControl.labelRect.x, resControl.labelRect.y + 30),
               UITheme::Fonts::FontFace, UITheme::Fonts::BodySize,
               UITheme::Colors::TextPrimary, UITheme::Fonts::BodyThickness);
    UITheme::drawDropdown(img, resText, resControl.controlRect);
    
    // FPS
    startY += rowHeight;
    SettingControl fpsControl;
    fpsControl.label = "FPS Cap";
    fpsControl.labelRect = cv::Rect(leftMargin, startY, labelWidth, 40);
    fpsControl.controlRect = cv::Rect(leftMargin + labelWidth + 40, startY, controlWidth, 40);
    currentControls_.push_back(fpsControl);
    
    cv::putText(img, fpsControl.label,
               cv::Point(fpsControl.labelRect.x, fpsControl.labelRect.y + 30),
               UITheme::Fonts::FontFace, UITheme::Fonts::BodySize,
               UITheme::Colors::TextPrimary, UITheme::Fonts::BodyThickness);
    UITheme::drawSlider(img, settings_.fps / 120.0f, fpsControl.controlRect, 0.0f, 1.0f);
    
    // Brightness
    startY += rowHeight;
    SettingControl brightControl;
    brightControl.label = "Brightness";
    brightControl.labelRect = cv::Rect(leftMargin, startY, labelWidth, 40);
    brightControl.controlRect = cv::Rect(leftMargin + labelWidth + 40, startY, controlWidth, 40);
    currentControls_.push_back(brightControl);
    
    cv::putText(img, brightControl.label,
               cv::Point(brightControl.labelRect.x, brightControl.labelRect.y + 30),
               UITheme::Fonts::FontFace, UITheme::Fonts::BodySize,
               UITheme::Colors::TextPrimary, UITheme::Fonts::BodyThickness);
    UITheme::drawSlider(img, settings_.brightness, brightControl.controlRect);
    
    // Re-run calibration button
    startY += rowHeight;
    cv::Rect calibButton(leftMargin, startY, controlWidth, 50);
    UITheme::drawButton(img, "Re-run Calibration Wizard", calibButton);
}

void SettingsPage::drawGameSettings(cv::Mat& img) {
    currentControls_.clear();
    
    int startY = 180;
    int labelWidth = 250;
    int controlWidth = 300;
    int rowHeight = 70;
    int leftMargin = (windowWidth_ - labelWidth - controlWidth - 40) / 2;
    
    // Default Game Type
    SettingControl gameControl;
    gameControl.label = "Default Game Type";
    gameControl.labelRect = cv::Rect(leftMargin, startY, labelWidth, 40);
    gameControl.controlRect = cv::Rect(leftMargin + labelWidth + 40, startY, controlWidth, 40);
    currentControls_.push_back(gameControl);
    
    cv::putText(img, gameControl.label,
               cv::Point(gameControl.labelRect.x, gameControl.labelRect.y + 30),
               UITheme::Fonts::FontFace, UITheme::Fonts::BodySize,
               UITheme::Colors::TextPrimary, UITheme::Fonts::BodyThickness);
    UITheme::drawDropdown(img, settings_.defaultGameType, gameControl.controlRect);
    
    // Rule Variant
    startY += rowHeight;
    SettingControl ruleControl;
    ruleControl.label = "Rule Variant";
    ruleControl.labelRect = cv::Rect(leftMargin, startY, labelWidth, 40);
    ruleControl.controlRect = cv::Rect(leftMargin + labelWidth + 40, startY, controlWidth, 40);
    currentControls_.push_back(ruleControl);
    
    cv::putText(img, ruleControl.label,
               cv::Point(ruleControl.labelRect.x, ruleControl.labelRect.y + 30),
               UITheme::Fonts::FontFace, UITheme::Fonts::BodySize,
               UITheme::Colors::TextPrimary, UITheme::Fonts::BodyThickness);
    UITheme::drawDropdown(img, settings_.ruleVariant, ruleControl.controlRect);
    
    // Auto-detection
    startY += rowHeight;
    SettingControl autoControl;
    autoControl.label = "Auto-Detection";
    autoControl.labelRect = cv::Rect(leftMargin, startY, labelWidth, 40);
    autoControl.controlRect = cv::Rect(leftMargin + labelWidth + 40, startY, 80, 40);
    currentControls_.push_back(autoControl);
    
    cv::putText(img, autoControl.label,
               cv::Point(autoControl.labelRect.x, autoControl.labelRect.y + 30),
               UITheme::Fonts::FontFace, UITheme::Fonts::BodySize,
               UITheme::Colors::TextPrimary, UITheme::Fonts::BodyThickness);
    UITheme::drawToggle(img, settings_.autoDetection, autoControl.controlRect);
    
    // Shot Timer
    startY += rowHeight;
    SettingControl timerControl;
    timerControl.label = "Shot Timer (seconds)";
    timerControl.labelRect = cv::Rect(leftMargin, startY, labelWidth, 40);
    timerControl.controlRect = cv::Rect(leftMargin + labelWidth + 40, startY, controlWidth, 40);
    currentControls_.push_back(timerControl);
    
    cv::putText(img, timerControl.label,
               cv::Point(timerControl.labelRect.x, timerControl.labelRect.y + 30),
               UITheme::Fonts::FontFace, UITheme::Fonts::BodySize,
               UITheme::Colors::TextPrimary, UITheme::Fonts::BodyThickness);
    
    std::string timerText = settings_.shotTimer == 0 ? "Disabled" : 
                           std::to_string(settings_.shotTimer) + "s";
    UITheme::drawDropdown(img, timerText, timerControl.controlRect);
}

void SettingsPage::drawDisplaySettings(cv::Mat& img) {
    currentControls_.clear();
    
    int startY = 180;
    int labelWidth = 250;
    int controlWidth = 300;
    int rowHeight = 70;
    int leftMargin = (windowWidth_ - labelWidth - controlWidth - 40) / 2;
    
    // Fullscreen
    SettingControl fullscreenControl;
    fullscreenControl.label = "Fullscreen";
    fullscreenControl.labelRect = cv::Rect(leftMargin, startY, labelWidth, 40);
    fullscreenControl.controlRect = cv::Rect(leftMargin + labelWidth + 40, startY, 80, 40);
    currentControls_.push_back(fullscreenControl);
    
    cv::putText(img, fullscreenControl.label,
               cv::Point(fullscreenControl.labelRect.x, fullscreenControl.labelRect.y + 30),
               UITheme::Fonts::FontFace, UITheme::Fonts::BodySize,
               UITheme::Colors::TextPrimary, UITheme::Fonts::BodyThickness);
    UITheme::drawToggle(img, settings_.fullscreen, fullscreenControl.controlRect);
    
    // UI Scale
    startY += rowHeight;
    SettingControl scaleControl;
    scaleControl.label = "UI Scale";
    scaleControl.labelRect = cv::Rect(leftMargin, startY, labelWidth, 40);
    scaleControl.controlRect = cv::Rect(leftMargin + labelWidth + 40, startY, controlWidth, 40);
    currentControls_.push_back(scaleControl);
    
    cv::putText(img, scaleControl.label,
               cv::Point(scaleControl.labelRect.x, scaleControl.labelRect.y + 30),
               UITheme::Fonts::FontFace, UITheme::Fonts::BodySize,
               UITheme::Colors::TextPrimary, UITheme::Fonts::BodyThickness);
    UITheme::drawSlider(img, (settings_.uiScale - 0.5f) / 1.5f, scaleControl.controlRect);
    
    // Show Overlay
    startY += rowHeight;
    SettingControl overlayControl;
    overlayControl.label = "Show Game Overlay";
    overlayControl.labelRect = cv::Rect(leftMargin, startY, labelWidth, 40);
    overlayControl.controlRect = cv::Rect(leftMargin + labelWidth + 40, startY, 80, 40);
    currentControls_.push_back(overlayControl);
    
    cv::putText(img, overlayControl.label,
               cv::Point(overlayControl.labelRect.x, overlayControl.labelRect.y + 30),
               UITheme::Fonts::FontFace, UITheme::Fonts::BodySize,
               UITheme::Colors::TextPrimary, UITheme::Fonts::BodyThickness);
    UITheme::drawToggle(img, settings_.showOverlay, overlayControl.controlRect);
    
    // Show Velocity Vectors
    startY += rowHeight;
    SettingControl velocityControl;
    velocityControl.label = "Show Velocity Vectors";
    velocityControl.labelRect = cv::Rect(leftMargin, startY, labelWidth, 40);
    velocityControl.controlRect = cv::Rect(leftMargin + labelWidth + 40, startY, 80, 40);
    currentControls_.push_back(velocityControl);
    
    cv::putText(img, velocityControl.label,
               cv::Point(velocityControl.labelRect.x, velocityControl.labelRect.y + 30),
               UITheme::Fonts::FontFace, UITheme::Fonts::BodySize,
               UITheme::Colors::TextPrimary, UITheme::Fonts::BodyThickness);
    UITheme::drawToggle(img, settings_.showVelocityVectors, velocityControl.controlRect);
    
    // Color Scheme
    startY += rowHeight;
    SettingControl colorControl;
    colorControl.label = "Color Scheme";
    colorControl.labelRect = cv::Rect(leftMargin, startY, labelWidth, 40);
    colorControl.controlRect = cv::Rect(leftMargin + labelWidth + 40, startY, controlWidth, 40);
    currentControls_.push_back(colorControl);
    
    cv::putText(img, colorControl.label,
               cv::Point(colorControl.labelRect.x, colorControl.labelRect.y + 30),
               UITheme::Fonts::FontFace, UITheme::Fonts::BodySize,
               UITheme::Colors::TextPrimary, UITheme::Fonts::BodyThickness);
    UITheme::drawDropdown(img, settings_.colorScheme, colorControl.controlRect);
}

void SettingsPage::drawButtons(cv::Mat& img) {
    // Back button
    UITheme::drawButton(img, "< Back", backButtonRect_);
    
    // Save button (highlighted)
    UITheme::drawButton(img, "Save", saveButtonRect_, false, true);
}

void SettingsPage::onMouse(int event, int x, int y, int flags) {
    mousePos_ = cv::Point(x, y);
    
    if (event == cv::EVENT_LBUTTONDOWN) {
        handleTabClick(x, y);
        handleControlClick(x, y);
        
        // Check button clicks
        if (UITheme::isPointInRect(mousePos_, backButtonRect_)) {
            goBack_ = true;
        }
        else if (UITheme::isPointInRect(mousePos_, saveButtonRect_)) {
            settings_.save();
            std::cout << "Settings saved!" << std::endl;
        }
    }
}

void SettingsPage::handleTabClick(int x, int y) {
    if (!tabBarRect_.contains(cv::Point(x, y))) {
        return;
    }
    
    int tabWidth = tabBarRect_.width / 4;
    int tabIndex = (x - tabBarRect_.x) / tabWidth;
    
    if (tabIndex >= 0 && tabIndex < 4) {
        currentTab_ = static_cast<Tab>(tabIndex);
    }
}

void SettingsPage::handleControlClick(int x, int y) {
    // Handle toggle clicks
    for (size_t i = 0; i < currentControls_.size(); ++i) {
        if (currentControls_[i].controlRect.contains(cv::Point(x, y))) {
            // Determine control type based on tab and index
            if (currentTab_ == Tab::General) {
                if (i == 2) settings_.soundEffects = !settings_.soundEffects;
                else if (i == 3) settings_.notifications = !settings_.notifications;
            }
            else if (currentTab_ == Tab::Game) {
                if (i == 2) settings_.autoDetection = !settings_.autoDetection;
            }
            else if (currentTab_ == Tab::Display) {
                if (i == 0) settings_.fullscreen = !settings_.fullscreen;
                else if (i == 2) settings_.showOverlay = !settings_.showOverlay;
                else if (i == 3) settings_.showVelocityVectors = !settings_.showVelocityVectors;
            }
        }
    }
}

void SettingsPage::onKey(int key) {
    // ESC to go back
    if (key == 27) {
        goBack_ = true;
    }
    
    // Tab navigation with number keys
    if (key >= '1' && key <= '4') {
        currentTab_ = static_cast<Tab>(key - '1');
    }
}

} // namespace pv
