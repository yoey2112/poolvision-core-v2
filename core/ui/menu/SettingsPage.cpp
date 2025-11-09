#include "SettingsPage.hpp"
#include "../ResponsiveLayout.hpp"
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
    , goBack_(false)
    , rootContainer_(nullptr)
    , animationTime_(0.0f) {
}

void SettingsPage::init() {
    settings_.load();
    currentTab_ = Tab::General;
    goBack_ = false;
    
    // Initialize UITheme for responsive scaling
    UITheme::init(windowWidth_, windowHeight_);
    
    createResponsiveLayout();
}

void SettingsPage::setWindowSize(int width, int height) {
    windowWidth_ = width;
    windowHeight_ = height;
    
    // Update UITheme scaling
    UITheme::setWindowSize(width, height);
    
    // Recreate responsive layout
    createResponsiveLayout();
}

void SettingsPage::createResponsiveLayout() {
    // Create root container with full window size
    cv::Rect windowRect(0, 0, windowWidth_, windowHeight_);
    rootContainer_ = std::make_unique<ResponsiveLayout::Container>(
        ResponsiveLayout::Direction::Column, windowRect);
    
    rootContainer_->setJustifyContent(ResponsiveLayout::Justify::SpaceBetween);
    rootContainer_->setAlignItems(ResponsiveLayout::Alignment::Center);
    rootContainer_->setPadding(UITheme::getResponsiveSpacing(20));
    
    // Header area (title) - 15% of height
    headerRect_ = UITheme::getResponsiveRect(0, 0, 100, 15, windowRect);
    
    // Tab bar area - 10% of height  
    tabBarRect_ = UITheme::getResponsiveRect(0, 15, 100, 10, windowRect);
    
    // Content area - 65% of height
    contentRect_ = UITheme::getResponsiveRect(0, 25, 100, 65, windowRect);
    
    // Button area - 10% of height
    buttonRect_ = UITheme::getResponsiveRect(0, 90, 100, 10, windowRect);
    
    updateButtonLayout();
}

void SettingsPage::updateButtonLayout() {
    // Calculate responsive button dimensions
    cv::Size buttonSize = UITheme::getResponsiveSize(150, 50);
    int margin = UITheme::getResponsiveSpacing(20);
    
    backButtonRect_ = cv::Rect(margin, 
                               buttonRect_.y + (buttonRect_.height - buttonSize.height) / 2,
                               buttonSize.width, buttonSize.height);
    
    saveButtonRect_ = cv::Rect(windowWidth_ - margin - buttonSize.width,
                               buttonRect_.y + (buttonRect_.height - buttonSize.height) / 2,
                               buttonSize.width, buttonSize.height);
}

// Legacy method for compatibility
void SettingsPage::updateLayout() {
    createResponsiveLayout();
}

cv::Mat SettingsPage::render() {
    cv::Mat img(windowHeight_, windowWidth_, CV_8UC3, UITheme::Colors::DarkBg);
    
    // Update animation time
    static auto startTime = std::chrono::steady_clock::now();
    auto currentTime = std::chrono::steady_clock::now();
    animationTime_ = std::chrono::duration<float>(currentTime - startTime).count();
    
    // Draw enhanced background
    drawBackground(img);
    
    // Draw title with glass effect
    drawTitle(img);
    
    // Draw modern tabs
    drawTabs(img);
    
    // Draw current tab content with responsive layout
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

void SettingsPage::drawBackground(cv::Mat& img) {
    // Fill with dark background
    img.setTo(UITheme::Colors::DarkBg);
    
    // Add subtle animated background
    UITheme::drawAnimatedBackground(img, animationTime_, 0.2f);
    
    // Add glass-morphism overlay
    cv::Rect overlayRect(0, 0, windowWidth_, windowHeight_);
    UITheme::applyGlassMorphism(img, overlayRect, 5, 0.05f, UITheme::Colors::MediumBg);
}

void SettingsPage::drawTitle(cv::Mat& img) {
    std::string title = "SETTINGS";
    double titleSize = UITheme::getResponsiveFontSize(UITheme::Fonts::TitleSize * 1.2);
    
    cv::Size titleTextSize = UITheme::getTextSize(title, UITheme::Fonts::FontFaceBold,
                                                 titleSize, UITheme::Fonts::TitleThickness);
    
    // Create glass card for title
    int titlePadding = UITheme::getResponsiveSpacing(25);
    cv::Rect titleCard(
        (windowWidth_ - titleTextSize.width - titlePadding * 2) / 2,
        headerRect_.y + UITheme::getResponsiveSpacing(10),
        titleTextSize.width + titlePadding * 2,
        titleTextSize.height + titlePadding
    );
    
    // Draw glass card
    UITheme::drawGlassCard(img, titleCard, 12.0f, 0.1f, UITheme::Colors::MediumBg);
    UITheme::drawRoundedRect(img, titleCard, UITheme::getResponsiveSpacing(12),
                           cv::Scalar(UITheme::Colors::MediumBg[0], 
                                     UITheme::Colors::MediumBg[1], 
                                     UITheme::Colors::MediumBg[2], 100), -1, true);
    
    // Draw border
    UITheme::drawRoundedRect(img, titleCard, UITheme::getResponsiveSpacing(12),
                           UITheme::Colors::NeonCyan, 2, true);
    
    // Draw title text
    cv::Point titlePos(titleCard.x + titlePadding,
                      titleCard.y + titlePadding + titleTextSize.height - 5);
    
    UITheme::drawTextWithShadow(img, title, titlePos, UITheme::Fonts::FontFaceBold,
                                titleSize, UITheme::Colors::NeonCyan,
                                UITheme::Fonts::TitleThickness, 
                                UITheme::getResponsiveSpacing(3), true);
}

void SettingsPage::drawTabs(cv::Mat& img) {
    std::vector<std::string> tabs = {"General", "Camera", "Game", "Display"};
    
    // Create enhanced tab bar with animation
    UITheme::AnimationState animState;
    animState.isAnimating = true;
    animState.progress = std::sin(animationTime_ * 2.0f) * 0.5f + 0.5f;
    
    UITheme::drawTabBar(img, tabs, static_cast<int>(currentTab_), tabBarRect_, animState);
}

void SettingsPage::drawGeneralSettings(cv::Mat& img) {
    currentControls_.clear();
    
    // Calculate responsive dimensions using percentages
    int contentPadding = UITheme::getResponsiveSpacing(40);
    int rowHeight = UITheme::getResponsiveSpacing(80);
    
    // Use percentage-based layout instead of fixed pixels
    int totalWidth = contentRect_.width - (contentPadding * 2);
    int labelWidth = static_cast<int>(totalWidth * 0.35f);  // 35% instead of fixed 250px
    int controlWidth = static_cast<int>(totalWidth * 0.45f); // 45% instead of fixed 300px
    int spacing = static_cast<int>(totalWidth * 0.05f);      // 5% spacing
    
    int leftMargin = contentRect_.x + contentPadding;
    int startY = contentRect_.y + UITheme::getResponsiveSpacing(20);
    
    // Create responsive form grid
    drawSettingRow(img, "Language", settings_.language, leftMargin, startY, 
                  labelWidth, controlWidth, spacing, SettingType::Dropdown);
    
    startY += rowHeight;
    drawSettingRow(img, "Theme", settings_.theme, leftMargin, startY,
                  labelWidth, controlWidth, spacing, SettingType::Dropdown);
    
    startY += rowHeight; 
    drawSettingRow(img, "Sound Effects", std::to_string(settings_.soundEffects), 
                  leftMargin, startY, labelWidth, controlWidth, spacing, SettingType::Toggle);
    
    startY += rowHeight;
    drawSettingRow(img, "Notifications", std::to_string(settings_.notifications),
                  leftMargin, startY, labelWidth, controlWidth, spacing, SettingType::Toggle);
}

void SettingsPage::drawSettingRow(cv::Mat& img, const std::string& label, const std::string& value,
                                  int x, int y, int labelWidth, int controlWidth, int spacing,
                                  SettingType type) {
    // Create setting control for interaction
    SettingControl control;
    control.label = label;
    control.type = type;
    control.labelRect = cv::Rect(x, y, labelWidth, UITheme::getResponsiveSpacing(40));
    
    // Determine control dimensions based on type
    int actualControlWidth = controlWidth;
    if (type == SettingType::Toggle) {
        actualControlWidth = UITheme::getResponsiveSpacing(80);
    }
    control.controlRect = cv::Rect(x + labelWidth + spacing, y, actualControlWidth, 
                                  UITheme::getResponsiveSpacing(40));
    currentControls_.push_back(control);
    
    // Draw glass card background for row
    cv::Rect rowCard(x - UITheme::getResponsiveSpacing(15), y - UITheme::getResponsiveSpacing(10),
                    labelWidth + spacing + actualControlWidth + UITheme::getResponsiveSpacing(30),
                    UITheme::getResponsiveSpacing(60));
    
    UITheme::drawGlassCard(img, rowCard, 8.0f, 0.03f, UITheme::Colors::LightBg);
    UITheme::drawRoundedRect(img, rowCard, UITheme::getResponsiveSpacing(8),
                           cv::Scalar(UITheme::Colors::LightBg[0], 
                                     UITheme::Colors::LightBg[1], 
                                     UITheme::Colors::LightBg[2], 30), -1, true);
    
    // Draw label with responsive font
    double fontSize = UITheme::getResponsiveFontSize(UITheme::Fonts::BodySize);
    cv::Point labelPos(control.labelRect.x, control.labelRect.y + 
                      static_cast<int>(fontSize * 20));
    
    UITheme::drawText(img, label, labelPos, fontSize,
                     UITheme::Colors::TextPrimary, UITheme::Fonts::BodyThickness, true);
    
    // Draw control based on type
    UITheme::ComponentState state = UITheme::ComponentState::Normal;
    
    switch (type) {
        case SettingType::Dropdown:
            UITheme::drawDropdown(img, value, control.controlRect, state, false);
            break;
        case SettingType::Toggle:
            UITheme::drawToggle(img, value == "1", control.controlRect, state);
            break;
        case SettingType::Slider: {
            float sliderValue = std::stof(value);
            UITheme::drawSlider(img, sliderValue, control.controlRect, 0.0f, 1.0f, state);
            break;
        }
        case SettingType::Button:
            UITheme::drawButton(img, value, control.controlRect, state);
            break;
    }
}
void SettingsPage::drawCameraSettings(cv::Mat& img) {
    currentControls_.clear();
    
    // Calculate responsive dimensions
    int contentPadding = UITheme::getResponsiveSpacing(40);
    int rowHeight = UITheme::getResponsiveSpacing(80);
    
    int totalWidth = contentRect_.width - (contentPadding * 2);
    int labelWidth = static_cast<int>(totalWidth * 0.35f);
    int controlWidth = static_cast<int>(totalWidth * 0.45f);
    int spacing = static_cast<int>(totalWidth * 0.05f);
    
    int leftMargin = contentRect_.x + contentPadding;
    int startY = contentRect_.y + UITheme::getResponsiveSpacing(20);
    
    // Camera Device
    std::string cameraText = "Camera " + std::to_string(settings_.cameraIndex);
    drawSettingRow(img, "Camera Device", cameraText, leftMargin, startY,
                  labelWidth, controlWidth, spacing, SettingType::Dropdown);
    
    // Resolution
    startY += rowHeight;
    std::string resText = std::to_string(settings_.resolutionWidth) + " × " + 
                         std::to_string(settings_.resolutionHeight);
    drawSettingRow(img, "Resolution", resText, leftMargin, startY,
                  labelWidth, controlWidth, spacing, SettingType::Dropdown);
    
    // FPS - using slider
    startY += rowHeight;
    std::string fpsValue = std::to_string(settings_.fps / 120.0f);
    drawSettingRow(img, "FPS Cap", fpsValue, leftMargin, startY,
                  labelWidth, controlWidth, spacing, SettingType::Slider);
    
    // Brightness - using slider
    startY += rowHeight;
    std::string brightnessValue = std::to_string(settings_.brightness);
    drawSettingRow(img, "Brightness", brightnessValue, leftMargin, startY,
                  labelWidth, controlWidth, spacing, SettingType::Slider);
    
    // Contrast - using slider  
    startY += rowHeight;
    std::string contrastValue = std::to_string(settings_.contrast);
    drawSettingRow(img, "Contrast", contrastValue, leftMargin, startY,
                  labelWidth, controlWidth, spacing, SettingType::Slider);
    
    // Re-run calibration button
    startY += rowHeight;
    drawSettingRow(img, "", "Re-run Calibration Wizard", leftMargin, startY,
                  0, controlWidth, 0, SettingType::Button);
}
}

void SettingsPage::drawGameSettings(cv::Mat& img) {
    currentControls_.clear();
    
    // Calculate responsive dimensions
    int contentPadding = UITheme::getResponsiveSpacing(40);
    int rowHeight = UITheme::getResponsiveSpacing(80);
    
    int totalWidth = contentRect_.width - (contentPadding * 2);
    int labelWidth = static_cast<int>(totalWidth * 0.35f);
    int controlWidth = static_cast<int>(totalWidth * 0.45f);
    int spacing = static_cast<int>(totalWidth * 0.05f);
    
    int leftMargin = contentRect_.x + contentPadding;
    int startY = contentRect_.y + UITheme::getResponsiveSpacing(20);
    
    // Default Game Type
    drawSettingRow(img, "Default Game Type", settings_.defaultGameType, leftMargin, startY,
                  labelWidth, controlWidth, spacing, SettingType::Dropdown);
    
    // Rule Variant
    startY += rowHeight;
    drawSettingRow(img, "Rule Variant", settings_.ruleVariant, leftMargin, startY,
                  labelWidth, controlWidth, spacing, SettingType::Dropdown);
    
    // Auto-detection
    startY += rowHeight;
    drawSettingRow(img, "Auto-Detection", std::to_string(settings_.autoDetection), 
                  leftMargin, startY, labelWidth, controlWidth, spacing, SettingType::Toggle);
    
    // Shot Timer
    startY += rowHeight;
    std::string timerText = settings_.shotTimer == 0 ? "Disabled" : 
                           std::to_string(settings_.shotTimer) + "s";
    drawSettingRow(img, "Shot Timer", timerText, leftMargin, startY,
                  labelWidth, controlWidth, spacing, SettingType::Dropdown);
}

void SettingsPage::drawDisplaySettings(cv::Mat& img) {
    currentControls_.clear();
    
    // Calculate responsive dimensions
    int contentPadding = UITheme::getResponsiveSpacing(40);
    int rowHeight = UITheme::getResponsiveSpacing(80);
    
    int totalWidth = contentRect_.width - (contentPadding * 2);
    int labelWidth = static_cast<int>(totalWidth * 0.35f);
    int controlWidth = static_cast<int>(totalWidth * 0.45f);
    int spacing = static_cast<int>(totalWidth * 0.05f);
    
    int leftMargin = contentRect_.x + contentPadding;
    int startY = contentRect_.y + UITheme::getResponsiveSpacing(20);
    
    // Fullscreen
    drawSettingRow(img, "Fullscreen", std::to_string(settings_.fullscreen), 
                  leftMargin, startY, labelWidth, controlWidth, spacing, SettingType::Toggle);
    
    // UI Scale
    startY += rowHeight;
    std::string scaleValue = std::to_string((settings_.uiScale - 0.5f) / 1.5f);
    drawSettingRow(img, "UI Scale", scaleValue, leftMargin, startY,
                  labelWidth, controlWidth, spacing, SettingType::Slider);
    
    // Show Overlay
    startY += rowHeight;
    drawSettingRow(img, "Show Game Overlay", std::to_string(settings_.showOverlay),
                  leftMargin, startY, labelWidth, controlWidth, spacing, SettingType::Toggle);
    
    // Show Velocity Vectors
    startY += rowHeight;
    drawSettingRow(img, "Show Velocity Vectors", std::to_string(settings_.showVelocityVectors),
                  leftMargin, startY, labelWidth, controlWidth, spacing, SettingType::Toggle);
    
    // Show Trajectories
    startY += rowHeight;
    drawSettingRow(img, "Show Trajectories", std::to_string(settings_.showTrajectories),
                  leftMargin, startY, labelWidth, controlWidth, spacing, SettingType::Toggle);
    
    // Color Scheme
    startY += rowHeight;
    drawSettingRow(img, "Color Scheme", settings_.colorScheme, leftMargin, startY,
                  labelWidth, controlWidth, spacing, SettingType::Dropdown);
}

void SettingsPage::drawButtons(cv::Mat& img) {
    // Modern enhanced buttons with state management
    UITheme::ComponentState backState = UITheme::ComponentState::Normal;
    UITheme::ComponentState saveState = UITheme::ComponentState::Normal; // Highlighted by default
    
    // Check hover states
    if (UITheme::isPointInRect(mousePos_, backButtonRect_)) {
        backState = UITheme::ComponentState::Hover;
    }
    if (UITheme::isPointInRect(mousePos_, saveButtonRect_)) {
        saveState = UITheme::ComponentState::Hover;
    }
    
    // Create button container background
    cv::Rect buttonArea = buttonRect_;
    buttonArea.x += UITheme::getResponsiveSpacing(20);
    buttonArea.width -= UITheme::getResponsiveSpacing(40);
    
    UITheme::drawGlassCard(img, buttonArea, 10.0f, 0.05f, UITheme::Colors::MediumBg);
    
    // Back button with icon
    UITheme::drawIconButton(img, "←", "Back", backButtonRect_, backState);
    
    // Save button (highlighted) with icon
    UITheme::drawIconButton(img, "✓", "Save", saveButtonRect_, saveState);
    
    // Add subtle pulsing effect to save button
    float pulse = std::sin(animationTime_ * 3.0f) * 0.1f + 0.9f;
    cv::Rect pulseRect = saveButtonRect_;
    pulseRect.x -= static_cast<int>(pulse * 3);
    pulseRect.y -= static_cast<int>(pulse * 3);
    pulseRect.width += static_cast<int>(pulse * 6);
    pulseRect.height += static_cast<int>(pulse * 6);
    
    cv::Scalar pulseColor(UITheme::Colors::NeonGreen[0] * pulse,
                         UITheme::Colors::NeonGreen[1] * pulse,
                         UITheme::Colors::NeonGreen[2] * pulse, 30);
    UITheme::drawRoundedRect(img, pulseRect, 
                           UITheme::getResponsiveSpacing(UITheme::Layout::BorderRadius + 2),
                           pulseColor, 2, true);
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
