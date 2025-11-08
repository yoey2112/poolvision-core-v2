#include "MainMenuPage.hpp"
#include <chrono>

namespace pv {

MainMenuPage::MainMenuPage()
    : selectedAction_(MenuAction::None)
    , windowWidth_(1280)
    , windowHeight_(720)
    , animationTime_(0.0f) {
}

void MainMenuPage::init() {
    selectedAction_ = MenuAction::None;
    animationTime_ = 0.0f;
    createMenuItems();
    updateLayout();
}

void MainMenuPage::setWindowSize(int width, int height) {
    windowWidth_ = width;
    windowHeight_ = height;
    updateLayout();
}

void MainMenuPage::createMenuItems() {
    menuItems_.clear();
    
    menuItems_.push_back({"New Game", "🎱", MenuAction::NewGame, cv::Rect()});
    menuItems_.push_back({"Drills & Practice", "🎯", MenuAction::Drills, cv::Rect()});
    menuItems_.push_back({"Player Profiles", "👤", MenuAction::PlayerProfiles, cv::Rect()});
    menuItems_.push_back({"Analytics", "📊", MenuAction::Analytics, cv::Rect()});
    menuItems_.push_back({"Settings", "⚙", MenuAction::Settings, cv::Rect()});
    menuItems_.push_back({"Calibration", "📷", MenuAction::Calibration, cv::Rect()});
    menuItems_.push_back({"Exit", "🚪", MenuAction::Exit, cv::Rect()});
}

void MainMenuPage::updateLayout() {
    int buttonWidth = UITheme::Layout::ButtonWidth;
    int buttonHeight = UITheme::Layout::ButtonHeight;
    int spacing = UITheme::Layout::Spacing;
    
    // Center buttons vertically and horizontally
    int startY = 220;  // Below logo/title
    int centerX = windowWidth_ / 2 - buttonWidth / 2;
    
    for (size_t i = 0; i < menuItems_.size(); ++i) {
        int y = startY + i * (buttonHeight + spacing);
        menuItems_[i].rect = cv::Rect(centerX, y, buttonWidth, buttonHeight);
    }
}

cv::Mat MainMenuPage::render() {
    // Create canvas
    cv::Mat img(windowHeight_, windowWidth_, CV_8UC3, UITheme::Colors::DarkBg);
    
    // Update animation time
    static auto startTime = std::chrono::steady_clock::now();
    auto currentTime = std::chrono::steady_clock::now();
    animationTime_ = std::chrono::duration<float>(currentTime - startTime).count();
    
    // Draw components
    drawBackground(img);
    drawLogo(img);
    drawMenuItems(img);
    drawFooter(img);
    
    return img;
}

void MainMenuPage::drawBackground(cv::Mat& img) {
    // Draw animated background
    UITheme::drawAnimatedBackground(img, animationTime_);
    
    // Add vignette effect
    cv::Mat overlay = img.clone();
    cv::circle(overlay, cv::Point(windowWidth_ / 2, windowHeight_ / 2),
              std::max(windowWidth_, windowHeight_) / 2,
              UITheme::Colors::DarkBg, -1);
    cv::addWeighted(overlay, 0.3, img, 0.7, 0, img);
}

void MainMenuPage::drawLogo(cv::Mat& img) {
    // Draw title area
    std::string title = "POOL VISION";
    std::string subtitle = "Computer Vision Billiards System";
    
    // Main title
    cv::Size titleSize = UITheme::getTextSize(title, UITheme::Fonts::FontFaceBold,
                                              UITheme::Fonts::TitleSize * 1.5,
                                              UITheme::Fonts::TitleThickness + 1);
    cv::Point titlePos((windowWidth_ - titleSize.width) / 2, 100);
    
    UITheme::drawTextWithShadow(img, title, titlePos, UITheme::Fonts::FontFaceBold,
                                UITheme::Fonts::TitleSize * 1.5,
                                UITheme::Colors::NeonCyan,
                                UITheme::Fonts::TitleThickness + 1, 4);
    
    // Subtitle
    cv::Size subtitleSize = UITheme::getTextSize(subtitle, UITheme::Fonts::FontFace,
                                                 UITheme::Fonts::BodySize,
                                                 UITheme::Fonts::BodyThickness);
    cv::Point subtitlePos((windowWidth_ - subtitleSize.width) / 2, 140);
    
    cv::putText(img, subtitle, subtitlePos, UITheme::Fonts::FontFace,
               UITheme::Fonts::BodySize, UITheme::Colors::TextSecondary,
               UITheme::Fonts::BodyThickness);
    
    // Draw decorative line
    int lineY = 160;
    int lineWidth = 400;
    int lineX = (windowWidth_ - lineWidth) / 2;
    cv::line(img, cv::Point(lineX, lineY), cv::Point(lineX + lineWidth, lineY),
            UITheme::Colors::NeonCyan, 2);
}

void MainMenuPage::drawMenuItems(cv::Mat& img) {
    for (auto& item : menuItems_) {
        // Draw button with hover effect
        bool isActive = (selectedAction_ == item.action);
        UITheme::drawButton(img, item.text, item.rect, 
                          item.isHovered, isActive, false);
        
        // Add icon indicator on the left
        if (!item.icon.empty()) {
            cv::Point iconPos(item.rect.x + 20, item.rect.y + item.rect.height / 2 + 10);
            cv::putText(img, item.icon, iconPos, UITheme::Fonts::FontFaceBold,
                       UITheme::Fonts::HeadingSize, UITheme::Colors::NeonYellow,
                       UITheme::Fonts::HeadingThickness);
        }
        
        // Add glow effect on hover
        if (item.isHovered) {
            cv::Rect glowRect = item.rect;
            glowRect.x -= 2;
            glowRect.y -= 2;
            glowRect.width += 4;
            glowRect.height += 4;
            UITheme::drawRoundedRect(img, glowRect, UITheme::Layout::BorderRadius,
                                   UITheme::Colors::NeonCyan, 1);
        }
    }
}

void MainMenuPage::drawFooter(cv::Mat& img) {
    // Version info
    std::string version = "v2.0.0 - Phase 2 Complete";
    cv::Size versionSize = UITheme::getTextSize(version, UITheme::Fonts::FontFace,
                                               UITheme::Fonts::SmallSize,
                                               UITheme::Fonts::BodyThickness);
    cv::Point versionPos((windowWidth_ - versionSize.width) / 2,
                        windowHeight_ - 20);
    
    cv::putText(img, version, versionPos, UITheme::Fonts::FontFace,
               UITheme::Fonts::SmallSize, UITheme::Colors::TextDisabled,
               UITheme::Fonts::BodyThickness);
    
    // Copyright
    std::string copyright = "(c) 2025 Pool Vision Core";
    cv::Size copyrightSize = UITheme::getTextSize(copyright, UITheme::Fonts::FontFace,
                                                  UITheme::Fonts::SmallSize,
                                                  UITheme::Fonts::BodyThickness);
    cv::Point copyrightPos((windowWidth_ - copyrightSize.width) / 2,
                          windowHeight_ - 5);
    
    cv::putText(img, copyright, copyrightPos, UITheme::Fonts::FontFace,
               UITheme::Fonts::SmallSize, UITheme::Colors::TextDisabled,
               UITheme::Fonts::BodyThickness);
}

void MainMenuPage::onMouse(int event, int x, int y, int flags) {
    mousePos_ = cv::Point(x, y);
    
    // Update hover state for all items
    for (auto& item : menuItems_) {
        item.isHovered = UITheme::isPointInRect(mousePos_, item.rect);
        
        // Handle click
        if (event == cv::EVENT_LBUTTONDOWN && item.isHovered) {
            selectedAction_ = item.action;
        }
    }
}

void MainMenuPage::onKey(int key) {
    // ESC to exit
    if (key == 27) {
        selectedAction_ = MenuAction::Exit;
    }
    
    // Number keys for quick selection
    if (key >= '1' && key <= '7') {
        int index = key - '1';
        if (index < static_cast<int>(menuItems_.size())) {
            selectedAction_ = menuItems_[index].action;
        }
    }
}

} // namespace pv
