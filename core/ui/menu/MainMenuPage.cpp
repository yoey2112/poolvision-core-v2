#include "MainMenuPage.hpp"
#include "../ResponsiveLayout.hpp"
#include <chrono>

namespace pv {

MainMenuPage::MainMenuPage()
    : selectedAction_(MenuAction::None)
    , windowWidth_(1280)
    , windowHeight_(720)
    , animationTime_(0.0f)
    , rootContainer_(nullptr) {
}

void MainMenuPage::init() {
    selectedAction_ = MenuAction::None;
    animationTime_ = 0.0f;
    
    // Initialize UITheme for responsive scaling
    UITheme::init(windowWidth_, windowHeight_);
    
    createMenuItems();
    createResponsiveLayout();
}

void MainMenuPage::setWindowSize(int width, int height) {
    windowWidth_ = width;
    windowHeight_ = height;
    
    // Update UITheme scaling
    UITheme::setWindowSize(width, height);
    
    // Recreate responsive layout with new dimensions
    createResponsiveLayout();
}

void MainMenuPage::createResponsiveLayout() {
    // Create root container with full window size
    cv::Rect windowRect(0, 0, windowWidth_, windowHeight_);
    rootContainer_ = std::make_unique<ResponsiveLayout::Container>(
        ResponsiveLayout::Direction::Column, windowRect);
    
    // Configure main layout structure
    rootContainer_->setJustify(ResponsiveLayout::Justify::SpaceBetween);
    rootContainer_->setAlignment(ResponsiveLayout::Alignment::Stretch);
    rootContainer_->setPadding(UITheme::getResponsiveSpacing(40));
    
    // Header section (logo area) - 20% of screen height
    headerRect_ = UITheme::getResponsiveRect(0, 0, 100, 20, windowRect);
    
    // Menu section (buttons area) - 60% of screen height  
    menuRect_ = UITheme::getResponsiveRect(0, 20, 100, 60, windowRect);
    
    // Footer section (version info) - 20% of screen height
    footerRect_ = UITheme::getResponsiveRect(0, 80, 100, 20, windowRect);
    
    // Update menu item positions within menu section
    updateMenuLayout();
}

void MainMenuPage::updateMenuLayout() {
    if (menuItems_.empty()) return;
    
    // Calculate responsive button dimensions
    cv::Size buttonSize = UITheme::getResponsiveSize(
        UITheme::Layout::ButtonWidth, 
        UITheme::Layout::ButtonHeight
    );
    
    int spacing = UITheme::getResponsiveSpacing(UITheme::Layout::Spacing);
    int maxWidth = static_cast<int>(windowWidth_ * 0.4f); // 40% of window width
    
    // Ensure button doesn't exceed max width
    buttonSize.width = std::min(buttonSize.width, maxWidth);
    
    // Calculate total height needed for all buttons
    int totalButtonsHeight = menuItems_.size() * buttonSize.height + 
                            (menuItems_.size() - 1) * spacing;
    
    // Center buttons in menu section
    int startY = menuRect_.y + (menuRect_.height - totalButtonsHeight) / 2;
    int centerX = windowWidth_ / 2 - buttonSize.width / 2;
    
    // Position each menu item
    for (size_t i = 0; i < menuItems_.size(); ++i) {
        int y = startY + i * (buttonSize.height + spacing);
        menuItems_[i].rect = cv::Rect(centerX, y, buttonSize.width, buttonSize.height);
        
        // Ensure buttons stay within screen bounds
        if (menuItems_[i].rect.y + menuItems_[i].rect.height > menuRect_.y + menuRect_.height) {
            // If buttons don't fit, reduce spacing
            spacing = UITheme::getResponsiveSpacing(UITheme::Layout::Spacing / 2);
            startY = menuRect_.y + (menuRect_.height - totalButtonsHeight) / 2;
            y = startY + i * (buttonSize.height + spacing);
            menuItems_[i].rect.y = y;
        }
    }
}

void MainMenuPage::createMenuItems() {
    menuItems_.clear();
    
    // Create menu items with vector-style icons (will be enhanced later)
    menuItems_.push_back({"New Game", "⚡", MenuAction::NewGame, cv::Rect()});
    menuItems_.push_back({"Drills & Practice", "🎯", MenuAction::Drills, cv::Rect()});
    menuItems_.push_back({"Player Profiles", "👤", MenuAction::PlayerProfiles, cv::Rect()});
    menuItems_.push_back({"Analytics", "📊", MenuAction::Analytics, cv::Rect()});
    menuItems_.push_back({"Settings", "⚙", MenuAction::Settings, cv::Rect()});
    menuItems_.push_back({"Calibration", "📷", MenuAction::Calibration, cv::Rect()});
    menuItems_.push_back({"Exit", "🚪", MenuAction::Exit, cv::Rect()});
}

// Legacy method for compatibility
void MainMenuPage::updateLayout() {
    updateMenuLayout();
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
    // Fill with dark background
    img.setTo(UITheme::Colors::DarkBg);
    
    // Draw animated background with particles
    UITheme::drawAnimatedBackground(img, animationTime_, 0.3f);
    
    // Add glass-morphism overlay for depth
    cv::Rect overlayRect(0, 0, windowWidth_, windowHeight_);
    UITheme::applyGlassMorphism(img, overlayRect, 10, 0.1f, UITheme::Colors::MediumBg);
    
    // Add subtle vignette effect
    cv::Point center(windowWidth_ / 2, windowHeight_ / 2);
    int maxRadius = std::max(windowWidth_, windowHeight_) / 2;
    
    cv::Mat overlay = cv::Mat::zeros(img.size(), img.type());
    cv::circle(overlay, center, maxRadius, UITheme::Colors::DarkBg, -1);
    cv::GaussianBlur(overlay, overlay, cv::Size(151, 151), 0);
    cv::addWeighted(overlay, 0.2, img, 0.8, 0, img);
}

void MainMenuPage::drawLogo(cv::Mat& img) {
    // Calculate responsive font sizes
    double titleSize = UITheme::getResponsiveFontSize(UITheme::Fonts::TitleSize * 1.8);
    double subtitleSize = UITheme::getResponsiveFontSize(UITheme::Fonts::BodySize * 1.2);
    
    // Main title with glass card background
    std::string title = "POOL VISION";
    std::string subtitle = "Computer Vision Billiards System";
    
    cv::Size titleTextSize = UITheme::getTextSize(title, UITheme::Fonts::FontFaceBold,
                                                 titleSize, UITheme::Fonts::TitleThickness + 1);
    
    // Create glass card for title area
    int titleCardPadding = UITheme::getResponsiveSpacing(30);
    cv::Rect titleCard(
        headerRect_.x + (headerRect_.width - titleTextSize.width - titleCardPadding * 2) / 2,
        headerRect_.y + UITheme::getResponsiveSpacing(20),
        titleTextSize.width + titleCardPadding * 2,
        titleTextSize.height + UITheme::getResponsiveSpacing(60)
    );
    
    // Draw glass card background
    UITheme::drawGlassCard(img, titleCard, 15.0f, 0.1f, UITheme::Colors::MediumBg);
    UITheme::drawRoundedRect(img, titleCard, UITheme::getResponsiveSpacing(15), 
                           cv::Scalar(UITheme::Colors::MediumBg[0], 
                                     UITheme::Colors::MediumBg[1], 
                                     UITheme::Colors::MediumBg[2], 120), -1, true);
    
    // Draw border with neon effect
    UITheme::drawRoundedRect(img, titleCard, UITheme::getResponsiveSpacing(15),
                           UITheme::Colors::NeonCyan, 2, true);
    
    // Draw main title with enhanced shadow
    cv::Point titlePos(titleCard.x + titleCardPadding,
                      titleCard.y + titleCardPadding + titleTextSize.height);
    
    UITheme::drawTextWithShadow(img, title, titlePos, UITheme::Fonts::FontFaceBold,
                                titleSize, UITheme::Colors::NeonCyan,
                                UITheme::Fonts::TitleThickness + 1, 
                                UITheme::getResponsiveSpacing(4), true);
    
    // Draw subtitle
    cv::Size subtitleTextSize = UITheme::getTextSize(subtitle, UITheme::Fonts::FontFace,
                                                    subtitleSize, UITheme::Fonts::BodyThickness);
    cv::Point subtitlePos((windowWidth_ - subtitleTextSize.width) / 2,
                         titleCard.y + titleCard.height + UITheme::getResponsiveSpacing(25));
    
    UITheme::drawText(img, subtitle, subtitlePos, subtitleSize,
                     UITheme::Colors::TextSecondary, UITheme::Fonts::BodyThickness, true);
    
    // Draw decorative elements
    int lineY = subtitlePos.y + UITheme::getResponsiveSpacing(15);
    int lineWidth = UITheme::getResponsiveSpacing(400);
    int lineX = (windowWidth_ - lineWidth) / 2;
    
    // Gradient line effect
    for (int i = 0; i < 3; ++i) {
        cv::Scalar lineColor = UITheme::Colors::NeonCyan;
        lineColor[3] = 255 - i * 80; // Fade effect
        cv::line(img, cv::Point(lineX, lineY + i), cv::Point(lineX + lineWidth, lineY + i),
                lineColor, 1);
    }
}

void MainMenuPage::drawMenuItems(cv::Mat& img) {
    for (size_t i = 0; i < menuItems_.size(); ++i) {
        auto& item = menuItems_[i];
        
        // Determine component state
        UITheme::ComponentState state = UITheme::ComponentState::Normal;
        if (selectedAction_ == item.action) {
            state = UITheme::ComponentState::Active;
        } else if (item.isHovered) {
            state = UITheme::ComponentState::Hover;
        }
        
        // Create animation state for smooth transitions
        UITheme::AnimationState animState;
        animState.isAnimating = item.isHovered || (selectedAction_ == item.action);
        animState.progress = std::sin(animationTime_ * 3.0f + i * 0.5f) * 0.5f + 0.5f;
        
        // Draw enhanced button with state and animation
        UITheme::drawButton(img, item.text, item.rect, state, animState);
        
        // Draw modern icon
        if (!item.icon.empty()) {
            int iconSize = UITheme::getResponsiveSpacing(24);
            cv::Point iconPos(item.rect.x + UITheme::getResponsiveSpacing(20), 
                            item.rect.y + (item.rect.height - iconSize) / 2);
            
            // Draw icon with state-based color
            cv::Scalar iconColor = UITheme::Colors::NeonYellow;
            if (state == UITheme::ComponentState::Active) {
                iconColor = UITheme::Colors::NeonCyan;
            } else if (state == UITheme::ComponentState::Hover) {
                iconColor = UITheme::Colors::NeonGreen;
            }
            
            // Enhanced icon rendering (simplified vector style)
            drawModernIcon(img, item.icon, iconPos, iconSize, iconColor);
        }
        
        // Add premium glow effect for active/hovered items
        if (item.isHovered || selectedAction_ == item.action) {
            cv::Rect glowRect = item.rect;
            int glowExpansion = UITheme::getResponsiveSpacing(8);
            glowRect.x -= glowExpansion;
            glowRect.y -= glowExpansion;
            glowRect.width += 2 * glowExpansion;
            glowRect.height += 2 * glowExpansion;
            
            cv::Scalar glowColor = (selectedAction_ == item.action) ? 
                                  UITheme::Colors::NeonCyan : UITheme::Colors::NeonYellow;
            glowColor[3] = 60; // Semi-transparent
            
            UITheme::drawRoundedRect(img, glowRect, 
                                   UITheme::getResponsiveSpacing(UITheme::Layout::BorderRadius + 4),
                                   glowColor, 3, true);
        }
        
        // Add entry animation for menu items
        if (animationTime_ < 2.0f) {
            float itemDelay = i * 0.1f;
            if (animationTime_ > itemDelay) {
                float itemProgress = std::min((animationTime_ - itemDelay) / 0.5f, 1.0f);
                float easeProgress = UITheme::easeOut(itemProgress);
                
                // Slide in from right
                cv::Rect animRect = item.rect;
                int slideOffset = static_cast<int>((1.0f - easeProgress) * 200);
                animRect.x += slideOffset;
                
                // Fade in
                float alpha = easeProgress;
                cv::Scalar overlayColor(0, 0, 0, static_cast<int>((1.0f - alpha) * 255));
                UITheme::drawRoundedRect(img, item.rect, 
                                       UITheme::getResponsiveSpacing(UITheme::Layout::BorderRadius),
                                       overlayColor, -1, true);
            }
        }
    }
}

void MainMenuPage::drawModernIcon(cv::Mat& img, const std::string& icon, 
                                 const cv::Point& pos, int size, const cv::Scalar& color) {
    // Enhanced icon rendering - replace with actual vector icons later
    cv::Point center(pos.x + size/2, pos.y + size/2);
    int radius = size / 3;
    
    if (icon == "⚡") {
        // Lightning bolt style
        std::vector<cv::Point> lightning = {
            cv::Point(center.x - radius/2, center.y - radius),
            cv::Point(center.x + radius/2, center.y),
            cv::Point(center.x, center.y),
            cv::Point(center.x + radius/2, center.y + radius),
            cv::Point(center.x - radius/2, center.y)
        };
        cv::fillPoly(img, lightning, color);
    } else if (icon == "🎯") {
        // Target circles
        cv::circle(img, center, radius, color, 2);
        cv::circle(img, center, radius/2, color, 2);
        cv::circle(img, center, 3, color, -1);
    } else if (icon == "👤") {
        // Person silhouette
        cv::circle(img, cv::Point(center.x, center.y - radius/3), radius/3, color, 2);
        cv::ellipse(img, cv::Point(center.x, center.y + radius/3), 
                   cv::Size(radius/2, radius/2), 0, 0, 180, color, 2);
    } else if (icon == "📊") {
        // Bar chart
        int barWidth = radius/3;
        for (int i = 0; i < 3; ++i) {
            int barHeight = (i + 1) * radius/2;
            cv::Rect bar(center.x - radius + i * barWidth * 2, 
                        center.y + radius - barHeight,
                        barWidth, barHeight);
            cv::rectangle(img, bar, color, -1);
        }
    } else if (icon == "⚙") {
        // Gear
        cv::circle(img, center, radius, color, 2);
        cv::circle(img, center, radius/3, color, -1);
        for (int i = 0; i < 8; ++i) {
            float angle = i * CV_PI / 4;
            cv::Point tooth = center + cv::Point(
                static_cast<int>(std::cos(angle) * radius * 1.3),
                static_cast<int>(std::sin(angle) * radius * 1.3)
            );
            cv::circle(img, tooth, 2, color, -1);
        }
    } else {
        // Fallback: simple circle
        cv::circle(img, center, radius, color, 2);
        cv::circle(img, center, 3, color, -1);
    }
}

void MainMenuPage::drawFooter(cv::Mat& img) {
    // Create glass card for footer
    cv::Rect footerCard(footerRect_.x + UITheme::getResponsiveSpacing(20),
                       footerRect_.y + UITheme::getResponsiveSpacing(20),
                       footerRect_.width - UITheme::getResponsiveSpacing(40),
                       footerRect_.height - UITheme::getResponsiveSpacing(40));
    
    UITheme::drawGlassCard(img, footerCard, 8.0f, 0.05f, UITheme::Colors::LightBg);
    
    // Version info with responsive sizing
    std::string version = "v2.0.0 - Phase 2 Complete";
    double versionSize = UITheme::getResponsiveFontSize(UITheme::Fonts::SmallSize);
    cv::Size versionTextSize = UITheme::getTextSize(version, UITheme::Fonts::FontFace,
                                                   versionSize, UITheme::Fonts::BodyThickness);
    cv::Point versionPos((windowWidth_ - versionTextSize.width) / 2,
                        footerCard.y + UITheme::getResponsiveSpacing(25));
    
    UITheme::drawText(img, version, versionPos, versionSize,
                     UITheme::Colors::TextSecondary, UITheme::Fonts::BodyThickness, true);
    
    // Copyright with glow effect
    std::string copyright = "© 2025 Pool Vision Core";
    cv::Size copyrightTextSize = UITheme::getTextSize(copyright, UITheme::Fonts::FontFace,
                                                     versionSize, UITheme::Fonts::BodyThickness);
    cv::Point copyrightPos((windowWidth_ - copyrightTextSize.width) / 2,
                          versionPos.y + UITheme::getResponsiveSpacing(20));
    
    UITheme::drawText(img, copyright, copyrightPos, versionSize,
                     UITheme::Colors::TextDisabled, UITheme::Fonts::BodyThickness, true);
    
    // Add subtle pulse effect to footer
    float pulse = std::sin(animationTime_ * 2.0f) * 0.2f + 0.8f;
    cv::Scalar pulseColor(UITheme::Colors::NeonCyan[0] * pulse,
                         UITheme::Colors::NeonCyan[1] * pulse,
                         UITheme::Colors::NeonCyan[2] * pulse,
                         30);
    UITheme::drawRoundedRect(img, footerCard, UITheme::getResponsiveSpacing(8),
                           pulseColor, 1, true);
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
