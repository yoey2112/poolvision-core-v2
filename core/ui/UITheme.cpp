#include "UITheme.hpp"
#include <cmath>

namespace pv {

// Color definitions
const cv::Scalar UITheme::Colors::TableGreen = cv::Scalar(58, 94, 13);      // BGR: #0D5E3A
const cv::Scalar UITheme::Colors::DarkBg = cv::Scalar(26, 26, 26);          // BGR: #1A1A1A
const cv::Scalar UITheme::Colors::MediumBg = cv::Scalar(42, 42, 42);        // BGR: #2A2A2A
const cv::Scalar UITheme::Colors::LightBg = cv::Scalar(58, 58, 58);         // BGR: #3A3A3A
const cv::Scalar UITheme::Colors::TextShadow = cv::Scalar(0, 0, 0, 150);    // Semi-transparent black

const cv::Scalar UITheme::Colors::NeonCyan = cv::Scalar(255, 255, 0);       // BGR: #00FFFF
const cv::Scalar UITheme::Colors::NeonYellow = cv::Scalar(0, 215, 255);     // BGR: #FFD700
const cv::Scalar UITheme::Colors::NeonGreen = cv::Scalar(0, 255, 0);        // BGR: #00FF00
const cv::Scalar UITheme::Colors::NeonRed = cv::Scalar(102, 0, 255);        // BGR: #FF0066

const cv::Scalar UITheme::Colors::TextPrimary = cv::Scalar(255, 255, 255);  // BGR: #FFFFFF
const cv::Scalar UITheme::Colors::TextSecondary = cv::Scalar(204, 204, 204);// BGR: #CCCCCC
const cv::Scalar UITheme::Colors::TextDisabled = cv::Scalar(102, 102, 102); // BGR: #666666

const cv::Scalar UITheme::Colors::ButtonDefault = UITheme::Colors::MediumBg;
const cv::Scalar UITheme::Colors::ButtonHover = UITheme::Colors::LightBg;
const cv::Scalar UITheme::Colors::ButtonActive = UITheme::Colors::TableGreen;
const cv::Scalar UITheme::Colors::ButtonDisabled = cv::Scalar(40, 40, 40);
const cv::Scalar UITheme::Colors::BorderColor = UITheme::Colors::NeonCyan;
const cv::Scalar UITheme::Colors::ShadowColor = cv::Scalar(0, 0, 0);

void UITheme::drawButton(cv::Mat& img, const std::string& text, 
                        const cv::Rect& rect, bool isHovered, 
                        bool isActive, bool isDisabled) {
    // Choose colors based on state
    cv::Scalar bgColor = Colors::ButtonDefault;
    cv::Scalar borderColor = Colors::BorderColor;
    cv::Scalar textColor = Colors::TextPrimary;
    
    if (isDisabled) {
        bgColor = Colors::ButtonDisabled;
        textColor = Colors::TextDisabled;
        borderColor = Colors::TextDisabled;
    } else if (isActive) {
        bgColor = Colors::ButtonActive;
        borderColor = Colors::NeonYellow;
    } else if (isHovered) {
        bgColor = Colors::ButtonHover;
        borderColor = Colors::NeonYellow;
    }
    
    // Draw shadow
    cv::Rect shadowRect = rect + cv::Point(Layout::ShadowOffset, Layout::ShadowOffset);
    drawRoundedRect(img, shadowRect, Layout::BorderRadius, 
                   cv::Scalar(0, 0, 0, 100), -1);
    
    // Draw button background
    drawRoundedRect(img, rect, Layout::BorderRadius, bgColor, -1);
    
    // Draw border
    drawRoundedRect(img, rect, Layout::BorderRadius, borderColor, 2);
    
    // Draw text
    cv::Size textSize = getTextSize(text, Fonts::FontFaceBold, 
                                    Fonts::ButtonSize, Fonts::ButtonThickness);
    cv::Point textPos(rect.x + (rect.width - textSize.width) / 2,
                      rect.y + (rect.height + textSize.height) / 2);
    
    drawTextWithShadow(img, text, textPos, Fonts::FontFaceBold,
                      Fonts::ButtonSize, textColor, Fonts::ButtonThickness);
}

void UITheme::drawIconButton(cv::Mat& img, const std::string& icon,
                            const std::string& text, const cv::Rect& rect,
                            bool isHovered) {
    drawButton(img, text, rect, isHovered);
    
    // Draw icon (simplified - just draw first letter as placeholder)
    if (!icon.empty()) {
        cv::Point iconPos(rect.x + Layout::Padding, 
                         rect.y + rect.height / 2 + 10);
        cv::putText(img, icon.substr(0, 1), iconPos, Fonts::FontFaceBold,
                   Fonts::HeadingSize, Colors::NeonCyan, Fonts::HeadingThickness);
    }
}

void UITheme::drawCard(cv::Mat& img, const cv::Rect& rect,
                      const cv::Scalar& bgColor, int alpha) {
    // Draw shadow
    cv::Rect shadowRect = rect + cv::Point(Layout::ShadowOffset, Layout::ShadowOffset);
    drawRoundedRect(img, shadowRect, Layout::BorderRadius,
                   cv::Scalar(0, 0, 0, 80), -1);
    
    // Draw card background with transparency
    cv::Mat overlay = img.clone();
    drawRoundedRect(overlay, rect, Layout::BorderRadius, bgColor, -1);
    
    // Blend with alpha
    double alphaVal = alpha / 255.0;
    cv::addWeighted(overlay, alphaVal, img, 1.0 - alphaVal, 0, img);
    
    // Draw subtle border
    drawRoundedRect(img, rect, Layout::BorderRadius, Colors::LightBg, 1);
}

void UITheme::drawTextWithShadow(cv::Mat& img, const std::string& text,
                                const cv::Point& pos, int fontFace,
                                double scale, const cv::Scalar& color,
                                int thickness, int shadowOffset) {
    // Draw shadow
    cv::Point shadowPos = pos + cv::Point(shadowOffset, shadowOffset);
    cv::putText(img, text, shadowPos, fontFace, scale, 
               Colors::ShadowColor, thickness);
    
    // Draw text
    cv::putText(img, text, pos, fontFace, scale, color, thickness);
}

void UITheme::drawTitleBar(cv::Mat& img, const std::string& title, int height) {
    // Draw background gradient effect
    cv::Rect titleRect(0, 0, img.cols, height);
    cv::rectangle(img, titleRect, Colors::DarkBg, -1);
    
    // Draw bottom border with gradient
    cv::line(img, cv::Point(0, height), cv::Point(img.cols, height),
            Colors::NeonCyan, 2);
    
    // Draw title text
    cv::Size textSize = getTextSize(title, Fonts::FontFaceBold,
                                    Fonts::TitleSize, Fonts::TitleThickness);
    cv::Point textPos((img.cols - textSize.width) / 2,
                     (height + textSize.height) / 2);
    
    drawTextWithShadow(img, title, textPos, Fonts::FontFaceBold,
                      Fonts::TitleSize, Colors::TextPrimary, 
                      Fonts::TitleThickness, 3);
}

void UITheme::drawProgressBar(cv::Mat& img, float progress,
                             const cv::Rect& rect, const cv::Scalar& color) {
    // Clamp progress
    progress = std::max(0.0f, std::min(1.0f, progress));
    
    // Draw background
    drawRoundedRect(img, rect, 5, Colors::MediumBg, -1);
    
    // Draw progress
    int fillWidth = static_cast<int>(rect.width * progress);
    if (fillWidth > 0) {
        cv::Rect fillRect(rect.x, rect.y, fillWidth, rect.height);
        drawRoundedRect(img, fillRect, 5, color, -1);
    }
    
    // Draw border
    drawRoundedRect(img, rect, 5, Colors::BorderColor, 1);
}

void UITheme::drawToggle(cv::Mat& img, bool isOn, const cv::Rect& rect) {
    // Toggle background
    cv::Scalar bgColor = isOn ? Colors::NeonGreen : Colors::MediumBg;
    drawRoundedRect(img, rect, rect.height / 2, bgColor, -1);
    
    // Toggle circle
    int circleRadius = rect.height / 2 - 4;
    int circleX = isOn ? rect.x + rect.width - circleRadius - 4 :
                        rect.x + circleRadius + 4;
    int circleY = rect.y + rect.height / 2;
    
    cv::circle(img, cv::Point(circleX, circleY), circleRadius,
              Colors::TextPrimary, -1);
}

void UITheme::drawSlider(cv::Mat& img, float value, const cv::Rect& rect,
                        float min, float max) {
    // Clamp value
    value = std::max(min, std::min(max, value));
    float normalized = (value - min) / (max - min);
    
    // Draw track
    cv::Rect trackRect(rect.x, rect.y + rect.height / 2 - 3,
                      rect.width, 6);
    drawRoundedRect(img, trackRect, 3, Colors::MediumBg, -1);
    
    // Draw fill
    int fillWidth = static_cast<int>(rect.width * normalized);
    if (fillWidth > 0) {
        cv::Rect fillRect(rect.x, trackRect.y, fillWidth, trackRect.height);
        drawRoundedRect(img, fillRect, 3, Colors::NeonCyan, -1);
    }
    
    // Draw handle
    int handleX = rect.x + fillWidth;
    int handleY = rect.y + rect.height / 2;
    cv::circle(img, cv::Point(handleX, handleY), 10, Colors::TextPrimary, -1);
    cv::circle(img, cv::Point(handleX, handleY), 10, Colors::NeonCyan, 2);
}

void UITheme::drawTabBar(cv::Mat& img, const std::vector<std::string>& tabs,
                        int activeTab, const cv::Rect& rect) {
    if (tabs.empty()) return;
    
    int tabWidth = rect.width / static_cast<int>(tabs.size());
    
    for (size_t i = 0; i < tabs.size(); ++i) {
        cv::Rect tabRect(rect.x + i * tabWidth, rect.y, tabWidth, rect.height);
        
        bool isActive = (static_cast<int>(i) == activeTab);
        cv::Scalar bgColor = isActive ? Colors::TableGreen : Colors::MediumBg;
        cv::Scalar textColor = isActive ? Colors::TextPrimary : Colors::TextSecondary;
        
        // Draw tab background
        cv::rectangle(img, tabRect, bgColor, -1);
        
        // Draw bottom border for active tab
        if (isActive) {
            cv::line(img, cv::Point(tabRect.x, tabRect.y + tabRect.height - 2),
                    cv::Point(tabRect.x + tabRect.width, tabRect.y + tabRect.height - 2),
                    Colors::NeonCyan, 3);
        }
        
        // Draw text
        cv::Size textSize = getTextSize(tabs[i], Fonts::FontFace,
                                       Fonts::BodySize, Fonts::BodyThickness);
        cv::Point textPos(tabRect.x + (tabRect.width - textSize.width) / 2,
                         tabRect.y + (tabRect.height + textSize.height) / 2);
        
        cv::putText(img, tabs[i], textPos, Fonts::FontFace,
                   Fonts::BodySize, textColor, Fonts::BodyThickness);
    }
}

void UITheme::drawDropdown(cv::Mat& img, const std::string& selected,
                          const cv::Rect& rect, bool isOpen) {
    // Draw background
    drawRoundedRect(img, rect, 5, Colors::MediumBg, -1);
    drawRoundedRect(img, rect, 5, Colors::BorderColor, 1);
    
    // Draw selected text
    cv::Point textPos(rect.x + Layout::Padding,
                     rect.y + rect.height / 2 + 5);
    cv::putText(img, selected, textPos, Fonts::FontFace,
               Fonts::BodySize, Colors::TextPrimary, Fonts::BodyThickness);
    
    // Draw arrow
    int arrowX = rect.x + rect.width - 20;
    int arrowY = rect.y + rect.height / 2;
    std::string arrow = isOpen ? "^" : "v";
    cv::putText(img, arrow, cv::Point(arrowX, arrowY + 5), Fonts::FontFace,
               Fonts::BodySize, Colors::NeonCyan, Fonts::BodyThickness);
}

void UITheme::drawAnimatedBackground(cv::Mat& img, float time) {
    // Create subtle animated gradient effect
    for (int y = 0; y < img.rows; y += 40) {
        float wave = std::sin(time + y * 0.01f) * 10.0f;
        int alpha = static_cast<int>(20 + wave);
        
        cv::line(img, cv::Point(0, y), cv::Point(img.cols, y),
                Colors::TableGreen, 1);
    }
}

void UITheme::applyGlassMorphism(cv::Mat& img, const cv::Rect& rect,
                                int blurAmount, int alpha) {
    // Extract region
    cv::Mat roi = img(rect);
    
    // Apply blur
    cv::Mat blurred;
    cv::GaussianBlur(roi, blurred, cv::Size(blurAmount, blurAmount), 0);
    
    // Blend back with alpha
    double alphaVal = alpha / 255.0;
    cv::addWeighted(blurred, alphaVal, roi, 1.0 - alphaVal, 0, roi);
}

void UITheme::drawRoundedRect(cv::Mat& img, const cv::Rect& rect,
                             int radius, const cv::Scalar& color,
                             int thickness) {
    // Simplified rounded rectangle (draws regular rectangle with slight rounding)
    // For full rounded corners, would need more complex polygon drawing
    
    if (thickness < 0) {
        // Filled rectangle
        cv::rectangle(img, rect, color, -1);
    } else {
        // Outlined rectangle
        cv::rectangle(img, rect, color, thickness);
    }
    
    // Add corner circles for visual rounding effect
    if (radius > 0 && thickness < 0) {
        cv::circle(img, cv::Point(rect.x, rect.y), radius, color, -1);
        cv::circle(img, cv::Point(rect.x + rect.width, rect.y), radius, color, -1);
        cv::circle(img, cv::Point(rect.x, rect.y + rect.height), radius, color, -1);
        cv::circle(img, cv::Point(rect.x + rect.width, rect.y + rect.height), 
                  radius, color, -1);
    }
}

bool UITheme::isPointInRect(const cv::Point& pt, const cv::Rect& rect,
                           int hoverRadius) {
    cv::Rect expandedRect = rect;
    if (hoverRadius > 0) {
        expandedRect.x -= hoverRadius;
        expandedRect.y -= hoverRadius;
        expandedRect.width += 2 * hoverRadius;
        expandedRect.height += 2 * hoverRadius;
    }
    
    return expandedRect.contains(pt);
}

cv::Size UITheme::getTextSize(const std::string& text, int fontFace,
                             double scale, int thickness) {
    int baseline = 0;
    return cv::getTextSize(text, fontFace, scale, thickness, &baseline);
}

} // namespace pv
