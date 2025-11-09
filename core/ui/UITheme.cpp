#include "UITheme.hpp"
#include "ResponsiveLayout.hpp"
#include <cmath>
#include <memory>
#include <chrono>

namespace pv {

// Static members
std::unique_ptr<ModernTheme> UITheme::modernTheme_;
int UITheme::windowWidth_ = 1280;
int UITheme::windowHeight_ = 720;
bool UITheme::initialized_ = false;

// Color definitions (backward compatibility)
const cv::Scalar UITheme::Colors::TableGreen = cv::Scalar(58, 94, 13);      // BGR: #0D5E3A
const cv::Scalar UITheme::Colors::DarkBg = cv::Scalar(26, 26, 26);          // BGR: #1A1A1A
const cv::Scalar UITheme::Colors::MediumBg = cv::Scalar(42, 42, 42);        // BGR: #2A2A2A
const cv::Scalar UITheme::Colors::LightBg = cv::Scalar(58, 58, 58);         // BGR: #3A3A3A
const cv::Scalar UITheme::Colors::TextShadow = cv::Scalar(0, 0, 0, 150);    // Semi-transparent black

const cv::Scalar UITheme::Colors::NeonCyan = cv::Scalar(255, 255, 0);       // BGR: #00FFFF
const cv::Scalar UITheme::Colors::NeonYellow = cv::Scalar(0, 215, 255);     // BGR: #FFD700
const cv::Scalar UITheme::Colors::NeonGreen = cv::Scalar(0, 255, 0);        // BGR: #00FF00
const cv::Scalar UITheme::Colors::NeonRed = cv::Scalar(102, 0, 255);        // BGR: #FF0066
const cv::Scalar UITheme::Colors::NeonOrange = cv::Scalar(0, 102, 255);     // BGR: #FF6600
const cv::Scalar UITheme::Colors::NeonBlue = cv::Scalar(255, 102, 0);       // BGR: #0066FF

const cv::Scalar UITheme::Colors::TextPrimary = cv::Scalar(255, 255, 255);  // BGR: #FFFFFF
const cv::Scalar UITheme::Colors::TextSecondary = cv::Scalar(204, 204, 204);// BGR: #CCCCCC
const cv::Scalar UITheme::Colors::TextDisabled = cv::Scalar(102, 102, 102); // BGR: #666666

const cv::Scalar UITheme::Colors::ButtonDefault = UITheme::Colors::MediumBg;
const cv::Scalar UITheme::Colors::ButtonHover = UITheme::Colors::LightBg;
const cv::Scalar UITheme::Colors::ButtonActive = UITheme::Colors::TableGreen;
const cv::Scalar UITheme::Colors::ButtonDisabled = cv::Scalar(40, 40, 40);
const cv::Scalar UITheme::Colors::BorderColor = UITheme::Colors::NeonCyan;
const cv::Scalar UITheme::Colors::ShadowColor = cv::Scalar(0, 0, 0);

// Global theme management
void UITheme::init(int windowWidth, int windowHeight) {
    if (!modernTheme_) {
        modernTheme_ = std::make_unique<ModernTheme>();
    }
    
    windowWidth_ = windowWidth;
    windowHeight_ = windowHeight;
    initialized_ = true;
    
    // Set initial scale based on window size
    float scale = std::min(
        static_cast<float>(windowWidth) / 1280.0f,
        static_cast<float>(windowHeight) / 720.0f
    );
    modernTheme_->setScale(scale);
}

void UITheme::setWindowSize(int width, int height) {
    windowWidth_ = width;
    windowHeight_ = height;
    
    if (modernTheme_) {
        // Update scale based on new size
        float scale = std::min(
            static_cast<float>(width) / 1280.0f,
            static_cast<float>(height) / 720.0f
        );
        modernTheme_->setScale(scale);
    }
}

void UITheme::setDarkMode(bool enabled) {
    if (!modernTheme_) init();
    modernTheme_->setDarkMode(enabled);
}

void UITheme::setHighContrast(bool enabled) {
    if (!modernTheme_) init();
    modernTheme_->setHighContrast(enabled);
}

void UITheme::setColorBlindMode(bool enabled) {
    if (!modernTheme_) init();
    modernTheme_->setColorBlindMode(enabled);
}

void UITheme::setScale(float scale) {
    if (!modernTheme_) init();
    modernTheme_->setScale(scale);
}

// Responsive utilities
cv::Size UITheme::getResponsiveSize(int baseWidth, int baseHeight) {
    return ResponsiveLayout::getResponsiveSize(baseWidth, baseHeight, 
                                             windowWidth_, windowHeight_);
}

double UITheme::getResponsiveFontSize(double baseSize) {
    if (!modernTheme_) return baseSize;
    return modernTheme_->getResponsiveFontSize(baseSize, windowWidth_, windowHeight_);
}

int UITheme::getResponsiveSpacing(int baseSpacing) {
    if (!modernTheme_) return baseSpacing;
    return modernTheme_->getResponsiveSpacing(baseSpacing, windowWidth_);
}

cv::Rect UITheme::getResponsiveRect(float x, float y, float width, float height, const cv::Rect& parent) {
    return ResponsiveLayout::getPercentRect(x, y, width, height, parent);
}

// State management
cv::Scalar UITheme::getStateColor(const cv::Scalar& baseColor, ComponentState state) {
    if (!modernTheme_) return baseColor;
    
    switch (state) {
        case ComponentState::Hover:
            return modernTheme_->getStateColor(baseColor, "hover");
        case ComponentState::Active:
            return modernTheme_->getStateColor(baseColor, "active");
        case ComponentState::Disabled:
            return modernTheme_->getStateColor(baseColor, "disabled");
        default:
            return baseColor;
    }
}

float UITheme::getStateOpacity(ComponentState state) {
    switch (state) {
        case ComponentState::Hover: return 0.9f;
        case ComponentState::Active: return 0.8f;
        case ComponentState::Disabled: return 0.5f;
        case ComponentState::Focused: return 1.0f;
        default: return 1.0f;
    }
}

// Enhanced drawing functions
void UITheme::drawButton(cv::Mat& img, const std::string& text, 
                        const cv::Rect& rect, ComponentState state,
                        const AnimationState& anim) {
    // Calculate responsive size
    cv::Rect responsiveRect = rect;
    if (anim.isAnimating) {
        float scale = 1.0f + (anim.progress * 0.05f); // Slight scale on animation
        int deltaW = static_cast<int>((responsiveRect.width * scale) - responsiveRect.width);
        int deltaH = static_cast<int>((responsiveRect.height * scale) - responsiveRect.height);
        responsiveRect.x -= deltaW / 2;
        responsiveRect.y -= deltaH / 2;
        responsiveRect.width += deltaW;
        responsiveRect.height += deltaH;
    }
    
    // Choose colors based on state
    cv::Scalar bgColor = getStateColor(Colors::ButtonDefault, state);
    cv::Scalar borderColor = (state == ComponentState::Focused) ? Colors::NeonYellow : Colors::BorderColor;
    cv::Scalar textColor = (state == ComponentState::Disabled) ? Colors::TextDisabled : Colors::TextPrimary;
    
    // Draw shadow with elevation
    int elevation = (state == ComponentState::Hover) ? 6 : 2;
    if (state != ComponentState::Disabled) {
        cv::Rect shadowRect = responsiveRect;
        shadowRect.x += elevation;
        shadowRect.y += elevation;
        drawRoundedRect(img, shadowRect, Layout::BorderRadius, 
                       cv::Scalar(0, 0, 0, 100), -1, true);
    }
    
    // Draw button background with glass effect
    drawGlassCard(img, responsiveRect, 10.0f, 0.9f);
    drawRoundedRect(img, responsiveRect, getResponsiveSpacing(Layout::BorderRadius), 
                   bgColor, -1, true);
    
    // Draw border
    if (state == ComponentState::Focused) {
        drawRoundedRect(img, responsiveRect, getResponsiveSpacing(Layout::BorderRadius), 
                       borderColor, 3, true);
    } else {
        drawRoundedRect(img, responsiveRect, getResponsiveSpacing(Layout::BorderRadius), 
                       borderColor, 2, true);
    }
    
    // Draw text with responsive sizing
    double fontSize = getResponsiveFontSize(Fonts::ButtonSize);
    cv::Size textSize = getTextSize(text, Fonts::FontFace, fontSize, Fonts::ButtonThickness, true);
    cv::Point textPos(
        responsiveRect.x + (responsiveRect.width - textSize.width) / 2,
        responsiveRect.y + (responsiveRect.height + textSize.height) / 2
    );
    
    drawText(img, text, textPos, fontSize, textColor, Fonts::ButtonThickness, true);
    
    // Draw focus ring for accessibility
    if (state == ComponentState::Focused) {
        drawFocusRing(img, responsiveRect, 3);
    }
}

void UITheme::drawIconButton(cv::Mat& img, const std::string& icon, 
                           const std::string& text, const cv::Rect& rect,
                           ComponentState state) {
    // Use the enhanced button as base
    drawButton(img, "", rect, state);
    
    // Calculate responsive sizes
    int iconSize = getResponsiveSpacing(Layout::IconSize);
    int spacing = getResponsiveSpacing(Layout::Spacing / 2);
    
    // Layout icon and text
    int totalWidth = iconSize + spacing + static_cast<int>(getTextSize(text, Fonts::FontFace, 
                     getResponsiveFontSize(Fonts::ButtonSize), Fonts::ButtonThickness).width);
    int startX = rect.x + (rect.width - totalWidth) / 2;
    
    // Draw icon (simplified vector-style)
    cv::Point iconCenter(startX + iconSize / 2, rect.y + rect.height / 2);
    cv::Scalar iconColor = (state == ComponentState::Disabled) ? Colors::TextDisabled : Colors::NeonCyan;
    
    // Simple icon rendering (can be replaced with actual vector icons)
    cv::circle(img, iconCenter, iconSize / 3, iconColor, -1);
    
    // Draw text
    cv::Point textPos(startX + iconSize + spacing, 
                     rect.y + rect.height / 2 + static_cast<int>(getResponsiveFontSize(Fonts::ButtonSize) * 10));
    drawText(img, text, textPos, getResponsiveFontSize(Fonts::ButtonSize), 
            (state == ComponentState::Disabled) ? Colors::TextDisabled : Colors::TextPrimary);
}

void UITheme::drawGlassCard(cv::Mat& img, const cv::Rect& rect, 
                           float blurRadius, float opacity,
                           const cv::Scalar& tint) {
    // Create blur effect
    if (blurRadius > 0) {
        applyGaussianBlur(img, rect, static_cast<int>(blurRadius));
    }
    
    // Add glass tint
    cv::Mat overlay = cv::Mat::zeros(rect.height, rect.width, CV_8UC3);
    overlay.setTo(tint);
    
    cv::Mat roi = img(rect);
    cv::addWeighted(roi, 1.0 - opacity, overlay, opacity, 0, roi);
}

void UITheme::drawCard(cv::Mat& img, const cv::Rect& rect, 
                      ComponentState state, const cv::Scalar& bgColor, int elevation) {
    // Calculate shadow based on elevation and state
    int shadowOffset = elevation + ((state == ComponentState::Hover) ? 2 : 0);
    cv::Scalar shadowColor = cv::Scalar(0, 0, 0, 
                                       static_cast<double>(100 + elevation * 20));
    
    // Draw shadow
    if (shadowOffset > 0 && state != ComponentState::Disabled) {
        cv::Rect shadowRect = rect;
        shadowRect.x += shadowOffset;
        shadowRect.y += shadowOffset;
        drawRoundedRect(img, shadowRect, getResponsiveSpacing(Layout::BorderRadius), 
                       shadowColor, -1, true);
    }
    
    // Draw card background
    cv::Scalar cardColor = getStateColor(bgColor, state);
    drawRoundedRect(img, rect, getResponsiveSpacing(Layout::BorderRadius), cardColor, -1, true);
    
    // Add border for focus
    if (state == ComponentState::Focused) {
        drawRoundedRect(img, rect, getResponsiveSpacing(Layout::BorderRadius), 
                       Colors::NeonCyan, 2, true);
    }
}

void UITheme::drawText(cv::Mat& img, const std::string& text,
                      const cv::Point& pos, double fontSize,
                      const cv::Scalar& color, int fontWeight, bool responsive) {
    double scaledSize = responsive ? getResponsiveFontSize(fontSize) : fontSize;
    
    // Draw text with improved rendering
    cv::putText(img, text, pos, Fonts::FontFace, scaledSize, color, fontWeight, cv::LINE_AA);
}

void UITheme::drawTextWithShadow(cv::Mat& img, const std::string& text,
                                 const cv::Point& pos, int fontFace,
                                 double scale, const cv::Scalar& color,
                                 int thickness, int shadowOffset, bool responsive) {
    double scaledSize = responsive ? getResponsiveFontSize(scale) : scale;
    int scaledOffset = responsive ? getResponsiveSpacing(shadowOffset) : shadowOffset;
    
    // Draw shadow
    cv::Point shadowPos = pos + cv::Point(scaledOffset, scaledOffset);
    cv::putText(img, text, shadowPos, fontFace, scaledSize, Colors::TextShadow, thickness, cv::LINE_AA);
    
    // Draw main text
    cv::putText(img, text, pos, fontFace, scaledSize, color, thickness, cv::LINE_AA);
}

void UITheme::drawProgressBar(cv::Mat& img, float progress, 
                             const cv::Rect& rect, const cv::Scalar& color,
                             const AnimationState& anim) {
    // Animate progress if specified
    float animatedProgress = progress;
    if (anim.isAnimating) {
        animatedProgress = progress * easeOut(anim.progress);
    }
    
    // Background
    drawRoundedRect(img, rect, rect.height / 2, Colors::LightBg, -1, true);
    
    // Progress fill
    int fillWidth = static_cast<int>(rect.width * std::clamp(animatedProgress, 0.0f, 1.0f));
    if (fillWidth > 0) {
        cv::Rect fillRect(rect.x, rect.y, fillWidth, rect.height);
        drawRoundedRect(img, fillRect, rect.height / 2, color, -1, true);
        
        // Add glow effect
        cv::Rect glowRect = fillRect;
        glowRect.x -= 2;
        glowRect.y -= 2;
        glowRect.width += 4;
        glowRect.height += 4;
        drawRoundedRect(img, glowRect, rect.height / 2 + 2, 
                       cv::Scalar(color[0], color[1], color[2], 50), -1, true);
    }
}

void UITheme::drawToggle(cv::Mat& img, bool isOn, const cv::Rect& rect,
                        ComponentState state, const AnimationState& anim) {
    // Calculate dimensions
    int trackWidth = rect.width;
    int trackHeight = rect.height;
    int knobSize = trackHeight - 4;
    
    // Colors
    cv::Scalar trackColor = isOn ? Colors::NeonCyan : Colors::LightBg;
    cv::Scalar knobColor = Colors::TextPrimary;
    
    if (state == ComponentState::Disabled) {
        trackColor = Colors::TextDisabled;
        knobColor = Colors::TextDisabled;
    }
    
    // Draw track
    drawRoundedRect(img, rect, trackHeight / 2, trackColor, -1, true);
    
    // Calculate knob position
    int knobX = isOn ? (rect.x + trackWidth - knobSize - 2) : (rect.x + 2);
    if (anim.isAnimating) {
        float animProgress = easeInOut(anim.progress);
        int startX = isOn ? (rect.x + 2) : (rect.x + trackWidth - knobSize - 2);
        int endX = isOn ? (rect.x + trackWidth - knobSize - 2) : (rect.x + 2);
        knobX = startX + static_cast<int>((endX - startX) * animProgress);
    }
    
    // Draw knob with shadow
    cv::Rect knobRect(knobX, rect.y + 2, knobSize, knobSize);
    cv::Rect shadowRect = knobRect;
    shadowRect.x += 1;
    shadowRect.y += 1;
    
    if (state != ComponentState::Disabled) {
        drawRoundedRect(img, shadowRect, knobSize / 2, cv::Scalar(0, 0, 0, 100), -1, true);
    }
    drawRoundedRect(img, knobRect, knobSize / 2, knobColor, -1, true);
    
    // Focus ring
    if (state == ComponentState::Focused) {
        drawFocusRing(img, rect, 2);
    }
}

void UITheme::drawSlider(cv::Mat& img, float value, const cv::Rect& rect,
                        float min, float max, ComponentState state) {
    // Calculate dimensions
    int trackHeight = getResponsiveSpacing(6);
    int knobSize = getResponsiveSpacing(20);
    
    // Track rectangle
    cv::Rect trackRect(rect.x, rect.y + (rect.height - trackHeight) / 2, 
                      rect.width, trackHeight);
    
    // Colors
    cv::Scalar trackColor = (state == ComponentState::Disabled) ? Colors::TextDisabled : Colors::LightBg;
    cv::Scalar fillColor = (state == ComponentState::Disabled) ? Colors::TextDisabled : Colors::NeonCyan;
    cv::Scalar knobColor = Colors::TextPrimary;
    
    // Draw track
    drawRoundedRect(img, trackRect, trackHeight / 2, trackColor, -1, true);
    
    // Draw fill
    float normalizedValue = std::clamp((value - min) / (max - min), 0.0f, 1.0f);
    int fillWidth = static_cast<int>(trackRect.width * normalizedValue);
    if (fillWidth > 0) {
        cv::Rect fillRect(trackRect.x, trackRect.y, fillWidth, trackHeight);
        drawRoundedRect(img, fillRect, trackHeight / 2, fillColor, -1, true);
    }
    
    // Draw knob
    int knobX = trackRect.x + fillWidth - knobSize / 2;
    cv::Rect knobRect(knobX, rect.y + (rect.height - knobSize) / 2, knobSize, knobSize);
    
    // Knob shadow
    if (state != ComponentState::Disabled) {
        cv::Rect shadowRect = knobRect;
        shadowRect.x += 2;
        shadowRect.y += 2;
        drawRoundedRect(img, shadowRect, knobSize / 2, cv::Scalar(0, 0, 0, 100), -1, true);
    }
    
    // Knob
    if (state == ComponentState::Hover) {
        // Larger knob on hover
        knobRect.x -= 2;
        knobRect.y -= 2;
        knobRect.width += 4;
        knobRect.height += 4;
    }
    
    drawRoundedRect(img, knobRect, knobSize / 2, knobColor, -1, true);
    
    // Focus ring
    if (state == ComponentState::Focused) {
        drawFocusRing(img, knobRect, 2);
    }
}

void UITheme::drawTabBar(cv::Mat& img, const std::vector<std::string>& tabs,
                        int activeTab, const cv::Rect& rect,
                        const AnimationState& anim) {
    if (tabs.empty()) return;
    
    int tabWidth = rect.width / static_cast<int>(tabs.size());
    
    // Draw background
    drawRoundedRect(img, rect, getResponsiveSpacing(Layout::BorderRadius), Colors::MediumBg, -1, true);
    
    // Draw tabs
    for (int i = 0; i < static_cast<int>(tabs.size()); ++i) {
        cv::Rect tabRect(rect.x + i * tabWidth, rect.y, tabWidth, rect.height);
        
        ComponentState state = (i == activeTab) ? ComponentState::Active : ComponentState::Normal;
        cv::Scalar textColor = (i == activeTab) ? Colors::NeonCyan : Colors::TextSecondary;
        
        // Active tab background
        if (i == activeTab) {
            cv::Rect activeRect = tabRect;
            if (anim.isAnimating) {
                // Animate tab transition
                float progress = easeInOut(anim.progress);
                // Add smooth transition animation here
            }
            drawRoundedRect(img, activeRect, getResponsiveSpacing(Layout::BorderRadius), 
                           Colors::TableGreen, -1, true);
        }
        
        // Tab text
        double fontSize = getResponsiveFontSize(Fonts::BodySize);
        cv::Size textSize = getTextSize(tabs[i], Fonts::FontFace, fontSize, Fonts::BodyThickness);
        cv::Point textPos(
            tabRect.x + (tabRect.width - textSize.width) / 2,
            tabRect.y + (tabRect.height + textSize.height) / 2
        );
        
        drawText(img, tabs[i], textPos, fontSize, textColor, Fonts::BodyThickness);
    }
}

void UITheme::drawDropdown(cv::Mat& img, const std::string& selected,
                          const cv::Rect& rect, ComponentState state, bool isOpen) {
    // Draw button-like background
    drawCard(img, rect, state);
    
    // Draw selected text
    cv::Scalar textColor = (state == ComponentState::Disabled) ? Colors::TextDisabled : Colors::TextPrimary;
    double fontSize = getResponsiveFontSize(Fonts::BodySize);
    
    cv::Point textPos(rect.x + getResponsiveSpacing(Layout::Padding), 
                     rect.y + (rect.height + static_cast<int>(fontSize * 20)) / 2);
    drawText(img, selected, textPos, fontSize, textColor);
    
    // Draw dropdown arrow
    int arrowSize = getResponsiveSpacing(8);
    cv::Point arrowCenter(rect.x + rect.width - getResponsiveSpacing(Layout::Padding) - arrowSize, 
                         rect.y + rect.height / 2);
    
    // Simple triangle arrow
    std::vector<cv::Point> arrow;
    if (isOpen) {
        // Up arrow
        arrow = {
            cv::Point(arrowCenter.x - arrowSize/2, arrowCenter.y + arrowSize/2),
            cv::Point(arrowCenter.x + arrowSize/2, arrowCenter.y + arrowSize/2),
            cv::Point(arrowCenter.x, arrowCenter.y - arrowSize/2)
        };
    } else {
        // Down arrow  
        arrow = {
            cv::Point(arrowCenter.x - arrowSize/2, arrowCenter.y - arrowSize/2),
            cv::Point(arrowCenter.x + arrowSize/2, arrowCenter.y - arrowSize/2),
            cv::Point(arrowCenter.x, arrowCenter.y + arrowSize/2)
        };
    }
    
    cv::fillPoly(img, arrow, textColor);
}

// Continue with remaining implementations...
void UITheme::drawAnimatedBackground(cv::Mat& img, float time, float intensity) {
    // Create subtle animated particle effect
    cv::RNG rng(static_cast<uint64_t>(time * 1000));
    
    for (int i = 0; i < static_cast<int>(50 * intensity); ++i) {
        float x = rng.uniform(0, img.cols);
        float y = rng.uniform(0, img.rows);
        
        // Animate position based on time
        x += std::sin(time * 0.5f + i * 0.1f) * 20;
        y += std::cos(time * 0.3f + i * 0.15f) * 15;
        
        // Keep within bounds
        if (x >= 0 && x < img.cols && y >= 0 && y < img.rows) {
            float alpha = 0.1f + std::sin(time + i * 0.2f) * 0.05f;
            cv::circle(img, cv::Point(static_cast<int>(x), static_cast<int>(y)), 
                      2, cv::Scalar(255, 255, 0, static_cast<int>(alpha * 255)), -1);
        }
    }
}

void UITheme::applyGlassMorphism(cv::Mat& img, const cv::Rect& rect,
                                int blurAmount, float opacity,
                                const cv::Scalar& tint) {
    if (rect.x < 0 || rect.y < 0 || 
        rect.x + rect.width > img.cols || 
        rect.y + rect.height > img.rows) {
        return;
    }
    
    // Extract region
    cv::Mat roi = img(rect).clone();
    
    // Apply blur
    if (blurAmount > 0) {
        cv::Size kernelSize(blurAmount * 2 + 1, blurAmount * 2 + 1);
        cv::GaussianBlur(roi, roi, kernelSize, 0);
    }
    
    // Create tint overlay
    cv::Mat tintOverlay(roi.size(), CV_8UC3);
    tintOverlay.setTo(cv::Scalar(tint[0], tint[1], tint[2]));
    
    // Blend with tint
    cv::addWeighted(roi, 1.0 - opacity, tintOverlay, opacity, 0, roi);
    
    // Copy back to original image
    roi.copyTo(img(rect));
}

void UITheme::drawRoundedRect(cv::Mat& img, const cv::Rect& rect,
                             int radius, const cv::Scalar& color,
                             int thickness, bool antiAlias) {
    if (radius <= 0) {
        if (thickness < 0) {
            cv::rectangle(img, rect, color, thickness);
        } else {
            cv::rectangle(img, rect, color, thickness);
        }
        return;
    }
    
    // Clamp radius to reasonable bounds
    int maxRadius = std::min(rect.width, rect.height) / 2;
    radius = std::min(radius, maxRadius);
    
    if (thickness < 0) {
        // Filled rounded rectangle
        cv::rectangle(img, cv::Point(rect.x + radius, rect.y), 
                     cv::Point(rect.x + rect.width - radius, rect.y + rect.height), 
                     color, -1);
        cv::rectangle(img, cv::Point(rect.x, rect.y + radius), 
                     cv::Point(rect.x + rect.width, rect.y + rect.height - radius), 
                     color, -1);
        
        // Corner circles
        cv::circle(img, cv::Point(rect.x + radius, rect.y + radius), radius, color, -1);
        cv::circle(img, cv::Point(rect.x + rect.width - radius, rect.y + radius), radius, color, -1);
        cv::circle(img, cv::Point(rect.x + radius, rect.y + rect.height - radius), radius, color, -1);
        cv::circle(img, cv::Point(rect.x + rect.width - radius, rect.y + rect.height - radius), radius, color, -1);
    } else {
        // Outlined rounded rectangle
        cv::line(img, cv::Point(rect.x + radius, rect.y), 
                cv::Point(rect.x + rect.width - radius, rect.y), color, thickness);
        cv::line(img, cv::Point(rect.x + radius, rect.y + rect.height), 
                cv::Point(rect.x + rect.width - radius, rect.y + rect.height), color, thickness);
        cv::line(img, cv::Point(rect.x, rect.y + radius), 
                cv::Point(rect.x, rect.y + rect.height - radius), color, thickness);
        cv::line(img, cv::Point(rect.x + rect.width, rect.y + radius), 
                cv::Point(rect.x + rect.width, rect.y + rect.height - radius), color, thickness);
        
        // Corner arcs (simplified as circles for now)
        cv::circle(img, cv::Point(rect.x + radius, rect.y + radius), radius, color, thickness);
        cv::circle(img, cv::Point(rect.x + rect.width - radius, rect.y + radius), radius, color, thickness);
        cv::circle(img, cv::Point(rect.x + radius, rect.y + rect.height - radius), radius, color, thickness);
        cv::circle(img, cv::Point(rect.x + rect.width - radius, rect.y + rect.height - radius), radius, color, thickness);
    }
}

bool UITheme::isPointInRect(const cv::Point& pt, const cv::Rect& rect, int hoverRadius) {
    cv::Rect expandedRect = rect;
    expandedRect.x -= hoverRadius;
    expandedRect.y -= hoverRadius;
    expandedRect.width += 2 * hoverRadius;
    expandedRect.height += 2 * hoverRadius;
    
    return expandedRect.contains(pt);
}

cv::Size UITheme::getTextSize(const std::string& text, int fontFace,
                             double scale, int thickness, bool responsive) {
    double scaledSize = responsive ? getResponsiveFontSize(scale) : scale;
    
    int baseline = 0;
    return cv::getTextSize(text, fontFace, scaledSize, thickness, &baseline);
}

// Additional modern features
void UITheme::drawToast(cv::Mat& img, const std::string& message,
                       const cv::Point& position, const std::string& type, float opacity) {
    cv::Size textSize = getTextSize(message, Fonts::FontFace, Fonts::BodySize, Fonts::BodyThickness);
    int padding = getResponsiveSpacing(Layout::Padding);
    
    cv::Rect toastRect(position.x - padding, position.y - padding, 
                      textSize.width + 2 * padding, textSize.height + 2 * padding);
    
    // Choose color based on type
    cv::Scalar bgColor = Colors::MediumBg;
    cv::Scalar borderColor = Colors::BorderColor;
    
    if (type == "success") {
        borderColor = Colors::NeonGreen;
    } else if (type == "warning") {
        borderColor = Colors::NeonYellow;
    } else if (type == "error") {
        borderColor = Colors::NeonRed;
    }
    
    // Draw toast with glass effect
    drawGlassCard(img, toastRect, 10.0f, opacity);
    drawRoundedRect(img, toastRect, getResponsiveSpacing(Layout::BorderRadius), bgColor, -1);
    drawRoundedRect(img, toastRect, getResponsiveSpacing(Layout::BorderRadius), borderColor, 2);
    
    // Draw message
    drawText(img, message, position, Fonts::BodySize, Colors::TextPrimary);
}

void UITheme::drawSpinner(cv::Mat& img, const cv::Point& center, int radius,
                         float rotation, const cv::Scalar& color) {
    int segments = 8;
    float angleStep = 2.0f * CV_PI / segments;
    
    for (int i = 0; i < segments; ++i) {
        float angle = rotation + i * angleStep;
        float alpha = 1.0f - (i / static_cast<float>(segments));
        
        cv::Point start = center + cv::Point(
            static_cast<int>(std::cos(angle) * radius * 0.7),
            static_cast<int>(std::sin(angle) * radius * 0.7)
        );
        cv::Point end = center + cv::Point(
            static_cast<int>(std::cos(angle) * radius),
            static_cast<int>(std::sin(angle) * radius)
        );
        
        cv::Scalar segmentColor(color[0], color[1], color[2], static_cast<int>(alpha * 255));
        cv::line(img, start, end, segmentColor, 3);
    }
}

void UITheme::drawFocusRing(cv::Mat& img, const cv::Rect& rect, int thickness) {
    // Draw dashed border for accessibility
    cv::Rect focusRect = rect;
    focusRect.x -= thickness;
    focusRect.y -= thickness;
    focusRect.width += 2 * thickness;
    focusRect.height += 2 * thickness;
    
    // Simple focus ring (can be enhanced with dashed lines)
    drawRoundedRect(img, focusRect, getResponsiveSpacing(Layout::BorderRadius + thickness), 
                   Colors::NeonYellow, thickness);
}

// Legacy compatibility
void UITheme::drawTitleBar(cv::Mat& img, const std::string& title, int height) {
    cv::Rect titleRect(0, 0, img.cols, height);
    
    // Background
    drawGlassCard(img, titleRect);
    drawRoundedRect(img, titleRect, 0, Colors::MediumBg, -1);
    
    // Title text
    double fontSize = getResponsiveFontSize(Fonts::HeadingSize);
    cv::Size textSize = getTextSize(title, Fonts::FontFaceBold, fontSize, Fonts::HeadingThickness);
    cv::Point textPos((img.cols - textSize.width) / 2, 
                     (height + textSize.height) / 2);
    
    drawTextWithShadow(img, title, textPos, Fonts::FontFaceBold, fontSize,
                      Colors::NeonCyan, Fonts::HeadingThickness, 3);
}

// Internal utilities
cv::Mat UITheme::createBlurKernel(int size) {
    cv::Mat kernel = cv::getGaussianKernel(size, -1);
    return kernel * kernel.t();
}

void UITheme::applyGaussianBlur(cv::Mat& img, const cv::Rect& rect, int blurRadius) {
    if (blurRadius <= 0) return;
    
    cv::Mat roi = img(rect);
    cv::Size kernelSize(blurRadius * 2 + 1, blurRadius * 2 + 1);
    cv::GaussianBlur(roi, roi, kernelSize, 0);
}

cv::Scalar UITheme::interpolateColor(const cv::Scalar& from, const cv::Scalar& to, float progress) {
    return cv::Scalar(
        from[0] + (to[0] - from[0]) * progress,
        from[1] + (to[1] - from[1]) * progress,
        from[2] + (to[2] - from[2]) * progress,
        from[3] + (to[3] - from[3]) * progress
    );
}

float UITheme::easeOut(float t) {
    return 1.0f - std::pow(1.0f - t, 3.0f);
}

float UITheme::easeInOut(float t) {
    return t < 0.5f ? 2.0f * t * t : 1.0f - std::pow(-2.0f * t + 2.0f, 3.0f) / 2.0f;
}

} // namespace pv
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
