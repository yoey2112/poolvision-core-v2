#include "ResponsiveLayout.hpp"
#include <algorithm>
#include <cmath>

namespace pv {

// ResponsiveLayout::Container implementation

ResponsiveLayout::Container::Container(const cv::Rect& bounds) 
    : bounds_(bounds) {
}

ResponsiveLayout::Container::Container(Direction direction, const cv::Rect& bounds) 
    : bounds_(bounds), direction_(direction) {
}

ResponsiveLayout::Container& ResponsiveLayout::Container::setDirection(Direction dir) {
    direction_ = dir;
    return *this;
}

ResponsiveLayout::Container& ResponsiveLayout::Container::setAlignment(Alignment align) {
    alignment_ = align;
    return *this;
}

ResponsiveLayout::Container& ResponsiveLayout::Container::setJustify(Justify justify) {
    justify_ = justify;
    return *this;
}

ResponsiveLayout::Container& ResponsiveLayout::Container::setPadding(int padding) {
    paddingTop_ = paddingRight_ = paddingBottom_ = paddingLeft_ = padding;
    return *this;
}

ResponsiveLayout::Container& ResponsiveLayout::Container::setPadding(int horizontal, int vertical) {
    paddingTop_ = paddingBottom_ = vertical;
    paddingLeft_ = paddingRight_ = horizontal;
    return *this;
}

ResponsiveLayout::Container& ResponsiveLayout::Container::setPadding(int top, int right, int bottom, int left) {
    paddingTop_ = top;
    paddingRight_ = right;
    paddingBottom_ = bottom;
    paddingLeft_ = left;
    return *this;
}

ResponsiveLayout::Container& ResponsiveLayout::Container::setSpacing(int spacing) {
    spacing_ = spacing;
    return *this;
}

ResponsiveLayout::Container& ResponsiveLayout::Container::setWrap(bool wrap) {
    wrap_ = wrap;
    return *this;
}

ResponsiveLayout::Container& ResponsiveLayout::Container::addItem(int flex, int minWidth, int minHeight) {
    LayoutItem item;
    item.flex = flex;
    item.minWidth = minWidth;
    item.minHeight = minHeight;
    items_.push_back(item);
    return *this;
}

ResponsiveLayout::Container& ResponsiveLayout::Container::addItem(const cv::Size& fixedSize) {
    LayoutItem item;
    item.flex = 0;
    item.preferredSize = fixedSize;
    item.minWidth = fixedSize.width;
    item.minHeight = fixedSize.height;
    item.maxWidth = fixedSize.width;
    item.maxHeight = fixedSize.height;
    items_.push_back(item);
    return *this;
}

ResponsiveLayout::Container& ResponsiveLayout::Container::addFixedItem(int width, int height) {
    return addItem(cv::Size(width, height));
}

ResponsiveLayout::Container& ResponsiveLayout::Container::addFlexItem(int flex, int minSize) {
    LayoutItem item;
    item.flex = flex;
    if (direction_ == Direction::Row) {
        item.minWidth = minSize;
    } else {
        item.minHeight = minSize;
    }
    items_.push_back(item);
    return *this;
}

ResponsiveLayout::Container& ResponsiveLayout::Container::addSpacer(int flex) {
    return addFlexItem(flex, 0);
}

std::vector<cv::Rect> ResponsiveLayout::Container::calculate() {
    std::vector<cv::Rect> results(items_.size());
    
    if (direction_ == Direction::Row) {
        calculateRowLayout(results);
    } else {
        calculateColumnLayout(results);
    }
    
    return results;
}

void ResponsiveLayout::Container::applyTo(std::vector<LayoutItem>& items) {
    auto rects = calculate();
    for (size_t i = 0; i < std::min(items.size(), rects.size()); ++i) {
        items[i].rect = rects[i];
    }
}

cv::Rect ResponsiveLayout::Container::getContentBounds() const {
    return getContentRect();
}

cv::Size ResponsiveLayout::Container::getMinimumSize() const {
    int totalMinWidth = 0;
    int totalMinHeight = 0;
    int maxWidth = 0;
    int maxHeight = 0;
    
    for (const auto& item : items_) {
        if (!item.visible) continue;
        
        if (direction_ == Direction::Row) {
            totalMinWidth += item.minWidth;
            maxHeight = std::max(maxHeight, item.minHeight);
        } else {
            totalMinHeight += item.minHeight;
            maxWidth = std::max(maxWidth, item.minWidth);
        }
    }
    
    // Add spacing
    int visibleItems = 0;
    for (const auto& item : items_) {
        if (item.visible) visibleItems++;
    }
    
    if (direction_ == Direction::Row) {
        totalMinWidth += (visibleItems - 1) * spacing_;
        totalMinWidth += paddingLeft_ + paddingRight_;
        maxHeight += paddingTop_ + paddingBottom_;
        return cv::Size(totalMinWidth, maxHeight);
    } else {
        totalMinHeight += (visibleItems - 1) * spacing_;
        totalMinHeight += paddingTop_ + paddingBottom_;
        maxWidth += paddingLeft_ + paddingRight_;
        return cv::Size(maxWidth, totalMinHeight);
    }
}

void ResponsiveLayout::Container::calculateRowLayout(std::vector<cv::Rect>& results) {
    cv::Rect contentRect = getContentRect();
    
    // Calculate flex items
    std::vector<int> widths(items_.size());
    int totalFlex = 0;
    int totalFixedWidth = 0;
    int visibleItems = 0;
    
    // First pass: calculate fixed widths and total flex
    for (size_t i = 0; i < items_.size(); ++i) {
        if (!items_[i].visible) continue;
        visibleItems++;
        
        if (items_[i].flex == 0) {
            // Fixed width item
            widths[i] = items_[i].preferredSize.width > 0 ? 
                       items_[i].preferredSize.width : items_[i].minWidth;
            totalFixedWidth += widths[i];
        } else {
            // Flex item
            totalFlex += items_[i].flex;
        }
    }
    
    // Calculate available space for flex items
    int availableWidth = contentRect.width - totalFixedWidth - (visibleItems - 1) * spacing_;
    
    // Second pass: calculate flex widths
    if (totalFlex > 0 && availableWidth > 0) {
        int flexUnit = availableWidth / totalFlex;
        for (size_t i = 0; i < items_.size(); ++i) {
            if (!items_[i].visible) continue;
            
            if (items_[i].flex > 0) {
                int flexWidth = flexUnit * items_[i].flex;
                widths[i] = std::max(flexWidth, items_[i].minWidth);
                widths[i] = std::min(widths[i], items_[i].maxWidth);
            }
        }
    }
    
    // Position items
    int currentX = contentRect.x;
    
    // Handle justify content
    if (justify_ != Justify::Start) {
        int totalUsedWidth = 0;
        for (size_t i = 0; i < items_.size(); ++i) {
            if (items_[i].visible) {
                totalUsedWidth += widths[i];
            }
        }
        totalUsedWidth += (visibleItems - 1) * spacing_;
        int remainingWidth = contentRect.width - totalUsedWidth;
        
        switch (justify_) {
            case Justify::Center:
                currentX += remainingWidth / 2;
                break;
            case Justify::End:
                currentX += remainingWidth;
                break;
            case Justify::SpaceBetween:
                if (visibleItems > 1) {
                    spacing_ = remainingWidth / (visibleItems - 1);
                }
                break;
            case Justify::SpaceAround:
                if (visibleItems > 0) {
                    int extraSpace = remainingWidth / visibleItems;
                    currentX += extraSpace / 2;
                    spacing_ += extraSpace;
                }
                break;
            case Justify::SpaceEvenly:
                if (visibleItems > 0) {
                    int extraSpace = remainingWidth / (visibleItems + 1);
                    currentX += extraSpace;
                    spacing_ += extraSpace;
                }
                break;
        }
    }
    
    // Create rectangles
    for (size_t i = 0; i < items_.size(); ++i) {
        if (!items_[i].visible) {
            results[i] = cv::Rect(0, 0, 0, 0);
            continue;
        }
        
        int height = contentRect.height;
        int y = contentRect.y;
        
        // Handle cross-axis alignment
        if (alignment_ == Alignment::Center) {
            int itemHeight = std::min(height, items_[i].minHeight);
            y = contentRect.y + (height - itemHeight) / 2;
            height = itemHeight;
        } else if (alignment_ == Alignment::End) {
            int itemHeight = std::min(height, items_[i].minHeight);
            y = contentRect.y + height - itemHeight;
            height = itemHeight;
        }
        
        results[i] = cv::Rect(currentX, y, widths[i], height);
        currentX += widths[i] + spacing_;
    }
}

void ResponsiveLayout::Container::calculateColumnLayout(std::vector<cv::Rect>& results) {
    cv::Rect contentRect = getContentRect();
    
    // Calculate flex items (similar to row but for heights)
    std::vector<int> heights(items_.size());
    int totalFlex = 0;
    int totalFixedHeight = 0;
    int visibleItems = 0;
    
    // First pass: calculate fixed heights and total flex
    for (size_t i = 0; i < items_.size(); ++i) {
        if (!items_[i].visible) continue;
        visibleItems++;
        
        if (items_[i].flex == 0) {
            heights[i] = items_[i].preferredSize.height > 0 ? 
                        items_[i].preferredSize.height : items_[i].minHeight;
            totalFixedHeight += heights[i];
        } else {
            totalFlex += items_[i].flex;
        }
    }
    
    // Calculate available space for flex items
    int availableHeight = contentRect.height - totalFixedHeight - (visibleItems - 1) * spacing_;
    
    // Second pass: calculate flex heights
    if (totalFlex > 0 && availableHeight > 0) {
        int flexUnit = availableHeight / totalFlex;
        for (size_t i = 0; i < items_.size(); ++i) {
            if (!items_[i].visible) continue;
            
            if (items_[i].flex > 0) {
                int flexHeight = flexUnit * items_[i].flex;
                heights[i] = std::max(flexHeight, items_[i].minHeight);
                heights[i] = std::min(heights[i], items_[i].maxHeight);
            }
        }
    }
    
    // Position items
    int currentY = contentRect.y;
    
    // Create rectangles
    for (size_t i = 0; i < items_.size(); ++i) {
        if (!items_[i].visible) {
            results[i] = cv::Rect(0, 0, 0, 0);
            continue;
        }
        
        int width = contentRect.width;
        int x = contentRect.x;
        
        // Handle cross-axis alignment
        if (alignment_ == Alignment::Center) {
            int itemWidth = std::min(width, items_[i].minWidth);
            x = contentRect.x + (width - itemWidth) / 2;
            width = itemWidth;
        } else if (alignment_ == Alignment::End) {
            int itemWidth = std::min(width, items_[i].minWidth);
            x = contentRect.x + width - itemWidth;
            width = itemWidth;
        }
        
        results[i] = cv::Rect(x, currentY, width, heights[i]);
        currentY += heights[i] + spacing_;
    }
}

cv::Rect ResponsiveLayout::Container::getContentRect() const {
    return cv::Rect(
        bounds_.x + paddingLeft_,
        bounds_.y + paddingTop_,
        bounds_.width - paddingLeft_ - paddingRight_,
        bounds_.height - paddingTop_ - paddingBottom_
    );
}

// Static utility functions

cv::Size ResponsiveLayout::getResponsiveSize(int baseWidth, int baseHeight, 
                                           int windowWidth, int windowHeight,
                                           float minScale, float maxScale) {
    float scaleX = static_cast<float>(windowWidth) / 1280.0f;
    float scaleY = static_cast<float>(windowHeight) / 720.0f;
    float scale = std::min(scaleX, scaleY);
    
    scale = std::max(minScale, std::min(maxScale, scale));
    
    return cv::Size(
        static_cast<int>(baseWidth * scale),
        static_cast<int>(baseHeight * scale)
    );
}

int ResponsiveLayout::getResponsiveValue(int baseValue, int windowSize, 
                                       int baseWindowSize,
                                       float minScale, float maxScale) {
    float scale = static_cast<float>(windowSize) / static_cast<float>(baseWindowSize);
    scale = std::max(minScale, std::min(maxScale, scale));
    return static_cast<int>(baseValue * scale);
}

double ResponsiveLayout::getFluidFontSize(double baseSize, int windowWidth, int windowHeight,
                                        int baseWidth, int baseHeight) {
    float scaleX = static_cast<float>(windowWidth) / static_cast<float>(baseWidth);
    float scaleY = static_cast<float>(windowHeight) / static_cast<float>(baseHeight);
    float scale = std::min(scaleX, scaleY);
    
    // Clamp scale to reasonable bounds
    scale = std::max(0.6f, std::min(2.0f, scale));
    
    return baseSize * scale;
}

cv::Rect ResponsiveLayout::getGridRect(int col, int row, int cols, int rows,
                                     const cv::Rect& container, int spacing) {
    int cellWidth = (container.width - spacing * (cols - 1)) / cols;
    int cellHeight = (container.height - spacing * (rows - 1)) / rows;
    
    int x = container.x + col * (cellWidth + spacing);
    int y = container.y + row * (cellHeight + spacing);
    
    return cv::Rect(x, y, cellWidth, cellHeight);
}

cv::Rect ResponsiveLayout::getPercentRect(float x, float y, float width, float height,
                                        const cv::Rect& parent) {
    return cv::Rect(
        parent.x + static_cast<int>(parent.width * x),
        parent.y + static_cast<int>(parent.height * y),
        static_cast<int>(parent.width * width),
        static_cast<int>(parent.height * height)
    );
}

ResponsiveLayout::Breakpoint ResponsiveLayout::getBreakpoint(int width) {
    if (width < 576) return Breakpoint::XSmall;
    if (width < 768) return Breakpoint::Small;
    if (width < 992) return Breakpoint::Medium;
    if (width < 1200) return Breakpoint::Large;
    return Breakpoint::XLarge;
}

bool ResponsiveLayout::isBreakpoint(int width, Breakpoint bp) {
    return getBreakpoint(width) == bp;
}

// ModernTheme implementation

ModernTheme::ModernTheme() {
    updateColors();
}

void ModernTheme::setDarkMode(bool dark) {
    if (darkMode_ != dark) {
        darkMode_ = dark;
        updateColors();
    }
}

void ModernTheme::setHighContrast(bool highContrast) {
    if (highContrast_ != highContrast) {
        highContrast_ = highContrast;
        updateColors();
    }
}

void ModernTheme::setColorBlindMode(bool colorBlind) {
    if (colorBlindMode_ != colorBlind) {
        colorBlindMode_ = colorBlind;
        updateColors();
    }
}

void ModernTheme::setScale(float scale) {
    scale_ = std::max(0.5f, std::min(3.0f, scale));
    
    // Update spacing with scale
    spacing_.xs = static_cast<int>(4 * scale_);
    spacing_.sm = static_cast<int>(8 * scale_);
    spacing_.md = static_cast<int>(16 * scale_);
    spacing_.lg = static_cast<int>(24 * scale_);
    spacing_.xl = static_cast<int>(32 * scale_);
    spacing_.xxl = static_cast<int>(48 * scale_);
    spacing_.xxxl = static_cast<int>(64 * scale_);
}

double ModernTheme::getResponsiveFontSize(double baseSize, int windowWidth, int windowHeight) const {
    return ResponsiveLayout::getFluidFontSize(baseSize * scale_, windowWidth, windowHeight);
}

int ModernTheme::getResponsiveSpacing(int baseSpacing, int windowWidth) const {
    float scale = static_cast<float>(windowWidth) / 1280.0f;
    scale = std::max(0.5f, std::min(2.0f, scale));
    return static_cast<int>(baseSpacing * scale * scale_);
}

cv::Scalar ModernTheme::getStateColor(const cv::Scalar& base, const std::string& state) const {
    if (state == "hover") {
        // Lighten color for hover
        return cv::Scalar(
            std::min(255.0, base[0] * 1.2),
            std::min(255.0, base[1] * 1.2),
            std::min(255.0, base[2] * 1.2),
            base[3]
        );
    } else if (state == "active") {
        // Darken color for active
        return cv::Scalar(
            base[0] * 0.8,
            base[1] * 0.8,
            base[2] * 0.8,
            base[3]
        );
    } else if (state == "disabled") {
        // Desaturate and reduce opacity
        double gray = (base[0] + base[1] + base[2]) / 3.0;
        return cv::Scalar(gray, gray, gray, base[3] * 0.5);
    }
    
    return base;
}

void ModernTheme::updateColors() {
    if (darkMode_) {
        // Dark theme (pool table inspired)
        colors_.background = cv::Scalar(26, 26, 26);        // #1A1A1A
        colors_.surface = cv::Scalar(42, 42, 42);           // #2A2A2A
        colors_.surfaceVariant = cv::Scalar(58, 58, 58);    // #3A3A3A
        
        colors_.primary = cv::Scalar(58, 94, 13);           // Pool table green #0D5E3A
        colors_.primaryHover = cv::Scalar(75, 122, 17);
        colors_.primaryActive = cv::Scalar(46, 75, 10);
        colors_.primaryDisabled = cv::Scalar(29, 47, 6);
        
        colors_.secondary = cv::Scalar(102, 102, 102);      // #666666
        colors_.secondaryHover = cv::Scalar(128, 128, 128);
        colors_.secondaryActive = cv::Scalar(76, 76, 76);
        colors_.secondaryDisabled = cv::Scalar(51, 51, 51);
        
        colors_.accent = cv::Scalar(255, 255, 0);           // Neon cyan #00FFFF
        colors_.accentHover = cv::Scalar(255, 255, 51);
        colors_.accentActive = cv::Scalar(204, 204, 0);
        
        colors_.text = cv::Scalar(255, 255, 255);           // White
        colors_.textSecondary = cv::Scalar(204, 204, 204);  // #CCCCCC
        colors_.textDisabled = cv::Scalar(102, 102, 102);   // #666666
        colors_.textOnPrimary = cv::Scalar(255, 255, 255);
        
    } else {
        // Light theme
        colors_.background = cv::Scalar(248, 249, 250);     // #F8F9FA
        colors_.surface = cv::Scalar(255, 255, 255);        // White
        colors_.surfaceVariant = cv::Scalar(241, 243, 244); // #F1F3F4
        
        colors_.primary = cv::Scalar(58, 94, 13);           // Pool table green
        colors_.primaryHover = cv::Scalar(75, 122, 17);
        colors_.primaryActive = cv::Scalar(46, 75, 10);
        colors_.primaryDisabled = cv::Scalar(29, 47, 6);
        
        colors_.text = cv::Scalar(33, 37, 41);              // Dark gray
        colors_.textSecondary = cv::Scalar(108, 117, 125);  // Medium gray
        colors_.textDisabled = cv::Scalar(173, 181, 189);   // Light gray
        colors_.textOnPrimary = cv::Scalar(255, 255, 255);
    }
    
    // Common colors
    colors_.border = cv::Scalar(64, 64, 64);
    colors_.borderFocus = colors_.accent;
    colors_.shadow = cv::Scalar(0, 0, 0, 100);
    
    colors_.success = cv::Scalar(0, 255, 0);              // Bright green
    colors_.warning = cv::Scalar(0, 215, 255);            // Gold #FFD700
    colors_.error = cv::Scalar(102, 0, 255);              // Neon red #FF0066
    colors_.info = cv::Scalar(255, 102, 0);               // Neon blue #0066FF
    
    // Adjust for high contrast mode
    if (highContrast_) {
        if (darkMode_) {
            colors_.text = cv::Scalar(255, 255, 255);
            colors_.background = cv::Scalar(0, 0, 0);
        } else {
            colors_.text = cv::Scalar(0, 0, 0);
            colors_.background = cv::Scalar(255, 255, 255);
        }
    }
    
    // Adjust for color blind mode
    if (colorBlindMode_) {
        // Use blue/yellow instead of red/green for better accessibility
        colors_.success = cv::Scalar(255, 255, 0);         // Bright yellow
        colors_.error = cv::Scalar(255, 0, 0);             // Bright blue
    }
}

} // namespace pv
