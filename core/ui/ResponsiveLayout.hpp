#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <functional>

namespace pv {

/**
 * @brief Responsive layout system for Pool Vision UI
 * 
 * Provides flexbox-like layout management with automatic sizing,
 * spacing, and alignment for modern responsive interfaces.
 */
class ResponsiveLayout {
public:
    enum class Direction {
        Row,        // Horizontal layout
        Column      // Vertical layout
    };
    
    enum class Alignment {
        Start,      // Top/Left
        Center,     // Center
        End,        // Bottom/Right
        Stretch     // Fill available space
    };
    
    enum class Justify {
        Start,          // Items at start
        Center,         // Items centered
        End,            // Items at end
        SpaceBetween,   // Equal space between items
        SpaceAround,    // Equal space around items
        SpaceEvenly     // Equal space everywhere
    };
    
    struct LayoutItem {
        cv::Rect rect;
        int flex = 0;           // Flex grow factor (0 = fixed size)
        int minWidth = 0;       // Minimum width
        int minHeight = 0;      // Minimum height
        int maxWidth = INT_MAX; // Maximum width
        int maxHeight = INT_MAX;// Maximum height
        cv::Size preferredSize = cv::Size(0, 0); // Preferred size
        bool visible = true;    // Whether item is visible
    };
    
    class Container {
    public:
        Container(const cv::Rect& bounds);
        
        // Configuration
        Container& setDirection(Direction dir);
        Container& setAlignment(Alignment align);
        Container& setJustify(Justify justify);
        Container& setPadding(int padding);
        Container& setPadding(int horizontal, int vertical);
        Container& setPadding(int top, int right, int bottom, int left);
        Container& setSpacing(int spacing);
        Container& setWrap(bool wrap);
        
        // Add items
        Container& addItem(int flex = 0, int minWidth = 0, int minHeight = 0);
        Container& addItem(const cv::Size& fixedSize);
        Container& addFixedItem(int width, int height);
        Container& addFlexItem(int flex, int minSize = 0);
        Container& addSpacer(int flex = 1);
        
        // Layout calculation
        std::vector<cv::Rect> calculate();
        void applyTo(std::vector<LayoutItem>& items);
        
        // Utility
        cv::Rect getContentBounds() const;
        cv::Size getMinimumSize() const;
        
    private:
        cv::Rect bounds_;
        Direction direction_ = Direction::Row;
        Alignment alignment_ = Alignment::Start;
        Justify justify_ = Justify::Start;
        int paddingTop_ = 0, paddingRight_ = 0, paddingBottom_ = 0, paddingLeft_ = 0;
        int spacing_ = 0;
        bool wrap_ = false;
        std::vector<LayoutItem> items_;
        
        void calculateRowLayout(std::vector<cv::Rect>& results);
        void calculateColumnLayout(std::vector<cv::Rect>& results);
        cv::Rect getContentRect() const;
    };
    
    // Responsive utilities
    static cv::Size getResponsiveSize(int baseWidth, int baseHeight, 
                                     int windowWidth, int windowHeight,
                                     float minScale = 0.5f, float maxScale = 2.0f);
    
    static int getResponsiveValue(int baseValue, int windowSize, 
                                 int baseWindowSize = 1280,
                                 float minScale = 0.5f, float maxScale = 2.0f);
    
    static double getFluidFontSize(double baseSize, int windowWidth, int windowHeight,
                                  int baseWidth = 1280, int baseHeight = 720);
    
    // Grid system
    static cv::Rect getGridRect(int col, int row, int cols, int rows,
                               const cv::Rect& container, int spacing = 0);
    
    // Percentage-based sizing
    static cv::Rect getPercentRect(float x, float y, float width, float height,
                                  const cv::Rect& parent);
    
    // Responsive breakpoints
    enum class Breakpoint {
        XSmall,     // < 576px
        Small,      // 576px - 768px
        Medium,     // 768px - 992px
        Large,      // 992px - 1200px
        XLarge      // >= 1200px
    };
    
    static Breakpoint getBreakpoint(int width);
    static bool isBreakpoint(int width, Breakpoint bp);
};

/**
 * @brief Modern theme manager with responsive design support
 */
class ModernTheme {
public:
    // Color system with states
    struct ColorPalette {
        cv::Scalar primary;
        cv::Scalar primaryHover;
        cv::Scalar primaryActive;
        cv::Scalar primaryDisabled;
        
        cv::Scalar secondary;
        cv::Scalar secondaryHover;
        cv::Scalar secondaryActive;
        cv::Scalar secondaryDisabled;
        
        cv::Scalar accent;
        cv::Scalar accentHover;
        cv::Scalar accentActive;
        
        cv::Scalar background;
        cv::Scalar surface;
        cv::Scalar surfaceVariant;
        
        cv::Scalar text;
        cv::Scalar textSecondary;
        cv::Scalar textDisabled;
        cv::Scalar textOnPrimary;
        
        cv::Scalar border;
        cv::Scalar borderFocus;
        cv::Scalar shadow;
        
        cv::Scalar success;
        cv::Scalar warning;
        cv::Scalar error;
        cv::Scalar info;
    };
    
    // Typography system
    struct Typography {
        // Base font settings
        int fontFace = cv::FONT_HERSHEY_SIMPLEX;
        int fontFaceBold = cv::FONT_HERSHEY_DUPLEX;
        
        // Responsive font sizes (base at 1280x720)
        double h1 = 2.0;       // Main titles
        double h2 = 1.5;       // Section headers  
        double h3 = 1.2;       // Subsection headers
        double h4 = 1.0;       // Card titles
        double body = 0.7;     // Body text
        double caption = 0.5;  // Small text
        double button = 0.8;   // Button text
        
        // Font weights (thickness)
        int thin = 1;
        int regular = 1;
        int medium = 2;
        int bold = 2;
        int extraBold = 3;
        
        // Line heights (multiplier of font size)
        double lineHeight = 1.4;
        double titleLineHeight = 1.2;
    };
    
    // Spacing system (8px base unit)
    struct Spacing {
        int xs = 4;    // 0.5 unit
        int sm = 8;    // 1 unit  
        int md = 16;   // 2 units
        int lg = 24;   // 3 units
        int xl = 32;   // 4 units
        int xxl = 48;  // 6 units
        int xxxl = 64; // 8 units
    };
    
    // Border radius
    struct Radius {
        int none = 0;
        int sm = 4;
        int md = 8;
        int lg = 16;
        int xl = 24;
        int full = 9999;  // Fully rounded
    };
    
    // Shadows
    struct Shadow {
        cv::Size none = cv::Size(0, 0);
        cv::Size sm = cv::Size(2, 2);
        cv::Size md = cv::Size(4, 4);
        cv::Size lg = cv::Size(8, 8);
        cv::Size xl = cv::Size(16, 16);
    };
    
    // Animation timing
    struct Animation {
        double fast = 0.15;      // Quick hover effects
        double normal = 0.25;    // Standard transitions
        double slow = 0.35;      // Complex animations
        double slower = 0.5;     // Page transitions
    };
    
public:
    ModernTheme();
    
    // Theme management
    void setDarkMode(bool dark);
    void setHighContrast(bool highContrast);
    void setColorBlindMode(bool colorBlind);
    void setScale(float scale);
    
    // Getters
    const ColorPalette& colors() const { return colors_; }
    const Typography& typography() const { return typography_; }
    const Spacing& spacing() const { return spacing_; }
    const Radius& radius() const { return radius_; }
    const Shadow& shadow() const { return shadow_; }
    const Animation& animation() const { return animation_; }
    
    // Responsive font size calculation
    double getResponsiveFontSize(double baseSize, int windowWidth, int windowHeight) const;
    int getResponsiveSpacing(int baseSpacing, int windowWidth) const;
    
    // State colors
    cv::Scalar getStateColor(const cv::Scalar& base, const std::string& state) const;
    
private:
    ColorPalette colors_;
    Typography typography_;
    Spacing spacing_;
    Radius radius_;
    Shadow shadow_;
    Animation animation_;
    
    bool darkMode_ = true;
    bool highContrast_ = false;
    bool colorBlindMode_ = false;
    float scale_ = 1.0f;
    
    void updateColors();
};

} // namespace pv