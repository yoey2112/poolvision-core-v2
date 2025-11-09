#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <memory>

namespace pv {

// Forward declarations
class ModernTheme;
class ResponsiveLayout;

/**
 * @brief Modern UI theme with responsive design and accessibility
 * 
 * Enhanced version of the original UITheme with:
 * - Responsive layout support
 * - State-based color management
 * - Fluid typography
 * - Glass morphism effects
 * - Animation support
 * - Accessibility features
 */
class UITheme {
public:
    // Legacy color palette for backward compatibility
    struct Colors {
        // Primary colors
        static const cv::Scalar TableGreen;      // #0D5E3A - Pool table green
        static const cv::Scalar DarkBg;          // #1A1A1A - Dark background
        static const cv::Scalar MediumBg;        // #2A2A2A - Medium background
        static const cv::Scalar LightBg;         // #3A3A3A - Light background
        
        // Accent colors
        static const cv::Scalar NeonCyan;        // #00FFFF - Bright cyan
        static const cv::Scalar NeonYellow;      // #FFD700 - Gold/Yellow
        static const cv::Scalar NeonGreen;       // #00FF00 - Bright green
        static const cv::Scalar NeonRed;         // #FF0066 - Error red
        static const cv::Scalar NeonOrange;      // #FF6600 - Orange
        static const cv::Scalar NeonBlue;        // #0066FF - Blue
        
        // Text colors
        static const cv::Scalar TextPrimary;     // #FFFFFF - White
        static const cv::Scalar TextSecondary;   // #CCCCCC - Light gray
        static const cv::Scalar TextDisabled;    // #666666 - Dark gray
        static const cv::Scalar TextShadow;      // Semi-transparent black
        
        // UI elements
        static const cv::Scalar ButtonDefault;
        static const cv::Scalar ButtonHover;
        static const cv::Scalar ButtonActive;
        static const cv::Scalar ButtonDisabled;
        static const cv::Scalar BorderColor;
        static const cv::Scalar ShadowColor;
    };
    
    // Enhanced typography with responsive support
    struct Fonts {
        static constexpr int FontFace = cv::FONT_HERSHEY_SIMPLEX;
        static constexpr int FontFaceBold = cv::FONT_HERSHEY_DUPLEX;
        
        // Base font sizes (responsive scaling applied automatically)
        static constexpr double TitleSize = 1.8;
        static constexpr double HeadingSize = 1.2;
        static constexpr double BodySize = 0.7;
        static constexpr double SmallSize = 0.5;
        static constexpr double ButtonSize = 0.8;
        
        static constexpr int TitleThickness = 3;
        static constexpr int HeadingThickness = 2;
        static constexpr int BodyThickness = 1;
        static constexpr int ButtonThickness = 2;
    };
    
    // Enhanced layout with responsive support
    struct Layout {
        // Base measurements (responsive scaling applied automatically)
        static constexpr int Margin = 20;
        static constexpr int Padding = 15;
        static constexpr int ButtonHeight = 60;
        static constexpr int ButtonWidth = 280;
        static constexpr int BorderRadius = 8;
        static constexpr int ShadowOffset = 4;
        static constexpr int IconSize = 40;
        static constexpr int Spacing = 15;
        
        // New responsive layout constants
        static constexpr float MinButtonScale = 0.7f;
        static constexpr float MaxButtonScale = 1.5f;
        static constexpr int MinTouchTarget = 44; // Minimum touch target size
    };
    
    // Component states for modern interactions
    enum class ComponentState {
        Normal,
        Hover,
        Active,
        Focused,
        Disabled
    };
    
    // Animation state for smooth transitions
    struct AnimationState {
        float progress = 0.0f;        // 0.0 to 1.0
        float duration = 0.25f;       // Animation duration in seconds
        bool isAnimating = false;
        std::string easing = "ease-out";
    };

public:
    // Global theme management
    static void init(int windowWidth = 1280, int windowHeight = 720);
    static void setWindowSize(int width, int height);
    static void setDarkMode(bool enabled);
    static void setHighContrast(bool enabled);
    static void setColorBlindMode(bool enabled);
    static void setScale(float scale);
    
    // Responsive utilities
    static cv::Size getResponsiveSize(int baseWidth, int baseHeight);
    static double getResponsiveFontSize(double baseSize);
    static int getResponsiveSpacing(int baseSpacing);
    static cv::Rect getResponsiveRect(float x, float y, float width, float height, const cv::Rect& parent);
    
    // State management
    static cv::Scalar getStateColor(const cv::Scalar& baseColor, ComponentState state);
    static float getStateOpacity(ComponentState state);
    
    /**
     * @brief Enhanced button with animation and states
     */
    static void drawButton(cv::Mat& img, const std::string& text, 
                          const cv::Rect& rect, ComponentState state = ComponentState::Normal,
                          const AnimationState& anim = AnimationState());
    
    /**
     * @brief Modern icon button with vector-like icons
     */
    static void drawIconButton(cv::Mat& img, const std::string& icon, 
                               const std::string& text, const cv::Rect& rect,
                               ComponentState state = ComponentState::Normal);
    
    /**
     * @brief Glass morphism card with blur and transparency
     */
    static void drawGlassCard(cv::Mat& img, const cv::Rect& rect, 
                             float blurRadius = 15.0f, float opacity = 0.8f,
                             const cv::Scalar& tint = cv::Scalar(255, 255, 255, 30));
    
    /**
     * @brief Enhanced card with shadow and hover effects
     */
    static void drawCard(cv::Mat& img, const cv::Rect& rect, 
                        ComponentState state = ComponentState::Normal,
                        const cv::Scalar& bgColor = Colors::MediumBg,
                        int elevation = 2);
    
    /**
     * @brief Text with responsive sizing and improved readability
     */
    static void drawText(cv::Mat& img, const std::string& text,
                        const cv::Point& pos, double fontSize = Fonts::BodySize,
                        const cv::Scalar& color = Colors::TextPrimary,
                        int fontWeight = Fonts::BodyThickness,
                        bool responsive = true);
    
    /**
     * @brief Enhanced text with shadow and glow effects
     */
    static void drawTextWithShadow(cv::Mat& img, const std::string& text,
                                   const cv::Point& pos, int fontFace,
                                   double scale, const cv::Scalar& color,
                                   int thickness = 1, int shadowOffset = 2,
                                   bool responsive = true);
    
    /**
     * @brief Animated progress bar with easing
     */
    static void drawProgressBar(cv::Mat& img, float progress, 
                               const cv::Rect& rect,
                               const cv::Scalar& color = Colors::NeonCyan,
                               const AnimationState& anim = AnimationState());
    
    /**
     * @brief Modern toggle switch with animation
     */
    static void drawToggle(cv::Mat& img, bool isOn, const cv::Rect& rect,
                          ComponentState state = ComponentState::Normal,
                          const AnimationState& anim = AnimationState());
    
    /**
     * @brief Responsive slider with touch-friendly design
     */
    static void drawSlider(cv::Mat& img, float value, const cv::Rect& rect,
                          float min = 0.0f, float max = 1.0f,
                          ComponentState state = ComponentState::Normal);
    
    /**
     * @brief Modern tab bar with animations
     */
    static void drawTabBar(cv::Mat& img, const std::vector<std::string>& tabs,
                          int activeTab, const cv::Rect& rect,
                          const AnimationState& anim = AnimationState());
    
    /**
     * @brief Enhanced dropdown with search and keyboard navigation
     */
    static void drawDropdown(cv::Mat& img, const std::string& selected,
                            const cv::Rect& rect, ComponentState state = ComponentState::Normal,
                            bool isOpen = false);
    
    /**
     * @brief Animated background with particle effects
     */
    static void drawAnimatedBackground(cv::Mat& img, float time, float intensity = 1.0f);
    
    /**
     * @brief Apply glass morphism effect with GPU acceleration
     */
    static void applyGlassMorphism(cv::Mat& img, const cv::Rect& rect,
                                   int blurAmount = 15, float opacity = 0.8f,
                                   const cv::Scalar& tint = cv::Scalar(255, 255, 255, 30));
    
    /**
     * @brief Draw rounded rectangle with anti-aliasing
     */
    static void drawRoundedRect(cv::Mat& img, const cv::Rect& rect,
                               int radius, const cv::Scalar& color,
                               int thickness = -1, bool antiAlias = true);
    
    /**
     * @brief Enhanced hit testing with hover radius
     */
    static bool isPointInRect(const cv::Point& pt, const cv::Rect& rect,
                             int hoverRadius = 0);
    
    /**
     * @brief Responsive text size calculation
     */
    static cv::Size getTextSize(const std::string& text, int fontFace,
                               double scale, int thickness, bool responsive = true);
    
    /**
     * @brief Draw notification toast
     */
    static void drawToast(cv::Mat& img, const std::string& message,
                         const cv::Point& position, const std::string& type = "info",
                         float opacity = 1.0f);
    
    /**
     * @brief Draw loading spinner
     */
    static void drawSpinner(cv::Mat& img, const cv::Point& center, int radius,
                           float rotation = 0.0f, const cv::Scalar& color = Colors::NeonCyan);
    
    /**
     * @brief Draw accessibility focus indicator
     */
    static void drawFocusRing(cv::Mat& img, const cv::Rect& rect, int thickness = 3);
    
    /**
     * @brief Animation easing functions
     */
    static float easeOut(float t);
    static float easeInOut(float t);
    
    // Legacy compatibility functions (maintained for backward compatibility)
    static void drawTitleBar(cv::Mat& img, const std::string& title, int height = 80);
    
private:
    static std::unique_ptr<ModernTheme> modernTheme_;
    static int windowWidth_;
    static int windowHeight_;
    static bool initialized_;
    
    // Internal utilities
    static cv::Mat createBlurKernel(int size);
    static void applyGaussianBlur(cv::Mat& img, const cv::Rect& rect, int blurRadius);
    static cv::Scalar interpolateColor(const cv::Scalar& from, const cv::Scalar& to, float progress);
};

} // namespace pv
