#pragma once
#include <opencv2/opencv.hpp>
#include <string>

namespace pv {

/**
 * @brief Modern UI theme with pool table inspired design
 * 
 * Provides color palette, typography, and drawing utilities for
 * creating a clean, stylish user interface.
 */
class UITheme {
public:
    // Color Palette
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
        
        // Text colors
        static const cv::Scalar TextPrimary;     // #FFFFFF - White
        static const cv::Scalar TextSecondary;   // #CCCCCC - Light gray
        static const cv::Scalar TextDisabled;    // #666666 - Dark gray
        
        // UI elements
        static const cv::Scalar ButtonDefault;
        static const cv::Scalar ButtonHover;
        static const cv::Scalar ButtonActive;
        static const cv::Scalar ButtonDisabled;
        static const cv::Scalar BorderColor;
        static const cv::Scalar ShadowColor;
    };
    
    // Typography
    struct Fonts {
        static constexpr int FontFace = cv::FONT_HERSHEY_SIMPLEX;
        static constexpr int FontFaceBold = cv::FONT_HERSHEY_DUPLEX;
        
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
    
    // Layout constants
    struct Layout {
        static constexpr int Margin = 20;
        static constexpr int Padding = 15;
        static constexpr int ButtonHeight = 60;
        static constexpr int ButtonWidth = 280;
        static constexpr int BorderRadius = 8;
        static constexpr int ShadowOffset = 4;
        static constexpr int IconSize = 40;
        static constexpr int Spacing = 15;
    };
    
    /**
     * @brief Draw a modern styled button
     */
    static void drawButton(cv::Mat& img, const std::string& text, 
                          const cv::Rect& rect, bool isHovered = false, 
                          bool isActive = false, bool isDisabled = false);
    
    /**
     * @brief Draw a button with icon
     */
    static void drawIconButton(cv::Mat& img, const std::string& icon, 
                               const std::string& text, const cv::Rect& rect,
                               bool isHovered = false);
    
    /**
     * @brief Draw a card panel with glass-morphism effect
     */
    static void drawCard(cv::Mat& img, const cv::Rect& rect, 
                        const cv::Scalar& bgColor = Colors::MediumBg,
                        int alpha = 200);
    
    /**
     * @brief Draw text with shadow for better readability
     */
    static void drawTextWithShadow(cv::Mat& img, const std::string& text,
                                   const cv::Point& pos, int fontFace,
                                   double scale, const cv::Scalar& color,
                                   int thickness = 1, int shadowOffset = 2);
    
    /**
     * @brief Draw a title bar
     */
    static void drawTitleBar(cv::Mat& img, const std::string& title,
                            int height = 80);
    
    /**
     * @brief Draw a progress bar
     */
    static void drawProgressBar(cv::Mat& img, float progress, 
                               const cv::Rect& rect,
                               const cv::Scalar& color = Colors::NeonCyan);
    
    /**
     * @brief Draw a toggle switch
     */
    static void drawToggle(cv::Mat& img, bool isOn, const cv::Rect& rect);
    
    /**
     * @brief Draw a slider
     */
    static void drawSlider(cv::Mat& img, float value, const cv::Rect& rect,
                          float min = 0.0f, float max = 1.0f);
    
    /**
     * @brief Draw a tab bar
     */
    static void drawTabBar(cv::Mat& img, const std::vector<std::string>& tabs,
                          int activeTab, const cv::Rect& rect);
    
    /**
     * @brief Draw a dropdown menu
     */
    static void drawDropdown(cv::Mat& img, const std::string& selected,
                            const cv::Rect& rect, bool isOpen = false);
    
    /**
     * @brief Draw an animated background
     */
    static void drawAnimatedBackground(cv::Mat& img, float time);
    
    /**
     * @brief Apply glass-morphism effect to a region
     */
    static void applyGlassMorphism(cv::Mat& img, const cv::Rect& rect,
                                   int blurAmount = 15, int alpha = 180);
    
    /**
     * @brief Draw a rounded rectangle
     */
    static void drawRoundedRect(cv::Mat& img, const cv::Rect& rect,
                               int radius, const cv::Scalar& color,
                               int thickness = -1);
    
    /**
     * @brief Check if point is inside rect (with optional hover radius)
     */
    static bool isPointInRect(const cv::Point& pt, const cv::Rect& rect,
                             int hoverRadius = 0);
    
    /**
     * @brief Get text size for proper layout
     */
    static cv::Size getTextSize(const std::string& text, int fontFace,
                               double scale, int thickness);
};

} // namespace pv
