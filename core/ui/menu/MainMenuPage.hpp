#pragma once
#include "../UITheme.hpp"
#include "../ResponsiveLayout.hpp"
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <memory>

namespace pv {

/**
 * @brief Menu item action types
 */
enum class MenuAction {
    NewGame,
    Drills,
    PlayerProfiles,
    Analytics,
    Settings,
    Calibration,
    Exit,
    None
};

/**
 * @brief Main menu page with modern responsive UI
 * 
 * Features:
 * - Responsive layout system
 * - Glass-morphism effects
 * - Animated background and transitions
 * - Modern icon system
 * - Hover and focus states
 * - Accessibility support
 */
class MainMenuPage {
public:
    MainMenuPage();
    ~MainMenuPage() = default;
    
    /**
     * @brief Initialize the page with responsive layout
     */
    void init();
    
    /**
     * @brief Render the main menu with modern effects
     * @return Rendered image
     */
    cv::Mat render();
    
    /**
     * @brief Handle mouse events with enhanced interaction
     */
    void onMouse(int event, int x, int y, int flags);
    
    /**
     * @brief Handle keyboard events with accessibility
     */
    void onKey(int key);
    
    /**
     * @brief Get the selected action
     */
    MenuAction getSelectedAction() const { return selectedAction_; }
    
    /**
     * @brief Reset the selected action
     */
    void resetAction() { selectedAction_ = MenuAction::None; }
    
    /**
     * @brief Set the window size and update responsive layout
     */
    void setWindowSize(int width, int height);
    
private:
    struct MenuItem {
        std::string text;
        std::string icon;
        MenuAction action;
        cv::Rect rect;
        bool isHovered = false;
        bool isFocused = false;
        float animationProgress = 0.0f;
    };
    
    void createMenuItems();
    void createResponsiveLayout();
    void updateMenuLayout();
    void updateLayout(); // Legacy compatibility
    
    void drawBackground(cv::Mat& img);
    void drawLogo(cv::Mat& img);
    void drawMenuItems(cv::Mat& img);
    void drawFooter(cv::Mat& img);
    void drawModernIcon(cv::Mat& img, const std::string& icon, 
                       const cv::Point& pos, int size, const cv::Scalar& color);
    
    std::vector<MenuItem> menuItems_;
    MenuAction selectedAction_;
    cv::Point mousePos_;
    int windowWidth_;
    int windowHeight_;
    float animationTime_;
    
    // Responsive layout system
    std::unique_ptr<ResponsiveLayout::Container> rootContainer_;
    cv::Rect headerRect_;
    cv::Rect menuRect_;
    cv::Rect footerRect_;
};

} // namespace pv
