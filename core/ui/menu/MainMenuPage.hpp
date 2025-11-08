#pragma once
#include "../UITheme.hpp"
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <functional>

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
 * @brief Main menu page with modern UI
 * 
 * Features:
 * - Animated background
 * - Large interactive buttons
 * - Clean, modern design
 * - Hover effects
 */
class MainMenuPage {
public:
    MainMenuPage();
    ~MainMenuPage() = default;
    
    /**
     * @brief Initialize the page
     */
    void init();
    
    /**
     * @brief Render the main menu
     * @return Rendered image
     */
    cv::Mat render();
    
    /**
     * @brief Handle mouse events
     */
    void onMouse(int event, int x, int y, int flags);
    
    /**
     * @brief Handle keyboard events
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
     * @brief Set the window size
     */
    void setWindowSize(int width, int height);
    
private:
    struct MenuItem {
        std::string text;
        std::string icon;
        MenuAction action;
        cv::Rect rect;
        bool isHovered = false;
    };
    
    void createMenuItems();
    void updateLayout();
    void drawBackground(cv::Mat& img);
    void drawLogo(cv::Mat& img);
    void drawMenuItems(cv::Mat& img);
    void drawFooter(cv::Mat& img);
    
    std::vector<MenuItem> menuItems_;
    MenuAction selectedAction_;
    cv::Point mousePos_;
    int windowWidth_;
    int windowHeight_;
    float animationTime_;
};

} // namespace pv
