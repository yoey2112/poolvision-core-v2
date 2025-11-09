#pragma once
#include "../UITheme.hpp"
#include "../ResponsiveLayout.hpp"
#include "../../db/Database.hpp"
#include "../../db/PlayerProfile.hpp"
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <memory>
#include <chrono>

namespace pv {

/**
 * @brief Player Profiles page with modern responsive card grid layout
 * 
 * Features:
 * - Responsive card grid that adapts to screen size
 * - Glass-morphism effects and modern card design
 * - Enhanced hover animations and state management
 * - Modern search and filtering controls
 * - Accessibility support with focus states
 */
class PlayerProfilesPage {
public:
    explicit PlayerProfilesPage(Database& db);
    ~PlayerProfilesPage() = default;
    
    /**
     * @brief Initialize the page with responsive layout
     */
    void init();
    
    /**
     * @brief Render the player profiles page with modern UI
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
     * @brief Check if user wants to go back
     */
    bool shouldGoBack() const { return goBack_; }
    
    /**
     * @brief Reset go back flag
     */
    void resetGoBack() { goBack_ = false; }
    
    /**
     * @brief Set window size and update responsive layout
     */
    void setWindowSize(int width, int height);
    
    /**
     * @brief Get database reference
     */
    Database& getDatabase() { return db_; }
    
private:
    enum class Mode {
        List,           // Show responsive card grid of players
        Add,            // Add new player form
        Edit,           // Edit existing player form
        View            // View player details and stats
    };
    
    struct PlayerListItem {
        PlayerProfile profile;
        cv::Rect rect;
        cv::Rect editButton;
        cv::Rect deleteButton;
        cv::Rect viewButton;
        bool isHovered = false;
        bool isFocused = false;
        float animationProgress = 0.0f;
    };
    
    // Responsive layout methods
    void createResponsiveLayout();
    void updateControlLayout();
    void updateFormLayout();
    void updateLayout(); // Legacy compatibility
    
    void loadPlayers();
    cv::Size calculateResponsiveCardSize();
    void calculateCardButtons(PlayerListItem& item, const cv::Size& cardSize);
    
    // Enhanced rendering methods
    void drawBackground(cv::Mat& img);
    void drawTitle(cv::Mat& img);
    void drawPlayerGrid(cv::Mat& img);
    void drawSearchControls(cv::Mat& img);
    void drawPlayerCard(cv::Mat& img, PlayerListItem& item);
    void drawPlayerStats(cv::Mat& img, const PlayerListItem& item, int x, int y);
    void drawCardButtons(cv::Mat& img, const PlayerListItem& item);
    void drawStatsFooter(cv::Mat& img);
    
    void drawPlayerForm(cv::Mat& img);
    void drawPlayerDetails(cv::Mat& img);
    void drawActionButtons(cv::Mat& img);
    
    // Form controls
    void drawTextInput(cv::Mat& img, const std::string& label, 
                      const std::string& value, const cv::Rect& rect, 
                      UITheme::ComponentState state);
    void drawSkillLevelSelector(cv::Mat& img, const cv::Rect& rect);
    void drawHandednessSelector(cv::Mat& img, const cv::Rect& rect);
    
    // Event handlers
    void handleListClick(int x, int y);
    void handleFormClick(int x, int y);
    void handleDetailsClick(int x, int y);
    
    // Actions
    void startAddPlayer();
    void startEditPlayer(int playerId);
    void startViewPlayer(int playerId);
    void savePlayer();
    void deletePlayer(int playerId);
    void cancelEdit();
    
    Database& db_;
    std::vector<PlayerListItem> playerItems_;
    PlayerProfile currentPlayer_;
    Mode currentMode_;
    
    int windowWidth_;
    int windowHeight_;
    cv::Point mousePos_;
    bool goBack_;
    float animationTime_;
    
    // UI state
    int scrollOffset_;
    std::string searchQuery_;
    int activeInputField_;  // 0 = none, 1 = name, 2 = search
    
    // Responsive layout system
    std::unique_ptr<ResponsiveLayout::Container> rootContainer_;
    cv::Rect headerRect_;
    cv::Rect contentRect_;
    cv::Rect buttonRect_;
    
    // Control rectangles
    cv::Rect addButton_;
    cv::Rect backButton_;
    cv::Rect saveButton_;
    cv::Rect cancelButton_;
    cv::Rect searchBox_;
    
    // Form inputs
    cv::Rect nameInput_;
    cv::Rect skillLevelDropdown_;
    cv::Rect handednessToggle_;
    cv::Rect gameTypeDropdown_;
};

} // namespace pv
