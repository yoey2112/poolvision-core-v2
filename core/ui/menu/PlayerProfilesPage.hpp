#pragma once
#include "../UITheme.hpp"
#include "../../db/Database.hpp"
#include "../../db/PlayerProfile.hpp"
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace pv {

/**
 * @brief Player Profiles page for managing player database
 */
class PlayerProfilesPage {
public:
    explicit PlayerProfilesPage(Database& db);
    ~PlayerProfilesPage() = default;
    
    /**
     * @brief Initialize the page
     */
    void init();
    
    /**
     * @brief Render the player profiles page
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
     * @brief Check if user wants to go back
     */
    bool shouldGoBack() const { return goBack_; }
    
    /**
     * @brief Reset go back flag
     */
    void resetGoBack() { goBack_ = false; }
    
    /**
     * @brief Set window size
     */
    void setWindowSize(int width, int height);
    
    /**
     * @brief Get database reference
     */
    Database& getDatabase() { return db_; }
    
private:
    enum class Mode {
        List,           // Show list of players
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
    };
    
    void updateLayout();
    void loadPlayers();
    
    // Rendering methods
    void drawPlayerList(cv::Mat& img);
    void drawPlayerForm(cv::Mat& img);
    void drawPlayerDetails(cv::Mat& img);
    void drawActionButtons(cv::Mat& img);
    
    // Form controls
    void drawTextInput(cv::Mat& img, const std::string& label, 
                      const std::string& value, const cv::Rect& rect, bool active);
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
    
    Database db_;
    std::vector<PlayerListItem> playerItems_;
    PlayerProfile currentPlayer_;
    Mode currentMode_;
    
    int windowWidth_;
    int windowHeight_;
    cv::Point mousePos_;
    bool goBack_;
    
    // UI state
    int scrollOffset_;
    std::string searchQuery_;
    int activeInputField_;  // 0 = none, 1 = name, 2 = search
    
    // Buttons
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
