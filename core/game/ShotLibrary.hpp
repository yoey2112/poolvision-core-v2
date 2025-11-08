#pragma once
#include "../db/Database.hpp"
#include "SessionPlayback.hpp"
#include <opencv2/opencv.hpp>
#include <memory>
#include <vector>
#include <string>
#include <map>

namespace pv {

/**
 * @brief Library of recorded shots for review and practice
 * 
 * Manages a collection of successful shots that players can save,
 * categorize, search, and practice
 */
class ShotLibrary {
public:
    /**
     * @brief Shot categories for organization
     */
    enum class ShotCategory {
        Break,          // Break shots
        Cut,            // Cut shots
        Bank,           // Bank shots
        Combination,    // Combination shots
        Safety,         // Safety/defensive shots
        Position,       // Position play shots
        Power,          // Power shots
        Finesse,        // Finesse/touch shots
        Trick,          // Trick shots
        Favorite,       // Player favorites
        All             // All shots (for browsing)
    };

    /**
     * @brief Shot difficulty rating
     */
    enum class ShotDifficulty {
        Beginner = 1,
        Intermediate = 2,
        Advanced = 3,
        Expert = 4,
        Master = 5
    };

    /**
     * @brief Shot entry in the library
     */
    struct LibraryShot {
        int id;                         // Unique shot ID
        int sessionId;                  // Source session
        int shotNumber;                 // Shot number in session
        int playerId;                   // Player who made the shot
        std::string playerName;         // Player name
        std::string title;              // Custom shot title
        std::string description;        // Shot description
        ShotCategory category;          // Shot category
        ShotDifficulty difficulty;      // Difficulty rating
        float successRate;              // Success rate when practiced
        int practiceCount;              // Times this shot was practiced
        cv::Point2f cueBallStart;       // Starting cue ball position
        cv::Point2f cueBallEnd;         // Ending cue ball position
        cv::Point2f targetBall;         // Target ball position
        cv::Point2f objectiveBall;      // Objective ball/pocket position
        float shotSpeed;                // Shot speed
        std::string shotType;           // Type of shot
        bool isFavorite;                // Marked as favorite
        std::chrono::time_point<std::chrono::system_clock> dateRecorded; // When recorded
        std::chrono::time_point<std::chrono::system_clock> lastPracticed; // Last practice time
        std::vector<std::string> tags;  // Custom tags
        cv::Mat thumbnail;              // Shot thumbnail image
    };

    /**
     * @brief Search/filter criteria
     */
    struct SearchCriteria {
        ShotCategory category = ShotCategory::All;
        ShotDifficulty minDifficulty = ShotDifficulty::Beginner;
        ShotDifficulty maxDifficulty = ShotDifficulty::Master;
        int playerId = -1;              // -1 for all players
        std::string searchText;         // Text search in title/description
        std::vector<std::string> tags;  // Required tags
        bool favoritesOnly = false;     // Show only favorites
        float minSuccessRate = 0.0f;    // Minimum success rate
        int sortBy = 0;                 // 0=date, 1=difficulty, 2=success rate
        bool ascending = false;         // Sort order
    };

public:
    /**
     * @brief Constructor
     * @param database Database reference for shot storage
     */
    explicit ShotLibrary(Database& database);
    
    /**
     * @brief Add a shot to the library
     * @param shot Shot record from database
     * @param title Custom title for the shot
     * @param description Description of the shot
     * @param category Shot category
     * @param difficulty Difficulty rating
     * @return true if shot was added successfully
     */
    bool addShot(const ShotRecord& shot, 
                const std::string& title,
                const std::string& description,
                ShotCategory category,
                ShotDifficulty difficulty);
    
    /**
     * @brief Remove a shot from the library
     * @param shotId Shot ID to remove
     * @return true if shot was removed
     */
    bool removeShot(int shotId);
    
    /**
     * @brief Update shot information
     * @param shot Updated shot data
     * @return true if shot was updated
     */
    bool updateShot(const LibraryShot& shot);
    
    /**
     * @brief Search shots by criteria
     * @param criteria Search/filter criteria
     * @return Vector of matching shots
     */
    std::vector<LibraryShot> searchShots(const SearchCriteria& criteria) const;
    
    /**
     * @brief Get shot by ID
     * @param shotId Shot ID to retrieve
     * @return Shot data, or empty shot if not found
     */
    LibraryShot getShot(int shotId) const;
    
    /**
     * @brief Get all shots in category
     * @param category Category to filter by
     * @return Vector of shots in category
     */
    std::vector<LibraryShot> getShotsByCategory(ShotCategory category) const;
    
    /**
     * @brief Get favorite shots
     * @param playerId Player ID (-1 for all players)
     * @return Vector of favorite shots
     */
    std::vector<LibraryShot> getFavoriteShots(int playerId = -1) const;
    
    /**
     * @brief Mark/unmark shot as favorite
     * @param shotId Shot ID
     * @param isFavorite Whether to mark as favorite
     * @return true if updated successfully
     */
    bool setFavorite(int shotId, bool isFavorite);
    
    /**
     * @brief Add tag to shot
     * @param shotId Shot ID
     * @param tag Tag to add
     * @return true if tag was added
     */
    bool addTag(int shotId, const std::string& tag);
    
    /**
     * @brief Remove tag from shot
     * @param shotId Shot ID
     * @param tag Tag to remove
     * @return true if tag was removed
     */
    bool removeTag(int shotId, const std::string& tag);
    
    /**
     * @brief Get all unique tags in library
     * @return Vector of all tags
     */
    std::vector<std::string> getAllTags() const;
    
    /**
     * @brief Record practice attempt of a library shot
     * @param shotId Shot being practiced
     * @param successful Whether attempt was successful
     */
    void recordPracticeAttempt(int shotId, bool successful);
    
    /**
     * @brief Render shot library interface
     * @param frame Frame to draw on
     */
    void render(cv::Mat& frame);
    
    /**
     * @brief Render shot details view
     * @param frame Frame to draw on
     * @param shot Shot to display details for
     */
    void renderShotDetails(cv::Mat& frame, const LibraryShot& shot);
    
    /**
     * @brief Handle mouse events
     * @param event OpenCV mouse event
     * @param x Mouse X coordinate  
     * @param y Mouse Y coordinate
     * @param flags Mouse event flags
     */
    void onMouse(int event, int x, int y, int flags);
    
    /**
     * @brief Handle keyboard events
     * @param key Key code
     * @return true if key was handled
     */
    bool onKeyboard(int key);
    
    /**
     * @brief Start practicing a library shot
     * @param shotId Shot ID to practice
     * @return true if practice mode started
     */
    bool startPractice(int shotId);
    
    /**
     * @brief Export shots to file
     * @param filename Output filename
     * @param shots Shots to export
     * @return true if exported successfully
     */
    bool exportShots(const std::string& filename, const std::vector<LibraryShot>& shots) const;
    
    /**
     * @brief Import shots from file
     * @param filename Input filename
     * @return Number of shots imported
     */
    int importShots(const std::string& filename);
    
    /**
     * @brief Get library statistics
     */
    struct LibraryStats {
        int totalShots;
        int favoriteShots;
        int categoryCounts[10]; // Count per category
        float averageDifficulty;
        float averageSuccessRate;
        std::string mostPopularTag;
    };
    LibraryStats getStats() const;

private:
    Database& database_;
    
    // UI state
    std::vector<LibraryShot> currentResults_;
    SearchCriteria currentSearch_;
    int selectedShotId_;
    int currentPage_;
    int shotsPerPage_;
    
    // View state
    enum class ViewMode {
        Grid,           // Grid view of shots
        List,           // List view with details
        Details,        // Detailed view of single shot
        Search          // Search interface
    };
    ViewMode currentView_;
    
    cv::Point mousePos_;
    std::vector<cv::Rect> clickableAreas_;
    cv::Rect searchRect_;
    cv::Rect gridRect_;
    cv::Rect detailsRect_;
    cv::Rect controlsRect_;
    
    // Playback for shot preview
    SessionPlayback playbackSystem_;
    bool showingPreview_;
    
    /**
     * @brief Load shots from database
     */
    void loadShots();
    
    /**
     * @brief Apply current search criteria
     */
    void applySearch();
    
    /**
     * @brief Sort shots by specified criteria
     */
    void sortShots(std::vector<LibraryShot>& shots, int sortBy, bool ascending) const;
    
    /**
     * @brief Render search interface
     */
    void renderSearchInterface(cv::Mat& frame);
    
    /**
     * @brief Render shot grid
     */
    void renderShotGrid(cv::Mat& frame);
    
    /**
     * @brief Render shot list
     */
    void renderShotList(cv::Mat& frame);
    
    /**
     * @brief Render library controls
     */
    void renderControls(cv::Mat& frame);
    
    /**
     * @brief Render shot thumbnail
     */
    void renderShotThumbnail(cv::Mat& frame, const cv::Rect& rect, const LibraryShot& shot);
    
    /**
     * @brief Handle button clicks
     */
    void handleButtonClick(int buttonIndex);
    
    /**
     * @brief Generate thumbnail for shot
     */
    cv::Mat generateThumbnail(const LibraryShot& shot);
    
    /**
     * @brief Convert category to string
     */
    static std::string categoryToString(ShotCategory category);
    
    /**
     * @brief Convert difficulty to string  
     */
    static std::string difficultyToString(ShotDifficulty difficulty);
    
    /**
     * @brief Convert string to category
     */
    static ShotCategory stringToCategory(const std::string& str);
    
    /**
     * @brief Convert string to difficulty
     */
    static ShotDifficulty stringToDifficulty(const std::string& str);
};

} // namespace pv