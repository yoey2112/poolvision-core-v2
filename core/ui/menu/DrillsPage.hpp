#pragma once
#include "../../game/DrillSystem.hpp"
#include "../../db/Database.hpp"
#include "../UITheme.hpp"
#include <opencv2/opencv.hpp>
#include <memory>
#include <functional>

namespace pv {

/**
 * @brief Drill selection and execution interface
 * 
 * Provides a comprehensive UI for browsing drills, starting sessions,
 * tracking progress, and managing custom drills.
 */
class DrillsPage {
public:
    /**
     * @brief UI states for different drill interfaces
     */
    enum class DrillUIState {
        DrillLibrary,     // Browse and select drills
        DrillExecution,   // Active drill session
        DrillResults,     // Session results and statistics
        DrillCreator,     // Custom drill creation
        DrillStats        // Historical performance
    };

    /**
     * @brief Filter options for drill library
     */
    struct DrillFilter {
        DrillSystem::Category category;
        DrillSystem::Difficulty difficulty;
        std::string searchQuery;
        bool showCustomOnly;
        bool showFavoritesOnly;
        
        DrillFilter() : category(DrillSystem::Category::CutShots),
                       difficulty(DrillSystem::Difficulty::Beginner),
                       showCustomOnly(false), showFavoritesOnly(false) {}
    };

    /**
     * @brief Drill execution UI state
     */
    struct DrillExecutionState {
        int currentAttempt;
        int totalAttempts;
        double currentAccuracy;
        double bestAccuracy;
        std::string feedback;
        bool isPaused;
        double timeRemaining;
        std::vector<DrillSystem::DrillAttempt> attempts;
        
        DrillExecutionState() : currentAttempt(0), totalAttempts(0),
                               currentAccuracy(0.0), bestAccuracy(0.0),
                               isPaused(false), timeRemaining(0.0) {}
    };

    /**
     * @brief Construct a new Drills Page
     */
    DrillsPage(Database& database, std::shared_ptr<DrillSystem> drillSystem);
    ~DrillsPage() = default;

    /**
     * @brief Render the drill interface
     */
    void render(cv::Mat& frame);

    /**
     * @brief Handle mouse click events
     */
    void handleClick(const cv::Point& clickPos);

    /**
     * @brief Handle keyboard input
     */
    void handleKeyPress(int key);

    /**
     * @brief Update drill execution state
     */
    void update();

    /**
     * @brief Set callbacks for navigation
     */
    void setNavigationCallback(std::function<void(const std::string&)> callback) {
        navigationCallback_ = callback;
    }

    /**
     * @brief Set active player
     */
    void setActivePlayer(int playerId) { activePlayerId_ = playerId; }

    /**
     * @brief Get current UI state
     */
    DrillUIState getCurrentState() const { return currentState_; }

    /**
     * @brief Set UI state
     */
    void setState(DrillUIState state) { currentState_ = state; }

private:
    Database& database_;
    std::shared_ptr<DrillSystem> drillSystem_;
    DrillUIState currentState_;
    DrillFilter currentFilter_;
    DrillExecutionState executionState_;
    std::function<void(const std::string&)> navigationCallback_;
    int activePlayerId_;
    int selectedDrillId_;
    int scrollOffset_;
    std::vector<DrillSystem::Drill> filteredDrills_;

    // UI button definitions
    struct Button {
        cv::Rect rect;
        std::string label;
        std::function<void()> action;
        bool enabled;
        bool visible;
        
        Button() : enabled(true), visible(true) {}
    };

    std::vector<Button> buttons_;
    cv::Point lastClickPos_;
    bool mousePressed_;

    // Rendering methods
    void renderDrillLibrary(cv::Mat& frame);
    void renderDrillExecution(cv::Mat& frame);
    void renderDrillResults(cv::Mat& frame);
    void renderDrillCreator(cv::Mat& frame);
    void renderDrillStats(cv::Mat& frame);

    // Drill library rendering
    void renderDrillList(cv::Mat& frame, const cv::Rect& listArea);
    void renderDrillCard(cv::Mat& frame, const DrillSystem::Drill& drill, const cv::Rect& cardRect, bool isSelected);
    void renderFilterPanel(cv::Mat& frame, const cv::Rect& filterArea);
    void renderDrillDetails(cv::Mat& frame, const DrillSystem::Drill& drill, const cv::Rect& detailArea);

    // Drill execution rendering
    void renderExecutionHUD(cv::Mat& frame);
    void renderAttemptHistory(cv::Mat& frame, const cv::Rect& historyArea);
    void renderDrillInstructions(cv::Mat& frame, const cv::Rect& instructionArea);
    void renderProgressBar(cv::Mat& frame, const cv::Rect& progressArea);
    void renderExecutionControls(cv::Mat& frame, const cv::Rect& controlArea);

    // Results rendering
    void renderResultsSummary(cv::Mat& frame, const cv::Rect& summaryArea);
    void renderAccuracyChart(cv::Mat& frame, const cv::Rect& chartArea);
    void renderPerformanceMetrics(cv::Mat& frame, const cv::Rect& metricsArea);

    // Statistics rendering
    void renderStatsOverview(cv::Mat& frame, const cv::Rect& overviewArea);
    void renderDrillProgress(cv::Mat& frame, const cv::Rect& progressArea);
    void renderImprovementTrends(cv::Mat& frame, const cv::Rect& trendsArea);

    // Custom drill creator
    void renderDrillDesigner(cv::Mat& frame, const cv::Rect& designArea);
    void renderBallPlacement(cv::Mat& frame, const cv::Rect& tableArea);
    void renderDrillProperties(cv::Mat& frame, const cv::Rect& propertiesArea);

    // Helper methods
    void updateFilteredDrills();
    void startSelectedDrill();
    void pauseCurrentDrill();
    void resumeCurrentDrill();
    void endCurrentDrill();
    void resetCurrentDrill();
    bool isPointInRect(const cv::Point& point, const cv::Rect& rect) const;
    void createButtons();
    void updateButtons();
    Button* findButtonAt(const cv::Point& point);

    // Data management
    std::vector<DrillSystem::DrillStats> getPlayerStats() const;
    void saveDrillResults();
    void loadDrillHistory();

    // UI constants
    static constexpr int CARD_HEIGHT = 120;
    static constexpr int CARD_MARGIN = 10;
    static constexpr int FILTER_PANEL_WIDTH = 250;
    static constexpr int DETAIL_PANEL_WIDTH = 300;
    static constexpr int BUTTON_HEIGHT = 40;
    static constexpr int BUTTON_WIDTH = 120;
    static constexpr int SCROLL_STEP = 20;
};

} // namespace pv