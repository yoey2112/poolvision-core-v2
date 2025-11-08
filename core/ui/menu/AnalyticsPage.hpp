#pragma once
#include "../../db/Database.hpp"
#include "../../db/PlayerProfile.hpp"
#include "../UITheme.hpp"
#include <opencv2/opencv.hpp>
#include <memory>
#include <vector>
#include <string>

namespace pv {

/**
 * @brief Analytics dashboard showing player statistics and game history
 * 
 * Displays comprehensive statistics, performance trends, heat maps,
 * and game history for analysis and improvement.
 */
class AnalyticsPage {
public:
    /**
     * @brief Player statistics summary
     */
    struct PlayerStats {
        std::string name;
        int gamesPlayed;
        float winRate;
        float shotSuccessRate;
        int totalShots;
        int successfulShots;
        int fouls;
    };
    
    /**
     * @brief Construct a new Analytics Page
     * 
     * @param database Database instance
     */
    AnalyticsPage(Database& database);
    ~AnalyticsPage() = default;
    
    /**
     * @brief Render the analytics page
     * 
     * @param frame Frame to render on
     * @return cv::Mat Rendered frame
     */
    cv::Mat render(cv::Mat& frame);
    
    /**
     * @brief Handle mouse events
     */
    void onMouse(int event, int x, int y, int flags);
    
    /**
     * @brief Handle keyboard input
     * 
     * @param key Key code
     * @return true if handled, false to propagate
     */
    bool onKey(int key);
    
    /**
     * @brief Set selected player for detailed view
     */
    void setSelectedPlayer(int playerId);
    
    /**
     * @brief Get result (selected action)
     * @return std::string Action ("back", "player_select", etc.)
     */
    std::string getResult() const { return result_; }
    
private:
    enum class ViewMode {
        Overview,      // Show all players summary
        PlayerDetail,  // Show specific player details
        GameHistory,   // Show game history list
        HeatMap       // Show shot position heat map
    };
    
    Database& database_;
    ViewMode currentView_ = ViewMode::Overview;
    int selectedPlayerId_ = -1;
    std::string result_;
    
    // UI state
    cv::Point mousePos_;
    std::vector<cv::Rect> clickableAreas_;
    int hoveredIndex_ = -1;
    int scrollOffset_ = 0;
    
    // Data cache
    std::vector<PlayerStats> playerStats_;
    std::vector<GameSession> recentGames_;
    std::vector<ShotRecord> playerShots_;
    
    // Rendering methods
    void renderOverview(cv::Mat& frame);
    void renderPlayerDetail(cv::Mat& frame);
    void renderGameHistory(cv::Mat& frame);
    void renderHeatMap(cv::Mat& frame);
    
    void drawStatCard(cv::Mat& frame, const cv::Rect& rect, 
                     const std::string& title, const std::string& value,
                     const cv::Scalar& color);
    
    void drawBarChart(cv::Mat& frame, const cv::Rect& rect,
                     const std::vector<float>& values,
                     const std::vector<std::string>& labels,
                     const std::string& title);
    
    void drawLineChart(cv::Mat& frame, const cv::Rect& rect,
                      const std::vector<float>& values,
                      const std::string& title, const std::string& yLabel);
    
    void drawHeatMapVisualization(cv::Mat& frame, const cv::Rect& rect,
                                 const std::vector<cv::Point2f>& positions,
                                 const std::vector<bool>& success);
    
    void drawBackButton(cv::Mat& frame);
    void drawTabBar(cv::Mat& frame);
    
    // Data loading methods
    void loadPlayerStats();
    void loadRecentGames();
    void loadPlayerShots(int playerId);
    
    // Helper methods
    PlayerStats calculatePlayerStats(const PlayerProfile& player);
    float calculateTrendDirection(const std::vector<float>& values);
};

} // namespace pv
