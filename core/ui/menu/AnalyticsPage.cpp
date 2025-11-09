#include "AnalyticsPage.hpp"
#include <algorithm>
#include <cmath>
#include <sstream>
#include <iomanip>

using namespace pv;

AnalyticsPage::AnalyticsPage(Database& database)
    : database_(database) {
    loadPlayerStats();
    loadRecentGames();
}

cv::Mat AnalyticsPage::render(cv::Mat& frame) {
    // Clear frame
    frame = cv::Mat(720, 1280, CV_8UC3, UITheme::Colors::DarkBg);
    
    // Draw title bar
    UITheme::drawTitleBar(frame, "Analytics & Statistics");
    
    // Draw tab bar for view selection
    drawTabBar(frame);
    
    // Render current view
    switch (currentView_) {
        case ViewMode::Overview:
            renderOverview(frame);
            break;
        case ViewMode::PlayerDetail:
            renderPlayerDetail(frame);
            break;
        case ViewMode::GameHistory:
            renderGameHistory(frame);
            break;
        case ViewMode::HeatMap:
            renderHeatMap(frame);
            break;
    }
    
    // Draw back button
    drawBackButton(frame);
    
    return frame;
}

void AnalyticsPage::renderOverview(cv::Mat& frame) {
    int x = 40;
    int y = 180;
    const int cardWidth = 360;
    const int cardHeight = 200;
    const int spacing = 40;
    
    clickableAreas_.clear();
    
    // Draw player stat cards
    for (size_t i = 0; i < playerStats_.size() && i < 6; ++i) {
        const auto& stats = playerStats_[i];
        
        int col = i % 3;
        int row = i / 3;
        cv::Rect cardRect(x + col * (cardWidth + spacing),
                         y + row * (cardHeight + spacing),
                         cardWidth, cardHeight);
        
        // Draw card background
        UITheme::drawCard(frame, cardRect, UITheme::ComponentState::Normal, UITheme::Colors::MediumBg);
        
        // Player name
        cv::putText(frame, stats.name, 
                   cv::Point(cardRect.x + 20, cardRect.y + 40),
                   UITheme::Fonts::FontFaceBold, UITheme::Fonts::HeadingSize,
                   UITheme::Colors::NeonCyan, UITheme::Fonts::HeadingThickness);
        
        // Stats grid
        int statY = cardRect.y + 80;
        
        // Games played
        std::string gamesText = "Games: " + std::to_string(stats.gamesPlayed);
        cv::putText(frame, gamesText, cv::Point(cardRect.x + 20, statY),
                   UITheme::Fonts::FontFace, UITheme::Fonts::BodySize,
                   UITheme::Colors::TextPrimary, UITheme::Fonts::BodyThickness);
        
        // Win rate
        statY += 30;
        std::stringstream winStream;
        winStream << "Win Rate: " << std::fixed << std::setprecision(1) 
                 << (stats.winRate * 100) << "%";
        cv::putText(frame, winStream.str(), cv::Point(cardRect.x + 20, statY),
                   UITheme::Fonts::FontFace, UITheme::Fonts::BodySize,
                   UITheme::Colors::NeonGreen, UITheme::Fonts::BodyThickness);
        
        // Shot success rate
        statY += 30;
        std::stringstream shotStream;
        shotStream << "Shot Success: " << std::fixed << std::setprecision(1)
                  << (stats.shotSuccessRate * 100) << "%";
        cv::putText(frame, shotStream.str(), cv::Point(cardRect.x + 20, statY),
                   UITheme::Fonts::FontFace, UITheme::Fonts::BodySize,
                   UITheme::Colors::TextSecondary, UITheme::Fonts::BodyThickness);
        
        clickableAreas_.push_back(cardRect);
    }
    
    // If no players, show message
    if (playerStats_.empty()) {
        std::string msg = "No player data available. Create players in Player Profiles.";
        cv::Size textSize = UITheme::getTextSize(msg, UITheme::Fonts::FontFace,
                                                UITheme::Fonts::BodySize,
                                                UITheme::Fonts::BodyThickness);
        cv::Point textPos((frame.cols - textSize.width) / 2, frame.rows / 2);
        cv::putText(frame, msg, textPos, UITheme::Fonts::FontFace,
                   UITheme::Fonts::BodySize, UITheme::Colors::TextSecondary,
                   UITheme::Fonts::BodyThickness);
    }
}

void AnalyticsPage::renderPlayerDetail(cv::Mat& frame) {
    if (selectedPlayerId_ < 0 || playerShots_.empty()) {
        std::string msg = "No detailed data available for this player.";
        cv::Size textSize = UITheme::getTextSize(msg, UITheme::Fonts::FontFace,
                                                UITheme::Fonts::BodySize,
                                                UITheme::Fonts::BodyThickness);
        cv::Point textPos((frame.cols - textSize.width) / 2, frame.rows / 2);
        cv::putText(frame, msg, textPos, UITheme::Fonts::FontFace,
                   UITheme::Fonts::BodySize, UITheme::Colors::TextSecondary,
                   UITheme::Fonts::BodyThickness);
        return;
    }
    
    // Find player stats
    auto playerIt = std::find_if(playerStats_.begin(), playerStats_.end(),
        [this](const PlayerStats& s) { return s.name == ""; }); // TODO: Match by ID
    
    // Draw charts
    int y = 180;
    
    // Win rate trend chart
    cv::Rect chartRect1(40, y, 580, 200);
    std::vector<float> winRates = calculatePlayerWinRateTrend();
    drawLineChart(frame, chartRect1, winRates, "Win Rate Trend", "Win %");
    
    // Shot success by type
    cv::Rect chartRect2(660, y, 580, 200);
    auto shotTypeStats = calculateShotSuccessByType();
    drawBarChart(frame, chartRect2, shotTypeStats.first, shotTypeStats.second, "Shot Success by Type");
    
    // Heat map
    y += 240;
    cv::Rect heatMapRect(40, y, 580, 240);
    std::vector<cv::Point2f> positions;
    std::vector<bool> success;
    for (const auto& shot : playerShots_) {
        positions.emplace_back(shot.ballX, shot.ballY);
        success.push_back(shot.successful);
    }
    drawHeatMapVisualization(frame, heatMapRect, positions, success);
}

void AnalyticsPage::renderGameHistory(cv::Mat& frame) {
    int y = 180;
    clickableAreas_.clear();
    
    // Draw game history list
    for (size_t i = 0; i < recentGames_.size() && i < 8; ++i) {
        const auto& game = recentGames_[i];
        
        cv::Rect gameRect(40, y, 1200, 60);
        UITheme::drawCard(frame, gameRect, UITheme::ComponentState::Normal, UITheme::Colors::MediumBg, 180);
        
        // Game type
        cv::putText(frame, game.gameType, cv::Point(gameRect.x + 20, gameRect.y + 35),
                   UITheme::Fonts::FontFaceBold, UITheme::Fonts::BodySize,
                   UITheme::Colors::NeonCyan, UITheme::Fonts::ButtonThickness);
        
        // Players and scores
        std::string scoreText = "Player " + std::to_string(game.player1Id) + " vs Player " +
                               std::to_string(game.player2Id) + " | Score: " +
                               std::to_string(game.player1Score) + " - " +
                               std::to_string(game.player2Score);
        cv::putText(frame, scoreText, cv::Point(gameRect.x + 200, gameRect.y + 35),
                   UITheme::Fonts::FontFace, UITheme::Fonts::BodySize,
                   UITheme::Colors::TextPrimary, UITheme::Fonts::BodyThickness);
        
        // Duration
        std::string durationText = std::to_string(game.durationSeconds / 60) + " min";
        cv::putText(frame, durationText, cv::Point(gameRect.x + 1000, gameRect.y + 35),
                   UITheme::Fonts::FontFace, UITheme::Fonts::BodySize,
                   UITheme::Colors::TextSecondary, UITheme::Fonts::BodyThickness);
        
        clickableAreas_.push_back(gameRect);
        y += 70;
    }
    
    // If no games, show message
    if (recentGames_.empty()) {
        std::string msg = "No game history available. Play some games to see history!";
        cv::Size textSize = UITheme::getTextSize(msg, UITheme::Fonts::FontFace,
                                                UITheme::Fonts::BodySize,
                                                UITheme::Fonts::BodyThickness);
        cv::Point textPos((frame.cols - textSize.width) / 2, frame.rows / 2);
        cv::putText(frame, msg, textPos, UITheme::Fonts::FontFace,
                   UITheme::Fonts::BodySize, UITheme::Colors::TextSecondary,
                   UITheme::Fonts::BodyThickness);
    }
}

void AnalyticsPage::renderHeatMap(cv::Mat& frame) {
    // Full-screen heat map
    cv::Rect heatMapRect(40, 180, 1200, 480);
    
    // Load all shots from database
    std::vector<cv::Point2f> allPositions;
    std::vector<bool> allSuccess;
    
    // Load real shot data from all players
    auto allShots = getAllPlayerShots();
    for (const auto& shot : allShots) {
        allPositions.emplace_back(shot.ballX, shot.ballY);
        allSuccess.push_back(shot.successful);
    }
    
    drawHeatMapVisualization(frame, heatMapRect, allPositions, allSuccess);
}

void AnalyticsPage::drawStatCard(cv::Mat& frame, const cv::Rect& rect,
                                 const std::string& title, const std::string& value,
                                 const cv::Scalar& color) {
    UITheme::drawCard(frame, rect, UITheme::ComponentState::Normal, UITheme::Colors::MediumBg, 200);
    
    // Title
    cv::putText(frame, title, cv::Point(rect.x + 20, rect.y + 30),
               UITheme::Fonts::FontFace, UITheme::Fonts::SmallSize,
               UITheme::Colors::TextSecondary, UITheme::Fonts::BodyThickness);
    
    // Value
    cv::putText(frame, value, cv::Point(rect.x + 20, rect.y + 70),
               UITheme::Fonts::FontFaceBold, UITheme::Fonts::TitleSize,
               color, UITheme::Fonts::TitleThickness);
}

void AnalyticsPage::drawBarChart(cv::Mat& frame, const cv::Rect& rect,
                                 const std::vector<float>& values,
                                 const std::vector<std::string>& labels,
                                 const std::string& title) {
    UITheme::drawCard(frame, rect, UITheme::ComponentState::Normal, UITheme::Colors::MediumBg, 180);
    
    // Title
    cv::putText(frame, title, cv::Point(rect.x + 20, rect.y + 30),
               UITheme::Fonts::FontFaceBold, UITheme::Fonts::BodySize,
               UITheme::Colors::TextPrimary, UITheme::Fonts::ButtonThickness);
    
    if (values.empty()) return;
    
    int chartY = rect.y + 60;
    int chartHeight = rect.height - 80;
    int barWidth = (rect.width - 80) / values.size();
    
    for (size_t i = 0; i < values.size(); ++i) {
        int x = rect.x + 40 + i * barWidth;
        int barHeight = static_cast<int>(chartHeight * values[i]);
        int y = chartY + chartHeight - barHeight;
        
        cv::Rect barRect(x, y, barWidth - 10, barHeight);
        cv::rectangle(frame, barRect, UITheme::Colors::NeonCyan, -1);
        
        // Label
        if (i < labels.size()) {
            cv::Size labelSize = UITheme::getTextSize(labels[i], UITheme::Fonts::FontFace,
                                                     UITheme::Fonts::SmallSize, 1);
            cv::putText(frame, labels[i],
                       cv::Point(x + (barWidth - labelSize.width) / 2,
                                chartY + chartHeight + 20),
                       UITheme::Fonts::FontFace, UITheme::Fonts::SmallSize,
                       UITheme::Colors::TextSecondary, 1);
        }
        
        // Value
        std::stringstream ss;
        ss << std::fixed << std::setprecision(0) << (values[i] * 100) << "%";
        cv::putText(frame, ss.str(), cv::Point(x + 5, y - 5),
                   UITheme::Fonts::FontFace, UITheme::Fonts::SmallSize,
                   UITheme::Colors::TextPrimary, 1);
    }
}

void AnalyticsPage::drawLineChart(cv::Mat& frame, const cv::Rect& rect,
                                  const std::vector<float>& values,
                                  const std::string& title, const std::string& yLabel) {
    UITheme::drawCard(frame, rect, UITheme::ComponentState::Normal, UITheme::Colors::MediumBg, 180);
    
    // Title
    cv::putText(frame, title, cv::Point(rect.x + 20, rect.y + 30),
               UITheme::Fonts::FontFaceBold, UITheme::Fonts::BodySize,
               UITheme::Colors::TextPrimary, UITheme::Fonts::ButtonThickness);
    
    if (values.size() < 2) return;
    
    int chartX = rect.x + 40;
    int chartY = rect.y + 60;
    int chartWidth = rect.width - 80;
    int chartHeight = rect.height - 80;
    
    // Draw axes
    cv::line(frame, cv::Point(chartX, chartY + chartHeight),
            cv::Point(chartX + chartWidth, chartY + chartHeight),
            UITheme::Colors::TextDisabled, 1);
    cv::line(frame, cv::Point(chartX, chartY),
            cv::Point(chartX, chartY + chartHeight),
            UITheme::Colors::TextDisabled, 1);
    
    // Draw line
    std::vector<cv::Point> points;
    float maxVal = *std::max_element(values.begin(), values.end());
    float minVal = *std::max_element(values.begin(), values.end());
    float range = maxVal - minVal;
    if (range < 0.01f) range = 1.0f;
    
    for (size_t i = 0; i < values.size(); ++i) {
        float x = chartX + (i * chartWidth) / (values.size() - 1);
        float normalizedValue = (values[i] - minVal) / range;
        float y = chartY + chartHeight - (normalizedValue * chartHeight);
        points.emplace_back(static_cast<int>(x), static_cast<int>(y));
    }
    
    // Draw line segments
    for (size_t i = 1; i < points.size(); ++i) {
        cv::line(frame, points[i-1], points[i], UITheme::Colors::NeonCyan, 2, cv::LINE_AA);
    }
    
    // Draw points
    for (const auto& pt : points) {
        cv::circle(frame, pt, 4, UITheme::Colors::NeonYellow, -1, cv::LINE_AA);
    }
}

void AnalyticsPage::drawHeatMapVisualization(cv::Mat& frame, const cv::Rect& rect,
                                            const std::vector<cv::Point2f>& positions,
                                            const std::vector<bool>& success) {
    UITheme::drawCard(frame, rect, UITheme::ComponentState::Normal, UITheme::Colors::MediumBg, 180);
    
    // Title
    cv::putText(frame, "Shot Position Heat Map", cv::Point(rect.x + 20, rect.y + 30),
               UITheme::Fonts::FontFaceBold, UITheme::Fonts::BodySize,
               UITheme::Colors::TextPrimary, UITheme::Fonts::ButtonThickness);
    
    if (positions.empty()) return;
    
    // Draw table outline
    cv::Rect tableRect(rect.x + 40, rect.y + 60, rect.width - 80, rect.height - 80);
    cv::rectangle(frame, tableRect, UITheme::Colors::TableGreen, 2);
    
    // Draw shot positions
    for (size_t i = 0; i < positions.size(); ++i) {
        // Scale position to fit in table rect
        cv::Point2f scaledPos(
            tableRect.x + (positions[i].x / 1280.0f) * tableRect.width,
            tableRect.y + (positions[i].y / 720.0f) * tableRect.height
        );
        
        cv::Scalar color = (i < success.size() && success[i]) ?
                          UITheme::Colors::NeonGreen : UITheme::Colors::NeonRed;
        
        cv::circle(frame, scaledPos, 3, color, -1, cv::LINE_AA);
    }
    
    // Legend
    cv::putText(frame, "Success", cv::Point(rect.x + rect.width - 150, rect.y + 30),
               UITheme::Fonts::FontFace, UITheme::Fonts::SmallSize,
               UITheme::Colors::NeonGreen, 1);
    cv::putText(frame, "Miss", cv::Point(rect.x + rect.width - 70, rect.y + 30),
               UITheme::Fonts::FontFace, UITheme::Fonts::SmallSize,
               UITheme::Colors::NeonRed, 1);
}

void AnalyticsPage::drawBackButton(cv::Mat& frame) {
    cv::Rect backButton(40, frame.rows - 80, 150, 50);
    UITheme::drawButton(frame, "Back", backButton, UITheme::ComponentState::Normal);
    clickableAreas_.push_back(backButton);
}

void AnalyticsPage::drawTabBar(cv::Mat& frame) {
    std::vector<std::string> tabs = {"Overview", "Game History", "Heat Map"};
    cv::Rect tabBarRect(0, 100, frame.cols, 60);
    int activeTab = static_cast<int>(currentView_);
    if (activeTab == static_cast<int>(ViewMode::PlayerDetail)) activeTab = 0;
    UITheme::drawTabBar(frame, tabs, activeTab, tabBarRect);
}

void AnalyticsPage::onMouse(int event, int x, int y, int flags) {
    mousePos_ = cv::Point(x, y);
    
    if (event == cv::EVENT_LBUTTONDOWN) {
        // Check tab bar clicks
        int tabWidth = 1280 / 3;
        if (y >= 100 && y <= 160) {
            int tabIndex = x / tabWidth;
            if (tabIndex == 0) currentView_ = ViewMode::Overview;
            else if (tabIndex == 1) currentView_ = ViewMode::GameHistory;
            else if (tabIndex == 2) currentView_ = ViewMode::HeatMap;
            return;
        }
        
        // Check clickable areas
        for (size_t i = 0; i < clickableAreas_.size(); ++i) {
            if (clickableAreas_[i].contains(mousePos_)) {
                // Last area is always back button
                if (i == clickableAreas_.size() - 1) {
                    result_ = "back";
                    return;
                }
                
                // Other areas depend on current view
                if (currentView_ == ViewMode::Overview && i < playerStats_.size()) {
                    // TODO: Get actual player ID
                    selectedPlayerId_ = i;
                    loadPlayerShots(selectedPlayerId_);
                    currentView_ = ViewMode::PlayerDetail;
                }
                break;
            }
        }
    }
}

bool AnalyticsPage::onKey(int key) {
    if (key == 27) { // ESC
        result_ = "back";
        return true;
    }
    return false;
}

void AnalyticsPage::setSelectedPlayer(int playerId) {
    selectedPlayerId_ = playerId;
    loadPlayerShots(playerId);
    currentView_ = ViewMode::PlayerDetail;
}

void AnalyticsPage::loadPlayerStats() {
    playerStats_.clear();
    auto players = database_.getAllPlayers();
    
    for (const auto& player : players) {
        playerStats_.push_back(calculatePlayerStats(player));
    }
}

void AnalyticsPage::loadRecentGames() {
    // Get sessions for all players - we need to call this for each player
    // or modify to get all sessions. For now, get first player's sessions
    auto players = database_.getAllPlayers();
    if (!players.empty()) {
        recentGames_ = database_.getPlayerSessions(players[0].id);
    }
    
    // Sort by date (most recent first)
    std::sort(recentGames_.begin(), recentGames_.end(),
        [](const GameSession& a, const GameSession& b) {
            return a.startedAt > b.startedAt;
        });
    
    // Keep only recent 20
    if (recentGames_.size() > 20) {
        recentGames_.resize(20);
    }
}

void AnalyticsPage::loadPlayerShots(int playerId) {
    playerShots_ = database_.getPlayerShots(playerId);
}

AnalyticsPage::PlayerStats AnalyticsPage::calculatePlayerStats(const PlayerProfile& player) {
    PlayerStats stats;
    stats.name = player.name;
    stats.gamesPlayed = player.gamesPlayed;
    stats.winRate = player.gamesWon / static_cast<float>(std::max(1, player.gamesPlayed));
    
    // Calculate shot statistics
    auto shots = database_.getPlayerShots(player.id);
    stats.totalShots = shots.size();
    stats.successfulShots = 0;
    stats.fouls = 0;
    
    for (const auto& shot : shots) {
        if (shot.successful) stats.successfulShots++;
        // TODO: Count fouls if shotType includes "foul"
    }
    
    stats.shotSuccessRate = stats.totalShots > 0 ?
        stats.successfulShots / static_cast<float>(stats.totalShots) : 0.0f;
    
    return stats;
}

float AnalyticsPage::calculateTrendDirection(const std::vector<float>& values) {
    if (values.size() < 2) return 0.0f;
    
    // Simple linear regression slope
    float n = static_cast<float>(values.size());
    float sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0;
    
    for (size_t i = 0; i < values.size(); ++i) {
        float x = static_cast<float>(i);
        float y = values[i];
        sumX += x;
        sumY += y;
        sumXY += x * y;
        sumX2 += x * x;
    }
    
    float slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
    return slope;
}

std::vector<float> AnalyticsPage::calculatePlayerWinRateTrend() {
    std::vector<float> winRates;
    
    // Get all game sessions and group by time periods (e.g., by week)
    auto allPlayers = database_.getAllPlayers();
    if (allPlayers.empty()) {
        // Return default data if no players
        return {0.5f, 0.5f, 0.5f, 0.5f, 0.5f};
    }
    
    // For simplicity, calculate win rate for the first player over their recent games
    auto sessions = database_.getPlayerSessions(allPlayers[0].id);
    
    if (sessions.size() < 10) {
        // Not enough data for meaningful trend
        float overallWinRate = 0.5f;
        if (!sessions.empty()) {
            int wins = 0;
            for (const auto& session : sessions) {
                if (session.winnerId == allPlayers[0].id) wins++;
            }
            overallWinRate = static_cast<float>(wins) / sessions.size();
        }
        // Return flat trend
        return std::vector<float>(10, overallWinRate);
    }
    
    // Calculate win rate in chunks of sessions
    size_t chunkSize = std::max(size_t(1), sessions.size() / 10);
    
    for (size_t i = 0; i < sessions.size(); i += chunkSize) {
        size_t end = std::min(i + chunkSize, sessions.size());
        int wins = 0;
        int total = 0;
        
        for (size_t j = i; j < end; ++j) {
            if (sessions[j].winnerId == allPlayers[0].id) wins++;
            total++;
        }
        
        float winRate = total > 0 ? static_cast<float>(wins) / total : 0.0f;
        winRates.push_back(winRate);
    }
    
    // Ensure we have at least some data points
    if (winRates.empty()) {
        return {0.5f, 0.5f, 0.5f, 0.5f, 0.5f};
    }
    
    return winRates;
}

std::pair<std::vector<float>, std::vector<std::string>> AnalyticsPage::calculateShotSuccessByType() {
    std::map<std::string, std::pair<int, int>> shotTypeStats; // type -> {successful, total}
    
    // Get all shots from all players
    auto allShots = getAllPlayerShots();
    
    for (const auto& shot : allShots) {
        if (!shotTypeStats.count(shot.shotType)) {
            shotTypeStats[shot.shotType] = {0, 0};
        }
        
        shotTypeStats[shot.shotType].second++; // total count
        if (shot.successful) {
            shotTypeStats[shot.shotType].first++; // successful count
        }
    }
    
    std::vector<float> successRates;
    std::vector<std::string> shotTypes;
    
    // Convert to vectors for chart display
    for (const auto& [shotType, stats] : shotTypeStats) {
        if (stats.second > 0) { // Only include types with shots
            shotTypes.push_back(shotType);
            successRates.push_back(static_cast<float>(stats.first) / stats.second);
        }
    }
    
    // If no real data, provide default categories
    if (successRates.empty()) {
        successRates = {0.75f, 0.65f, 0.55f, 0.80f};
        shotTypes = {"Basic", "Bank", "Combo", "Cut"};
    }
    
    return {successRates, shotTypes};
}

std::vector<ShotRecord> AnalyticsPage::getAllPlayerShots() {
    std::vector<ShotRecord> allShots;
    
    auto allPlayers = database_.getAllPlayers();
    for (const auto& player : allPlayers) {
        auto playerShots = database_.getPlayerShots(player.id);
        allShots.insert(allShots.end(), playerShots.begin(), playerShots.end());
    }
    
    return allShots;
}
