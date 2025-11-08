#pragma once
#include "../util/Types.hpp"
#include "../db/Database.hpp"
#include "GameState.hpp"
#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <chrono>

namespace pv {

// Forward declarations
class DrillLibrary;

/**
 * @brief Drill execution and management system
 * 
 * Manages drill execution, tracks performance, provides feedback,
 * and maintains drill session statistics for player improvement.
 */
class DrillSystem {
public:
    /**
     * @brief Drill difficulty levels
     */
    enum class Difficulty {
        Beginner = 1,
        Intermediate = 2,
        Advanced = 3,
        Professional = 4,
        Expert = 5
    };

    /**
     * @brief Drill categories for organization
     */
    enum class Category {
        Breaking,
        CutShots,
        BankShots,
        Combinations,
        PositionPlay,
        SpeedControl,
        RailShots,
        RunOut,
        Safety,
        Specialty
    };

    /**
     * @brief Drill execution state
     */
    enum class DrillState {
        NotStarted,
        Setup,
        InProgress,
        Paused,
        Completed,
        Failed
    };

    /**
     * @brief Individual drill definition
     */
    struct Drill {
        int id;
        std::string name;
        std::string description;
        Category category;
        Difficulty difficulty;
        std::string instructions;
        std::vector<Ball> initialSetup;  // Starting ball positions
        std::vector<cv::Point2f> targets;  // Target positions/pockets
        int maxAttempts;
        double timeLimit;  // Seconds, 0 = no limit
        double successThreshold;  // Accuracy threshold (0.0-1.0)
        bool isCustom;  // User-created vs built-in
        
        Drill() : id(0), category(Category::CutShots), difficulty(Difficulty::Beginner),
                  maxAttempts(10), timeLimit(0.0), successThreshold(0.8), isCustom(false) {}
    };

    /**
     * @brief Drill attempt result
     */
    struct DrillAttempt {
        int attemptNumber;
        bool success;
        double accuracy;  // 0.0-1.0
        double timeTaken;
        cv::Point2f shotPosition;
        cv::Point2f targetPosition;
        cv::Point2f actualResult;
        std::string feedback;
        
        DrillAttempt() : attemptNumber(0), success(false), accuracy(0.0), timeTaken(0.0) {}
    };

    /**
     * @brief Active drill session data
     */
    struct DrillSession {
        int sessionId;
        int drillId;
        int playerId;
        std::chrono::steady_clock::time_point startTime;
        DrillState state;
        std::vector<DrillAttempt> attempts;
        int currentAttempt;
        double bestAccuracy;
        double averageAccuracy;
        bool sessionCompleted;
        
        DrillSession() : sessionId(0), drillId(0), playerId(0), state(DrillState::NotStarted),
                        currentAttempt(0), bestAccuracy(0.0), averageAccuracy(0.0), sessionCompleted(false) {}
    };

    /**
     * @brief Drill performance statistics
     */
    struct DrillStats {
        int totalAttempts;
        int successfulAttempts;
        double successRate;
        double averageAccuracy;
        double bestAccuracy;
        double averageTime;
        double bestTime;
        int totalSessions;
        std::chrono::steady_clock::time_point lastPlayed;
        double improvementRate;  // Trend over time
        
        DrillStats() : totalAttempts(0), successfulAttempts(0), successRate(0.0),
                      averageAccuracy(0.0), bestAccuracy(0.0), averageTime(0.0),
                      bestTime(0.0), totalSessions(0), improvementRate(0.0) {}
    };

    /**
     * @brief Construct a new Drill System
     */
    DrillSystem(Database& database, GameState& gameState);
    ~DrillSystem() = default;

    // Drill management
    /**
     * @brief Start a new drill session
     */
    bool startDrill(int drillId, int playerId);
    
    /**
     * @brief End current drill session
     */
    void endDrill();
    
    /**
     * @brief Pause/resume current drill
     */
    void pauseDrill();
    void resumeDrill();
    
    /**
     * @brief Process a shot attempt during drill
     */
    void processShot(const cv::Point2f& shotPosition, const std::vector<Ball>& ballsBefore, 
                    const std::vector<Ball>& ballsAfter);
    
    /**
     * @brief Reset current drill to beginning
     */
    void resetDrill();
    
    /**
     * @brief Skip to next attempt
     */
    void nextAttempt();

    // Drill session queries
    /**
     * @brief Get current drill session info
     */
    const DrillSession& getCurrentSession() const { return currentSession_; }
    
    /**
     * @brief Check if drill is active
     */
    bool isDrillActive() const { return currentSession_.state == DrillState::InProgress; }
    
    /**
     * @brief Check if drill is paused
     */
    bool isDrillPaused() const { return currentSession_.state == DrillState::Paused; }
    
    /**
     * @brief Get current drill definition
     */
    const Drill* getCurrentDrill() const;

    // Statistics and performance
    /**
     * @brief Get drill statistics for a player
     */
    DrillStats getDrillStats(int drillId, int playerId) const;
    
    /**
     * @brief Get overall player drill performance
     */
    std::vector<DrillStats> getPlayerDrillStats(int playerId) const;
    
    /**
     * @brief Get recent drill sessions
     */
    std::vector<DrillSession> getRecentSessions(int playerId, int limit = 10) const;
    
    /**
     * @brief Get improvement trends for specific drill
     */
    std::vector<double> getImprovementTrend(int drillId, int playerId, int sessionCount = 20) const;

    // Drill evaluation and feedback
    /**
     * @brief Calculate shot accuracy for current drill
     */
    double calculateShotAccuracy(const cv::Point2f& target, const cv::Point2f& actual) const;
    
    /**
     * @brief Generate feedback for shot attempt
     */
    std::string generateFeedback(const DrillAttempt& attempt) const;
    
    /**
     * @brief Evaluate if attempt meets success criteria
     */
    bool evaluateSuccess(const DrillAttempt& attempt) const;
    
    /**
     * @brief Get hint for improving performance
     */
    std::string getPerformanceHint(const DrillStats& stats) const;

    // Session management
    /**
     * @brief Save current session to database
     */
    void saveSession();
    
    /**
     * @brief Load previous session
     */
    bool loadSession(int sessionId);
    
    /**
     * @brief Delete drill session
     */
    bool deleteSession(int sessionId);

    // Callback system for UI updates
    /**
     * @brief Set callback for drill state changes
     */
    void setStateChangeCallback(std::function<void(DrillState)> callback) {
        stateChangeCallback_ = callback;
    }
    
    /**
     * @brief Set callback for attempt completion
     */
    void setAttemptCallback(std::function<void(const DrillAttempt&)> callback) {
        attemptCallback_ = callback;
    }
    
    /**
     * @brief Set callback for session completion
     */
    void setSessionCallback(std::function<void(const DrillSession&)> callback) {
        sessionCallback_ = callback;
    }

    // Utility functions
    /**
     * @brief Convert difficulty to string
     */
    static std::string difficultyToString(Difficulty difficulty);
    
    /**
     * @brief Convert category to string
     */
    static std::string categoryToString(Category category);
    
    /**
     * @brief Convert state to string
     */
    static std::string stateToString(DrillState state);

private:
    Database& database_;
    GameState& gameState_;
    DrillSession currentSession_;
    std::unique_ptr<DrillLibrary> drillLibrary_;
    
    // Callbacks for UI updates
    std::function<void(DrillState)> stateChangeCallback_;
    std::function<void(const DrillAttempt&)> attemptCallback_;
    std::function<void(const DrillSession&)> sessionCallback_;
    
    // Internal methods
    void changeState(DrillState newState);
    void updateSessionStats();
    double calculateDistance(const cv::Point2f& p1, const cv::Point2f& p2) const;
    void recordAttempt(const DrillAttempt& attempt);
    bool setupDrillBalls(const Drill& drill);
};

} // namespace pv