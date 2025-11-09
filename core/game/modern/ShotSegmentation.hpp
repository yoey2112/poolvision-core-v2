#pragma once

#include <vector>
#include <deque>
#include <chrono>
#include <memory>
#include <map>
#include <atomic>
#include "../../util/Types.hpp"
#include "../../track/modern/ByteTrackMOT.hpp"
#include "../GameState.hpp"

namespace pv {
namespace modern {

/**
 * Advanced shot segmentation system that uses high-precision tracking data
 * to detect shot boundaries and analyze ball motion physics
 */
class ShotSegmentation {
public:
    struct MotionAnalysis {
        cv::Point2f velocity;
        float speed;
        float acceleration;
        float kinematicEnergy;
        bool isMoving;
        bool isDecelerating;
        bool isStationary;
    };
    
    struct ShotEvent {
        uint64_t startTimestamp;
        uint64_t endTimestamp;
        double duration;
        
        // Motion analysis
        cv::Point2f cueBallStartPos;
        cv::Point2f cueBallEndPos;
        float maxSpeed;
        float totalDistance;
        
        // Game context
        std::vector<int> ballsInMotion;
        std::vector<int> ballsPotted;
        std::vector<int> ballsContacted;  // Balls that were contacted during the shot
        bool isLegalShot;
        bool hasCollisions;
        
        // Shot classification
        enum Type {
            Unknown,
            Break,
            Standard,
            SafetyShot,
            MassShot,
            JumpShot,
            BankShot
        } shotType;
        
        ShotEvent() : startTimestamp(0), endTimestamp(0), duration(0.0),
                     maxSpeed(0.0f), totalDistance(0.0f), isLegalShot(false),
                     hasCollisions(false), shotType(Unknown) {}
    };
    
    struct BallState {
        int ballId;
        cv::Point2f position;
        cv::Point2f velocity;
        double timestamp;
        bool isMoving;
        
        std::deque<cv::Point2f> positionHistory;
        std::deque<double> timestampHistory;
        std::deque<float> speedHistory;
        
        BallState(int id = -1) : ballId(id), isMoving(false) {}
        
        void updatePosition(const cv::Point2f& pos, double ts);
        MotionAnalysis getMotionAnalysis() const;
        float getCurrentSpeed() const;
        bool hasSignificantMotion() const;
    };
    
    // Configuration
    struct Config {
        float motionThreshold = 5.0f;        // Minimum speed to consider ball moving (pixels/second)
        float stationaryThreshold = 2.0f;    // Speed below which ball is considered stationary
        double shotTimeoutMs = 30000.0;      // Maximum shot duration (30 seconds)
        double stationaryTimeMs = 3000.0;    // Time all balls must be stationary to end shot
        int historySize = 30;                // Number of tracking frames to keep in history
        float collisionDistanceThreshold = 60.0f; // Distance threshold for collision detection
    };

private:
    Config config_;
    
    // State management
    bool shotInProgress_;
    uint64_t shotStartTime_;
    uint64_t lastMotionTime_;
    
    std::map<int, BallState> ballStates_;
    std::deque<ShotEvent> recentShots_;
    ShotEvent currentShot_;
    
    // Motion detection
    int movingBallCount_;
    bool cueBallInMotion_;
    
    // Performance tracking
    uint64_t framesProcessed_;
    double avgProcessingTime_;

public:
    explicit ShotSegmentation(const Config& config = Config{});
    ~ShotSegmentation() = default;
    
    // Main processing interface
    bool processTracks(const std::vector<Track>& tracks, double timestamp);
    
    // Shot event queries
    bool isShotInProgress() const { return shotInProgress_; }
    const ShotEvent& getCurrentShot() const { return currentShot_; }
    std::vector<ShotEvent> getRecentShots(int count = 10) const;
    
    // Motion analysis
    MotionAnalysis getBallMotion(int ballId) const;
    std::vector<int> getMovingBalls() const;
    bool isTableStationary() const;
    
    // Configuration
    void updateConfig(const Config& config) { config_ = config; }
    const Config& getConfig() const { return config_; }
    
    // Performance monitoring
    uint64_t getFramesProcessed() const { return framesProcessed_; }
    double getAvgProcessingTime() const { return avgProcessingTime_; }

private:
    // Shot detection logic
    void startShot(double timestamp);
    void endShot(double timestamp);
    void updateCurrentShot(double timestamp);
    
    // Motion analysis
    void updateBallStates(const std::vector<Track>& tracks, double timestamp);
    void detectCollisions(double timestamp);
    void classifyShot();
    
    // Utility functions
    uint64_t getCurrentTimeMs() const;
    bool areAllBallsStationary(double stationaryDurationMs = 0.0) const;
    cv::Point2f getCueBallPosition() const;
    
    void updateMetrics(double processingTime);
};

/**
 * Enhanced pool rules engine that integrates with shot segmentation
 * for accurate rule validation and game state management
 */
class PoolRulesEngine {
public:
    enum class GameType {
        EightBall,
        NineBall,
        TenBall,
        StraightPool
    };
    
    enum class FoulType {
        None,
        Scratch,                 // Cue ball potted
        NoBallHit,              // Cue ball didn't hit any ball
        WrongBallFirst,         // Hit wrong ball first
        NoCushionAfterContact,  // No ball hit cushion after contact
        BallOffTable,           // Ball jumped off table
        DoubleFoul,             // Multiple fouls in one shot
        IllegalBreak,           // Invalid break shot
        CueStickFoul,           // Touching balls with cue stick
        TimeViolation           // Shot clock violation
    };
    
    struct RuleValidationResult {
        bool isLegalShot;
        FoulType foulType;
        std::string foulDescription;
        std::vector<int> legalTargets;
        std::vector<int> ballsContacted;
        std::vector<int> ballsPotted;
        bool requiresCushionContact;
        bool hadCushionContact;
        
        RuleValidationResult() : isLegalShot(true), foulType(FoulType::None),
                               requiresCushionContact(false), hadCushionContact(false) {}
    };

private:
    GameType gameType_;
    ShotSegmentation* shotSegmentation_;
    
    // Game state
    std::map<int, bool> ballsOnTable_;
    std::map<int, cv::Point2f> ballPositions_;
    bool isBreakShot_;
    int targetBall_;
    
    // Physics analysis
    struct CollisionEvent {
        double timestamp;
        int ball1Id;
        int ball2Id;
        cv::Point2f contactPoint;
        float impactSpeed;
    };
    
    std::vector<CollisionEvent> shotCollisions_;

public:
    explicit PoolRulesEngine(GameType type, ShotSegmentation* segmentation);
    ~PoolRulesEngine() = default;
    
    // Main rule validation
    RuleValidationResult validateShot(const ShotSegmentation::ShotEvent& shot);
    
    // Game state management
    void updateGameState(const std::vector<Track>& tracks);
    void setBreakShot(bool isBreak) { isBreakShot_ = isBreak; }
    void setTargetBall(int ballId) { targetBall_ = ballId; }
    
    // Rule queries
    std::vector<int> getLegalTargets() const;
    bool isLegalTarget(int ballId) const;
    std::string getGameStateDescription() const;
    
    // Configuration
    void setGameType(GameType type) { gameType_ = type; }
    GameType getGameType() const { return gameType_; }

private:
    // Rule validation logic
    bool validateEightBallRules(const ShotSegmentation::ShotEvent& shot, RuleValidationResult& result);
    bool validateNineBallRules(const ShotSegmentation::ShotEvent& shot, RuleValidationResult& result);
    
    // Physics analysis
    void analyzeCollisions(const ShotSegmentation::ShotEvent& shot);
    bool detectBallContact(int ball1Id, int ball2Id, double timestamp);
    bool detectCushionContact(int ballId, double timestamp);
    
    // Utility functions
    std::vector<int> getEightBallTargets() const;
    std::vector<int> getNineBallTargets() const;
    bool isCueBall(int ballId) const { return ballId == 0; }
    bool isEightBall(int ballId) const { return ballId == 8; }
};

/**
 * Integrated game logic manager that coordinates shot segmentation
 * and rules validation with the existing GameState system
 */
class GameLogicManager {
public:
    struct Config {
        ShotSegmentation::Config shotConfig;
        PoolRulesEngine::GameType gameType;
        bool enableAdvancedPhysics = true;
        bool enableRealTimeValidation = true;
    };

private:
    Config config_;
    std::unique_ptr<ShotSegmentation> shotSegmentation_;
    std::unique_ptr<PoolRulesEngine> rulesEngine_;
    
    // Integration with legacy system
    GameState* legacyGameState_;
    
    // Performance tracking
    std::atomic<uint64_t> shotsProcessed_{0};
    std::atomic<double> avgValidationTime_{0.0};

public:
    explicit GameLogicManager(GameState* legacyGame, const Config& config = Config{});
    ~GameLogicManager() = default;
    
    // Main processing interface
    void processTracks(const std::vector<Track>& tracks, double timestamp);
    
    // Shot and rule queries
    bool isShotInProgress() const;
    ShotSegmentation::ShotEvent getCurrentShot() const;
    PoolRulesEngine::RuleValidationResult getLastShotValidation() const;
    
    // Enhanced game state
    std::vector<int> getLegalTargets() const;
    std::string getAdvancedGameState() const;
    std::vector<ShotSegmentation::ShotEvent> getShotHistory(int count = 10) const;
    
    // Performance monitoring
    uint64_t getShotsProcessed() const { return shotsProcessed_.load(); }
    double getAvgValidationTime() const { return avgValidationTime_.load(); }

private:
    void updateMetrics(double validationTime);
    void synchronizeWithLegacyGameState();
};

} // namespace modern
} // namespace pv