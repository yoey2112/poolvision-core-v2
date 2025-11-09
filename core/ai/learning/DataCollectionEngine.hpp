#pragma once

#include <memory>
#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>
#include <cmath>
#include <opencv2/opencv.hpp>

#include "../CoachingEngine.hpp"
#include "../../track/modern/ByteTrackMOT.hpp"
#include "../../game/modern/ShotSegmentation.hpp"
#include "../../performance/ProcessingIsolation.hpp"
#include "../../util/Types.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace pv {
namespace ai {
namespace learning {

/**
 * CPU-optimized AI Data Collection Engine
 * 
 * Collects comprehensive game data for AI learning without interfering
 * with the GPU inference pipeline. Designed for background processing
 * with automatic performance scaling.
 */
class DataCollectionEngine {
public:
    struct PlayerBehaviorData {
        int playerId;
        float aimingTime;                    // Time spent aiming (seconds)
        int aimingAdjustments;               // Number of position changes
        bool hesitationDetected;             // Uncertainty indicators
        float confidenceLevel;               // Estimated confidence (0-1)
        cv::Point2f preferredShotAngle;      // Player's comfortable angles
        std::vector<cv::Point2f> aimingPattern; // Mouse/aiming movement pattern
        
        // Session context
        std::string sessionType;             // "practice", "match", "tournament"
        int sessionDuration;                 // Minutes played
        float fatigueLevel;                  // Estimated fatigue (0-1)
        
        PlayerBehaviorData() : playerId(-1), aimingTime(0), aimingAdjustments(0),
                             hesitationDetected(false), confidenceLevel(0.5f),
                             sessionDuration(0), fatigueLevel(0) {}
    };
    
    struct ShotOutcomeData {
        int playerId;
        cv::Point2f shotPosition;            // Cue ball start position
        cv::Point2f targetPosition;         // Intended target ball
        cv::Point2f actualOutcome;           // Actual result position
        bool successful;                     // Shot success flag
        float shotDifficulty;                // Calculated difficulty (0-1)
        float shotSpeed;                     // Ball velocity
        float shotAngle;                     // Shot angle in radians
        
        // Game context
        pv::modern::ShotSegmentation::ShotEvent shotEvent; // Complete shot data
        GameState gameStateBefore;           // Game state before shot
        GameState gameStateAfter;            // Game state after shot
        std::vector<Ball> ballPositions;     // Table state
        
        // Shot classification
        enum ShotType {
            Straight,
            Cut,
            Bank,
            Combo,
            Safety,
            Break,
            Masse,
            Jump
        } shotType;
        
        std::chrono::system_clock::time_point timestamp;
        
        ShotOutcomeData() : playerId(-1), successful(false), shotDifficulty(0),
                          shotSpeed(0), shotAngle(0), shotType(Straight) {}
    };
    
    struct LearningDataPacket {
        ShotOutcomeData shotData;
        PlayerBehaviorData behaviorData;
        std::vector<cv::Point2f> ballTrajectory;    // Complete ball path
        float environmentalFactors;                  // Lighting, table condition
        bool pressureSituation;                     // High-stakes shot
        std::string contextTags;                    // Additional metadata
        
        LearningDataPacket() : environmentalFactors(1.0f), pressureSituation(false) {}
    };

private:
    // CPU thread management
    std::thread dataProcessingThread_;
    std::atomic<bool> processingActive_{false};
    std::atomic<bool> pauseRequested_{false};
    
    // Performance isolation
    ProcessingIsolation* isolation_;
    std::vector<int> dedicatedCpuCores_;
    int threadPriority_;
    
    // Data collection queues
    std::queue<LearningDataPacket> pendingData_;
    std::queue<ShotOutcomeData> shotQueue_;
    std::queue<PlayerBehaviorData> behaviorQueue_;
    std::mutex dataQueueMutex_;
    std::condition_variable dataCondition_;
    
    static constexpr size_t MAX_QUEUE_SIZE = 1000;
    
    // Background learning storage
    class LearningDatabase {
    private:
        std::string databasePath_;
        mutable std::mutex dbMutex_;
        
    public:
        explicit LearningDatabase(const std::string& dbPath);
        
        void storeShotData(const ShotOutcomeData& data);
        void storeBehaviorData(const PlayerBehaviorData& data);
        void storeSessionData(const LearningDataPacket& packet);
        
        std::vector<ShotOutcomeData> getPlayerShots(int playerId, int limit = 100);
        std::vector<PlayerBehaviorData> getPlayerBehavior(int playerId, int limit = 50);
        std::vector<LearningDataPacket> getSessionData(const std::string& sessionType);
        
        void cleanupOldData(int daysToKeep = 90);
        size_t getDatabaseSize() const;
    };
    
    std::unique_ptr<LearningDatabase> database_;
    
    // Performance monitoring
    struct CollectionMetrics {
        std::atomic<uint64_t> shotsRecorded{0};
        std::atomic<uint64_t> behaviorEventsRecorded{0};
        std::atomic<double> avgProcessingTime{0.0};
        std::atomic<size_t> queueLength{0};
        std::atomic<bool> performanceImpacted{false};
        
        std::chrono::steady_clock::time_point lastMetricsUpdate;
        
        CollectionMetrics() {
            lastMetricsUpdate = std::chrono::steady_clock::now();
        }
    };
    
    CollectionMetrics metrics_;
    std::mutex metricsMutex_;

public:
    explicit DataCollectionEngine(ProcessingIsolation* isolation);
    ~DataCollectionEngine();
    
    // Lifecycle management
    bool initializeCollection(const std::string& databasePath);
    void startCollection();
    void stopCollection();
    bool isCollectionActive() const { return processingActive_.load(); }
    
    // Data input interface (called from Agent Groups 1-5)
    void recordShotOutcome(const ShotOutcomeData& shotData);
    void recordPlayerBehavior(const PlayerBehaviorData& behaviorData);
    void recordGameSession(const LearningDataPacket& sessionData);
    
    // Performance management
    void pauseCollection();     // Called when GPU needs maximum performance
    void resumeCollection();    // Called when GPU load is low
    bool isPerformanceImpacted() const { return metrics_.performanceImpacted.load(); }
    
    // Data access for learning systems
    std::vector<ShotOutcomeData> getPlayerShotHistory(int playerId, int shotCount = 100);
    std::vector<PlayerBehaviorData> getPlayerBehaviorHistory(int playerId);
    std::vector<LearningDataPacket> getTrainingDataset(const std::string& filterCriteria = "");
    
    // Analytics support
    struct PlayerStatistics {
        int totalShots;
        float overallSuccessRate;
        float averageAimingTime;
        std::map<ShotOutcomeData::ShotType, float> shotTypeSuccessRates;
        std::vector<float> performanceTrend;    // Last 30 days
        float improvementRate;                  // Shots per week improvement
        
        PlayerStatistics() : totalShots(0), overallSuccessRate(0), 
                           averageAimingTime(0), improvementRate(0) {}
    };
    
    PlayerStatistics calculatePlayerStatistics(int playerId);
    
    // Performance monitoring
    CollectionMetrics getMetrics() const;
    void logPerformanceReport();
    
    // Configuration
    void setCpuCores(const std::vector<int>& cores);
    void setThreadPriority(int priority);
    void setMaxQueueSize(size_t maxSize);

private:
    // Background processing loop
    void processingLoop();
    void processDataBatch();
    void waitForGpuIdle();
    bool isGpuBusy();
    
    // Data processing
    void analyzeShotPattern(const ShotOutcomeData& shot);
    void updateBehaviorModel(const PlayerBehaviorData& behavior);
    void extractLearningFeatures(LearningDataPacket& packet);
    
    // Performance optimization
    void setCpuAffinity(const std::vector<int>& cores);
    void adjustProcessingLoad();
    void updatePerformanceMetrics();
    
    // Utility functions
    float calculateShotDifficulty(const ShotOutcomeData& shot);
    ShotOutcomeData::ShotType classifyShotType(const ShotOutcomeData& shot);
    float estimatePlayerConfidence(const PlayerBehaviorData& behavior);
};

/**
 * Factory for creating optimized data collection engines
 */
class DataCollectionFactory {
public:
    static std::unique_ptr<DataCollectionEngine> createOptimized(ProcessingIsolation* isolation);
    static std::unique_ptr<DataCollectionEngine> createLowImpact(ProcessingIsolation* isolation);
    static std::unique_ptr<DataCollectionEngine> createHighThroughput(ProcessingIsolation* isolation);
};

} // namespace learning
} // namespace ai
} // namespace pv