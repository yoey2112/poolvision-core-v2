#ifndef PV_AI_LEARNING_AI_LEARNING_SYSTEM_HPP
#define PV_AI_LEARNING_AI_LEARNING_SYSTEM_HPP

#include "DataCollectionEngine.hpp"
#include "ShotAnalysisEngine.hpp"
#include "AdaptiveCoachingEngine.hpp"
#include "PerformanceAnalyticsEngine.hpp"
#include "../../events/EventEngine.hpp"
#include "../../track/Tracker.hpp"
#include "../../detect/classical/BallDetector.hpp"
#include <memory>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>

namespace pv {
namespace ai {
namespace learning {

/**
 * AI Learning System Integration
 * 
 * Integrates all AI Learning components with the existing Pool Vision pipeline:
 * - Coordinates data flow between Agent Groups 1-5 and AI components
 * - Manages CPU resource allocation to avoid GPU pipeline interference
 * - Provides unified API for AI learning features
 * - Handles system lifecycle and configuration
 */
class AILearningSystem {
public:
    // System configuration
    struct SystemConfig {
        // CPU allocation (cores 4-7 for AI processing)
        std::vector<int> cpuCores = {4, 5, 6, 7};
        
        // Data collection settings
        bool enableDataCollection = true;
        int dataCollectionFrequency = 10; // Hz
        std::string databasePath = "ai_learning.db";
        
        // Shot analysis settings
        bool enableShotAnalysis = true;
        bool enableMLPredictions = true;
        int analysisUpdateFrequency = 5; // Hz
        
        // Coaching settings
        bool enableAdaptiveCoaching = true;
        std::string ollamaEndpoint = "http://localhost:11434";
        float coachingIntensity = 0.7f;
        
        // Analytics settings
        bool enablePerformanceAnalytics = true;
        int analyticsDepth = 2; // 1=basic, 2=advanced, 3=comprehensive
        bool enableVisualization = true;
        
        // Performance monitoring
        bool enablePerformanceIsolation = true;
        int maxCpuUsage = 25; // Percentage of total CPU
    };
    
    // System status
    struct SystemStatus {
        bool dataCollectionActive = false;
        bool shotAnalysisActive = false;
        bool adaptiveCoachingActive = false;
        bool performanceAnalyticsActive = false;
        
        // Performance metrics
        double cpuUsage = 0.0;
        int playersTracked = 0;
        int activeModels = 0;
        
        // Quality metrics
        double dataQuality = 1.0;
        double predictionAccuracy = 0.0;
        double coachingEffectiveness = 0.0;
        
        std::chrono::time_point<std::chrono::steady_clock> lastUpdate;
    };
    
    // Integration events for Agent Groups communication
    struct AILearningEvent {
        enum Type {
            ShotAnalyzed,
            CoachingGenerated,
            PerformanceInsight,
            PlayerModelUpdated,
            SystemAlert
        };
        
        Type type;
        int playerId;
        std::string data;
        double confidence;
        std::chrono::time_point<std::chrono::steady_clock> timestamp;
    };

private:
    // Performance isolation manager
    class PerformanceIsolation {
    public:
        explicit PerformanceIsolation(const SystemConfig& config);
        
        void setCpuAffinity(std::thread& thread, const std::vector<int>& cores);
        void monitorCpuUsage();
        void throttleIfNeeded();
        double getCurrentCpuUsage() const { return currentCpuUsage_; }
        
    private:
        std::vector<int> allowedCores_;
        int maxCpuUsage_;
        std::atomic<double> currentCpuUsage_{0.0};
        std::thread monitoringThread_;
        std::atomic<bool> monitoring_{false};
    };
    
    // Data flow coordinator
    class DataFlowCoordinator {
    public:
        DataFlowCoordinator(DataCollectionEngine* dataEngine,
                           ShotAnalysisEngine* analysisEngine,
                           AdaptiveCoachingEngine* coachingEngine,
                           PerformanceAnalyticsEngine* analyticsEngine);
        
        // Route data between components
        void routeShotData(const DataCollectionEngine::ShotOutcomeData& shot);
        void routeBehaviorData(const DataCollectionEngine::PlayerBehaviorData& behavior);
        void routeGameState(const GameState& gameState);
        
        // Coordinate analysis pipeline
        void triggerShotAnalysis(int playerId, const GameState& gameState, 
                               const cv::Point2f& cueBallPos, const std::vector<Ball>& targetBalls);
        void processAnalysisResults(const ShotAnalysisEngine::ShotAnalysisResult& result);
        
    private:
        DataCollectionEngine* dataEngine_;
        ShotAnalysisEngine* analysisEngine_;
        AdaptiveCoachingEngine* coachingEngine_;
        PerformanceAnalyticsEngine* analyticsEngine_;
        
        std::queue<AILearningEvent> eventQueue_;
        std::mutex queueMutex_;
    };

public:
    explicit AILearningSystem(const SystemConfig& config = SystemConfig{});
    ~AILearningSystem();
    
    // System lifecycle
    bool initialize();
    void start();
    void stop();
    void shutdown();
    
    // Integration with Agent Groups
    void connectToTracker(std::shared_ptr<Tracker> tracker);
    void connectToEventEngine(std::shared_ptr<EventEngine> eventEngine);
    void connectToBallDetector(std::shared_ptr<BallDetector> detector);
    
    // Player management
    void addPlayer(int playerId, const std::string& playerName = "");
    void removePlayer(int playerId);
    void updatePlayerPosition(int playerId, const cv::Point2f& position);
    
    // Real-time AI features
    ShotAnalysisEngine::ShotAnalysisResult analyzeShotSituation(
        int playerId, const GameState& gameState, 
        const cv::Point2f& cueBallPos, const std::vector<Ball>& targetBalls);
    
    AdaptiveCoachingEngine::CoachingMessage generateCoaching(
        int playerId, const GameState& gameState,
        const ShotAnalysisEngine::ShotAnalysisResult& analysis);
    
    PerformanceAnalyticsEngine::PerformanceMetrics getPlayerMetrics(int playerId);
    std::vector<PerformanceAnalyticsEngine::PerformanceInsight> getPlayerInsights(int playerId);
    
    // Session management
    void startPlayerSession(int playerId);
    void endPlayerSession(int playerId);
    PerformanceAnalyticsEngine::SessionReport getSessionReport(int playerId);
    
    // Data input from Agent Groups
    void onShotCompleted(int playerId, const cv::Point2f& startPos, const cv::Point2f& endPos, 
                        bool successful, float difficulty = 0.5f);
    void onBallPositionsUpdate(const std::vector<Ball>& balls);
    void onGameStateChange(const GameState& newState);
    void onPlayerBehavior(int playerId, float aimingTime, float confidence);
    
    // Configuration and tuning
    void updateConfig(const SystemConfig& newConfig);
    SystemConfig getConfig() const { return config_; }
    
    // Monitoring and status
    SystemStatus getSystemStatus() const;
    void logSystemReport();
    
    // Performance optimization
    void optimizeForPerformance();
    void enableLowLatencyMode(bool enable);
    void setResourceLimits(double maxCpuPercent, size_t maxMemoryMB);
    
    // Data export and visualization
    cv::Mat generatePlayerPerformanceChart(int playerId, const std::string& chartType = "trend");
    std::string exportPlayerData(int playerId, const std::string& format = "json");
    std::vector<std::string> getSystemInsights();
    
    // Integration events
    void setEventCallback(std::function<void(const AILearningEvent&)> callback);
    
    // Component access (for advanced integration)
    DataCollectionEngine* getDataEngine() const { return dataEngine_.get(); }
    ShotAnalysisEngine* getAnalysisEngine() const { return analysisEngine_.get(); }
    AdaptiveCoachingEngine* getCoachingEngine() const { return coachingEngine_.get(); }
    PerformanceAnalyticsEngine* getAnalyticsEngine() const { return analyticsEngine_.get(); }

private:
    // Configuration
    SystemConfig config_;
    
    // Core AI components
    std::unique_ptr<DataCollectionEngine> dataEngine_;
    std::unique_ptr<ShotAnalysisEngine> analysisEngine_;
    std::unique_ptr<AdaptiveCoachingEngine> coachingEngine_;
    std::unique_ptr<PerformanceAnalyticsEngine> analyticsEngine_;
    
    // Integration infrastructure
    std::unique_ptr<PerformanceIsolation> performanceIsolation_;
    std::unique_ptr<DataFlowCoordinator> dataFlowCoordinator_;
    
    // Agent Groups connections
    std::weak_ptr<Tracker> tracker_;
    std::weak_ptr<EventEngine> eventEngine_;
    std::weak_ptr<BallDetector> ballDetector_;
    
    // System state
    std::atomic<bool> systemActive_{false};
    std::atomic<bool> initialized_{false};
    mutable std::mutex systemMutex_;
    
    // Event handling
    std::function<void(const AILearningEvent&)> eventCallback_;
    std::thread eventProcessingThread_;
    std::condition_variable eventCondition_;
    
    // Performance monitoring
    SystemStatus currentStatus_;
    std::thread statusUpdateThread_;
    std::atomic<bool> statusMonitoring_{false};
    
    // Background processing
    void eventProcessingLoop();
    void statusUpdateLoop();
    void updateSystemStatus();
    
    // Initialization helpers
    bool initializeDataEngine();
    bool initializeShotAnalysis();
    bool initializeAdaptiveCoaching();
    bool initializePerformanceAnalytics();
    bool setupPerformanceIsolation();
    
    // Data conversion helpers
    DataCollectionEngine::ShotOutcomeData createShotData(
        int playerId, const cv::Point2f& startPos, const cv::Point2f& endPos,
        bool successful, float difficulty);
    DataCollectionEngine::PlayerBehaviorData createBehaviorData(
        int playerId, float aimingTime, float confidence);
    
    // Resource management
    void allocateResources();
    void deallocateResources();
    void checkResourceUsage();
    
    // Error handling
    void handleComponentError(const std::string& component, const std::string& error);
    void attemptRecovery(const std::string& component);
};

// Factory for creating AI Learning System
class AILearningSystemFactory {
public:
    static std::unique_ptr<AILearningSystem> createDefault();
    static std::unique_ptr<AILearningSystem> createOptimized(bool lowLatency = false);
    static std::unique_ptr<AILearningSystem> createWithConfig(const AILearningSystem::SystemConfig& config);
};

// Global AI Learning System instance for easy access
class GlobalAILearning {
public:
    static AILearningSystem& getInstance();
    static void initialize(std::unique_ptr<AILearningSystem> system);
    static void shutdown();
    
private:
    static std::unique_ptr<AILearningSystem> instance_;
    static std::mutex instanceMutex_;
};

} // namespace learning
} // namespace ai
} // namespace pv

#endif // PV_AI_LEARNING_AI_LEARNING_SYSTEM_HPP