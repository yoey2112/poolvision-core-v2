#pragma once

#include "OllamaClient.hpp"
#include "CoachingPrompts.hpp"
#include "../game/modern/ShotSegmentation.hpp"
#include "../db/Database.hpp"
#include <memory>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <thread>
#include <functional>

namespace pv {
namespace ai {

/**
 * Main coaching engine that coordinates LLM analysis with pool game events
 * Provides non-blocking coaching advice using async processing
 */
class CoachingEngine {
public:
    struct CoachingConfig {
        OllamaClient::OllamaConfig ollamaConfig;
        CoachingPrompts::CoachingPersonality personality = CoachingPrompts::CoachingPersonality::Supportive;
        
        // Processing settings
        int maxConcurrentRequests = 3;
        int requestQueueSize = 50;
        float minTimeBetweenCoaching = 5.0f;  // seconds
        
        // Feature flags
        bool enableRealTimeCoaching = true;
        bool enableSessionAnalysis = true;
        bool enableDrillRecommendations = true;
        bool enablePerformanceTracking = true;
        
        // Performance tuning
        int maxHistoryShots = 10;
        float coachingTriggerThreshold = 0.6f;  // Success rate threshold for coaching
    };

    struct CoachingRequest {
        uint64_t requestId;
        CoachingPrompts::CoachingType type;
        CoachingPrompts::CoachingContext context;
        std::chrono::steady_clock::time_point timestamp;
        int priority = 0;  // Higher = more urgent
        
        CoachingRequest(uint64_t id, CoachingPrompts::CoachingType t, 
                       const CoachingPrompts::CoachingContext& ctx)
            : requestId(id), type(t), context(ctx), 
              timestamp(std::chrono::steady_clock::now()) {}
    };

    struct CoachingResponse {
        uint64_t requestId;
        CoachingPrompts::CoachingType type;
        std::string advice;
        bool success;
        float responseTime;
        std::chrono::steady_clock::time_point timestamp;
        
        CoachingResponse() : requestId(0), success(false), responseTime(0.0f),
                           timestamp(std::chrono::steady_clock::now()) {}
    };

    using CoachingCallback = std::function<void(const CoachingResponse&)>;
    using PerformanceCallback = std::function<void(const std::string&, float)>;

private:
    CoachingConfig config_;
    std::unique_ptr<OllamaClient> ollamaClient_;
    std::unique_ptr<CoachingPrompts> promptGenerator_;
    
    // Async processing
    std::queue<CoachingRequest> requestQueue_;
    mutable std::mutex queueMutex_;
    std::condition_variable queueCondition_;
    std::atomic<bool> processingActive_;
    std::vector<std::thread> workerThreads_;
    
    // Callbacks
    CoachingCallback responseCallback_;
    PerformanceCallback performanceCallback_;
    
    // Performance tracking
    std::atomic<uint64_t> nextRequestId_;
    std::chrono::steady_clock::time_point lastCoachingTime_;
    mutable std::mutex performanceMutex_;
    
    // Session tracking
    CoachingPrompts::CoachingContext::SessionInfo currentSession_;
    std::vector<pv::modern::ShotSegmentation::ShotEvent> shotHistory_;
    mutable std::mutex sessionMutex_;

public:
    explicit CoachingEngine(const CoachingConfig& config = CoachingConfig{});
    ~CoachingEngine();

    // Non-copyable but movable
    CoachingEngine(const CoachingEngine&) = delete;
    CoachingEngine& operator=(const CoachingEngine&) = delete;
    CoachingEngine(CoachingEngine&& other) noexcept = delete;
    CoachingEngine& operator=(CoachingEngine&& other) noexcept = delete;

    // Core functionality
    bool initialize();
    void shutdown();
    
    // Async coaching requests
    uint64_t requestCoaching(CoachingPrompts::CoachingType type, 
                           const CoachingPrompts::CoachingContext& context,
                           int priority = 0);
    
    uint64_t requestShotAnalysis(const pv::modern::ShotSegmentation::ShotEvent& shot,
                               const GameState& gameState,
                               const CoachingPrompts::CoachingContext::PlayerInfo& player);
    
    uint64_t requestDrillRecommendation(const CoachingPrompts::CoachingContext::PlayerInfo& player,
                                      const std::vector<pv::modern::ShotSegmentation::ShotEvent>& recentShots);
    
    uint64_t requestSessionReview();
    
    // Synchronous methods (for testing/immediate needs)
    CoachingResponse getImmediateCoaching(CoachingPrompts::CoachingType type,
                                        const CoachingPrompts::CoachingContext& context);
    
    // Session management
    void startSession(const std::string& sessionType = "casual");
    void endSession();
    void addShotToHistory(const pv::modern::ShotSegmentation::ShotEvent& shot);
    void updatePlayerInfo(const CoachingPrompts::CoachingContext::PlayerInfo& player);
    
    // Configuration
    void setConfig(const CoachingConfig& config);
    const CoachingConfig& getConfig() const { return config_; }
    void setResponseCallback(CoachingCallback callback) { responseCallback_ = callback; }
    void setPerformanceCallback(PerformanceCallback callback) { performanceCallback_ = callback; }
    void setPersonality(CoachingPrompts::CoachingPersonality personality);
    
    // Status and monitoring
    bool isAvailable() const;
    bool isOllamaConnected() const;
    size_t getQueueSize() const;
    float getAverageResponseTime() const;
    void getPerformanceStats(int& totalRequests, float& avgTime, int& queueSize) const;
    
    // Auto-coaching triggers
    void enableAutoCoaching(bool enable) { config_.enableRealTimeCoaching = enable; }
    bool shouldTriggerCoaching(const pv::modern::ShotSegmentation::ShotEvent& shot) const;
    void processAutoCoaching(const pv::modern::ShotSegmentation::ShotEvent& shot,
                           const GameState& gameState,
                           const CoachingPrompts::CoachingContext::PlayerInfo& player);

private:
    void workerThread();
    void processRequest(const CoachingRequest& request);
    CoachingResponse generateCoachingResponse(const CoachingRequest& request);
    
    void updatePerformanceMetrics(const std::string& operation, float duration);
    void cleanupOldRequests();
    bool isRateLimited() const;
    
    CoachingPrompts::CoachingContext buildContext(
        const pv::modern::ShotSegmentation::ShotEvent& shot,
        const GameState& gameState,
        const CoachingPrompts::CoachingContext::PlayerInfo& player) const;
    
    void logRequest(const CoachingRequest& request);
    void logResponse(const CoachingResponse& response);
};

/**
 * Coaching engine factory for easy configuration
 */
class CoachingEngineFactory {
public:
    static std::unique_ptr<CoachingEngine> createDefault();
    static std::unique_ptr<CoachingEngine> createForTesting();
    static std::unique_ptr<CoachingEngine> createHighPerformance();
    static std::unique_ptr<CoachingEngine> createOfflineMode();
    
    static CoachingEngine::CoachingConfig getDefaultConfig();
    static CoachingEngine::CoachingConfig getTestingConfig();
    static CoachingEngine::CoachingConfig getHighPerformanceConfig();
    static CoachingEngine::CoachingConfig getOfflineConfig();
};

} // namespace ai
} // namespace pv