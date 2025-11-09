#pragma once

#include <opencv2/opencv.hpp>
#include <memory>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <vector>

#include "../detect/modern/TensorRtBallDetector.hpp"
#include "../track/modern/ByteTrackMOT.hpp"
#include "../game/modern/ShotSegmentation.hpp"
#include "../ui/modern/SeparatedUIRenderer.hpp"
#include "../performance/ProcessingIsolation.hpp"

#ifdef USE_OLLAMA
#include "../ai/CoachingEngine.hpp"
#endif

namespace pv {
namespace modern {

/**
 * ModernPipelineIntegrator - Agent Group 5: Complete System Integration
 * 
 * Coordinates all Agent Groups (1-4) with the separated UI renderer
 * to create the complete modern Pool Vision pipeline:
 * 
 * Agent Group 1: GPU inference (detection)
 * Agent Group 2: CPU tracking (ByteTrack)
 * Agent Group 3: Game logic (shot segmentation, rules)
 * Agent Group 4: LLM coaching (Ollama integration)
 * Agent Group 5: UI integration (this component)
 * 
 * Features:
 * - Lock-free inter-agent communication
 * - Thread coordination and CPU affinity management
 * - Performance monitoring across entire pipeline
 * - Graceful degradation and error handling
 * - Configuration management for all components
 */
class ModernPipelineIntegrator {
public:
    struct PipelineConfig {
        // Component configurations
        TensorRtBallDetector::ModernPipelineConfig detectionConfig;
        pv::modern::ByteTrackMOT::Config trackingConfig;
        pv::modern::ShotSegmentation::Config gameLogicConfig;
        pv::modern::SeparatedUIRenderer::UIRenderConfig uiConfig;
        
#ifdef USE_OLLAMA
        pv::ai::CoachingEngine::CoachingConfig coachingConfig;
        bool enableCoaching = false;
#endif
        
        // Pipeline coordination
        int gpuDeviceId = 0;
        std::vector<int> cpuCoresForTracking = {0, 1};
        std::vector<int> cpuCoresForGameLogic = {2};
        std::vector<int> cpuCoresForUI = {3};
        
#ifdef USE_OLLAMA
        std::vector<int> cpuCoresForLLM = {4, 5, 6, 7};
#endif
        
        // Performance targets and thresholds
        float targetInferenceFPS = 200.0f;
        float targetTrackingFPS = 300.0f;
        float targetUIFPS = 60.0f;
        float maxLatencyMs = 50.0f;
        
        // Queue sizes for inter-agent communication
        size_t detectionQueueSize = 10;
        size_t trackingQueueSize = 10;
        size_t gameLogicQueueSize = 5;
        size_t uiQueueSize = 3;
    };
    
    struct PipelineData {
        // Raw frame from video source
        cv::Mat originalFrame;
        uint64_t frameTimestamp;
        
        // Agent Group 1 outputs (GPU detection)
        std::vector<Ball> detections;
        float inferenceTime;
        
        // Agent Group 2 outputs (CPU tracking)
        std::vector<pv::modern::ByteTrackMOT::Track> tracks;
        float trackingTime;
        
        // Agent Group 3 outputs (game logic)
        pv::modern::ShotSegmentation::ShotEvent shotEvent;
        GameState gameState;
        float gameLogicTime;
        
#ifdef USE_OLLAMA
        // Agent Group 4 outputs (LLM coaching)
        std::string latestCoachingAdvice;
        bool hasNewCoaching = false;
        float coachingTime;
#endif
        
        // Processing metadata
        bool hasValidDetections = false;
        bool hasValidTracks = false;
        bool hasValidGameLogic = false;
        
        PipelineData() = default;
    };

private:
    PipelineConfig config_;
    
    // Agent Group components
    std::unique_ptr<TensorRtBallDetector> detector_;
    std::unique_ptr<pv::modern::ByteTrackMOT> tracker_;
    std::unique_ptr<pv::modern::ShotSegmentation> shotSegmenter_;
    std::unique_ptr<pv::modern::SeparatedUIRenderer> uiRenderer_;
    
#ifdef USE_OLLAMA
    std::unique_ptr<pv::ai::CoachingEngine> coachingEngine_;
#endif
    
    std::unique_ptr<ProcessingIsolation> processingIsolation_;
    
    // Pipeline threads (one for each agent group)
    std::thread detectionThread_;     // Agent Group 1
    std::thread trackingThread_;      // Agent Group 2
    std::thread gameLogicThread_;     // Agent Group 3
    std::thread coordinatorThread_;   // Main coordination
    
    std::atomic<bool> pipelineActive_{false};
    std::atomic<bool> shutdownRequested_{false};
    
    // Inter-agent communication queues
    struct DetectionResult {
        cv::Mat frame;
        std::vector<Ball> detections;
        uint64_t timestamp;
        float inferenceTime;
    };
    
    struct TrackingResult {
        std::vector<pv::modern::ByteTrackMOT::Track> tracks;
        uint64_t timestamp;
        float trackingTime;
        bool hasValidTracks;
    };
    
    struct GameLogicResult {
        pv::modern::ShotSegmentation::ShotEvent shotEvent;
        GameState gameState;
        uint64_t timestamp;
        float gameLogicTime;
        bool hasValidGameLogic;
    };
    
    // Lock-free queues for high-performance communication
    std::queue<DetectionResult> detectionQueue_;
    std::queue<TrackingResult> trackingQueue_;
    std::queue<GameLogicResult> gameLogicQueue_;
    std::queue<PipelineData> uiQueue_;
    
    // Mutexes for queue synchronization
    std::mutex detectionQueueMutex_;
    std::mutex trackingQueueMutex_;
    std::mutex gameLogicQueueMutex_;
    std::mutex uiQueueMutex_;
    
    // Condition variables for thread coordination
    std::condition_variable detectionCondition_;
    std::condition_variable trackingCondition_;
    std::condition_variable gameLogicCondition_;
    
    // Performance monitoring
    struct PipelineMetrics {
        float detectionFPS = 0.0f;
        float trackingFPS = 0.0f;
        float gameLogicFPS = 0.0f;
        float uiFPS = 0.0f;
        float overallLatency = 0.0f;
        size_t droppedFrames = 0;
        size_t totalFramesProcessed = 0;
        
        // Queue lengths for monitoring
        size_t detectionQueueLength = 0;
        size_t trackingQueueLength = 0;
        size_t gameLogicQueueLength = 0;
        size_t uiQueueLength = 0;
        
        std::chrono::steady_clock::time_point lastMetricsUpdate;
        
        PipelineMetrics() {
            lastMetricsUpdate = std::chrono::steady_clock::now();
        }
    };
    
    PipelineMetrics metrics_;
    std::mutex metricsMutex_;

public:
    ModernPipelineIntegrator(const PipelineConfig& config = PipelineConfig{});
    ~ModernPipelineIntegrator();
    
    // Pipeline lifecycle management
    bool initializePipeline();
    bool startPipeline();
    void stopPipeline();
    bool isPipelineActive() const { return pipelineActive_.load(); }
    
    // Frame input (main entry point)
    bool processFrame(const cv::Mat& frame, uint64_t timestamp);
    
    // Results access
    std::vector<pv::modern::ByteTrackMOT::Track> getCurrentTracks();
    GameState getCurrentGameState();
    pv::modern::ShotSegmentation::ShotEvent getCurrentShotEvent();
    
    // UI frame access
    cv::Mat getCurrentUIFrame();
    cv::Mat getCurrentBirdsEyeView();
    cv::Mat getSideBySideView();
    
#ifdef USE_OLLAMA
    std::optional<pv::ai::CoachingEngine::CoachingResponse> getLatestCoaching();
    void requestCoaching(pv::ai::CoachingPrompts::CoachingType type);
#endif
    
    // Performance monitoring
    PipelineMetrics getPerformanceMetrics();
    void logPerformanceReport();
    bool isPerformanceWithinTargets();
    
    // Configuration management
    void updateDetectionConfig(const TensorRtBallDetector::ModernPipelineConfig& config);
    void updateTrackingConfig(const pv::modern::ByteTrackMOT::Config& config);
    void updateGameLogicConfig(const pv::modern::ShotSegmentation::Config& config);
    void updateUIConfig(const pv::modern::SeparatedUIRenderer::UIRenderConfig& config);
    
#ifdef USE_OLLAMA
    void updateCoachingConfig(const pv::ai::CoachingEngine::CoachingConfig& config);
    void enableCoaching(bool enable);
#endif
    
    // Error handling and recovery
    void handleComponentFailure(const std::string& componentName);
    bool isComponentHealthy(const std::string& componentName);
    void restartComponent(const std::string& componentName);

private:
    // Agent Group thread functions
    void detectionLoop();          // Agent Group 1: GPU inference
    void trackingLoop();           // Agent Group 2: CPU tracking  
    void gameLogicLoop();          // Agent Group 3: Game logic
    void coordinatorLoop();        // Main coordination and UI
    
    // Queue management
    void enqueueDetectionResult(const DetectionResult& result);
    void enqueueTrackingResult(const TrackingResult& result);
    void enqueueGameLogicResult(const GameLogicResult& result);
    void enqueueUIData(const PipelineData& data);
    
    bool dequeueDetectionResult(DetectionResult& result);
    bool dequeueTrackingResult(TrackingResult& result);
    bool dequeueGameLogicResult(GameLogicResult& result);
    
    // Performance and monitoring
    void updateMetrics();
    void monitorQueueLengths();
    void checkPerformanceTargets();
    void logComponentPerformance(const std::string& component, float processingTime);
    
    // CPU affinity management
    void setCpuAffinity(const std::vector<int>& cores);
    void setThreadPriority(int priority);
    
    // Error handling
    void handleQueueOverflow(const std::string& queueName);
    void handleProcessingTimeout(const std::string& component);
    
    // Utility functions
    void cleanupResources();
    bool validateConfiguration();
    void initializeMetrics();
};

/**
 * PipelineIntegratorFactory - Helper factory for creating pipeline integrators
 */
class PipelineIntegratorFactory {
public:
    static std::unique_ptr<ModernPipelineIntegrator> createDefault();
    static std::unique_ptr<ModernPipelineIntegrator> createHighPerformance();
    static std::unique_ptr<ModernPipelineIntegrator> createLowLatency();
    static std::unique_ptr<ModernPipelineIntegrator> createStreamingOptimized();
    static std::unique_ptr<ModernPipelineIntegrator> createCpuOnlyMode();
    
private:
    static ModernPipelineIntegrator::PipelineConfig getDefaultConfig();
    static ModernPipelineIntegrator::PipelineConfig getHighPerformanceConfig();
    static ModernPipelineIntegrator::PipelineConfig getLowLatencyConfig();
    static ModernPipelineIntegrator::PipelineConfig getStreamingConfig();
    static ModernPipelineIntegrator::PipelineConfig getCpuOnlyConfig();
};

} // namespace modern
} // namespace pv