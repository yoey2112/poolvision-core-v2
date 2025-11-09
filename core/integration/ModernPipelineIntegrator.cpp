#include "ModernPipelineIntegrator.hpp"
#include <algorithm>
#include <iostream>
#include <sstream>
#include <iomanip>

#ifdef _WIN32
#include <Windows.h>
#elif defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif

namespace pv {
namespace modern {

ModernPipelineIntegrator::ModernPipelineIntegrator(const PipelineConfig& config)
    : config_(config) {
    
    initializeMetrics();
}

ModernPipelineIntegrator::~ModernPipelineIntegrator() {
    stopPipeline();
    cleanupResources();
}

bool ModernPipelineIntegrator::initializePipeline() {
    if (!validateConfiguration()) {
        std::cerr << "Invalid pipeline configuration" << std::endl;
        return false;
    }
    
    try {
        // Initialize processing isolation
        processingIsolation_ = std::make_unique<ProcessingIsolation>();
        if (!processingIsolation_->initialize(2, 4)) {
            std::cerr << "Failed to initialize processing isolation" << std::endl;
            return false;
        }
        
        // Initialize Agent Group 1: GPU Detection
        detector_ = std::make_unique<TensorRtBallDetector>();
        if (!detector_->initializeNvdecPipeline(config_.detectionConfig)) {
            std::cout << "GPU detection not available, using CPU fallback" << std::endl;
            detector_.reset();
        }
        
        // Initialize Agent Group 2: CPU Tracking
        tracker_ = std::make_unique<pv::modern::ByteTrackMOT>(config_.trackingConfig);
        if (!tracker_) {
            std::cerr << "Failed to initialize ByteTrack tracker" << std::endl;
            return false;
        }
        
        // Initialize Agent Group 3: Game Logic
        shotSegmenter_ = std::make_unique<pv::modern::ShotSegmentation>(config_.gameLogicConfig);
        if (!shotSegmenter_) {
            std::cerr << "Failed to initialize shot segmentation" << std::endl;
            return false;
        }
        
        // Initialize Agent Group 4: LLM Coaching (optional)
#ifdef USE_OLLAMA
        if (config_.enableCoaching) {
            coachingEngine_ = std::make_unique<pv::ai::CoachingEngine>(config_.coachingConfig);
            if (!coachingEngine_->initialize()) {
                std::cout << "AI coaching not available, continuing without coaching" << std::endl;
                coachingEngine_.reset();
            }
        }
#endif
        
        // Initialize Agent Group 5: UI Renderer
        uiRenderer_ = std::make_unique<pv::modern::SeparatedUIRenderer>(config_.uiConfig);
        if (!uiRenderer_) {
            std::cerr << "Failed to initialize UI renderer" << std::endl;
            return false;
        }
        
        std::cout << "Modern pipeline initialized successfully" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Exception during pipeline initialization: " << e.what() << std::endl;
        cleanupResources();
        return false;
    }
}

bool ModernPipelineIntegrator::startPipeline() {
    if (pipelineActive_.load()) {
        return true;
    }
    
    if (!detector_ && !tracker_ && !shotSegmenter_ && !uiRenderer_) {
        std::cerr << "Pipeline not initialized" << std::endl;
        return false;
    }
    
    pipelineActive_ = true;
    shutdownRequested_ = false;
    
    try {
        // Start UI renderer first
        if (uiRenderer_ && !uiRenderer_->startUIRendering()) {
            std::cerr << "Failed to start UI renderer" << std::endl;
            pipelineActive_ = false;
            return false;
        }
        
#ifdef USE_OLLAMA
        // Start coaching engine if available
        if (coachingEngine_) {
            // Coaching engine starts its own threads internally
        }
#endif
        
        // Start pipeline threads
        detectionThread_ = std::thread(&ModernPipelineIntegrator::detectionLoop, this);
        trackingThread_ = std::thread(&ModernPipelineIntegrator::trackingLoop, this);
        gameLogicThread_ = std::thread(&ModernPipelineIntegrator::gameLogicLoop, this);
        coordinatorThread_ = std::thread(&ModernPipelineIntegrator::coordinatorLoop, this);
        
        std::cout << "Modern pipeline started successfully" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Exception during pipeline startup: " << e.what() << std::endl;
        stopPipeline();
        return false;
    }
}

void ModernPipelineIntegrator::stopPipeline() {
    if (!pipelineActive_.load()) return;
    
    std::cout << "Stopping modern pipeline..." << std::endl;
    
    // Signal shutdown
    shutdownRequested_ = true;
    pipelineActive_ = false;
    
    // Wake up all waiting threads
    detectionCondition_.notify_all();
    trackingCondition_.notify_all();
    gameLogicCondition_.notify_all();
    
    // Wait for threads to finish
    if (detectionThread_.joinable()) detectionThread_.join();
    if (trackingThread_.joinable()) trackingThread_.join();
    if (gameLogicThread_.joinable()) gameLogicThread_.join();
    if (coordinatorThread_.joinable()) coordinatorThread_.join();
    
    // Stop UI renderer
    if (uiRenderer_) {
        uiRenderer_->stopUIRendering();
    }
    
#ifdef USE_OLLAMA
    // Stop coaching engine
    if (coachingEngine_) {
        coachingEngine_->shutdown();
    }
#endif
    
    // Clear all queues
    {
        std::lock_guard<std::mutex> lock(detectionQueueMutex_);
        while (!detectionQueue_.empty()) detectionQueue_.pop();
    }
    {
        std::lock_guard<std::mutex> lock(trackingQueueMutex_);
        while (!trackingQueue_.empty()) trackingQueue_.pop();
    }
    {
        std::lock_guard<std::mutex> lock(gameLogicQueueMutex_);
        while (!gameLogicQueue_.empty()) gameLogicQueue_.pop();
    }
    {
        std::lock_guard<std::mutex> lock(uiQueueMutex_);
        while (!uiQueue_.empty()) uiQueue_.pop();
    }
    
    std::cout << "Modern pipeline stopped" << std::endl;
}

bool ModernPipelineIntegrator::processFrame(const cv::Mat& frame, uint64_t timestamp) {
    if (!pipelineActive_.load() || frame.empty()) {
        return false;
    }
    
    // Create detection result for Agent Group 1
    DetectionResult detectionResult;
    detectionResult.frame = frame.clone();
    detectionResult.timestamp = timestamp;
    
    // Perform detection (simplified for integration)
    auto detectionStart = std::chrono::high_resolution_clock::now();
    
    if (detector_) {
        // GPU detection path
        detectionResult.detections = detector_->detectFromNvdec(
            frame.data, frame.cols, frame.rows, timestamp);
    } else {
        // CPU detection fallback - use classical detector
        // For now, create mock detections
        detectionResult.detections.clear();
    }
    
    auto detectionEnd = std::chrono::high_resolution_clock::now();
    detectionResult.inferenceTime = std::chrono::duration<float, std::milli>(
        detectionEnd - detectionStart).count();
    
    // Enqueue for tracking
    enqueueDetectionResult(detectionResult);
    
    return true;
}

std::vector<pv::modern::ByteTrackMOT::Track> ModernPipelineIntegrator::getCurrentTracks() {
    if (!tracker_) return {};
    return tracker_->getActiveTracks();
}

GameState ModernPipelineIntegrator::getCurrentGameState() {
    // Return current game state from shot segmenter
    if (shotSegmenter_) {
        return shotSegmenter_->getCurrentGameState();
    }
    return GameState(GameType::EightBall);  // Default fallback
}

pv::modern::ShotSegmentation::ShotEvent ModernPipelineIntegrator::getCurrentShotEvent() {
    if (shotSegmenter_) {
        return shotSegmenter_->getCurrentShotEvent();
    }
    return pv::modern::ShotSegmentation::ShotEvent{};
}

cv::Mat ModernPipelineIntegrator::getCurrentUIFrame() {
    if (!uiRenderer_) return cv::Mat();
    return uiRenderer_->getCurrentCompositeFrame();
}

cv::Mat ModernPipelineIntegrator::getCurrentBirdsEyeView() {
    if (!uiRenderer_) return cv::Mat();
    return uiRenderer_->getCurrentBirdsEyeView();
}

cv::Mat ModernPipelineIntegrator::getSideBySideView() {
    if (!uiRenderer_) return cv::Mat();
    return uiRenderer_->getSideBySideFrame();
}

#ifdef USE_OLLAMA
std::optional<pv::ai::CoachingEngine::CoachingResponse> ModernPipelineIntegrator::getLatestCoaching() {
    if (!coachingEngine_) return std::nullopt;
    // Would implement getting latest coaching response
    return std::nullopt;
}

void ModernPipelineIntegrator::requestCoaching(pv::ai::CoachingPrompts::CoachingType type) {
    if (!coachingEngine_) return;
    
    // Create coaching context from current game state
    pv::ai::CoachingPrompts::CoachingContext context;
    // Fill context with current game data
    
    coachingEngine_->requestCoaching(type, context, 1);
}
#endif

ModernPipelineIntegrator::PipelineMetrics ModernPipelineIntegrator::getPerformanceMetrics() {
    std::lock_guard<std::mutex> lock(metricsMutex_);
    return metrics_;
}

void ModernPipelineIntegrator::logPerformanceReport() {
    auto metrics = getPerformanceMetrics();
    
    std::stringstream report;
    report << "\n=== Modern Pipeline Performance Report ===\n";
    report << "Detection FPS: " << std::fixed << std::setprecision(1) << metrics.detectionFPS << "\n";
    report << "Tracking FPS: " << metrics.trackingFPS << "\n";
    report << "Game Logic FPS: " << metrics.gameLogicFPS << "\n";
    report << "UI FPS: " << metrics.uiFPS << "\n";
    report << "Overall Latency: " << metrics.overallLatency << "ms\n";
    report << "Total Frames: " << metrics.totalFramesProcessed << "\n";
    report << "Dropped Frames: " << metrics.droppedFrames << "\n";
    report << "Queue Lengths: D=" << metrics.detectionQueueLength 
           << " T=" << metrics.trackingQueueLength
           << " G=" << metrics.gameLogicQueueLength
           << " U=" << metrics.uiQueueLength << "\n";
    report << "==========================================\n";
    
    std::cout << report.str();
}

bool ModernPipelineIntegrator::isPerformanceWithinTargets() {
    auto metrics = getPerformanceMetrics();
    
    bool detectionOk = metrics.detectionFPS >= config_.targetInferenceFPS * 0.8f;
    bool trackingOk = metrics.trackingFPS >= config_.targetTrackingFPS * 0.8f;
    bool uiOk = metrics.uiFPS >= config_.targetUIFPS * 0.8f;
    bool latencyOk = metrics.overallLatency <= config_.maxLatencyMs;
    
    return detectionOk && trackingOk && uiOk && latencyOk;
}

void ModernPipelineIntegrator::detectionLoop() {
    setCpuAffinity({config_.gpuDeviceId});  // GPU-related processing
    
    while (pipelineActive_.load()) {
        // Detection processing happens in processFrame()
        // This thread could handle additional GPU processing if needed
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
}

void ModernPipelineIntegrator::trackingLoop() {
    setCpuAffinity(config_.cpuCoresForTracking);
    
    while (pipelineActive_.load()) {
        DetectionResult detectionResult;
        
        // Wait for detection results
        if (dequeueDetectionResult(detectionResult)) {
            auto trackingStart = std::chrono::high_resolution_clock::now();
            
            // Convert detections to ByteTrack format
            std::vector<pv::DetectionResult::Detection> byteDetections;
            for (const auto& ball : detectionResult.detections) {
                pv::DetectionResult::Detection det;
                det.x = ball.c.x - ball.r;
                det.y = ball.c.y - ball.r;
                det.w = ball.r * 2;
                det.h = ball.r * 2;
                det.confidence = 0.8f;
                det.classId = 0;
                byteDetections.push_back(det);
            }
            
            // Update tracking
            auto tracks = tracker_->update(byteDetections, 
                static_cast<double>(detectionResult.timestamp) / 1000.0);
            
            auto trackingEnd = std::chrono::high_resolution_clock::now();
            float trackingTime = std::chrono::duration<float, std::milli>(
                trackingEnd - trackingStart).count();
            
            // Create tracking result
            TrackingResult trackingResult;
            trackingResult.tracks = tracks;
            trackingResult.timestamp = detectionResult.timestamp;
            trackingResult.trackingTime = trackingTime;
            trackingResult.hasValidTracks = !tracks.empty();
            
            enqueueTrackingResult(trackingResult);
            
            logComponentPerformance("tracking", trackingTime);
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
}

void ModernPipelineIntegrator::gameLogicLoop() {
    setCpuAffinity(config_.cpuCoresForGameLogic);
    
    while (pipelineActive_.load()) {
        TrackingResult trackingResult;
        
        // Wait for tracking results
        if (dequeueTrackingResult(trackingResult)) {
            auto gameLogicStart = std::chrono::high_resolution_clock::now();
            
            // Update shot segmentation
            auto shotEvent = shotSegmenter_->update(trackingResult.tracks);
            
            // Update game state
            GameState gameState = shotSegmenter_->getCurrentGameState();
            
            auto gameLogicEnd = std::chrono::high_resolution_clock::now();
            float gameLogicTime = std::chrono::duration<float, std::milli>(
                gameLogicEnd - gameLogicStart).count();
            
            // Create game logic result
            GameLogicResult gameLogicResult;
            gameLogicResult.shotEvent = shotEvent.value_or(pv::modern::ShotSegmentation::ShotEvent{});
            gameLogicResult.gameState = gameState;
            gameLogicResult.timestamp = trackingResult.timestamp;
            gameLogicResult.gameLogicTime = gameLogicTime;
            gameLogicResult.hasValidGameLogic = shotEvent.has_value();
            
            enqueueGameLogicResult(gameLogicResult);
            
            logComponentPerformance("gameLogic", gameLogicTime);
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
}

void ModernPipelineIntegrator::coordinatorLoop() {
    setCpuAffinity(config_.cpuCoresForUI);
    
    while (pipelineActive_.load()) {
        GameLogicResult gameLogicResult;
        
        // Wait for game logic results
        if (dequeueGameLogicResult(gameLogicResult)) {
            // Assemble complete pipeline data for UI
            PipelineData pipelineData;
            
            // Would need to correlate with original frame data
            pipelineData.frameTimestamp = gameLogicResult.timestamp;
            pipelineData.shotEvent = gameLogicResult.shotEvent;
            pipelineData.gameState = gameLogicResult.gameState;
            pipelineData.gameLogicTime = gameLogicResult.gameLogicTime;
            pipelineData.hasValidGameLogic = gameLogicResult.hasValidGameLogic;
            
#ifdef USE_OLLAMA
            // Check for coaching updates
            if (coachingEngine_ && coachingEngine_->isAvailable()) {
                // Handle coaching integration
                pipelineData.hasNewCoaching = false;
                pipelineData.latestCoachingAdvice = "";
                pipelineData.coachingTime = 0.0f;
            }
#endif
            
            // Convert pipeline data to UI frame data
            pv::modern::SeparatedUIRenderer::FrameData uiFrameData;
            uiFrameData.frameTimestamp = pipelineData.frameTimestamp;
            uiFrameData.currentShot = pipelineData.shotEvent;
            uiFrameData.gameState = pipelineData.gameState;
            uiFrameData.inferenceTime = 0.0f;  // Would get from detection result
            uiFrameData.trackingTime = pipelineData.gameLogicTime;
            uiFrameData.hasValidData = true;
            
#ifdef USE_OLLAMA
            uiFrameData.latestCoachingAdvice = pipelineData.latestCoachingAdvice;
            uiFrameData.hasNewCoaching = pipelineData.hasNewCoaching;
#endif
            
            // Submit to UI renderer
            if (uiRenderer_) {
                uiRenderer_->submitFrameData(uiFrameData);
            }
            
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        
        // Update metrics periodically
        updateMetrics();
    }
}

void ModernPipelineIntegrator::enqueueDetectionResult(const DetectionResult& result) {
    std::lock_guard<std::mutex> lock(detectionQueueMutex_);
    
    if (detectionQueue_.size() >= config_.detectionQueueSize) {
        detectionQueue_.pop();  // Drop oldest
        handleQueueOverflow("detection");
    }
    
    detectionQueue_.push(result);
    detectionCondition_.notify_one();
}

void ModernPipelineIntegrator::enqueueTrackingResult(const TrackingResult& result) {
    std::lock_guard<std::mutex> lock(trackingQueueMutex_);
    
    if (trackingQueue_.size() >= config_.trackingQueueSize) {
        trackingQueue_.pop();
        handleQueueOverflow("tracking");
    }
    
    trackingQueue_.push(result);
    trackingCondition_.notify_one();
}

void ModernPipelineIntegrator::enqueueGameLogicResult(const GameLogicResult& result) {
    std::lock_guard<std::mutex> lock(gameLogicQueueMutex_);
    
    if (gameLogicQueue_.size() >= config_.gameLogicQueueSize) {
        gameLogicQueue_.pop();
        handleQueueOverflow("gameLogic");
    }
    
    gameLogicQueue_.push(result);
    gameLogicCondition_.notify_one();
}

bool ModernPipelineIntegrator::dequeueDetectionResult(DetectionResult& result) {
    std::unique_lock<std::mutex> lock(detectionQueueMutex_);
    
    if (detectionCondition_.wait_for(lock, std::chrono::milliseconds(10),
        [this] { return !detectionQueue_.empty() || shutdownRequested_.load(); })) {
        
        if (!detectionQueue_.empty()) {
            result = detectionQueue_.front();
            detectionQueue_.pop();
            return true;
        }
    }
    
    return false;
}

bool ModernPipelineIntegrator::dequeueTrackingResult(TrackingResult& result) {
    std::unique_lock<std::mutex> lock(trackingQueueMutex_);
    
    if (trackingCondition_.wait_for(lock, std::chrono::milliseconds(10),
        [this] { return !trackingQueue_.empty() || shutdownRequested_.load(); })) {
        
        if (!trackingQueue_.empty()) {
            result = trackingQueue_.front();
            trackingQueue_.pop();
            return true;
        }
    }
    
    return false;
}

bool ModernPipelineIntegrator::dequeueGameLogicResult(GameLogicResult& result) {
    std::unique_lock<std::mutex> lock(gameLogicQueueMutex_);
    
    if (gameLogicCondition_.wait_for(lock, std::chrono::milliseconds(10),
        [this] { return !gameLogicQueue_.empty() || shutdownRequested_.load(); })) {
        
        if (!gameLogicQueue_.empty()) {
            result = gameLogicQueue_.front();
            gameLogicQueue_.pop();
            return true;
        }
    }
    
    return false;
}

void ModernPipelineIntegrator::updateMetrics() {
    std::lock_guard<std::mutex> lock(metricsMutex_);
    
    auto currentTime = std::chrono::steady_clock::now();
    auto timeDiff = currentTime - metrics_.lastMetricsUpdate;
    
    // Update every second
    if (timeDiff >= std::chrono::seconds(1)) {
        monitorQueueLengths();
        checkPerformanceTargets();
        metrics_.lastMetricsUpdate = currentTime;
    }
}

void ModernPipelineIntegrator::monitorQueueLengths() {
    std::lock_guard<std::mutex> dLock(detectionQueueMutex_);
    std::lock_guard<std::mutex> tLock(trackingQueueMutex_);
    std::lock_guard<std::mutex> gLock(gameLogicQueueMutex_);
    std::lock_guard<std::mutex> uLock(uiQueueMutex_);
    
    metrics_.detectionQueueLength = detectionQueue_.size();
    metrics_.trackingQueueLength = trackingQueue_.size();
    metrics_.gameLogicQueueLength = gameLogicQueue_.size();
    metrics_.uiQueueLength = uiQueue_.size();
}

void ModernPipelineIntegrator::setCpuAffinity(const std::vector<int>& cores) {
#ifdef _WIN32
    if (cores.empty()) return;
    
    HANDLE thread = GetCurrentThread();
    DWORD_PTR affinity = 0;
    for (int core : cores) {
        affinity |= (1ULL << core);
    }
    SetThreadAffinityMask(thread, affinity);
#elif defined(__linux__)
    if (cores.empty()) return;
    
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    for (int core : cores) {
        CPU_SET(core, &cpuset);
    }
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
#endif
}

void ModernPipelineIntegrator::logComponentPerformance(const std::string& component, float processingTime) {
    // Update component-specific FPS
    std::lock_guard<std::mutex> lock(metricsMutex_);
    
    float alpha = 0.1f;  // Smoothing factor
    
    if (component == "detection") {
        float fps = (processingTime > 0) ? 1000.0f / processingTime : 0.0f;
        metrics_.detectionFPS = alpha * fps + (1.0f - alpha) * metrics_.detectionFPS;
    } else if (component == "tracking") {
        float fps = (processingTime > 0) ? 1000.0f / processingTime : 0.0f;
        metrics_.trackingFPS = alpha * fps + (1.0f - alpha) * metrics_.trackingFPS;
    } else if (component == "gameLogic") {
        float fps = (processingTime > 0) ? 1000.0f / processingTime : 0.0f;
        metrics_.gameLogicFPS = alpha * fps + (1.0f - alpha) * metrics_.gameLogicFPS;
    }
}

void ModernPipelineIntegrator::handleQueueOverflow(const std::string& queueName) {
    std::lock_guard<std::mutex> lock(metricsMutex_);
    metrics_.droppedFrames++;
    
    if (metrics_.droppedFrames % 100 == 0) {
        std::cout << "Warning: Queue overflow in " << queueName 
                  << " queue. Dropped " << metrics_.droppedFrames << " frames total." << std::endl;
    }
}

void ModernPipelineIntegrator::checkPerformanceTargets() {
    if (!isPerformanceWithinTargets()) {
        static int warningCount = 0;
        if (++warningCount % 10 == 0) {  // Log every 10 checks
            std::cout << "Warning: Performance below targets. Consider reducing quality settings." << std::endl;
        }
    }
}

bool ModernPipelineIntegrator::validateConfiguration() {
    if (config_.cpuCoresForTracking.empty() || 
        config_.cpuCoresForGameLogic.empty() || 
        config_.cpuCoresForUI.empty()) {
        std::cerr << "Invalid CPU core configuration" << std::endl;
        return false;
    }
    
    if (config_.targetInferenceFPS <= 0 || config_.targetTrackingFPS <= 0 || config_.targetUIFPS <= 0) {
        std::cerr << "Invalid FPS targets" << std::endl;
        return false;
    }
    
    return true;
}

void ModernPipelineIntegrator::initializeMetrics() {
    std::lock_guard<std::mutex> lock(metricsMutex_);
    metrics_ = PipelineMetrics{};
}

void ModernPipelineIntegrator::cleanupResources() {
    detector_.reset();
    tracker_.reset();
    shotSegmenter_.reset();
    uiRenderer_.reset();
    
#ifdef USE_OLLAMA
    coachingEngine_.reset();
#endif
    
    processingIsolation_.reset();
}

// PipelineIntegratorFactory Implementation
std::unique_ptr<ModernPipelineIntegrator> PipelineIntegratorFactory::createDefault() {
    return std::make_unique<ModernPipelineIntegrator>(getDefaultConfig());
}

std::unique_ptr<ModernPipelineIntegrator> PipelineIntegratorFactory::createHighPerformance() {
    return std::make_unique<ModernPipelineIntegrator>(getHighPerformanceConfig());
}

std::unique_ptr<ModernPipelineIntegrator> PipelineIntegratorFactory::createLowLatency() {
    return std::make_unique<ModernPipelineIntegrator>(getLowLatencyConfig());
}

std::unique_ptr<ModernPipelineIntegrator> PipelineIntegratorFactory::createStreamingOptimized() {
    return std::make_unique<ModernPipelineIntegrator>(getStreamingConfig());
}

std::unique_ptr<ModernPipelineIntegrator> PipelineIntegratorFactory::createCpuOnlyMode() {
    return std::make_unique<ModernPipelineIntegrator>(getCpuOnlyConfig());
}

ModernPipelineIntegrator::PipelineConfig PipelineIntegratorFactory::getDefaultConfig() {
    ModernPipelineIntegrator::PipelineConfig config;
    
    // Default CPU core assignments
    config.cpuCoresForTracking = {0, 1};
    config.cpuCoresForGameLogic = {2};
    config.cpuCoresForUI = {3};
    
#ifdef USE_OLLAMA
    config.cpuCoresForLLM = {4, 5, 6, 7};
    config.enableCoaching = false;  // Disabled by default
#endif
    
    // Performance targets
    config.targetInferenceFPS = 120.0f;    // Conservative default
    config.targetTrackingFPS = 200.0f;
    config.targetUIFPS = 60.0f;
    config.maxLatencyMs = 50.0f;
    
    // Queue sizes
    config.detectionQueueSize = 5;
    config.trackingQueueSize = 5;
    config.gameLogicQueueSize = 3;
    config.uiQueueSize = 3;
    
    return config;
}

ModernPipelineIntegrator::PipelineConfig PipelineIntegratorFactory::getHighPerformanceConfig() {
    auto config = getDefaultConfig();
    
    // Higher performance targets
    config.targetInferenceFPS = 200.0f;
    config.targetTrackingFPS = 300.0f;
    config.maxLatencyMs = 30.0f;
    
    // Larger queues for better throughput
    config.detectionQueueSize = 10;
    config.trackingQueueSize = 10;
    
    return config;
}

ModernPipelineIntegrator::PipelineConfig PipelineIntegratorFactory::getLowLatencyConfig() {
    auto config = getDefaultConfig();
    
    // Smaller queues for lower latency
    config.detectionQueueSize = 2;
    config.trackingQueueSize = 2;
    config.gameLogicQueueSize = 1;
    config.uiQueueSize = 1;
    config.maxLatencyMs = 20.0f;
    
    return config;
}

ModernPipelineIntegrator::PipelineConfig PipelineIntegratorFactory::getStreamingConfig() {
    auto config = getDefaultConfig();
    
    // Optimized for streaming
    config.targetUIFPS = 60.0f;
    config.uiConfig.overlayResolution = cv::Size(1920, 1080);
    config.uiConfig.enableBirdsEyeView = true;
    
    return config;
}

ModernPipelineIntegrator::PipelineConfig PipelineIntegratorFactory::getCpuOnlyConfig() {
    auto config = getDefaultConfig();
    
    // CPU-only performance targets
    config.targetInferenceFPS = 60.0f;
    config.targetTrackingFPS = 120.0f;
    config.maxLatencyMs = 100.0f;
    
    return config;
}

} // namespace modern
} // namespace pv