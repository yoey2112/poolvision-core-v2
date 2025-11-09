#include "CoachingEngine.hpp"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <mutex>
#include <thread>

namespace pv {
namespace ai {

CoachingEngine::CoachingEngine(const CoachingConfig& config) 
    : config_(config), processingActive_(false), nextRequestId_(1),
      lastCoachingTime_(std::chrono::steady_clock::now()) {
    
    currentSession_.sessionStart = std::chrono::system_clock::now();
    currentSession_.sessionType = "casual";
}

CoachingEngine::~CoachingEngine() {
    shutdown();
}

bool CoachingEngine::initialize() {
    // Initialize Ollama client
    ollamaClient_ = std::make_unique<OllamaClient>(config_.ollamaConfig);
    if (!ollamaClient_->isAvailable()) {
        std::cerr << "Warning: Ollama server not available. Coaching will be disabled." << std::endl;
        return false;
    }
    
    // Initialize prompt generator
    promptGenerator_ = std::make_unique<CoachingPrompts>(config_.personality);
    
    // Start worker threads
    processingActive_ = true;
    int numThreads = std::min(config_.maxConcurrentRequests, 3);  // Cap at 3 for performance
    
    for (int i = 0; i < numThreads; ++i) {
        workerThreads_.emplace_back(&CoachingEngine::workerThread, this);
    }
    
    return true;
}

void CoachingEngine::shutdown() {
    if (!processingActive_) return;
    
    // Signal shutdown
    processingActive_ = false;
    queueCondition_.notify_all();
    
    // Wait for worker threads to finish
    for (auto& thread : workerThreads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    
    workerThreads_.clear();
    
    // Clear queue
    std::lock_guard<std::mutex> lock(queueMutex_);
    while (!requestQueue_.empty()) {
        requestQueue_.pop();
    }
}

uint64_t CoachingEngine::requestCoaching(CoachingPrompts::CoachingType type,
                                        const CoachingPrompts::CoachingContext& context,
                                        int priority) {
    if (!processingActive_ || !isOllamaConnected()) {
        return 0;  // Failed to queue request
    }
    
    if (isRateLimited()) {
        return 0;  // Rate limited
    }
    
    uint64_t requestId = nextRequestId_++;
    CoachingRequest request(requestId, type, context);
    request.priority = priority;
    
    {
        std::lock_guard<std::mutex> lock(queueMutex_);
        
        // Check queue size limit
        if (requestQueue_.size() >= static_cast<size_t>(config_.requestQueueSize)) {
            return 0;  // Queue full
        }
        
        requestQueue_.push(request);
    }
    
    queueCondition_.notify_one();
    logRequest(request);
    
    return requestId;
}

uint64_t CoachingEngine::requestShotAnalysis(const pv::modern::ShotSegmentation::ShotEvent& shot,
                                           const GameState& gameState,
                                           const CoachingPrompts::CoachingContext::PlayerInfo& player) {
    auto context = buildContext(shot, gameState, player);
    return requestCoaching(CoachingPrompts::CoachingType::ShotAnalysis, context, 1);
}

uint64_t CoachingEngine::requestDrillRecommendation(const CoachingPrompts::CoachingContext::PlayerInfo& player,
                                                   const std::vector<pv::modern::ShotSegmentation::ShotEvent>& recentShots) {
    CoachingPrompts::CoachingContext context;
    context.player = player;
    context.recentShots = recentShots;
    
    {
        std::lock_guard<std::mutex> lock(sessionMutex_);
        context.session = currentSession_;
    }
    
    return requestCoaching(CoachingPrompts::CoachingType::DrillRecommendation, context, 2);
}

uint64_t CoachingEngine::requestSessionReview() {
    CoachingPrompts::CoachingContext context;
    
    {
        std::lock_guard<std::mutex> lock(sessionMutex_);
        context.session = currentSession_;
        context.recentShots = shotHistory_;
    }
    
    return requestCoaching(CoachingPrompts::CoachingType::PerformanceReview, context, 3);
}

CoachingEngine::CoachingResponse CoachingEngine::getImmediateCoaching(CoachingPrompts::CoachingType type,
                                                                      const CoachingPrompts::CoachingContext& context) {
    if (!ollamaClient_ || !promptGenerator_) {
        CoachingResponse response;
        response.success = false;
        response.advice = "Coaching system not initialized";
        return response;
    }
    
    auto startTime = std::chrono::high_resolution_clock::now();
    
    CoachingRequest request(nextRequestId_++, type, context);
    CoachingResponse response = generateCoachingResponse(request);
    
    auto endTime = std::chrono::high_resolution_clock::now();
    response.responseTime = std::chrono::duration<float, std::milli>(endTime - startTime).count();
    
    return response;
}

void CoachingEngine::startSession(const std::string& sessionType) {
    std::lock_guard<std::mutex> lock(sessionMutex_);
    
    currentSession_ = CoachingPrompts::CoachingContext::SessionInfo{};
    currentSession_.sessionType = sessionType;
    currentSession_.sessionStart = std::chrono::system_clock::now();
    
    shotHistory_.clear();
}

void CoachingEngine::endSession() {
    if (config_.enableSessionAnalysis) {
        requestSessionReview();
    }
    
    std::lock_guard<std::mutex> lock(sessionMutex_);
    shotHistory_.clear();
}

void CoachingEngine::addShotToHistory(const pv::modern::ShotSegmentation::ShotEvent& shot) {
    std::lock_guard<std::mutex> lock(sessionMutex_);
    
    shotHistory_.push_back(shot);
    
    // Limit history size
    if (shotHistory_.size() > static_cast<size_t>(config_.maxHistoryShots)) {
        shotHistory_.erase(shotHistory_.begin());
    }
    
    // Update session stats
    currentSession_.shotsAttempted++;
    if (shot.isLegalShot) {
        currentSession_.successfulShots++;
    }
    
    // Update average shot time
    float totalTime = currentSession_.avgShotTime * (currentSession_.shotsAttempted - 1) + shot.duration;
    currentSession_.avgShotTime = totalTime / currentSession_.shotsAttempted;
}

void CoachingEngine::updatePlayerInfo(const CoachingPrompts::CoachingContext::PlayerInfo& player) {
    // Could be used to update persistent player information
    // For now, just store in context for next requests
}

void CoachingEngine::setConfig(const CoachingConfig& config) {
    config_ = config;
    
    if (ollamaClient_) {
        ollamaClient_->setConfig(config.ollamaConfig);
    }
    
    if (promptGenerator_) {
        promptGenerator_->setPersonality(config.personality);
    }
}

void CoachingEngine::setPersonality(CoachingPrompts::CoachingPersonality personality) {
    config_.personality = personality;
    if (promptGenerator_) {
        promptGenerator_->setPersonality(personality);
    }
}

bool CoachingEngine::isAvailable() const {
    return processingActive_ && ollamaClient_ && ollamaClient_->isAvailable();
}

bool CoachingEngine::isOllamaConnected() const {
    return ollamaClient_ && ollamaClient_->isAvailable();
}

size_t CoachingEngine::getQueueSize() const {
    std::lock_guard<std::mutex> lock(queueMutex_);
    return requestQueue_.size();
}

float CoachingEngine::getAverageResponseTime() const {
    if (ollamaClient_) {
        return ollamaClient_->getAverageResponseTime();
    }
    return 0.0f;
}

void CoachingEngine::getPerformanceStats(int& totalRequests, float& avgTime, int& queueSize) const {
    if (ollamaClient_) {
        totalRequests = ollamaClient_->getRequestCount();
        avgTime = ollamaClient_->getAverageResponseTime();
    } else {
        totalRequests = 0;
        avgTime = 0.0f;
    }
    
    queueSize = static_cast<int>(getQueueSize());
}

bool CoachingEngine::shouldTriggerCoaching(const pv::modern::ShotSegmentation::ShotEvent& shot) const {
    if (!config_.enableRealTimeCoaching) return false;
    
    // Don't coach on legal shots unless performance is poor
    if (shot.isLegalShot) {
        std::lock_guard<std::mutex> lock(sessionMutex_);
        if (currentSession_.shotsAttempted > 3) {
            float successRate = static_cast<float>(currentSession_.successfulShots) / currentSession_.shotsAttempted;
            return successRate < config_.coachingTriggerThreshold;
        }
        return false;
    }
    
    // Always coach on illegal shots
    return true;
}

void CoachingEngine::processAutoCoaching(const pv::modern::ShotSegmentation::ShotEvent& shot,
                                        const GameState& gameState,
                                        const CoachingPrompts::CoachingContext::PlayerInfo& player) {
    if (!shouldTriggerCoaching(shot)) return;
    
    // Add shot to history first
    addShotToHistory(shot);
    
    // Determine coaching type based on shot characteristics
    CoachingPrompts::CoachingType type = CoachingPrompts::CoachingType::ShotAnalysis;
    
    if (!shot.isLegalShot) {
        type = CoachingPrompts::CoachingType::TechnicalCorrection;
    } else if (shot.duration > 10.0f) {
        type = CoachingPrompts::CoachingType::StrategyAdvice;
    }
    
    requestShotAnalysis(shot, gameState, player);
}

void CoachingEngine::workerThread() {
    while (processingActive_) {
        CoachingRequest request(0, CoachingPrompts::CoachingType::ShotAnalysis, 
                              CoachingPrompts::CoachingContext{});
        
        // Wait for request
        {
            std::unique_lock<std::mutex> lock(queueMutex_);
            queueCondition_.wait(lock, [this] { 
                return !requestQueue_.empty() || !processingActive_; 
            });
            
            if (!processingActive_) break;
            
            if (!requestQueue_.empty()) {
                request = requestQueue_.front();
                requestQueue_.pop();
            } else {
                continue;
            }
        }
        
        // Process request
        CoachingResponse response = generateCoachingResponse(request);
        
        // Call callback if set
        if (responseCallback_) {
            try {
                responseCallback_(response);
            } catch (const std::exception& e) {
                std::cerr << "Error in coaching response callback: " << e.what() << std::endl;
            }
        }
        
        logResponse(response);
    }
}

void CoachingEngine::processRequest(const CoachingRequest& request) {
    auto response = generateCoachingResponse(request);
    
    if (responseCallback_) {
        responseCallback_(response);
    }
}

CoachingEngine::CoachingResponse CoachingEngine::generateCoachingResponse(const CoachingRequest& request) {
    CoachingResponse response;
    response.requestId = request.requestId;
    response.type = request.type;
    
    if (!ollamaClient_ || !promptGenerator_) {
        response.success = false;
        response.advice = "Coaching system components not initialized";
        return response;
    }
    
    auto startTime = std::chrono::high_resolution_clock::now();
    
    try {
        // Generate prompt
        std::string prompt = promptGenerator_->createCoachingPrompt(request.type, request.context);
        
        // Get response from Ollama
        auto ollamaResponse = ollamaClient_->generateResponse(prompt);
        
        auto endTime = std::chrono::high_resolution_clock::now();
        response.responseTime = std::chrono::duration<float, std::milli>(endTime - startTime).count();
        
        if (ollamaResponse.success) {
            response.success = true;
            response.advice = ollamaResponse.content;
        } else {
            response.success = false;
            response.advice = "Failed to generate coaching advice: " + ollamaResponse.error;
        }
        
        updatePerformanceMetrics("coaching_request", response.responseTime);
        
    } catch (const std::exception& e) {
        auto endTime = std::chrono::high_resolution_clock::now();
        response.responseTime = std::chrono::duration<float, std::milli>(endTime - startTime).count();
        response.success = false;
        response.advice = "Error generating coaching advice: " + std::string(e.what());
    }
    
    return response;
}

void CoachingEngine::updatePerformanceMetrics(const std::string& operation, float duration) {
    if (performanceCallback_) {
        performanceCallback_(operation, duration);
    }
}

void CoachingEngine::cleanupOldRequests() {
    std::lock_guard<std::mutex> lock(queueMutex_);
    
    // Remove requests older than 30 seconds
    auto cutoffTime = std::chrono::steady_clock::now() - std::chrono::seconds(30);
    
    std::queue<CoachingRequest> cleanedQueue;
    while (!requestQueue_.empty()) {
        if (requestQueue_.front().timestamp > cutoffTime) {
            cleanedQueue.push(requestQueue_.front());
        }
        requestQueue_.pop();
    }
    requestQueue_ = std::move(cleanedQueue);
}

bool CoachingEngine::isRateLimited() const {
    auto now = std::chrono::steady_clock::now();
    auto timeSinceLastCoaching = std::chrono::duration<float>(now - lastCoachingTime_).count();
    
    return timeSinceLastCoaching < config_.minTimeBetweenCoaching;
}

CoachingPrompts::CoachingContext CoachingEngine::buildContext(
    const pv::modern::ShotSegmentation::ShotEvent& shot,
    const GameState& gameState,
    const CoachingPrompts::CoachingContext::PlayerInfo& player) const {
    
    CoachingPrompts::CoachingContext context;
    context.currentShot = shot;
    context.gameState = gameState;
    context.player = player;
    context.isLegalShot = shot.isLegalShot;
    
    {
        std::lock_guard<std::mutex> lock(sessionMutex_);
        context.session = currentSession_;
        context.recentShots = shotHistory_;
    }
    
    return context;
}

void CoachingEngine::logRequest(const CoachingRequest& request) {
    // Log coaching request for debugging/analytics
    // Could be enhanced with proper logging system
}

void CoachingEngine::logResponse(const CoachingResponse& response) {
    // Log coaching response for debugging/analytics
    // Could be enhanced with proper logging system
}

// CoachingEngineFactory implementation

std::unique_ptr<CoachingEngine> CoachingEngineFactory::createDefault() {
    return std::make_unique<CoachingEngine>(getDefaultConfig());
}

std::unique_ptr<CoachingEngine> CoachingEngineFactory::createForTesting() {
    return std::make_unique<CoachingEngine>(getTestingConfig());
}

std::unique_ptr<CoachingEngine> CoachingEngineFactory::createHighPerformance() {
    return std::make_unique<CoachingEngine>(getHighPerformanceConfig());
}

std::unique_ptr<CoachingEngine> CoachingEngineFactory::createOfflineMode() {
    return std::make_unique<CoachingEngine>(getOfflineConfig());
}

CoachingEngine::CoachingConfig CoachingEngineFactory::getDefaultConfig() {
    CoachingEngine::CoachingConfig config;
    config.ollamaConfig.model = "phi3:mini";
    config.ollamaConfig.temperature = 0.7f;
    config.ollamaConfig.maxTokens = 512;
    config.personality = CoachingPrompts::CoachingPersonality::Supportive;
    config.maxConcurrentRequests = 2;
    config.minTimeBetweenCoaching = 5.0f;
    return config;
}

CoachingEngine::CoachingConfig CoachingEngineFactory::getTestingConfig() {
    auto config = getDefaultConfig();
    config.ollamaConfig.timeout = 10;  // Faster timeout for testing
    config.ollamaConfig.maxTokens = 256;  // Shorter responses
    config.requestQueueSize = 10;  // Smaller queue
    config.enableRealTimeCoaching = false;  // Manual control for testing
    return config;
}

CoachingEngine::CoachingConfig CoachingEngineFactory::getHighPerformanceConfig() {
    auto config = getDefaultConfig();
    config.maxConcurrentRequests = 3;
    config.requestQueueSize = 100;
    config.ollamaConfig.timeout = 15;
    config.enablePerformanceTracking = true;
    config.maxHistoryShots = 20;
    return config;
}

CoachingEngine::CoachingConfig CoachingEngineFactory::getOfflineConfig() {
    auto config = getDefaultConfig();
    config.enableRealTimeCoaching = false;
    config.enableSessionAnalysis = false;
    config.enableDrillRecommendations = false;
    return config;
}

} // namespace ai
} // namespace pv