#include "AILearningSystem.hpp"
#include <iostream>
#include <sstream>
#include <chrono>
#include <algorithm>

#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#else
#include <sys/resource.h>
#include <unistd.h>
#include <thread>
#endif

namespace pv {
namespace ai {
namespace learning {

// PerformanceIsolation Implementation
AILearningSystem::PerformanceIsolation::PerformanceIsolation(const SystemConfig& config)
    : allowedCores_(config.cpuCores), maxCpuUsage_(config.maxCpuUsage) {
}

void AILearningSystem::PerformanceIsolation::setCpuAffinity(std::thread& thread, const std::vector<int>& cores) {
#ifdef _WIN32
    HANDLE threadHandle = thread.native_handle();
    DWORD_PTR affinityMask = 0;
    
    for (int core : cores) {
        if (core >= 0 && core < 64) { // Windows supports up to 64 cores
            affinityMask |= (1ULL << core);
        }
    }
    
    if (affinityMask != 0) {
        SetThreadAffinityMask(threadHandle, affinityMask);
        std::cout << "Set AI thread affinity to cores: ";
        for (int core : cores) {
            std::cout << core << " ";
        }
        std::cout << std::endl;
    }
#else
    // Linux implementation would use pthread_setaffinity_np
    // For now, just log the intention
    std::cout << "CPU affinity setting requested for cores: ";
    for (int core : cores) {
        std::cout << core << " ";
    }
    std::cout << "(not implemented on this platform)" << std::endl;
#endif
}

void AILearningSystem::PerformanceIsolation::monitorCpuUsage() {
    monitoring_ = true;
    monitoringThread_ = std::thread([this]() {
        while (monitoring_.load()) {
#ifdef _WIN32
            HANDLE process = GetCurrentProcess();
            FILETIME createTime, exitTime, kernelTime, userTime;
            
            if (GetProcessTimes(process, &createTime, &exitTime, &kernelTime, &userTime)) {
                static FILETIME lastKernel = kernelTime;
                static FILETIME lastUser = userTime;
                
                ULARGE_INTEGER currentKernel, currentUser, prevKernel, prevUser;
                currentKernel.LowPart = kernelTime.dwLowDateTime;
                currentKernel.HighPart = kernelTime.dwHighDateTime;
                currentUser.LowPart = userTime.dwLowDateTime;
                currentUser.HighPart = userTime.dwHighDateTime;
                
                prevKernel.LowPart = lastKernel.dwLowDateTime;
                prevKernel.HighPart = lastKernel.dwHighDateTime;
                prevUser.LowPart = lastUser.dwLowDateTime;
                prevUser.HighPart = lastUser.dwHighDateTime;
                
                ULONGLONG totalTime = (currentKernel.QuadPart - prevKernel.QuadPart) + 
                                     (currentUser.QuadPart - prevUser.QuadPart);
                
                // Rough CPU usage estimate (simplified)
                double usage = static_cast<double>(totalTime) / 10000000.0; // Convert to percentage
                currentCpuUsage_.store(std::min(usage, 100.0));
                
                lastKernel = kernelTime;
                lastUser = userTime;
            }
#else
            // Simplified CPU usage monitoring for other platforms
            currentCpuUsage_.store(15.0); // Placeholder value
#endif
            
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    });
}

void AILearningSystem::PerformanceIsolation::throttleIfNeeded() {
    double usage = currentCpuUsage_.load();
    if (usage > maxCpuUsage_) {
        // Simple throttling: sleep proportional to excess usage
        int sleepMs = static_cast<int>((usage - maxCpuUsage_) * 10);
        std::this_thread::sleep_for(std::chrono::milliseconds(sleepMs));
    }
}

// DataFlowCoordinator Implementation
AILearningSystem::DataFlowCoordinator::DataFlowCoordinator(
    DataCollectionEngine* dataEngine, ShotAnalysisEngine* analysisEngine,
    AdaptiveCoachingEngine* coachingEngine, PerformanceAnalyticsEngine* analyticsEngine)
    : dataEngine_(dataEngine), analysisEngine_(analysisEngine),
      coachingEngine_(coachingEngine), analyticsEngine_(analyticsEngine) {
}

void AILearningSystem::DataFlowCoordinator::routeShotData(const DataCollectionEngine::ShotOutcomeData& shot) {
    // Route to all components that need shot data
    if (analysisEngine_) {
        analysisEngine_->updatePlayerModel(shot.playerId, shot);
    }
    
    if (coachingEngine_) {
        coachingEngine_->updateFromShotOutcome(shot.playerId, shot);
    }
    
    if (analyticsEngine_) {
        analyticsEngine_->updatePlayerPerformance(shot.playerId, shot);
    }
    
    // Generate event
    AILearningEvent event;
    event.type = AILearningEvent::ShotAnalyzed;
    event.playerId = shot.playerId;
    event.confidence = shot.successful ? 1.0 : 0.0;
    event.timestamp = std::chrono::steady_clock::now();
    
    std::lock_guard<std::mutex> lock(queueMutex_);
    eventQueue_.push(event);
}

void AILearningSystem::DataFlowCoordinator::routeBehaviorData(const DataCollectionEngine::PlayerBehaviorData& behavior) {
    // Route to coaching engine for adaptation
    if (coachingEngine_) {
        coachingEngine_->updatePlayerProfile(behavior.playerId, behavior);
    }
}

void AILearningSystem::DataFlowCoordinator::triggerShotAnalysis(
    int playerId, const GameState& gameState, const cv::Point2f& cueBallPos, 
    const std::vector<Ball>& targetBalls) {
    
    if (!analysisEngine_) return;
    
    auto result = analysisEngine_->analyzeShotSituation(playerId, gameState, cueBallPos, targetBalls);
    processAnalysisResults(result);
}

void AILearningSystem::DataFlowCoordinator::processAnalysisResults(const ShotAnalysisEngine::ShotAnalysisResult& result) {
    // Generate coaching based on analysis
    if (coachingEngine_) {
        // This would need the current game state, which we'd need to track
        // For now, just update the player model
    }
    
    // Generate event
    AILearningEvent event;
    event.type = AILearningEvent::ShotAnalyzed;
    event.confidence = result.mainPrediction.confidence;
    event.timestamp = std::chrono::steady_clock::now();
    
    std::stringstream ss;
    ss << "Success probability: " << result.mainPrediction.successProbability;
    event.data = ss.str();
    
    std::lock_guard<std::mutex> lock(queueMutex_);
    eventQueue_.push(event);
}

// AILearningSystem Implementation
AILearningSystem::AILearningSystem(const SystemConfig& config) : config_(config) {
    std::cout << "Initializing AI Learning System..." << std::endl;
}

AILearningSystem::~AILearningSystem() {
    shutdown();
}

bool AILearningSystem::initialize() {
    std::lock_guard<std::mutex> lock(systemMutex_);
    
    if (initialized_.load()) {
        std::cout << "AI Learning System already initialized" << std::endl;
        return true;
    }
    
    std::cout << "Initializing AI Learning System components..." << std::endl;
    
    // Initialize core components
    if (!initializeDataEngine()) {
        std::cerr << "Failed to initialize Data Collection Engine" << std::endl;
        return false;
    }
    
    if (!initializeShotAnalysis()) {
        std::cerr << "Failed to initialize Shot Analysis Engine" << std::endl;
        return false;
    }
    
    if (!initializeAdaptiveCoaching()) {
        std::cerr << "Failed to initialize Adaptive Coaching Engine" << std::endl;
        return false;
    }
    
    if (!initializePerformanceAnalytics()) {
        std::cerr << "Failed to initialize Performance Analytics Engine" << std::endl;
        return false;
    }
    
    // Setup data flow coordination
    dataFlowCoordinator_ = std::make_unique<DataFlowCoordinator>(
        dataEngine_.get(), analysisEngine_.get(), 
        coachingEngine_.get(), analyticsEngine_.get());
    
    // Setup performance isolation if enabled
    if (config_.enablePerformanceIsolation) {
        if (!setupPerformanceIsolation()) {
            std::cout << "Warning: Performance isolation setup failed, continuing without it" << std::endl;
        }
    }
    
    initialized_ = true;
    std::cout << "AI Learning System initialized successfully!" << std::endl;
    return true;
}

void AILearningSystem::start() {
    if (!initialized_.load()) {
        std::cerr << "Cannot start AI Learning System: not initialized" << std::endl;
        return;
    }
    
    if (systemActive_.load()) {
        std::cout << "AI Learning System already running" << std::endl;
        return;
    }
    
    std::cout << "Starting AI Learning System..." << std::endl;
    
    // Start core components
    if (config_.enableDataCollection && dataEngine_) {
        dataEngine_->startCollection();
    }
    
    if (config_.enableShotAnalysis && analysisEngine_) {
        analysisEngine_->startAnalysis();
    }
    
    if (config_.enableAdaptiveCoaching && coachingEngine_) {
        coachingEngine_->startCoaching();
    }
    
    if (config_.enablePerformanceAnalytics && analyticsEngine_) {
        analyticsEngine_->startAnalytics();
    }
    
    // Start background threads
    systemActive_ = true;
    statusMonitoring_ = true;
    
    eventProcessingThread_ = std::thread(&AILearningSystem::eventProcessingLoop, this);
    statusUpdateThread_ = std::thread(&AILearningSystem::statusUpdateLoop, this);
    
    // Set CPU affinity for background threads
    if (performanceIsolation_ && !config_.cpuCores.empty()) {
        performanceIsolation_->setCpuAffinity(eventProcessingThread_, config_.cpuCores);
        performanceIsolation_->setCpuAffinity(statusUpdateThread_, config_.cpuCores);
        performanceIsolation_->monitorCpuUsage();
    }
    
    std::cout << "AI Learning System started successfully!" << std::endl;
    logSystemReport();
}

void AILearningSystem::stop() {
    if (!systemActive_.load()) {
        return;
    }
    
    std::cout << "Stopping AI Learning System..." << std::endl;
    
    systemActive_ = false;
    statusMonitoring_ = false;
    eventCondition_.notify_all();
    
    // Stop core components
    if (dataEngine_) {
        dataEngine_->stopCollection();
    }
    if (analysisEngine_) {
        analysisEngine_->stopAnalysis();
    }
    if (coachingEngine_) {
        coachingEngine_->stopCoaching();
    }
    if (analyticsEngine_) {
        analyticsEngine_->stopAnalytics();
    }
    
    // Join background threads
    if (eventProcessingThread_.joinable()) {
        eventProcessingThread_.join();
    }
    if (statusUpdateThread_.joinable()) {
        statusUpdateThread_.join();
    }
    
    std::cout << "AI Learning System stopped" << std::endl;
}

void AILearningSystem::shutdown() {
    stop();
    
    std::lock_guard<std::mutex> lock(systemMutex_);
    
    // Reset components
    dataFlowCoordinator_.reset();
    performanceIsolation_.reset();
    analyticsEngine_.reset();
    coachingEngine_.reset();
    analysisEngine_.reset();
    dataEngine_.reset();
    
    initialized_ = false;
    std::cout << "AI Learning System shutdown complete" << std::endl;
}

void AILearningSystem::addPlayer(int playerId, const std::string& playerName) {
    if (!systemActive_.load()) return;
    
    std::cout << "Adding player " << playerId << " (" << playerName << ") to AI Learning System" << std::endl;
    
    if (coachingEngine_) {
        coachingEngine_->addPlayer(playerId, playerName);
    }
    
    if (analyticsEngine_) {
        analyticsEngine_->startPerformanceSession(playerId);
    }
}

void AILearningSystem::startPlayerSession(int playerId) {
    if (!systemActive_.load()) return;
    
    if (coachingEngine_) {
        coachingEngine_->startCoachingSession(playerId);
    }
    
    if (analyticsEngine_) {
        analyticsEngine_->startPerformanceSession(playerId);
    }
}

void AILearningSystem::endPlayerSession(int playerId) {
    if (!systemActive_.load()) return;
    
    if (coachingEngine_) {
        coachingEngine_->endCoachingSession(playerId);
    }
    
    if (analyticsEngine_) {
        analyticsEngine_->endPerformanceSession(playerId);
    }
}

ShotAnalysisEngine::ShotAnalysisResult AILearningSystem::analyzeShotSituation(
    int playerId, const GameState& gameState, const cv::Point2f& cueBallPos, 
    const std::vector<Ball>& targetBalls) {
    
    if (!systemActive_.load() || !analysisEngine_) {
        return ShotAnalysisEngine::ShotAnalysisResult{};
    }
    
    return analysisEngine_->analyzeShotSituation(playerId, gameState, cueBallPos, targetBalls);
}

AdaptiveCoachingEngine::CoachingMessage AILearningSystem::generateCoaching(
    int playerId, const GameState& gameState, const ShotAnalysisEngine::ShotAnalysisResult& analysis) {
    
    if (!systemActive_.load() || !coachingEngine_) {
        return AdaptiveCoachingEngine::CoachingMessage{};
    }
    
    return coachingEngine_->generateRealtimeCoaching(playerId, gameState, analysis);
}

PerformanceAnalyticsEngine::PerformanceMetrics AILearningSystem::getPlayerMetrics(int playerId) {
    if (!systemActive_.load() || !analyticsEngine_) {
        return PerformanceAnalyticsEngine::PerformanceMetrics{};
    }
    
    return analyticsEngine_->getPlayerMetrics(playerId);
}

void AILearningSystem::onShotCompleted(int playerId, const cv::Point2f& startPos, const cv::Point2f& endPos, 
                                     bool successful, float difficulty) {
    if (!systemActive_.load() || !dataFlowCoordinator_) return;
    
    auto shotData = createShotData(playerId, startPos, endPos, successful, difficulty);
    
    // Store in data collection engine
    if (dataEngine_) {
        dataEngine_->recordShotOutcome(shotData);
    }
    
    // Route to other components
    dataFlowCoordinator_->routeShotData(shotData);
}

void AILearningSystem::onPlayerBehavior(int playerId, float aimingTime, float confidence) {
    if (!systemActive_.load() || !dataFlowCoordinator_) return;
    
    auto behaviorData = createBehaviorData(playerId, aimingTime, confidence);
    
    // Store in data collection engine
    if (dataEngine_) {
        dataEngine_->recordPlayerBehavior(behaviorData);
    }
    
    // Route to other components
    dataFlowCoordinator_->routeBehaviorData(behaviorData);
}

AILearningSystem::SystemStatus AILearningSystem::getSystemStatus() const {
    std::lock_guard<std::mutex> lock(systemMutex_);
    return currentStatus_;
}

void AILearningSystem::logSystemReport() {
    auto status = getSystemStatus();
    
    std::cout << "\n=== AI Learning System Status ===" << std::endl;
    std::cout << "Data Collection: " << (status.dataCollectionActive ? "Active" : "Inactive") << std::endl;
    std::cout << "Shot Analysis: " << (status.shotAnalysisActive ? "Active" : "Inactive") << std::endl;
    std::cout << "Adaptive Coaching: " << (status.adaptiveCoachingActive ? "Active" : "Inactive") << std::endl;
    std::cout << "Performance Analytics: " << (status.performanceAnalyticsActive ? "Active" : "Inactive") << std::endl;
    std::cout << "Players Tracked: " << status.playersTracked << std::endl;
    std::cout << "CPU Usage: " << std::fixed << std::setprecision(1) << status.cpuUsage << "%" << std::endl;
    std::cout << "Data Quality: " << std::setprecision(2) << status.dataQuality << std::endl;
    std::cout << "===================================" << std::endl;
}

cv::Mat AILearningSystem::generatePlayerPerformanceChart(int playerId, const std::string& chartType) {
    if (!systemActive_.load() || !analyticsEngine_) {
        cv::Mat emptyChart(400, 600, CV_8UC3, cv::Scalar(255, 255, 255));
        cv::putText(emptyChart, "AI Learning System not active", cv::Point(150, 200),
                   cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 0), 2);
        return emptyChart;
    }
    
    return analyticsEngine_->generatePerformanceChart(playerId, chartType);
}

// Background processing methods
void AILearningSystem::eventProcessingLoop() {
    while (systemActive_.load()) {
        std::unique_lock<std::mutex> lock(systemMutex_);
        
        eventCondition_.wait_for(lock, std::chrono::milliseconds(100), [this] {
            return !systemActive_.load() || eventCallback_ != nullptr;
        });
        
        if (!systemActive_.load()) break;
        
        // Process events if callback is set
        if (eventCallback_ && dataFlowCoordinator_) {
            // Process any queued events
            // This is simplified - in practice you'd process the event queue
        }
        
        // Throttle if CPU usage is too high
        if (performanceIsolation_) {
            performanceIsolation_->throttleIfNeeded();
        }
        
        lock.unlock();
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

void AILearningSystem::statusUpdateLoop() {
    while (statusMonitoring_.load()) {
        updateSystemStatus();
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
}

void AILearningSystem::updateSystemStatus() {
    std::lock_guard<std::mutex> lock(systemMutex_);
    
    currentStatus_.dataCollectionActive = dataEngine_ && dataEngine_->isCollectionActive();
    currentStatus_.shotAnalysisActive = analysisEngine_ != nullptr;
    currentStatus_.adaptiveCoachingActive = coachingEngine_ != nullptr;
    currentStatus_.performanceAnalyticsActive = analyticsEngine_ != nullptr;
    
    if (performanceIsolation_) {
        currentStatus_.cpuUsage = performanceIsolation_->getCurrentCpuUsage();
    }
    
    currentStatus_.lastUpdate = std::chrono::steady_clock::now();
}

// Initialization helpers
bool AILearningSystem::initializeDataEngine() {
    if (!config_.enableDataCollection) return true;
    
    try {
        dataEngine_ = DataCollectionFactory::createWithDatabase(config_.databasePath);
        if (!dataEngine_) {
            std::cerr << "Failed to create Data Collection Engine" << std::endl;
            return false;
        }
        
        std::cout << "Data Collection Engine initialized" << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Exception initializing Data Collection Engine: " << e.what() << std::endl;
        return false;
    }
}

bool AILearningSystem::initializeShotAnalysis() {
    if (!config_.enableShotAnalysis) return true;
    
    try {
        analysisEngine_ = ShotAnalysisFactory::createRealTime(dataEngine_.get());
        if (!analysisEngine_) {
            std::cerr << "Failed to create Shot Analysis Engine" << std::endl;
            return false;
        }
        
        std::cout << "Shot Analysis Engine initialized" << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Exception initializing Shot Analysis Engine: " << e.what() << std::endl;
        return false;
    }
}

bool AILearningSystem::initializeAdaptiveCoaching() {
    if (!config_.enableAdaptiveCoaching) return true;
    
    try {
        if (config_.ollamaEndpoint.empty()) {
            coachingEngine_ = AdaptiveCoachingFactory::createStandalone(
                dataEngine_.get(), analysisEngine_.get());
        } else {
            coachingEngine_ = AdaptiveCoachingFactory::createWithOllama(
                dataEngine_.get(), analysisEngine_.get(), config_.ollamaEndpoint);
        }
        
        if (!coachingEngine_) {
            std::cerr << "Failed to create Adaptive Coaching Engine" << std::endl;
            return false;
        }
        
        coachingEngine_->setCoachingIntensity(config_.coachingIntensity);
        
        std::cout << "Adaptive Coaching Engine initialized" << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Exception initializing Adaptive Coaching Engine: " << e.what() << std::endl;
        return false;
    }
}

bool AILearningSystem::initializePerformanceAnalytics() {
    if (!config_.enablePerformanceAnalytics) return true;
    
    try {
        analysisEngine_ = ShotAnalysisFactory::createRealTime(dataEngine_.get());
        analyticsEngine_ = PerformanceAnalyticsFactory::createRealTime(
            dataEngine_.get(), analysisEngine_.get(), coachingEngine_.get());
        
        if (!analyticsEngine_) {
            std::cerr << "Failed to create Performance Analytics Engine" << std::endl;
            return false;
        }
        
        analyticsEngine_->setAnalyticsDepth(config_.analyticsDepth);
        analyticsEngine_->setVisualizationQuality(config_.enableVisualization ? 2 : 1);
        
        std::cout << "Performance Analytics Engine initialized" << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Exception initializing Performance Analytics Engine: " << e.what() << std::endl;
        return false;
    }
}

bool AILearningSystem::setupPerformanceIsolation() {
    try {
        performanceIsolation_ = std::make_unique<PerformanceIsolation>(config_);
        std::cout << "Performance isolation initialized" << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Exception setting up performance isolation: " << e.what() << std::endl;
        return false;
    }
}

// Data conversion helpers
DataCollectionEngine::ShotOutcomeData AILearningSystem::createShotData(
    int playerId, const cv::Point2f& startPos, const cv::Point2f& endPos,
    bool successful, float difficulty) {
    
    DataCollectionEngine::ShotOutcomeData shotData;
    shotData.playerId = playerId;
    shotData.shotPosition = startPos;
    shotData.targetPosition = endPos;
    shotData.successful = successful;
    shotData.shotDifficulty = difficulty;
    shotData.shotSpeed = cv::norm(endPos - startPos); // Simple speed calculation
    shotData.shotAngle = std::atan2(endPos.y - startPos.y, endPos.x - startPos.x);
    shotData.shotType = DataCollectionEngine::ShotOutcomeData::Straight; // Default type
    shotData.timestamp = std::chrono::steady_clock::now();
    
    return shotData;
}

DataCollectionEngine::PlayerBehaviorData AILearningSystem::createBehaviorData(
    int playerId, float aimingTime, float confidence) {
    
    DataCollectionEngine::PlayerBehaviorData behaviorData;
    behaviorData.playerId = playerId;
    behaviorData.aimingTime = aimingTime;
    behaviorData.confidenceLevel = confidence;
    behaviorData.timestamp = std::chrono::steady_clock::now();
    
    return behaviorData;
}

// Factory Implementation
std::unique_ptr<AILearningSystem> AILearningSystemFactory::createDefault() {
    return std::make_unique<AILearningSystem>();
}

std::unique_ptr<AILearningSystem> AILearningSystemFactory::createOptimized(bool lowLatency) {
    AILearningSystem::SystemConfig config;
    
    if (lowLatency) {
        config.dataCollectionFrequency = 20; // Higher frequency
        config.analysisUpdateFrequency = 10;
        config.maxCpuUsage = 30; // Allow more CPU usage
    } else {
        config.dataCollectionFrequency = 5;  // Lower frequency for efficiency
        config.analysisUpdateFrequency = 2;
        config.maxCpuUsage = 15; // Conservative CPU usage
    }
    
    return std::make_unique<AILearningSystem>(config);
}

std::unique_ptr<AILearningSystem> AILearningSystemFactory::createWithConfig(
    const AILearningSystem::SystemConfig& config) {
    return std::make_unique<AILearningSystem>(config);
}

// Global AI Learning System
std::unique_ptr<AILearningSystem> GlobalAILearning::instance_;
std::mutex GlobalAILearning::instanceMutex_;

AILearningSystem& GlobalAILearning::getInstance() {
    std::lock_guard<std::mutex> lock(instanceMutex_);
    
    if (!instance_) {
        instance_ = AILearningSystemFactory::createDefault();
    }
    
    return *instance_;
}

void GlobalAILearning::initialize(std::unique_ptr<AILearningSystem> system) {
    std::lock_guard<std::mutex> lock(instanceMutex_);
    instance_ = std::move(system);
}

void GlobalAILearning::shutdown() {
    std::lock_guard<std::mutex> lock(instanceMutex_);
    if (instance_) {
        instance_->shutdown();
        instance_.reset();
    }
}

} // namespace learning
} // namespace ai
} // namespace pv