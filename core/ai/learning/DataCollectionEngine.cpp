#include "DataCollectionEngine.hpp"
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <filesystem>

#ifdef _WIN32
#include <Windows.h>
#elif defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif

namespace pv {
namespace ai {
namespace learning {

// LearningDatabase Implementation
DataCollectionEngine::LearningDatabase::LearningDatabase(const std::string& dbPath)
    : databasePath_(dbPath) {
    
    // Ensure directory exists
    std::filesystem::path dir = std::filesystem::path(dbPath).parent_path();
    std::filesystem::create_directories(dir);
    
    // Initialize SQLite database for learning data
    // For now, use file-based storage as a starting point
    std::cout << "Learning database initialized at: " << databasePath_ << std::endl;
}

void DataCollectionEngine::LearningDatabase::storeShotData(const ShotOutcomeData& data) {
    std::lock_guard<std::mutex> lock(dbMutex_);
    
    // Store shot data in structured format
    std::string filename = databasePath_ + "/shots_player_" + std::to_string(data.playerId) + ".log";
    std::ofstream file(filename, std::ios::app);
    
    if (file.is_open()) {
        auto timestamp = std::chrono::system_clock::to_time_t(data.timestamp);
        file << timestamp << ","
             << data.playerId << ","
             << data.shotPosition.x << "," << data.shotPosition.y << ","
             << data.targetPosition.x << "," << data.targetPosition.y << ","
             << data.actualOutcome.x << "," << data.actualOutcome.y << ","
             << (data.successful ? 1 : 0) << ","
             << data.shotDifficulty << ","
             << data.shotSpeed << ","
             << data.shotAngle << ","
             << static_cast<int>(data.shotType) << "\n";
    }
}

void DataCollectionEngine::LearningDatabase::storeBehaviorData(const PlayerBehaviorData& data) {
    std::lock_guard<std::mutex> lock(dbMutex_);
    
    std::string filename = databasePath_ + "/behavior_player_" + std::to_string(data.playerId) + ".log";
    std::ofstream file(filename, std::ios::app);
    
    if (file.is_open()) {
        auto timestamp = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
        file << timestamp << ","
             << data.playerId << ","
             << data.aimingTime << ","
             << data.aimingAdjustments << ","
             << (data.hesitationDetected ? 1 : 0) << ","
             << data.confidenceLevel << ","
             << data.sessionType << ","
             << data.sessionDuration << ","
             << data.fatigueLevel << "\n";
    }
}

void DataCollectionEngine::LearningDatabase::storeSessionData(const LearningDataPacket& packet) {
    std::lock_guard<std::mutex> lock(dbMutex_);
    
    // Store complete session data for advanced learning
    std::string filename = databasePath_ + "/sessions.log";
    std::ofstream file(filename, std::ios::app);
    
    if (file.is_open()) {
        auto timestamp = std::chrono::system_clock::to_time_t(packet.shotData.timestamp);
        file << timestamp << ","
             << packet.shotData.playerId << ","
             << packet.environmentalFactors << ","
             << (packet.pressureSituation ? 1 : 0) << ","
             << "\"" << packet.contextTags << "\"\n";
    }
}

std::vector<DataCollectionEngine::ShotOutcomeData> 
DataCollectionEngine::LearningDatabase::getPlayerShots(int playerId, int limit) {
    std::lock_guard<std::mutex> lock(dbMutex_);
    std::vector<ShotOutcomeData> shots;
    
    std::string filename = databasePath_ + "/shots_player_" + std::to_string(playerId) + ".log";
    std::ifstream file(filename);
    
    if (file.is_open()) {
        std::string line;
        std::vector<std::string> lines;
        
        // Read all lines
        while (std::getline(file, line) && lines.size() < static_cast<size_t>(limit)) {
            lines.push_back(line);
        }
        
        // Parse most recent shots (reverse order)
        for (auto it = lines.rbegin(); it != lines.rend() && shots.size() < static_cast<size_t>(limit); ++it) {
            ShotOutcomeData shot;
            std::istringstream ss(*it);
            std::string token;
            
            try {
                std::getline(ss, token, ','); // timestamp
                std::getline(ss, token, ','); shot.playerId = std::stoi(token);
                std::getline(ss, token, ','); shot.shotPosition.x = std::stof(token);
                std::getline(ss, token, ','); shot.shotPosition.y = std::stof(token);
                std::getline(ss, token, ','); shot.targetPosition.x = std::stof(token);
                std::getline(ss, token, ','); shot.targetPosition.y = std::stof(token);
                std::getline(ss, token, ','); shot.actualOutcome.x = std::stof(token);
                std::getline(ss, token, ','); shot.actualOutcome.y = std::stof(token);
                std::getline(ss, token, ','); shot.successful = (std::stoi(token) == 1);
                std::getline(ss, token, ','); shot.shotDifficulty = std::stof(token);
                std::getline(ss, token, ','); shot.shotSpeed = std::stof(token);
                std::getline(ss, token, ','); shot.shotAngle = std::stof(token);
                std::getline(ss, token, ','); shot.shotType = static_cast<ShotOutcomeData::ShotType>(std::stoi(token));
                
                shots.push_back(shot);
            } catch (const std::exception& e) {
                // Skip malformed lines
                continue;
            }
        }
    }
    
    return shots;
}

std::vector<DataCollectionEngine::PlayerBehaviorData> 
DataCollectionEngine::LearningDatabase::getPlayerBehavior(int playerId, int limit) {
    std::lock_guard<std::mutex> lock(dbMutex_);
    std::vector<PlayerBehaviorData> behaviors;
    
    std::string filename = databasePath_ + "/behavior_player_" + std::to_string(playerId) + ".log";
    std::ifstream file(filename);
    
    if (file.is_open()) {
        std::string line;
        std::vector<std::string> lines;
        
        while (std::getline(file, line) && lines.size() < static_cast<size_t>(limit)) {
            lines.push_back(line);
        }
        
        for (auto it = lines.rbegin(); it != lines.rend() && behaviors.size() < static_cast<size_t>(limit); ++it) {
            PlayerBehaviorData behavior;
            std::istringstream ss(*it);
            std::string token;
            
            try {
                std::getline(ss, token, ','); // timestamp
                std::getline(ss, token, ','); behavior.playerId = std::stoi(token);
                std::getline(ss, token, ','); behavior.aimingTime = std::stof(token);
                std::getline(ss, token, ','); behavior.aimingAdjustments = std::stoi(token);
                std::getline(ss, token, ','); behavior.hesitationDetected = (std::stoi(token) == 1);
                std::getline(ss, token, ','); behavior.confidenceLevel = std::stof(token);
                std::getline(ss, token, ','); behavior.sessionType = token;
                std::getline(ss, token, ','); behavior.sessionDuration = std::stoi(token);
                std::getline(ss, token, ','); behavior.fatigueLevel = std::stof(token);
                
                behaviors.push_back(behavior);
            } catch (const std::exception& e) {
                continue;
            }
        }
    }
    
    return behaviors;
}

void DataCollectionEngine::LearningDatabase::cleanupOldData(int daysToKeep) {
    std::lock_guard<std::mutex> lock(dbMutex_);
    
    auto cutoffTime = std::chrono::system_clock::now() - std::chrono::hours(24 * daysToKeep);
    auto cutoffTimeT = std::chrono::system_clock::to_time_t(cutoffTime);
    
    // For file-based storage, we'll implement a simple cleanup later
    // In production, this would use proper database operations
    std::cout << "Cleaning up data older than " << daysToKeep << " days (cutoff: " << cutoffTimeT << ")" << std::endl;
}

size_t DataCollectionEngine::LearningDatabase::getDatabaseSize() const {
    std::lock_guard<std::mutex> lock(dbMutex_);
    
    size_t totalSize = 0;
    try {
        for (const auto& entry : std::filesystem::directory_iterator(std::filesystem::path(databasePath_).parent_path())) {
            if (entry.is_regular_file()) {
                totalSize += entry.file_size();
            }
        }
    } catch (const std::exception& e) {
        // Directory might not exist yet
    }
    
    return totalSize;
}

// DataCollectionEngine Implementation
DataCollectionEngine::DataCollectionEngine(ProcessingIsolation* isolation)
    : isolation_(isolation), threadPriority_(-1) {
    
    dedicatedCpuCores_ = {4, 5}; // Default CPU cores for AI processing
}

DataCollectionEngine::~DataCollectionEngine() {
    stopCollection();
}

bool DataCollectionEngine::initializeCollection(const std::string& databasePath) {
    try {
        database_ = std::make_unique<LearningDatabase>(databasePath);
        
        std::cout << "AI Data Collection Engine initialized successfully" << std::endl;
        std::cout << "Database path: " << databasePath << std::endl;
        std::cout << "CPU cores: ";
        for (int core : dedicatedCpuCores_) std::cout << core << " ";
        std::cout << std::endl;
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize data collection: " << e.what() << std::endl;
        return false;
    }
}

void DataCollectionEngine::startCollection() {
    if (processingActive_.load()) return;
    
    processingActive_ = true;
    pauseRequested_ = false;
    
    dataProcessingThread_ = std::thread(&DataCollectionEngine::processingLoop, this);
    
    std::cout << "AI data collection started" << std::endl;
}

void DataCollectionEngine::stopCollection() {
    if (!processingActive_.load()) return;
    
    processingActive_ = false;
    dataCondition_.notify_all();
    
    if (dataProcessingThread_.joinable()) {
        dataProcessingThread_.join();
    }
    
    std::cout << "AI data collection stopped" << std::endl;
}

void DataCollectionEngine::recordShotOutcome(const ShotOutcomeData& shotData) {
    std::unique_lock<std::mutex> lock(dataQueueMutex_);
    
    if (shotQueue_.size() >= MAX_QUEUE_SIZE) {
        shotQueue_.pop(); // Drop oldest data if queue is full
    }
    
    shotQueue_.push(shotData);
    metrics_.shotsRecorded.fetch_add(1);
    
    lock.unlock();
    dataCondition_.notify_one();
}

void DataCollectionEngine::recordPlayerBehavior(const PlayerBehaviorData& behaviorData) {
    std::unique_lock<std::mutex> lock(dataQueueMutex_);
    
    if (behaviorQueue_.size() >= MAX_QUEUE_SIZE) {
        behaviorQueue_.pop();
    }
    
    behaviorQueue_.push(behaviorData);
    metrics_.behaviorEventsRecorded.fetch_add(1);
    
    lock.unlock();
    dataCondition_.notify_one();
}

void DataCollectionEngine::recordGameSession(const LearningDataPacket& sessionData) {
    std::unique_lock<std::mutex> lock(dataQueueMutex_);
    
    if (pendingData_.size() >= MAX_QUEUE_SIZE) {
        pendingData_.pop();
    }
    
    pendingData_.push(sessionData);
    
    lock.unlock();
    dataCondition_.notify_one();
}

void DataCollectionEngine::pauseCollection() {
    pauseRequested_ = true;
    std::cout << "AI data collection paused for GPU performance" << std::endl;
}

void DataCollectionEngine::resumeCollection() {
    pauseRequested_ = false;
    dataCondition_.notify_one();
    std::cout << "AI data collection resumed" << std::endl;
}

std::vector<DataCollectionEngine::ShotOutcomeData> 
DataCollectionEngine::getPlayerShotHistory(int playerId, int shotCount) {
    if (!database_) return {};
    return database_->getPlayerShots(playerId, shotCount);
}

std::vector<DataCollectionEngine::PlayerBehaviorData> 
DataCollectionEngine::getPlayerBehaviorHistory(int playerId) {
    if (!database_) return {};
    return database_->getPlayerBehavior(playerId, 50);
}

DataCollectionEngine::PlayerStatistics 
DataCollectionEngine::calculatePlayerStatistics(int playerId) {
    PlayerStatistics stats;
    
    auto shots = getPlayerShotHistory(playerId, 1000); // Get recent shots
    auto behaviors = getPlayerBehaviorHistory(playerId);
    
    if (shots.empty()) return stats;
    
    stats.totalShots = static_cast<int>(shots.size());
    
    // Calculate success rate
    int successfulShots = 0;
    float totalAimingTime = 0;
    std::map<ShotOutcomeData::ShotType, int> shotTypeCounts;
    std::map<ShotOutcomeData::ShotType, int> shotTypeSuccesses;
    
    for (const auto& shot : shots) {
        if (shot.successful) successfulShots++;
        
        shotTypeCounts[shot.shotType]++;
        if (shot.successful) shotTypeSuccesses[shot.shotType]++;
    }
    
    stats.overallSuccessRate = static_cast<float>(successfulShots) / stats.totalShots;
    
    // Calculate shot type success rates
    for (const auto& [shotType, count] : shotTypeCounts) {
        if (count > 0) {
            stats.shotTypeSuccessRates[shotType] = 
                static_cast<float>(shotTypeSuccesses[shotType]) / count;
        }
    }
    
    // Calculate average aiming time from behavior data
    if (!behaviors.empty()) {
        for (const auto& behavior : behaviors) {
            totalAimingTime += behavior.aimingTime;
        }
        stats.averageAimingTime = totalAimingTime / behaviors.size();
    }
    
    // Calculate performance trend (last 30 entries)
    int trendSize = std::min(30, static_cast<int>(shots.size()));
    for (int i = 0; i < trendSize; ++i) {
        stats.performanceTrend.push_back(shots[i].successful ? 1.0f : 0.0f);
    }
    
    // Calculate improvement rate (simple linear trend)
    if (stats.performanceTrend.size() >= 2) {
        float firstHalf = 0, secondHalf = 0;
        int halfSize = trendSize / 2;
        
        for (int i = 0; i < halfSize; ++i) {
            firstHalf += stats.performanceTrend[i];
            secondHalf += stats.performanceTrend[i + halfSize];
        }
        
        stats.improvementRate = (secondHalf / halfSize) - (firstHalf / halfSize);
    }
    
    return stats;
}

void DataCollectionEngine::processingLoop() {
    setCpuAffinity(dedicatedCpuCores_);
    
    std::cout << "AI data collection processing loop started on CPU cores: ";
    for (int core : dedicatedCpuCores_) std::cout << core << " ";
    std::cout << std::endl;
    
    while (processingActive_.load()) {
        std::unique_lock<std::mutex> lock(dataQueueMutex_);
        
        // Wait for data or shutdown signal
        dataCondition_.wait(lock, [this] {
            return !processingActive_.load() || 
                   !shotQueue_.empty() || 
                   !behaviorQueue_.empty() || 
                   !pendingData_.empty();
        });
        
        if (!processingActive_.load()) break;
        
        // Check if we should pause for GPU performance
        if (pauseRequested_.load() || isGpuBusy()) {
            lock.unlock();
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }
        
        lock.unlock();
        
        auto startTime = std::chrono::high_resolution_clock::now();
        
        // Process pending data
        processDataBatch();
        
        auto endTime = std::chrono::high_resolution_clock::now();
        double processingTime = std::chrono::duration<double, std::milli>(endTime - startTime).count();
        
        // Update performance metrics
        updatePerformanceMetrics();
        
        // Adaptive load adjustment
        if (processingTime > 50.0) { // If processing takes more than 50ms
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
    
    std::cout << "AI data collection processing loop ended" << std::endl;
}

void DataCollectionEngine::processDataBatch() {
    std::lock_guard<std::mutex> lock(dataQueueMutex_);
    
    int processedCount = 0;
    const int maxBatchSize = 10; // Process up to 10 items per batch
    
    // Process shot outcome data
    while (!shotQueue_.empty() && processedCount < maxBatchSize) {
        auto shot = shotQueue_.front();
        shotQueue_.pop();
        
        if (database_) {
            database_->storeShotData(shot);
            analyzeShotPattern(shot);
        }
        
        processedCount++;
    }
    
    // Process behavior data
    while (!behaviorQueue_.empty() && processedCount < maxBatchSize) {
        auto behavior = behaviorQueue_.front();
        behaviorQueue_.pop();
        
        if (database_) {
            database_->storeBehaviorData(behavior);
            updateBehaviorModel(behavior);
        }
        
        processedCount++;
    }
    
    // Process session data
    while (!pendingData_.empty() && processedCount < maxBatchSize) {
        auto sessionData = pendingData_.front();
        pendingData_.pop();
        
        if (database_) {
            database_->storeSessionData(sessionData);
        }
        
        processedCount++;
    }
    
    // Update queue length metric
    size_t totalQueueLength = shotQueue_.size() + behaviorQueue_.size() + pendingData_.size();
    metrics_.queueLength.store(totalQueueLength);
}

void DataCollectionEngine::analyzeShotPattern(const ShotOutcomeData& shot) {
    // Analyze shot patterns for learning
    // This is where machine learning pattern recognition would go
    
    // For now, simple difficulty calculation
    float difficulty = calculateShotDifficulty(shot);
    
    // Log interesting patterns
    if (difficulty > 0.8f && shot.successful) {
        std::cout << "Difficult shot succeeded: Player " << shot.playerId 
                  << " made a " << difficulty << " difficulty shot" << std::endl;
    }
}

void DataCollectionEngine::updateBehaviorModel(const PlayerBehaviorData& behavior) {
    // Update player behavior models
    // This would feed into adaptive coaching systems
    
    if (behavior.hesitationDetected && behavior.confidenceLevel < 0.3f) {
        std::cout << "Low confidence detected for player " << behavior.playerId 
                  << " - potential coaching opportunity" << std::endl;
    }
}

float DataCollectionEngine::calculateShotDifficulty(const ShotOutcomeData& shot) {
    // Calculate shot difficulty based on various factors
    float distance = cv::norm(shot.targetPosition - shot.shotPosition);
    float angle = std::abs(shot.shotAngle);
    float speed = shot.shotSpeed;
    
    // Normalize and combine factors
    float distanceFactor = std::min(distance / 500.0f, 1.0f); // Max distance 500 pixels
    float angleFactor = std::min(angle / (M_PI / 2), 1.0f);   // Max angle 90 degrees
    float speedFactor = std::min(speed / 1000.0f, 1.0f);     // Max speed 1000 px/s
    
    return (distanceFactor + angleFactor + speedFactor) / 3.0f;
}

DataCollectionEngine::ShotOutcomeData::ShotType 
DataCollectionEngine::classifyShotType(const ShotOutcomeData& shot) {
    float angle = std::abs(shot.shotAngle);
    
    if (angle < 0.1f) return ShotOutcomeData::Straight;
    if (angle < M_PI / 4) return ShotOutcomeData::Cut;
    
    return ShotOutcomeData::Bank; // Default classification
}

bool DataCollectionEngine::isGpuBusy() {
    // Check if GPU pipeline is under heavy load
    if (isolation_) {
        auto metrics = isolation_->getMetrics();
        return metrics.avgGpuLatency.load() > 10.0; // GPU latency > 10ms indicates heavy load
    }
    return false;
}

void DataCollectionEngine::setCpuAffinity(const std::vector<int>& cores) {
#ifdef _WIN32
    if (cores.empty()) return;
    
    HANDLE thread = GetCurrentThread();
    DWORD_PTR affinity = 0;
    for (int core : cores) {
        affinity |= (1ULL << core);
    }
    SetThreadAffinityMask(thread, affinity);
    SetThreadPriority(thread, THREAD_PRIORITY_BELOW_NORMAL);
    
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

void DataCollectionEngine::updatePerformanceMetrics() {
    auto currentTime = std::chrono::steady_clock::now();
    auto timeDiff = currentTime - metrics_.lastMetricsUpdate;
    
    if (timeDiff >= std::chrono::seconds(1)) {
        size_t queueLength = metrics_.queueLength.load();
        
        // Check for performance impact
        bool impacted = queueLength > MAX_QUEUE_SIZE * 0.8 || isGpuBusy();
        metrics_.performanceImpacted.store(impacted);
        
        metrics_.lastMetricsUpdate = currentTime;
    }
}

DataCollectionEngine::CollectionMetrics DataCollectionEngine::getMetrics() const {
    return metrics_;
}

void DataCollectionEngine::logPerformanceReport() {
    auto metrics = getMetrics();
    
    std::cout << "\n=== AI Data Collection Performance Report ===" << std::endl;
    std::cout << "Shots recorded: " << metrics.shotsRecorded.load() << std::endl;
    std::cout << "Behavior events: " << metrics.behaviorEventsRecorded.load() << std::endl;
    std::cout << "Queue length: " << metrics.queueLength.load() << std::endl;
    std::cout << "Performance impacted: " << (metrics.performanceImpacted.load() ? "Yes" : "No") << std::endl;
    
    if (database_) {
        std::cout << "Database size: " << (database_->getDatabaseSize() / 1024) << " KB" << std::endl;
    }
    
    std::cout << "=============================================" << std::endl;
}

// Factory Implementation
std::unique_ptr<DataCollectionEngine> 
DataCollectionFactory::createOptimized(ProcessingIsolation* isolation) {
    auto engine = std::make_unique<DataCollectionEngine>(isolation);
    engine->setCpuCores({4, 5}); // Moderate CPU usage
    engine->setThreadPriority(-1); // Below normal priority
    return engine;
}

std::unique_ptr<DataCollectionEngine> 
DataCollectionFactory::createLowImpact(ProcessingIsolation* isolation) {
    auto engine = std::make_unique<DataCollectionEngine>(isolation);
    engine->setCpuCores({6, 7}); // Lower priority cores
    engine->setThreadPriority(-2); // Low priority
    engine->setMaxQueueSize(100); // Smaller queue
    return engine;
}

std::unique_ptr<DataCollectionEngine> 
DataCollectionFactory::createHighThroughput(ProcessingIsolation* isolation) {
    auto engine = std::make_unique<DataCollectionEngine>(isolation);
    engine->setCpuCores({4, 5, 6, 7}); // More CPU cores
    engine->setThreadPriority(0); // Normal priority
    engine->setMaxQueueSize(2000); // Larger queue
    return engine;
}

} // namespace learning
} // namespace ai
} // namespace pv