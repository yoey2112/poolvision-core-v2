#include "ShotSegmentation.hpp"
#include <algorithm>
#include <cmath>
#include <chrono>
#include <map>
#include <atomic>

namespace pv {
namespace modern {

// BallState implementation
void ShotSegmentation::BallState::updatePosition(const cv::Point2f& pos, double ts) {
    position = pos;
    timestamp = ts;
    
    positionHistory.push_back(pos);
    timestampHistory.push_back(ts);
    
    // Calculate speed
    if (positionHistory.size() >= 2 && timestampHistory.size() >= 2) {
        const auto& prevPos = positionHistory[positionHistory.size() - 2];
        double deltaTime = timestampHistory.back() - timestampHistory[timestampHistory.size() - 2];
        
        if (deltaTime > 0) {
            float distance = cv::norm(pos - prevPos);
            float speed = static_cast<float>(distance / deltaTime * 1000.0); // pixels per second
            speedHistory.push_back(speed);
            
            // Update velocity
            velocity.x = static_cast<float>((pos.x - prevPos.x) / deltaTime * 1000.0);
            velocity.y = static_cast<float>((pos.y - prevPos.y) / deltaTime * 1000.0);
        }
    }
    
    // Keep history size bounded
    while (positionHistory.size() > 30) {
        positionHistory.pop_front();
        timestampHistory.pop_front();
    }
    while (speedHistory.size() > 30) {
        speedHistory.pop_front();
    }
}

ShotSegmentation::MotionAnalysis ShotSegmentation::BallState::getMotionAnalysis() const {
    MotionAnalysis analysis;
    analysis.velocity = velocity;
    analysis.speed = getCurrentSpeed();
    
    // Calculate acceleration
    analysis.acceleration = 0.0f;
    if (speedHistory.size() >= 3) {
        float recentSpeed = speedHistory.back();
        float prevSpeed = speedHistory[speedHistory.size() - 2];
        double deltaTime = 0.033; // Assume ~30 FPS
        analysis.acceleration = (recentSpeed - prevSpeed) / static_cast<float>(deltaTime);
    }
    
    // Calculate kinematic energy (proportional to speed^2)
    analysis.kinematicEnergy = analysis.speed * analysis.speed;
    
    // Motion state classification
    analysis.isMoving = analysis.speed > 5.0f;
    analysis.isDecelerating = analysis.acceleration < -10.0f;
    analysis.isStationary = analysis.speed < 2.0f;
    
    return analysis;
}

float ShotSegmentation::BallState::getCurrentSpeed() const {
    if (speedHistory.empty()) return 0.0f;
    
    // Use weighted average of recent speeds for stability
    float totalWeight = 0.0f;
    float weightedSum = 0.0f;
    int count = std::min(5, static_cast<int>(speedHistory.size()));
    
    for (int i = 0; i < count; ++i) {
        float weight = static_cast<float>(i + 1);
        weightedSum += speedHistory[speedHistory.size() - 1 - i] * weight;
        totalWeight += weight;
    }
    
    return totalWeight > 0 ? weightedSum / totalWeight : 0.0f;
}

bool ShotSegmentation::BallState::hasSignificantMotion() const {
    return getCurrentSpeed() > 5.0f;
}

// ShotSegmentation implementation
ShotSegmentation::ShotSegmentation(const Config& config)
    : config_(config), shotInProgress_(false), shotStartTime_(0), 
      lastMotionTime_(0), movingBallCount_(0), cueBallInMotion_(false),
      framesProcessed_(0), avgProcessingTime_(0.0) {
}

bool ShotSegmentation::processTracks(const std::vector<Track>& tracks, double timestamp) {
    auto startTime = std::chrono::high_resolution_clock::now();
    
    // Update ball states with new tracking data
    updateBallStates(tracks, timestamp);
    
    // Count moving balls
    movingBallCount_ = 0;
    cueBallInMotion_ = false;
    
    for (const auto& [ballId, state] : ballStates_) {
        if (state.hasSignificantMotion()) {
            movingBallCount_++;
            if (ballId == 0) { // Cue ball
                cueBallInMotion_ = true;
            }
        }
    }
    
    // Shot detection logic
    bool wasInProgress = shotInProgress_;
    
    if (!shotInProgress_ && movingBallCount_ > 0) {
        // Start new shot
        startShot(timestamp);
    } else if (shotInProgress_) {
        // Update current shot
        updateCurrentShot(timestamp);
        
        // Check if shot should end
        double timeSinceLastMotion = timestamp - lastMotionTime_;
        bool allStationary = areAllBallsStationary(config_.stationaryTimeMs);
        bool shotTimeout = (timestamp - shotStartTime_) > config_.shotTimeoutMs;
        
        if (allStationary || shotTimeout) {
            endShot(timestamp);
        }
    }
    
    // Update timing
    if (movingBallCount_ > 0) {
        lastMotionTime_ = static_cast<uint64_t>(timestamp);
    }
    
    // Detect collisions during active shots
    if (shotInProgress_) {
        detectCollisions(timestamp);
    }
    
    framesProcessed_++;
    
    auto endTime = std::chrono::high_resolution_clock::now();
    double processingTime = std::chrono::duration<double, std::milli>(endTime - startTime).count();
    updateMetrics(processingTime);
    
    return shotInProgress_ != wasInProgress; // Return true if shot state changed
}

void ShotSegmentation::startShot(double timestamp) {
    shotInProgress_ = true;
    shotStartTime_ = static_cast<uint64_t>(timestamp);
    lastMotionTime_ = static_cast<uint64_t>(timestamp);
    
    // Initialize shot event
    currentShot_ = ShotEvent();
    currentShot_.startTimestamp = static_cast<uint64_t>(timestamp);
    currentShot_.cueBallStartPos = getCueBallPosition();
    currentShot_.ballsInMotion.clear();
    
    // Record initial moving balls
    for (const auto& [ballId, state] : ballStates_) {
        if (state.hasSignificantMotion()) {
            currentShot_.ballsInMotion.push_back(ballId);
        }
    }
}

void ShotSegmentation::endShot(double timestamp) {
    if (!shotInProgress_) return;
    
    shotInProgress_ = false;
    currentShot_.endTimestamp = static_cast<uint64_t>(timestamp);
    currentShot_.duration = timestamp - currentShot_.startTimestamp;
    currentShot_.cueBallEndPos = getCueBallPosition();
    
    // Calculate shot statistics
    cv::Point2f startPos = currentShot_.cueBallStartPos;
    cv::Point2f endPos = currentShot_.cueBallEndPos;
    currentShot_.totalDistance = cv::norm(endPos - startPos);
    
    // Find maximum speed during shot
    currentShot_.maxSpeed = 0.0f;
    for (const auto& [ballId, state] : ballStates_) {
        for (float speed : state.speedHistory) {
            currentShot_.maxSpeed = std::max(currentShot_.maxSpeed, speed);
        }
    }
    
    // Classify shot type
    classifyShot();
    
    // Add to recent shots history
    recentShots_.push_back(currentShot_);
    while (recentShots_.size() > 50) {
        recentShots_.pop_front();
    }
}

void ShotSegmentation::updateCurrentShot(double timestamp) {
    if (!shotInProgress_) return;
    
    // Update balls in motion
    currentShot_.ballsInMotion.clear();
    for (const auto& [ballId, state] : ballStates_) {
        if (state.hasSignificantMotion()) {
            currentShot_.ballsInMotion.push_back(ballId);
        }
    }
    
    // Update current duration
    currentShot_.duration = timestamp - currentShot_.startTimestamp;
}

void ShotSegmentation::updateBallStates(const std::vector<Track>& tracks, double timestamp) {
    // Update existing ball states
    for (const auto& track : tracks) {
        ballStates_[track.id].updatePosition(track.c, timestamp);
        ballStates_[track.id].ballId = track.id;
    }
    
    // Remove outdated ball states
    auto it = ballStates_.begin();
    while (it != ballStates_.end()) {
        if (timestamp - it->second.timestamp > 5000.0) { // 5 second timeout
            it = ballStates_.erase(it);
        } else {
            ++it;
        }
    }
}

void ShotSegmentation::detectCollisions(double timestamp) {
    if (ballStates_.size() < 2) return;
    
    // Check all pairs of balls for potential collisions
    auto ballList = std::vector<std::pair<int, BallState*>>();
    for (auto& [ballId, state] : ballStates_) {
        ballList.emplace_back(ballId, &state);
    }
    
    for (size_t i = 0; i < ballList.size(); ++i) {
        for (size_t j = i + 1; j < ballList.size(); ++j) {
            auto& [ball1Id, ball1] = ballList[i];
            auto& [ball2Id, ball2] = ballList[j];
            
            float distance = cv::norm(ball1->position - ball2->position);
            if (distance < config_.collisionDistanceThreshold) {
                // Potential collision detected
                currentShot_.hasCollisions = true;
            }
        }
    }
}

void ShotSegmentation::classifyShot() {
    // Basic shot classification based on motion analysis
    if (currentShot_.ballsInMotion.empty()) {
        currentShot_.shotType = ShotEvent::Unknown;
        return;
    }
    
    // Check if cue ball was involved
    bool cueBallInvolved = std::find(currentShot_.ballsInMotion.begin(),
                                   currentShot_.ballsInMotion.end(), 0) 
                          != currentShot_.ballsInMotion.end();
    
    if (!cueBallInvolved) {
        currentShot_.shotType = ShotEvent::Unknown;
        return;
    }
    
    // Classify based on speed and distance
    if (currentShot_.maxSpeed > 1000.0f) {
        currentShot_.shotType = ShotEvent::Break;
    } else if (currentShot_.totalDistance < 100.0f) {
        currentShot_.shotType = ShotEvent::SafetyShot;
    } else {
        currentShot_.shotType = ShotEvent::Standard;
    }
}

ShotSegmentation::MotionAnalysis ShotSegmentation::getBallMotion(int ballId) const {
    auto it = ballStates_.find(ballId);
    if (it != ballStates_.end()) {
        return it->second.getMotionAnalysis();
    }
    return MotionAnalysis{};
}

std::vector<int> ShotSegmentation::getMovingBalls() const {
    std::vector<int> movingBalls;
    for (const auto& [ballId, state] : ballStates_) {
        if (state.hasSignificantMotion()) {
            movingBalls.push_back(ballId);
        }
    }
    return movingBalls;
}

bool ShotSegmentation::isTableStationary() const {
    return movingBallCount_ == 0;
}

std::vector<ShotSegmentation::ShotEvent> ShotSegmentation::getRecentShots(int count) const {
    std::vector<ShotEvent> shots;
    int start = std::max(0, static_cast<int>(recentShots_.size()) - count);
    for (int i = start; i < static_cast<int>(recentShots_.size()); ++i) {
        shots.push_back(recentShots_[i]);
    }
    return shots;
}

uint64_t ShotSegmentation::getCurrentTimeMs() const {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count();
}

bool ShotSegmentation::areAllBallsStationary(double stationaryDurationMs) const {
    if (ballStates_.empty()) return true;
    
    for (const auto& [ballId, state] : ballStates_) {
        if (state.hasSignificantMotion()) {
            return false;
        }
    }
    
    // If duration check is required
    if (stationaryDurationMs > 0.0) {
        double currentTime = getCurrentTimeMs();
        return (currentTime - lastMotionTime_) >= stationaryDurationMs;
    }
    
    return true;
}

cv::Point2f ShotSegmentation::getCueBallPosition() const {
    auto it = ballStates_.find(0); // Assume cue ball ID is 0
    if (it != ballStates_.end()) {
        return it->second.position;
    }
    return cv::Point2f(-1, -1); // Invalid position
}

void ShotSegmentation::updateMetrics(double processingTime) {
    double currentAvg = avgProcessingTime_;
    double newAvg = currentAvg * 0.95 + processingTime * 0.05; // Exponential moving average
    avgProcessingTime_ = newAvg;
}

// PoolRulesEngine implementation
PoolRulesEngine::PoolRulesEngine(GameType type, ShotSegmentation* segmentation)
    : gameType_(type), shotSegmentation_(segmentation), isBreakShot_(true), targetBall_(-1) {
}

PoolRulesEngine::RuleValidationResult PoolRulesEngine::validateShot(const ShotSegmentation::ShotEvent& shot) {
    RuleValidationResult result;
    
    switch (gameType_) {
        case GameType::EightBall:
            validateEightBallRules(shot, result);
            break;
        case GameType::NineBall:
            validateNineBallRules(shot, result);
            break;
        default:
            result.isLegalShot = true; // Default to legal for unsupported games
            break;
    }
    
    return result;
}

bool PoolRulesEngine::validateEightBallRules(const ShotSegmentation::ShotEvent& shot, RuleValidationResult& result) {
    result.legalTargets = getEightBallTargets();
    
    // Basic validation - no balls potted means check for contact
    if (shot.ballsPotted.empty()) {
        // Must make contact with target ball
        if (targetBall_ != -1) {
            bool hitTarget = std::find(shot.ballsContacted.begin(), 
                                     shot.ballsContacted.end(), targetBall_) 
                           != shot.ballsContacted.end();
            if (!hitTarget) {
                result.isLegalShot = false;
                result.foulType = FoulType::WrongBallFirst;
                result.foulDescription = "Failed to contact target ball first";
                return false;
            }
        }
        
        // Must have cushion contact after hitting target
        if (!result.hadCushionContact && result.requiresCushionContact) {
            result.isLegalShot = false;
            result.foulType = FoulType::NoCushionAfterContact;
            result.foulDescription = "No cushion contact after ball contact";
            return false;
        }
    }
    
    return true;
}

bool PoolRulesEngine::validateNineBallRules(const ShotSegmentation::ShotEvent& shot, RuleValidationResult& result) {
    result.legalTargets = getNineBallTargets();
    
    // Must hit lowest numbered ball first
    if (!shot.ballsContacted.empty()) {
        auto lowestBallIt = std::min_element(ballsOnTable_.begin(), ballsOnTable_.end());
        if (lowestBallIt != ballsOnTable_.end()) {
            int lowestBall = lowestBallIt->first;
            if (shot.ballsContacted[0] != lowestBall) {
                result.isLegalShot = false;
                result.foulType = FoulType::WrongBallFirst;
                result.foulDescription = "Must contact lowest numbered ball first";
                return false;
            }
        }
    }
    
    return true;
}

std::vector<int> PoolRulesEngine::getEightBallTargets() const {
    // Simplified - return all remaining balls except 8-ball
    std::vector<int> targets;
    for (const auto& [ballId, onTable] : ballsOnTable_) {
        if (onTable && ballId != 0 && ballId != 8) {
            targets.push_back(ballId);
        }
    }
    return targets;
}

std::vector<int> PoolRulesEngine::getNineBallTargets() const {
    // Return lowest numbered ball
    std::vector<int> targets;
    for (int i = 1; i <= 9; ++i) {
        auto it = ballsOnTable_.find(i);
        if (it != ballsOnTable_.end() && it->second) {
            targets.push_back(i);
            break; // Only need the lowest
        }
    }
    return targets;
}

void PoolRulesEngine::updateGameState(const std::vector<Track>& tracks) {
    // Update ball positions and table state
    for (const auto& track : tracks) {
        ballPositions_[track.id] = track.c;
        ballsOnTable_[track.id] = true;
    }
}

std::vector<int> PoolRulesEngine::getLegalTargets() const {
    switch (gameType_) {
        case GameType::EightBall:
            return getEightBallTargets();
        case GameType::NineBall:
            return getNineBallTargets();
        default:
            return {};
    }
}

std::string PoolRulesEngine::getGameStateDescription() const {
    std::string desc = "Game Type: ";
    switch (gameType_) {
        case GameType::EightBall: desc += "8-Ball"; break;
        case GameType::NineBall: desc += "9-Ball"; break;
        case GameType::TenBall: desc += "10-Ball"; break;
        case GameType::StraightPool: desc += "Straight Pool"; break;
    }
    
    if (isBreakShot_) {
        desc += " (Break Shot)";
    }
    
    return desc;
}

// GameLogicManager implementation
GameLogicManager::GameLogicManager(GameState* legacyGame, const Config& config)
    : config_(config), legacyGameState_(legacyGame) {
    
    shotSegmentation_ = std::make_unique<ShotSegmentation>(config.shotConfig);
    rulesEngine_ = std::make_unique<PoolRulesEngine>(config.gameType, shotSegmentation_.get());
}

void GameLogicManager::processTracks(const std::vector<Track>& tracks, double timestamp) {
    auto startTime = std::chrono::high_resolution_clock::now();
    
    // Process shot segmentation
    bool shotStateChanged = shotSegmentation_->processTracks(tracks, timestamp);
    
    // Update rules engine
    rulesEngine_->updateGameState(tracks);
    
    // If shot ended, validate it
    if (shotStateChanged && !shotSegmentation_->isShotInProgress()) {
        auto recentShots = shotSegmentation_->getRecentShots(1);
        if (!recentShots.empty()) {
            auto validation = rulesEngine_->validateShot(recentShots.back());
            shotsProcessed_.fetch_add(1);
            
            // Synchronize with legacy game state if needed
            synchronizeWithLegacyGameState();
        }
    }
    
    auto endTime = std::chrono::high_resolution_clock::now();
    double validationTime = std::chrono::duration<double, std::milli>(endTime - startTime).count();
    updateMetrics(validationTime);
}

bool GameLogicManager::isShotInProgress() const {
    return shotSegmentation_->isShotInProgress();
}

ShotSegmentation::ShotEvent GameLogicManager::getCurrentShot() const {
    return shotSegmentation_->getCurrentShot();
}

PoolRulesEngine::RuleValidationResult GameLogicManager::getLastShotValidation() const {
    auto recentShots = shotSegmentation_->getRecentShots(1);
    if (!recentShots.empty()) {
        return rulesEngine_->validateShot(recentShots.back());
    }
    return PoolRulesEngine::RuleValidationResult{};
}

std::vector<int> GameLogicManager::getLegalTargets() const {
    return rulesEngine_->getLegalTargets();
}

std::string GameLogicManager::getAdvancedGameState() const {
    std::string state = rulesEngine_->getGameStateDescription();
    
    if (shotSegmentation_->isShotInProgress()) {
        auto movingBalls = shotSegmentation_->getMovingBalls();
        state += " | Balls in motion: " + std::to_string(movingBalls.size());
    } else {
        state += " | Table stationary";
    }
    
    return state;
}

std::vector<ShotSegmentation::ShotEvent> GameLogicManager::getShotHistory(int count) const {
    return shotSegmentation_->getRecentShots(count);
}

void GameLogicManager::updateMetrics(double validationTime) {
    double currentAvg = avgValidationTime_.load();
    double newAvg = currentAvg * 0.9 + validationTime * 0.1;
    avgValidationTime_.store(newAvg);
}

void GameLogicManager::synchronizeWithLegacyGameState() {
    // This would sync advanced game logic with the existing GameState
    // For now, just a placeholder for future integration
}

} // namespace modern
} // namespace pv