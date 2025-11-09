#include "DrillSystem.hpp"
#include "DrillLibrary.hpp"
#include "../util/Config.hpp"
#include <cmath>
#include <algorithm>
#include <sstream>

namespace pv {

DrillSystem::DrillSystem(Database& database, GameState& gameState)
    : database_(database), gameState_(gameState),
      drillLibrary_(std::make_unique<DrillLibrary>()) {
}

bool DrillSystem::startDrill(int drillId, int playerId) {
    // End any current drill
    if (isDrillActive() || isDrillPaused()) {
        endDrill();
    }
    
    // Get drill definition from library
    auto drill = drillLibrary_->getDrill(drillId);
    if (!drill) {
        return false;
    }
    
    // Initialize new session
    currentSession_ = DrillSession();
    currentSession_.drillId = drillId;
    currentSession_.playerId = playerId;
    currentSession_.startTime = std::chrono::steady_clock::now();
    currentSession_.state = DrillState::Setup;
    currentSession_.currentAttempt = 1;
    
    // Setup initial ball positions
    if (!setupDrillBalls(*drill)) {
        return false;
    }
    
    changeState(DrillState::InProgress);
    return true;
}

void DrillSystem::endDrill() {
    if (currentSession_.state == DrillState::NotStarted) {
        return;
    }
    
    // Mark session as completed
    currentSession_.sessionCompleted = true;
    updateSessionStats();
    
    // Save to database
    saveSession();
    
    // Notify completion
    if (sessionCallback_) {
        sessionCallback_(currentSession_);
    }
    
    changeState(DrillState::Completed);
}

void DrillSystem::pauseDrill() {
    if (currentSession_.state == DrillState::InProgress) {
        changeState(DrillState::Paused);
    }
}

void DrillSystem::resumeDrill() {
    if (currentSession_.state == DrillState::Paused) {
        changeState(DrillState::InProgress);
    }
}

void DrillSystem::processShot(const cv::Point2f& shotPosition, 
                             const std::vector<Ball>& ballsBefore,
                             const std::vector<Ball>& ballsAfter) {
    if (!isDrillActive()) {
        return;
    }
    
    auto drill = getCurrentDrill();
    if (!drill) {
        return;
    }
    
    // Create attempt record
    DrillAttempt attempt;
    attempt.attemptNumber = currentSession_.currentAttempt;
    attempt.shotPosition = shotPosition;
    attempt.timeTaken = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - currentSession_.startTime).count();
    
    // Find target and actual result
    if (!drill->targets.empty()) {
        attempt.targetPosition = drill->targets[0];  // Use first target for now
        
        // Find cue ball final position as actual result
        for (const auto& ball : ballsAfter) {
            if (ball.label == 0) {  // Cue ball
                attempt.actualResult = cv::Point2f(ball.c.x, ball.c.y);
                break;
            }
        }
        
        // Calculate accuracy
        attempt.accuracy = calculateShotAccuracy(attempt.targetPosition, attempt.actualResult);
        attempt.success = evaluateSuccess(attempt);
    }
    
    // Generate feedback
    attempt.feedback = generateFeedback(attempt);
    
    // Record attempt
    recordAttempt(attempt);
    
    // Notify UI
    if (attemptCallback_) {
        attemptCallback_(attempt);
    }
    
    // Check if drill is complete
    if (attempt.success || currentSession_.currentAttempt >= drill->maxAttempts) {
        endDrill();
    } else {
        nextAttempt();
    }
}

void DrillSystem::resetDrill() {
    if (currentSession_.state == DrillState::NotStarted) {
        return;
    }
    
    int drillId = currentSession_.drillId;
    int playerId = currentSession_.playerId;
    
    endDrill();
    startDrill(drillId, playerId);
}

void DrillSystem::nextAttempt() {
    if (!isDrillActive()) {
        return;
    }
    
    auto drill = getCurrentDrill();
    if (!drill) {
        return;
    }
    
    currentSession_.currentAttempt++;
    
    // Reset ball positions for next attempt
    setupDrillBalls(*drill);
    
    // Update session start time for this attempt
    currentSession_.startTime = std::chrono::steady_clock::now();
}

const DrillSystem::Drill* DrillSystem::getCurrentDrill() const {
    if (currentSession_.drillId == 0) {
        return nullptr;
    }
    return drillLibrary_->getDrill(currentSession_.drillId);
}

DrillSystem::DrillStats DrillSystem::getDrillStats(int drillId, int playerId) const {
    DrillStats stats;
    
    // Query database for drill history
    // This would typically involve SQL queries to get session data
    // For now, providing basic structure
    
    return stats;
}

std::vector<DrillSystem::DrillStats> DrillSystem::getPlayerDrillStats(int playerId) const {
    std::vector<DrillStats> stats;
    
    // Get all drills the player has attempted
    auto drillIds = drillLibrary_->getAllDrillIds();
    
    for (int drillId : drillIds) {
        auto drillStats = getDrillStats(drillId, playerId);
        if (drillStats.totalAttempts > 0) {
            stats.push_back(drillStats);
        }
    }
    
    return stats;
}

std::vector<DrillSystem::DrillSession> DrillSystem::getRecentSessions(int playerId, int limit) const {
    std::vector<DrillSession> sessions;
    
    // Query database for recent sessions
    // Implementation would involve SQL queries
    
    return sessions;
}

std::vector<double> DrillSystem::getImprovementTrend(int drillId, int playerId, int sessionCount) const {
    std::vector<double> trend;
    
    // Get recent session accuracies to show improvement trend
    auto sessions = getRecentSessions(playerId, sessionCount);
    
    for (const auto& session : sessions) {
        if (session.drillId == drillId) {
            trend.push_back(session.averageAccuracy);
        }
    }
    
    return trend;
}

double DrillSystem::calculateShotAccuracy(const cv::Point2f& target, const cv::Point2f& actual) const {
    double distance = calculateDistance(target, actual);
    
    // Convert distance to accuracy (closer = higher accuracy)
    // Using exponential decay function
    double maxDistance = 100.0;  // Maximum meaningful distance in pixels
    double accuracy = std::exp(-distance / (maxDistance / 3.0));
    
    return std::max(0.0, std::min(1.0, accuracy));
}

std::string DrillSystem::generateFeedback(const DrillAttempt& attempt) const {
    std::stringstream feedback;
    
    if (attempt.success) {
        feedback << "Excellent shot! ";
        if (attempt.accuracy > 0.9) {
            feedback << "Perfect accuracy!";
        } else if (attempt.accuracy > 0.7) {
            feedback << "Great precision.";
        } else {
            feedback << "Good execution.";
        }
    } else {
        feedback << "Try again. ";
        if (attempt.accuracy < 0.3) {
            feedback << "Focus on your aim and alignment.";
        } else if (attempt.accuracy < 0.6) {
            feedback << "Close! Adjust your power slightly.";
        } else {
            feedback << "Very close! Minor adjustment needed.";
        }
    }
    
    return feedback.str();
}

bool DrillSystem::evaluateSuccess(const DrillAttempt& attempt) const {
    auto drill = getCurrentDrill();
    if (!drill) {
        return false;
    }
    
    return attempt.accuracy >= drill->successThreshold;
}

std::string DrillSystem::getPerformanceHint(const DrillStats& stats) const {
    if (stats.totalAttempts < 5) {
        return "Keep practicing to build consistency.";
    }
    
    if (stats.successRate < 0.3) {
        return "Focus on basic technique and take your time.";
    } else if (stats.successRate < 0.6) {
        return "Good progress! Work on consistency.";
    } else if (stats.successRate < 0.8) {
        return "Excellent improvement! Fine-tune your precision.";
    } else {
        return "Outstanding performance! Try harder drills.";
    }
}

void DrillSystem::saveSession() {
    // Save current session to database
    // Implementation would involve SQL INSERT statements
}

bool DrillSystem::loadSession(int sessionId) {
    // Load session from database
    // Implementation would involve SQL SELECT statements
    return false;
}

bool DrillSystem::deleteSession(int sessionId) {
    // Delete session from database
    // Implementation would involve SQL DELETE statements
    return false;
}

std::string DrillSystem::difficultyToString(Difficulty difficulty) {
    switch (difficulty) {
        case Difficulty::Beginner: return "Beginner";
        case Difficulty::Intermediate: return "Intermediate";
        case Difficulty::Advanced: return "Advanced";
        case Difficulty::Professional: return "Professional";
        case Difficulty::Expert: return "Expert";
        default: return "Unknown";
    }
}

std::string DrillSystem::categoryToString(Category category) {
    switch (category) {
        case Category::Breaking: return "Breaking";
        case Category::CutShots: return "Cut Shots";
        case Category::BankShots: return "Bank Shots";
        case Category::Combinations: return "Combinations";
        case Category::PositionPlay: return "Position Play";
        case Category::SpeedControl: return "Speed Control";
        case Category::RailShots: return "Rail Shots";
        case Category::RunOut: return "Run Out";
        case Category::Safety: return "Safety";
        case Category::Specialty: return "Specialty";
        default: return "Unknown";
    }
}

std::string DrillSystem::stateToString(DrillState state) {
    switch (state) {
        case DrillState::NotStarted: return "Not Started";
        case DrillState::Setup: return "Setup";
        case DrillState::InProgress: return "In Progress";
        case DrillState::Paused: return "Paused";
        case DrillState::Completed: return "Completed";
        case DrillState::Failed: return "Failed";
        default: return "Unknown";
    }
}

void DrillSystem::changeState(DrillState newState) {
    currentSession_.state = newState;
    
    if (stateChangeCallback_) {
        stateChangeCallback_(newState);
    }
}

void DrillSystem::updateSessionStats() {
    if (currentSession_.attempts.empty()) {
        return;
    }
    
    double totalAccuracy = 0.0;
    int successCount = 0;
    
    for (const auto& attempt : currentSession_.attempts) {
        totalAccuracy += attempt.accuracy;
        if (attempt.success) {
            successCount++;
        }
        
        if (attempt.accuracy > currentSession_.bestAccuracy) {
            currentSession_.bestAccuracy = attempt.accuracy;
        }
    }
    
    currentSession_.averageAccuracy = totalAccuracy / currentSession_.attempts.size();
}

double DrillSystem::calculateDistance(const cv::Point2f& p1, const cv::Point2f& p2) const {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    return std::sqrt(dx * dx + dy * dy);
}

void DrillSystem::recordAttempt(const DrillAttempt& attempt) {
    currentSession_.attempts.push_back(attempt);
    updateSessionStats();
}

bool DrillSystem::setupDrillBalls(const Drill& drill) {
    // This would interface with the game state to position balls
    // For now, just return success
    // In real implementation, would use gameState_.setupBalls(drill.initialSetup);
    return true;
}

} // namespace pv
