#include "SimpleAILearningSystem.hpp"
#include <algorithm>
#include <sstream>
#include <iomanip>

namespace pv {
namespace ai {
namespace learning {

SimpleAILearningSystem::SimpleAILearningSystem(const Config& config) 
    : config_(config), initialized_(false), active_(false) {
}

SimpleAILearningSystem::~SimpleAILearningSystem() {
    shutdown();
}

bool SimpleAILearningSystem::initialize() {
    std::lock_guard<std::mutex> lock(systemMutex_);
    
    if (initialized_) {
        return true;
    }
    
    try {
        // Create data collection engine
        dataEngine_ = std::make_unique<SimpleDataCollectionEngine>();
        
        initialized_ = true;
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

void SimpleAILearningSystem::start() {
    if (!initialized_) {
        if (!initialize()) {
            return;
        }
    }
    
    active_ = true;
    if (dataEngine_) {
        dataEngine_->start();
    }
}

void SimpleAILearningSystem::stop() {
    active_ = false;
    if (dataEngine_) {
        dataEngine_->stop();
    }
}

void SimpleAILearningSystem::shutdown() {
    stop();
    
    std::lock_guard<std::mutex> lock(systemMutex_);
    dataEngine_.reset();
    initialized_ = false;
}

void SimpleAILearningSystem::analyzeShot(int playerId, int shotType, bool successful, float accuracy) {
    if (!active_ || !dataEngine_) {
        return;
    }
    
    // Record shot data
    SimpleDataCollectionEngine::ShotData shot;
    shot.playerId = playerId;
    shot.shotType = shotType;
    shot.successful = successful;
    shot.accuracy = accuracy;
    shot.difficulty = std::min(1.0f - accuracy + 0.2f, 1.0f); // Simple difficulty estimate
    shot.timestamp = std::chrono::steady_clock::now();
    
    dataEngine_->recordShot(shot);
    
    // Update player analysis
    updatePlayerAnalysis(playerId);
}

SimpleAILearningSystem::AnalysisResult SimpleAILearningSystem::getPlayerAnalysis(int playerId) {
    AnalysisResult result;
    
    if (!dataEngine_) {
        return result;
    }
    
    auto stats = dataEngine_->getPlayerStats(playerId);
    if (stats.totalShots == 0) {
        result.recommendation = "Start playing to build your skill profile!";
        return result;
    }
    
    result.confidence = stats.successRate;
    result.difficulty = 1.0f - stats.avgAccuracy;
    
    // Generate recommendation
    if (stats.successRate > 0.8f) {
        result.recommendation = "Excellent performance! Try more challenging shots.";
    } else if (stats.successRate > 0.6f) {
        result.recommendation = "Good consistency. Focus on precision for improvement.";
    } else {
        result.recommendation = "Work on shot selection and basic technique.";
    }
    
    result.insights = generateInsights(playerId);
    
    return result;
}

std::vector<std::string> SimpleAILearningSystem::getCoachingInsights(int playerId) {
    if (!dataEngine_) {
        return {"AI learning system not available"};
    }
    
    return dataEngine_->getPlayerInsights(playerId);
}

float SimpleAILearningSystem::getPlayerSkillLevel(int playerId) const {
    if (!dataEngine_) {
        return 0.0f;
    }
    
    return dataEngine_->getPlayerSkillLevel(playerId);
}

std::string SimpleAILearningSystem::getPlayerPerformanceSummary(int playerId) const {
    if (!dataEngine_) {
        return "AI learning system not available";
    }
    
    auto stats = dataEngine_->getPlayerStats(playerId);
    if (stats.totalShots == 0) {
        return "No performance data available yet";
    }
    
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(1);
    oss << "Performance Summary:\n";
    oss << "Total Shots: " << stats.totalShots << "\n";
    oss << "Success Rate: " << (stats.successRate * 100) << "%\n";
    oss << "Average Accuracy: " << (stats.avgAccuracy * 100) << "%\n";
    oss << "Skill Level: " << (getPlayerSkillLevel(playerId) * 100) << "%";
    
    return oss.str();
}

void SimpleAILearningSystem::setConfig(const Config& config) {
    std::lock_guard<std::mutex> lock(systemMutex_);
    config_ = config;
}

SimpleAILearningSystem::Config SimpleAILearningSystem::getConfig() const {
    std::lock_guard<std::mutex> lock(systemMutex_);
    return config_;
}

void SimpleAILearningSystem::updatePlayerAnalysis(int playerId) {
    if (!config_.enableAnalytics) {
        return;
    }
    
    // Update skill level
    playerSkillLevels_[playerId] = calculateSkillProgression(playerId);
    
    // Update insights
    playerInsights_[playerId] = generateInsights(playerId);
}

std::vector<std::string> SimpleAILearningSystem::generateInsights(int playerId) {
    std::vector<std::string> insights;
    
    if (!dataEngine_) {
        return insights;
    }
    
    auto stats = dataEngine_->getPlayerStats(playerId);
    auto recentShots = dataEngine_->getRecentShots(playerId, 10);
    
    if (recentShots.empty()) {
        return {"No recent shots to analyze"};
    }
    
    // Analyze recent performance trends
    int recentSuccesses = 0;
    float avgRecentAccuracy = 0.0f;
    
    for (const auto& shot : recentShots) {
        if (shot.successful) recentSuccesses++;
        avgRecentAccuracy += shot.accuracy;
    }
    avgRecentAccuracy /= recentShots.size();
    
    float recentSuccessRate = static_cast<float>(recentSuccesses) / recentShots.size();
    
    // Generate insights based on trends
    if (recentSuccessRate > stats.successRate + 0.1f) {
        insights.push_back("Improving! Your recent performance is trending upward.");
    } else if (recentSuccessRate < stats.successRate - 0.1f) {
        insights.push_back("Consider taking a break or reviewing your technique.");
    }
    
    if (avgRecentAccuracy > 0.8f) {
        insights.push_back("Your shot precision is excellent!");
    } else if (avgRecentAccuracy < 0.5f) {
        insights.push_back("Focus on shot alignment and follow-through.");
    }
    
    if (insights.empty()) {
        insights.push_back("Keep practicing to improve your game!");
    }
    
    return insights;
}

float SimpleAILearningSystem::calculateSkillProgression(int playerId) {
    if (!dataEngine_) {
        return 0.0f;
    }
    
    auto stats = dataEngine_->getPlayerStats(playerId);
    if (stats.totalShots < 10) {
        return 0.0f; // Not enough data
    }
    
    // Simple skill progression calculation
    float baseSkill = stats.successRate * 0.6f + stats.avgAccuracy * 0.4f;
    float experienceBonus = std::min(stats.totalShots / 1000.0f, 0.2f);
    
    return std::min(baseSkill + experienceBonus, 1.0f);
}

} // namespace learning
} // namespace ai
} // namespace pv