#include "SimpleDataCollectionEngine.hpp"
#include <algorithm>
#include <numeric>

namespace pv {
namespace ai {
namespace learning {

SimpleDataCollectionEngine::SimpleDataCollectionEngine() : active_(false) {
}

SimpleDataCollectionEngine::~SimpleDataCollectionEngine() {
    stop();
}

void SimpleDataCollectionEngine::start() {
    active_ = true;
}

void SimpleDataCollectionEngine::stop() {
    active_ = false;
}

void SimpleDataCollectionEngine::recordShot(const ShotData& shot) {
    if (!active_) return;
    
    std::lock_guard<std::mutex> lock(dataMutex_);
    
    // Update player statistics
    updatePlayerStats(shot.playerId, shot);
    
    // Add to player history (keep last 100 shots)
    auto& history = playerHistory_[shot.playerId];
    history.push_back(shot);
    if (history.size() > 100) {
        history.erase(history.begin());
    }
}

SimpleDataCollectionEngine::PlayerStats SimpleDataCollectionEngine::getPlayerStats(int playerId) const {
    std::lock_guard<std::mutex> lock(dataMutex_);
    auto it = playerStats_.find(playerId);
    return (it != playerStats_.end()) ? it->second : PlayerStats{};
}

std::vector<SimpleDataCollectionEngine::ShotData> SimpleDataCollectionEngine::getRecentShots(int playerId, int count) const {
    std::lock_guard<std::mutex> lock(dataMutex_);
    
    auto it = playerHistory_.find(playerId);
    if (it == playerHistory_.end()) {
        return {};
    }
    
    const auto& history = it->second;
    if (history.size() <= static_cast<size_t>(count)) {
        return history;
    }
    
    return std::vector<ShotData>(history.end() - count, history.end());
}

float SimpleDataCollectionEngine::getPlayerSkillLevel(int playerId) const {
    auto stats = getPlayerStats(playerId);
    
    if (stats.totalShots == 0) {
        return 0.0f;
    }
    
    // Simple skill calculation based on success rate and accuracy
    float skillLevel = (stats.successRate * 0.7f) + (stats.avgAccuracy * 0.3f);
    return std::min(1.0f, skillLevel);
}

std::vector<std::string> SimpleDataCollectionEngine::getPlayerInsights(int playerId) const {
    std::vector<std::string> insights;
    auto stats = getPlayerStats(playerId);
    
    if (stats.totalShots < 5) {
        insights.push_back("Keep practicing to build your skill profile!");
        return insights;
    }
    
    if (stats.successRate > 0.8f) {
        insights.push_back("Excellent shot success rate!");
    } else if (stats.successRate > 0.6f) {
        insights.push_back("Good consistency, keep it up!");
    } else {
        insights.push_back("Focus on shot selection to improve success rate.");
    }
    
    if (stats.avgAccuracy > 0.8f) {
        insights.push_back("Great precision in your shots!");
    } else if (stats.avgAccuracy > 0.6f) {
        insights.push_back("Good accuracy, room for improvement.");
    } else {
        insights.push_back("Work on shot alignment and technique.");
    }
    
    return insights;
}

void SimpleDataCollectionEngine::updatePlayerStats(int playerId, const ShotData& shot) {
    auto& stats = playerStats_[playerId];
    
    stats.totalShots++;
    if (shot.successful) {
        stats.successfulShots++;
    }
    
    // Update success rate
    stats.successRate = static_cast<float>(stats.successfulShots) / stats.totalShots;
    
    // Update average accuracy using exponential moving average
    float alpha = 0.1f;
    stats.avgAccuracy = alpha * shot.accuracy + (1.0f - alpha) * stats.avgAccuracy;
    
    stats.lastPlay = shot.timestamp;
}

} // namespace learning
} // namespace ai
} // namespace pv