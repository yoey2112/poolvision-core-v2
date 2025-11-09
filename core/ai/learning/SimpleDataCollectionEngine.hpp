#pragma once

#include <vector>
#include <map>
#include <string>
#include <memory>
#include <chrono>
#include <atomic>
#include <mutex>

namespace pv {
namespace ai {
namespace learning {

/**
 * Simplified Data Collection Engine for Pool Vision AI Learning System
 * 
 * A streamlined implementation focused on essential functionality:
 * - Basic shot data collection and storage
 * - Simple metrics tracking
 * - Minimal overhead for real-time operation
 */
class SimpleDataCollectionEngine {
public:
    // Simplified shot outcome data
    struct ShotData {
        int playerId;
        int shotType;           // 0-6 for different shot types
        bool successful;
        float accuracy;         // 0.0 to 1.0
        float difficulty;       // 0.0 to 1.0
        std::chrono::time_point<std::chrono::steady_clock> timestamp;
    };
    
    // Basic player statistics
    struct PlayerStats {
        int totalShots = 0;
        int successfulShots = 0;
        float successRate = 0.0f;
        float avgAccuracy = 0.0f;
        std::chrono::time_point<std::chrono::steady_clock> lastPlay;
    };

public:
    SimpleDataCollectionEngine();
    ~SimpleDataCollectionEngine();
    
    // Core functionality
    void start();
    void stop();
    
    // Data collection
    void recordShot(const ShotData& shot);
    
    // Data retrieval
    PlayerStats getPlayerStats(int playerId) const;
    std::vector<ShotData> getRecentShots(int playerId, int count = 10) const;
    
    // Simple analytics
    float getPlayerSkillLevel(int playerId) const;
    std::vector<std::string> getPlayerInsights(int playerId) const;

private:
    std::atomic<bool> active_;
    mutable std::mutex dataMutex_;
    
    std::map<int, PlayerStats> playerStats_;
    std::map<int, std::vector<ShotData>> playerHistory_;
    
    void updatePlayerStats(int playerId, const ShotData& shot);
};

} // namespace learning
} // namespace ai
} // namespace pv