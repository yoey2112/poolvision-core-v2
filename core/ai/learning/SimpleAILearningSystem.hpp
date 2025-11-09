#pragma once

#include "SimpleDataCollectionEngine.hpp"
#include <string>
#include <vector>
#include <memory>
#include <atomic>
#include <mutex>

namespace pv {
namespace ai {
namespace learning {

/**
 * Simplified AI Learning System Integration
 * 
 * A minimal but functional implementation that provides:
 * - Basic shot analysis and learning
 * - Simple coaching insights
 * - Performance tracking
 * - Clean integration with existing pool vision pipeline
 */
class SimpleAILearningSystem {
public:
    // Learning configuration
    struct Config {
        bool enableLearning = true;
        bool enableCoaching = true;
        bool enableAnalytics = true;
        int maxHistorySize = 100;
        float learningRate = 0.1f;
    };
    
    // Simple analysis result
    struct AnalysisResult {
        float confidence = 0.0f;
        float difficulty = 0.0f;
        std::string recommendation;
        std::vector<std::string> insights;
    };

public:
    explicit SimpleAILearningSystem(const Config& config = Config{});
    ~SimpleAILearningSystem();
    
    // System lifecycle
    bool initialize();
    void start();
    void stop();
    void shutdown();
    
    // Core AI learning functionality
    void analyzeShot(int playerId, int shotType, bool successful, float accuracy);
    AnalysisResult getPlayerAnalysis(int playerId);
    std::vector<std::string> getCoachingInsights(int playerId);
    
    // Performance monitoring
    float getPlayerSkillLevel(int playerId) const;
    std::string getPlayerPerformanceSummary(int playerId) const;
    
    // Configuration
    void setConfig(const Config& config);
    Config getConfig() const;

private:
    Config config_;
    std::atomic<bool> initialized_;
    std::atomic<bool> active_;
    mutable std::mutex systemMutex_;
    
    std::unique_ptr<SimpleDataCollectionEngine> dataEngine_;
    
    // Simple analytics
    std::map<int, float> playerSkillLevels_;
    std::map<int, std::vector<std::string>> playerInsights_;
    
    void updatePlayerAnalysis(int playerId);
    std::vector<std::string> generateInsights(int playerId);
    float calculateSkillProgression(int playerId);
};

} // namespace learning
} // namespace ai
} // namespace pv