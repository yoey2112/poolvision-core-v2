#ifndef PV_AI_LEARNING_ADAPTIVE_COACHING_HPP
#define PV_AI_LEARNING_ADAPTIVE_COACHING_HPP

#include "DataCollectionEngine.hpp"
#include "ShotAnalysisEngine.hpp"
#include "../../events/EventEngine.hpp"
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <queue>
#include <thread>
#include <mutex>
#include <atomic>
#include <condition_variable>
#include <chrono>

namespace pv {
namespace ai {
namespace learning {

/**
 * Adaptive Coaching System
 * 
 * Extends the existing Ollama coaching engine with AI-driven personalization:
 * - Player-specific coaching models based on performance data
 * - Adaptive difficulty progression 
 * - Personalized feedback generation
 * - Context-aware coaching tips
 * - Integration with existing coaching infrastructure
 */
class AdaptiveCoachingEngine {
public:
    // Player Profile Data
    struct PlayerProfile {
        int playerId;
        std::string playerName;
        
        // Skill Assessment
        struct SkillLevel {
            float overall;          // 0-1 overall skill rating
            float accuracy;         // Shot accuracy skill
            float positioning;      // Position play skill
            float strategy;         // Game strategy skill
            float consistency;      // Performance consistency
            float pressure;         // Performance under pressure
        } skillLevel;
        
        // Learning Preferences
        struct LearningStyle {
            enum Type {
                Visual,         // Prefers visual demonstrations
                Analytical,     // Prefers detailed analysis
                Encouraging,    // Prefers positive reinforcement
                Technical,      // Prefers technical details
                Progressive     // Prefers gradual challenges
            } primaryStyle;
            
            float detailLevel;      // 0-1 how much detail they want
            float feedbackFrequency; // 0-1 how often they want feedback
            bool wantsChallenges;   // Whether they want difficult suggestions
        } learningStyle;
        
        // Current Session Data
        struct SessionState {
            int shotsAttempted;
            int shotsSuccessful;
            float currentConfidence;
            float sessionTrend;     // Improving/declining performance
            std::chrono::time_point<std::chrono::steady_clock> sessionStart;
            std::vector<std::string> recentFeedback;
        } sessionState;
        
        // Historical Performance
        struct HistoricalData {
            int totalSessions;
            float bestSessionRating;
            float averageImprovement;
            std::map<DataCollectionEngine::ShotOutcomeData::ShotType, float> shotTypeProgression;
            std::vector<float> sessionRatings; // Last 20 sessions
        } history;
    };
    
    // Coaching Message Types
    struct CoachingMessage {
        enum Priority {
            Low,        // General tips
            Medium,     // Important feedback
            High,       // Critical corrections
            Urgent      // Immediate attention needed
        };
        
        enum Category {
            Technique,      // Shot technique
            Strategy,       // Game strategy
            Positioning,    // Position play
            Mental,         // Psychological aspects
            Encouragement,  // Motivation/support
            Analysis,       // Performance analysis
            Challenge       // Skill challenges
        };
        
        std::string message;
        Priority priority;
        Category category;
        float relevanceScore;   // 0-1 how relevant to current situation
        std::chrono::time_point<std::chrono::steady_clock> timestamp;
        bool hasVisualAid;      // Whether message includes visual demonstration
        std::string context;    // Game context when message was generated
    };
    
    // Adaptive Lesson Plan
    struct LessonPlan {
        std::string title;
        std::string description;
        std::vector<std::string> objectives;
        std::vector<std::string> exercises;
        float difficultyLevel;  // 0-1 difficulty rating
        int estimatedDuration;  // Minutes
        std::vector<std::string> prerequisites;
        std::string successCriteria;
    };
    
    // Coaching Session Data
    struct CoachingSession {
        int playerId;
        std::chrono::time_point<std::chrono::steady_clock> startTime;
        std::vector<CoachingMessage> messages;
        LessonPlan currentLesson;
        float playerEngagement;  // 0-1 how engaged player seems
        bool lessonCompleted;
        std::string sessionSummary;
    };

private:
    // Coaching Strategy Engine
    class StrategyEngine {
    public:
        // Generate personalized coaching strategy
        std::string generatePersonalizedStrategy(const PlayerProfile& profile,
                                                const ShotAnalysisEngine::PatternAnalysis& patterns);
        
        // Adapt coaching based on performance
        void adaptCoachingStrategy(PlayerProfile& profile, 
                                 const DataCollectionEngine::ShotOutcomeData& recentShot);
        
        // Generate difficulty progression
        float calculateNextDifficulty(const PlayerProfile& profile);
        
        // Context-aware message generation
        CoachingMessage generateContextualMessage(const PlayerProfile& profile,
                                                const GameState& gameState,
                                                const ShotAnalysisEngine::ShotAnalysisResult& analysis);
    };
    
    // Integration with Ollama Engine
    class OllamaIntegration {
    public:
        // Initialize connection to existing Ollama service
        bool connectToOllama(const std::string& endpoint);
        
        // Generate personalized prompt for Ollama
        std::string generatePersonalizedPrompt(const PlayerProfile& profile,
                                              const std::string& context,
                                              const CoachingMessage::Category& category);
        
        // Process Ollama response with player context
        CoachingMessage processOllamaResponse(const std::string& response,
                                            const PlayerProfile& profile);
        
        // Enhance existing coaching with AI insights
        std::string enhanceCoachingWithAI(const std::string& baseCoaching,
                                         const PlayerProfile& profile);
    private:
        std::string ollamaEndpoint_;
        bool connected_;
    };
    
    // Performance Assessment
    class PerformanceAssessor {
    public:
        // Update skill assessment based on recent performance
        void updateSkillAssessment(PlayerProfile& profile,
                                 const std::vector<DataCollectionEngine::ShotOutcomeData>& recentShots);
        
        // Assess learning progress
        float assessLearningProgress(const PlayerProfile& profile);
        
        // Detect performance patterns
        std::vector<std::string> detectPerformancePatterns(const PlayerProfile& profile);
        
        // Calculate coaching effectiveness
        float assessCoachingEffectiveness(const PlayerProfile& profile,
                                        const std::vector<CoachingMessage>& recentCoaching);
    };

public:
    explicit AdaptiveCoachingEngine(DataCollectionEngine* dataEngine,
                                   ShotAnalysisEngine* analysisEngine);
    ~AdaptiveCoachingEngine();
    
    // Core functionality
    void startCoaching();
    void stopCoaching();
    
    // Player management
    void addPlayer(int playerId, const std::string& playerName);
    void updatePlayerProfile(int playerId, const DataCollectionEngine::PlayerBehaviorData& behavior);
    PlayerProfile getPlayerProfile(int playerId);
    
    // Coaching session management
    void startCoachingSession(int playerId);
    void endCoachingSession(int playerId);
    CoachingSession getCurrentSession(int playerId);
    
    // Real-time coaching
    CoachingMessage generateRealtimeCoaching(int playerId, 
                                           const GameState& gameState,
                                           const ShotAnalysisEngine::ShotAnalysisResult& analysis);
    
    // Lesson planning
    LessonPlan generatePersonalizedLesson(int playerId, 
                                        const std::string& focusArea = "");
    void trackLessonProgress(int playerId, const LessonPlan& lesson, float completion);
    
    // Feedback and adaptation
    void provideFeedback(int playerId, const CoachingMessage& message);
    void adaptToPlayerResponse(int playerId, const std::string& response);
    void updateFromShotOutcome(int playerId, const DataCollectionEngine::ShotOutcomeData& shot);
    
    // Analytics and insights
    std::vector<std::string> generatePlayerInsights(int playerId);
    float calculateImprovementRate(int playerId);
    std::map<std::string, float> getSkillBreakdown(int playerId);
    
    // Integration with existing coaching
    void integrateWithOllama(const std::string& ollamaEndpoint);
    std::string enhanceExistingCoaching(int playerId, const std::string& baseCoaching);
    
    // Configuration
    void setCoachingIntensity(float intensity); // 0-1 how much coaching to provide
    void setAdaptationRate(float rate);         // 0-1 how quickly to adapt
    void enableVisualAids(bool enable);
    
    // Metrics and monitoring
    struct CoachingMetrics {
        std::atomic<int> sessionsStarted{0};
        std::atomic<int> messagesGenerated{0};
        std::atomic<int> lessonsCompleted{0};
        std::atomic<double> avgMessageRelevance{0.0};
        std::atomic<double> avgPlayerEngagement{0.0};
        std::atomic<double> avgImprovementRate{0.0};
    };
    
    CoachingMetrics getCoachingMetrics() const { return metrics_; }
    void logCoachingReport();

private:
    // Core components
    DataCollectionEngine* dataEngine_;
    ShotAnalysisEngine* analysisEngine_;
    
    std::unique_ptr<StrategyEngine> strategyEngine_;
    std::unique_ptr<OllamaIntegration> ollamaIntegration_;
    std::unique_ptr<PerformanceAssessor> performanceAssessor_;
    
    // Player data
    std::map<int, PlayerProfile> playerProfiles_;
    std::map<int, CoachingSession> activeSessions_;
    
    // Threading and synchronization
    std::atomic<bool> coachingActive_{false};
    std::thread coachingThread_;
    std::mutex profileMutex_;
    std::mutex sessionMutex_;
    std::condition_variable coachingCondition_;
    
    // Processing queue
    std::queue<std::pair<int, DataCollectionEngine::ShotOutcomeData>> shotQueue_;
    std::mutex queueMutex_;
    
    // Configuration
    float coachingIntensity_ = 0.7f;
    float adaptationRate_ = 0.3f;
    bool visualAidsEnabled_ = true;
    
    // Metrics
    CoachingMetrics metrics_;
    
    // Background processing
    void coachingLoop();
    void processShotQueue();
    void updatePlayerProfiles();
    void generateScheduledCoaching();
    
    // Helper methods
    void initializePlayerProfile(int playerId, const std::string& playerName);
    void updatePlayerSkills(PlayerProfile& profile, 
                          const DataCollectionEngine::ShotOutcomeData& shot);
    void updateLearningStyle(PlayerProfile& profile,
                           const std::vector<CoachingMessage>& messageHistory);
    std::string formatCoachingMessage(const CoachingMessage& message,
                                    const PlayerProfile& profile);
    float calculateMessageRelevance(const CoachingMessage& message,
                                  const PlayerProfile& profile,
                                  const GameState& gameState);
};

// Factory for creating coaching engines
class AdaptiveCoachingFactory {
public:
    static std::unique_ptr<AdaptiveCoachingEngine> createWithOllama(
        DataCollectionEngine* dataEngine,
        ShotAnalysisEngine* analysisEngine,
        const std::string& ollamaEndpoint);
    
    static std::unique_ptr<AdaptiveCoachingEngine> createStandalone(
        DataCollectionEngine* dataEngine,
        ShotAnalysisEngine* analysisEngine);
};

} // namespace learning
} // namespace ai
} // namespace pv

#endif // PV_AI_LEARNING_ADAPTIVE_COACHING_HPP