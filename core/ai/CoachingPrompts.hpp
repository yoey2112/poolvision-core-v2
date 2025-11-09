#pragma once

#include <string>
#include <unordered_map>
#include <vector>
#include "../game/GameState.hpp"
#include "../game/modern/ShotSegmentation.hpp"
#include "../db/Database.hpp"

namespace pv {
namespace ai {

/**
 * Pool-specific prompt engineering for LLM coaching
 * Creates contextual prompts based on shot analysis, player history, and game state
 */
class CoachingPrompts {
public:
    struct CoachingContext {
        // Shot information
        pv::modern::ShotSegmentation::ShotEvent currentShot;
        std::vector<pv::modern::ShotSegmentation::ShotEvent> recentShots;
        
        // Game state
        GameState gameState;
        
        // Player information
        struct PlayerInfo {
            std::string name;
            int skillLevel = 1;  // 1-10 scale
            std::string playingStyle;  // "aggressive", "defensive", "calculated", etc.
            std::vector<std::string> weaknesses;
            std::vector<std::string> strengths;
            float successRate = 0.0f;
            int gamesPlayed = 0;
        } player;
        
        // Session context
        struct SessionInfo {
            int shotsAttempted = 0;
            int successfulShots = 0;
            float avgShotTime = 0.0f;
            std::string sessionType = "casual";  // "casual", "practice", "tournament"
            std::chrono::system_clock::time_point sessionStart;
        } session;
        
        // Analysis flags
        bool isLegalShot = true;
        std::vector<std::string> ruleViolations;
        std::string lastError;
        
        CoachingContext() = default;
    };

    enum class CoachingType {
        ShotAnalysis,
        DrillRecommendation, 
        MotivationalFeedback,
        StrategyAdvice,
        TechnicalCorrection,
        GamePlanGuidance,
        PerformanceReview
    };

    enum class CoachingPersonality {
        Supportive,      // Encouraging, positive reinforcement
        Analytical,      // Data-driven, technical focus
        Challenging,     // Push boundaries, high expectations
        Patient,         // Methodical, step-by-step
        Competitive      // Win-focused, aggressive mindset
    };

private:
    std::unordered_map<CoachingType, std::string> promptTemplates_;
    std::unordered_map<CoachingPersonality, std::string> personalityModifiers_;
    std::string baseSystemPrompt_;
    CoachingPersonality currentPersonality_;

public:
    explicit CoachingPrompts(CoachingPersonality personality = CoachingPersonality::Supportive);
    
    // Main prompt generation methods
    std::string createCoachingPrompt(CoachingType type, const CoachingContext& context) const;
    std::string createShotAnalysisPrompt(const CoachingContext& context) const;
    std::string createDrillRecommendationPrompt(const CoachingContext& context) const;
    std::string createMotivationalPrompt(const CoachingContext& context) const;
    std::string createStrategyPrompt(const CoachingContext& context) const;
    std::string createTechnicalCorrectionPrompt(const CoachingContext& context) const;
    std::string createGamePlanPrompt(const CoachingContext& context) const;
    std::string createPerformanceReviewPrompt(const CoachingContext& context) const;
    
    // Configuration
    void setPersonality(CoachingPersonality personality);
    CoachingPersonality getPersonality() const { return currentPersonality_; }
    void updatePersonalityModifier(CoachingPersonality personality, const std::string& modifier);
    
    // Template management
    void updatePromptTemplate(CoachingType type, const std::string& templateStr);
    std::string getPromptTemplate(CoachingType type) const;

private:
    void initializePromptTemplates();
    void initializePersonalityModifiers();
    
    // Context formatting helpers
    std::string formatShotDetails(const pv::modern::ShotSegmentation::ShotEvent& shot) const;
    std::string formatPlayerProfile(const CoachingContext::PlayerInfo& profile) const;
    std::string formatGameContext(const GameState& state) const;
    std::string formatSessionStats(const CoachingContext::SessionInfo& session) const;
    std::string formatRecentPerformance(const std::vector<pv::modern::ShotSegmentation::ShotEvent>& shots) const;
    
    // Prompt building utilities
    std::string applyPersonality(const std::string& basePrompt) const;
    std::string replaceTokens(const std::string& templateStr, const CoachingContext& context) const;
    std::string formatSkillLevel(int level) const;
    std::string generateContextSummary(const CoachingContext& context) const;
    
    // Analysis helpers
    std::string analyzeShotDifficulty(const pv::modern::ShotSegmentation::ShotEvent& shot) const;
    std::string identifyMistakes(const CoachingContext& context) const;
    std::vector<std::string> suggestImprovements(const CoachingContext& context) const;
    std::string recommendDrills(const CoachingContext& context) const;
};

/**
 * Advanced coaching prompt builder for complex scenarios
 */
class AdvancedPromptBuilder {
public:
    struct PromptConfig {
        int maxTokens = 512;
        float creativity = 0.7f;
        bool includeHistory = true;
        bool includeTechnicalDetails = true;
        bool includeMotivation = true;
        std::vector<std::string> focusAreas;
    };

private:
    PromptConfig config_;
    
public:
    explicit AdvancedPromptBuilder(const PromptConfig& config = PromptConfig{});
    
    std::string buildCustomPrompt(
        const CoachingPrompts::CoachingContext& context,
        const std::vector<CoachingPrompts::CoachingType>& types,
        const std::string& specificInstructions = ""
    ) const;
    
    std::string buildMultiShotAnalysis(
        const std::vector<pv::modern::ShotSegmentation::ShotEvent>& shots,
        const CoachingPrompts::CoachingContext& context
    ) const;
    
    void setConfig(const PromptConfig& config) { config_ = config; }
    const PromptConfig& getConfig() const { return config_; }
};

} // namespace ai
} // namespace pv