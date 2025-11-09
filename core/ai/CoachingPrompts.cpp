#include "CoachingPrompts.hpp"
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <ctime>

namespace pv {
namespace ai {

CoachingPrompts::CoachingPrompts(CoachingPersonality personality) 
    : currentPersonality_(personality) {
    initializePromptTemplates();
    initializePersonalityModifiers();
    
    baseSystemPrompt_ = 
        "You are an expert pool/billiards coach with over 20 years of experience teaching players "
        "of all skill levels. You have deep knowledge of 8-ball, 9-ball, and straight pool games. "
        "Your coaching style adapts to each player's skill level and learning preferences. "
        "You provide clear, actionable advice focusing on technique, strategy, and mental game. "
        "Always maintain a professional yet encouraging tone.";
}

void CoachingPrompts::initializePromptTemplates() {
    promptTemplates_[CoachingType::ShotAnalysis] = 
        "Analyze this pool shot:\n\n"
        "Shot Details: {SHOT_DETAILS}\n"
        "Player Profile: {PLAYER_PROFILE}\n"
        "Game Context: {GAME_CONTEXT}\n"
        "Result: {SHOT_RESULT}\n\n"
        "Please provide:\n"
        "1. Technical analysis of the shot execution\n"
        "2. What went well vs. what could be improved\n"
        "3. Specific technique recommendations\n"
        "4. Strategic considerations for similar future shots\n"
        "Keep analysis concise but thorough (2-3 paragraphs).";

    promptTemplates_[CoachingType::DrillRecommendation] = 
        "Based on this player's performance:\n\n"
        "Player Profile: {PLAYER_PROFILE}\n"
        "Recent Performance: {RECENT_PERFORMANCE}\n"
        "Identified Weaknesses: {WEAKNESSES}\n"
        "Session Context: {SESSION_STATS}\n\n"
        "Recommend 3 specific practice drills that would help this player improve. "
        "For each drill, provide:\n"
        "1. Clear setup instructions\n"
        "2. Objectives and success criteria\n"
        "3. Progression levels (beginner to advanced)\n"
        "4. Expected time commitment\n"
        "Focus on drills that address their specific weaknesses.";

    promptTemplates_[CoachingType::MotivationalFeedback] = 
        "Provide motivational feedback for this player:\n\n"
        "Player Profile: {PLAYER_PROFILE}\n"
        "Session Progress: {SESSION_STATS}\n"
        "Recent Achievement: {RECENT_PERFORMANCE}\n"
        "Current Mood/Context: {CONTEXT_SUMMARY}\n\n"
        "Create an encouraging message that:\n"
        "1. Acknowledges their effort and progress\n"
        "2. Highlights specific improvements\n"
        "3. Sets positive expectations for continued growth\n"
        "4. Maintains motivation for practice\n"
        "Keep it genuine, specific, and appropriately challenging.";

    promptTemplates_[CoachingType::StrategyAdvice] = 
        "Provide strategic advice for this game situation:\n\n"
        "Game State: {GAME_CONTEXT}\n"
        "Player Profile: {PLAYER_PROFILE}\n"
        "Current Position: {SHOT_DETAILS}\n"
        "Opponent Analysis: {OPPONENT_INFO}\n\n"
        "Recommend the best strategic approach considering:\n"
        "1. Risk vs. reward analysis of available shots\n"
        "2. Table control and positioning\n"
        "3. Defensive vs. offensive options\n"
        "4. Player's skill level and playing style\n"
        "Provide 2-3 specific shot options with rationale.";

    promptTemplates_[CoachingType::TechnicalCorrection] = 
        "Provide technical correction for this issue:\n\n"
        "Identified Problem: {IDENTIFIED_MISTAKES}\n"
        "Player Skill Level: {SKILL_LEVEL}\n"
        "Shot Context: {SHOT_DETAILS}\n"
        "Previous Attempts: {RECENT_PERFORMANCE}\n\n"
        "Provide specific technical guidance:\n"
        "1. Root cause analysis of the problem\n"
        "2. Step-by-step correction technique\n"
        "3. Common mistakes to avoid\n"
        "4. Practice exercises for muscle memory\n"
        "Focus on the most impactful change for their level.";

    promptTemplates_[CoachingType::GamePlanGuidance] = 
        "Create a game plan for this match:\n\n"
        "Player Profile: {PLAYER_PROFILE}\n"
        "Game Type: {GAME_TYPE}\n"
        "Opponent Analysis: {OPPONENT_INFO}\n"
        "Match Importance: {MATCH_CONTEXT}\n\n"
        "Develop a comprehensive game plan including:\n"
        "1. Overall strategic approach\n"
        "2. Shot selection priorities\n"
        "3. Safety play considerations\n"
        "4. Pressure point management\n"
        "5. Mental preparation strategies\n"
        "Tailor advice to their playing style and strengths.";

    promptTemplates_[CoachingType::PerformanceReview] = 
        "Provide a performance review for this session:\n\n"
        "Session Overview: {SESSION_STATS}\n"
        "Player Profile: {PLAYER_PROFILE}\n"
        "Performance Metrics: {PERFORMANCE_METRICS}\n"
        "Key Shots Analyzed: {SHOT_ANALYSIS}\n\n"
        "Create a comprehensive review covering:\n"
        "1. Overall session assessment\n"
        "2. Improvement areas identified\n"
        "3. Strengths demonstrated\n"
        "4. Recommended focus for next session\n"
        "5. Long-term development goals\n"
        "Provide constructive, actionable feedback.";
}

void CoachingPrompts::initializePersonalityModifiers() {
    personalityModifiers_[CoachingPersonality::Supportive] = 
        "Adopt a supportive and encouraging coaching style. Use positive reinforcement, "
        "celebrate small wins, and frame challenges as opportunities for growth. "
        "Be patient and understanding while maintaining high standards.";

    personalityModifiers_[CoachingPersonality::Analytical] = 
        "Take a data-driven, analytical approach. Focus on technical details, statistics, "
        "and systematic improvement. Use precise language and provide detailed explanations "
        "of mechanics and physics behind shots.";

    personalityModifiers_[CoachingPersonality::Challenging] = 
        "Adopt a challenging coaching style that pushes the player to exceed their comfort zone. "
        "Set high expectations, be direct about areas needing improvement, and motivate through "
        "achievement-oriented goals.";

    personalityModifiers_[CoachingPersonality::Patient] = 
        "Be extremely patient and methodical. Break down complex concepts into simple steps. "
        "Repeat key points, provide detailed explanations, and ensure understanding before "
        "moving to the next concept.";

    personalityModifiers_[CoachingPersonality::Competitive] = 
        "Focus on winning strategies and competitive advantage. Emphasize mental toughness, "
        "pressure situations, and tactical approaches that give the player an edge over opponents. "
        "Be results-oriented and performance-focused.";
}

std::string CoachingPrompts::createCoachingPrompt(CoachingType type, const CoachingContext& context) const {
    switch (type) {
        case CoachingType::ShotAnalysis:
            return createShotAnalysisPrompt(context);
        case CoachingType::DrillRecommendation:
            return createDrillRecommendationPrompt(context);
        case CoachingType::MotivationalFeedback:
            return createMotivationalPrompt(context);
        case CoachingType::StrategyAdvice:
            return createStrategyPrompt(context);
        case CoachingType::TechnicalCorrection:
            return createTechnicalCorrectionPrompt(context);
        case CoachingType::GamePlanGuidance:
            return createGamePlanPrompt(context);
        case CoachingType::PerformanceReview:
            return createPerformanceReviewPrompt(context);
        default:
            return createShotAnalysisPrompt(context);
    }
}

std::string CoachingPrompts::createShotAnalysisPrompt(const CoachingContext& context) const {
    auto it = promptTemplates_.find(CoachingType::ShotAnalysis);
    if (it == promptTemplates_.end()) {
        return "Error: Shot analysis template not found.";
    }
    
    std::string prompt = replaceTokens(it->second, context);
    return applyPersonality(prompt);
}

std::string CoachingPrompts::createDrillRecommendationPrompt(const CoachingContext& context) const {
    auto it = promptTemplates_.find(CoachingType::DrillRecommendation);
    if (it == promptTemplates_.end()) {
        return "Error: Drill recommendation template not found.";
    }
    
    std::string prompt = replaceTokens(it->second, context);
    return applyPersonality(prompt);
}

std::string CoachingPrompts::createMotivationalPrompt(const CoachingContext& context) const {
    auto it = promptTemplates_.find(CoachingType::MotivationalFeedback);
    if (it == promptTemplates_.end()) {
        return "Error: Motivational template not found.";
    }
    
    std::string prompt = replaceTokens(it->second, context);
    return applyPersonality(prompt);
}

std::string CoachingPrompts::createStrategyPrompt(const CoachingContext& context) const {
    auto it = promptTemplates_.find(CoachingType::StrategyAdvice);
    if (it == promptTemplates_.end()) {
        return "Error: Strategy template not found.";
    }
    
    std::string prompt = replaceTokens(it->second, context);
    return applyPersonality(prompt);
}

std::string CoachingPrompts::createTechnicalCorrectionPrompt(const CoachingContext& context) const {
    auto it = promptTemplates_.find(CoachingType::TechnicalCorrection);
    if (it == promptTemplates_.end()) {
        return "Error: Technical correction template not found.";
    }
    
    std::string prompt = replaceTokens(it->second, context);
    return applyPersonality(prompt);
}

std::string CoachingPrompts::createGamePlanPrompt(const CoachingContext& context) const {
    auto it = promptTemplates_.find(CoachingType::GamePlanGuidance);
    if (it == promptTemplates_.end()) {
        return "Error: Game plan template not found.";
    }
    
    std::string prompt = replaceTokens(it->second, context);
    return applyPersonality(prompt);
}

std::string CoachingPrompts::createPerformanceReviewPrompt(const CoachingContext& context) const {
    auto it = promptTemplates_.find(CoachingType::PerformanceReview);
    if (it == promptTemplates_.end()) {
        return "Error: Performance review template not found.";
    }
    
    std::string prompt = replaceTokens(it->second, context);
    return applyPersonality(prompt);
}

void CoachingPrompts::setPersonality(CoachingPersonality personality) {
    currentPersonality_ = personality;
}

void CoachingPrompts::updatePersonalityModifier(CoachingPersonality personality, const std::string& modifier) {
    personalityModifiers_[personality] = modifier;
}

void CoachingPrompts::updatePromptTemplate(CoachingType type, const std::string& templateStr) {
    promptTemplates_[type] = templateStr;
}

std::string CoachingPrompts::getPromptTemplate(CoachingType type) const {
    auto it = promptTemplates_.find(type);
    return (it != promptTemplates_.end()) ? it->second : "";
}

std::string CoachingPrompts::formatShotDetails(const pv::modern::ShotSegmentation::ShotEvent& shot) const {
    std::ostringstream oss;
    oss << "Shot Duration: " << shot.duration << "s\n";
    oss << "Max Speed: " << shot.maxSpeed << " pixels/s\n";
    oss << "Total Distance: " << shot.totalDistance << " pixels\n";
    oss << "Shot Type: ";
    
    switch (shot.shotType) {
        case pv::modern::ShotSegmentation::ShotEvent::Break:
            oss << "Break Shot";
            break;
        case pv::modern::ShotSegmentation::ShotEvent::Standard:
            oss << "Standard Shot";
            break;
        case pv::modern::ShotSegmentation::ShotEvent::SafetyShot:
            oss << "Safety Shot";
            break;
        case pv::modern::ShotSegmentation::ShotEvent::BankShot:
            oss << "Bank Shot";
            break;
        case pv::modern::ShotSegmentation::ShotEvent::JumpShot:
            oss << "Jump Shot";
            break;
        case pv::modern::ShotSegmentation::ShotEvent::MassShot:
            oss << "Masse Shot";
            break;
        default:
            oss << "Unknown";
            break;
    }
    
    oss << "\nBalls in Motion: " << shot.ballsInMotion.size();
    oss << "\nBalls Potted: " << shot.ballsPotted.size();
    oss << "\nBalls Contacted: " << shot.ballsContacted.size();
    oss << "\nLegal Shot: " << (shot.isLegalShot ? "Yes" : "No");
    oss << "\nCollisions Detected: " << (shot.hasCollisions ? "Yes" : "No");
    
    return oss.str();
}

std::string CoachingPrompts::formatPlayerProfile(const CoachingContext::PlayerInfo& profile) const {
    std::ostringstream oss;
    oss << "Name: " << profile.name << "\n";
    oss << "Skill Level: " << formatSkillLevel(profile.skillLevel) << "\n";
    oss << "Playing Style: " << profile.playingStyle << "\n";
    oss << "Success Rate: " << std::fixed << std::setprecision(1) << (profile.successRate * 100) << "%\n";
    oss << "Games Played: " << profile.gamesPlayed << "\n";
    
    if (!profile.strengths.empty()) {
        oss << "Strengths: ";
        for (size_t i = 0; i < profile.strengths.size(); ++i) {
            oss << profile.strengths[i];
            if (i < profile.strengths.size() - 1) oss << ", ";
        }
        oss << "\n";
    }
    
    if (!profile.weaknesses.empty()) {
        oss << "Areas for Improvement: ";
        for (size_t i = 0; i < profile.weaknesses.size(); ++i) {
            oss << profile.weaknesses[i];
            if (i < profile.weaknesses.size() - 1) oss << ", ";
        }
        oss << "\n";
    }
    
    return oss.str();
}

std::string CoachingPrompts::formatGameContext(const GameState& state) const {
    std::ostringstream oss;
    oss << "Game Type: ";
    
    switch (state.getGameType()) {
        case GameState::EightBall:
            oss << "8-Ball";
            break;
        case GameState::NineBall:
            oss << "9-Ball";
            break;
        case GameState::StraightPool:
            oss << "Straight Pool";
            break;
        default:
            oss << "Unknown";
            break;
    }
    
    oss << "\nCurrent Player: " << (state.getCurrentPlayer() + 1);
    oss << "\nScore: Player 1: " << state.getScore(0) << ", Player 2: " << state.getScore(1);
    oss << "\nBalls on Table: " << state.getBallsOnTable().size();
    oss << "\nGame Over: " << (state.isGameOver() ? "Yes" : "No");
    
    if (state.isGameOver()) {
        oss << "\nWinner: Player " << (state.getWinner() + 1);
    }
    
    return oss.str();
}

std::string CoachingPrompts::formatSessionStats(const CoachingContext::SessionInfo& session) const {
    std::ostringstream oss;
    oss << "Session Type: " << session.sessionType << "\n";
    oss << "Shots Attempted: " << session.shotsAttempted << "\n";
    oss << "Successful Shots: " << session.successfulShots << "\n";
    
    if (session.shotsAttempted > 0) {
        float successRate = static_cast<float>(session.successfulShots) / session.shotsAttempted * 100.0f;
        oss << "Success Rate: " << std::fixed << std::setprecision(1) << successRate << "%\n";
    }
    
    oss << "Average Shot Time: " << std::fixed << std::setprecision(1) << session.avgShotTime << "s\n";
    
    auto now = std::chrono::system_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::minutes>(now - session.sessionStart);
    oss << "Session Duration: " << duration.count() << " minutes";
    
    return oss.str();
}

std::string CoachingPrompts::formatRecentPerformance(const std::vector<pv::modern::ShotSegmentation::ShotEvent>& shots) const {
    if (shots.empty()) {
        return "No recent shots to analyze.";
    }
    
    std::ostringstream oss;
    int legalShots = 0;
    float totalDuration = 0.0f;
    float maxSpeed = 0.0f;
    
    for (const auto& shot : shots) {
        if (shot.isLegalShot) legalShots++;
        totalDuration += shot.duration;
        if (shot.maxSpeed > maxSpeed) maxSpeed = shot.maxSpeed;
    }
    
    oss << "Recent " << shots.size() << " shots:\n";
    oss << "Legal Shots: " << legalShots << "/" << shots.size() << " (";
    oss << std::fixed << std::setprecision(1) << (static_cast<float>(legalShots) / shots.size() * 100) << "%)\n";
    oss << "Average Shot Duration: " << std::fixed << std::setprecision(1) << (totalDuration / shots.size()) << "s\n";
    oss << "Highest Speed Achieved: " << maxSpeed << " pixels/s";
    
    return oss.str();
}

std::string CoachingPrompts::applyPersonality(const std::string& basePrompt) const {
    auto it = personalityModifiers_.find(currentPersonality_);
    if (it == personalityModifiers_.end()) {
        return basePrompt;
    }
    
    return baseSystemPrompt_ + "\n\n" + it->second + "\n\n" + basePrompt;
}

std::string CoachingPrompts::replaceTokens(const std::string& templateStr, const CoachingContext& context) const {
    std::string result = templateStr;
    
    // Replace all tokens with actual context data
    size_t pos = 0;
    while ((pos = result.find("{SHOT_DETAILS}", pos)) != std::string::npos) {
        result.replace(pos, 14, formatShotDetails(context.currentShot));
        pos += formatShotDetails(context.currentShot).length();
    }
    
    pos = 0;
    while ((pos = result.find("{PLAYER_PROFILE}", pos)) != std::string::npos) {
        result.replace(pos, 16, formatPlayerProfile(context.player));
        pos += formatPlayerProfile(context.player).length();
    }
    
    pos = 0;
    while ((pos = result.find("{GAME_CONTEXT}", pos)) != std::string::npos) {
        result.replace(pos, 14, formatGameContext(context.gameState));
        pos += formatGameContext(context.gameState).length();
    }
    
    pos = 0;
    while ((pos = result.find("{SESSION_STATS}", pos)) != std::string::npos) {
        result.replace(pos, 15, formatSessionStats(context.session));
        pos += formatSessionStats(context.session).length();
    }
    
    pos = 0;
    while ((pos = result.find("{RECENT_PERFORMANCE}", pos)) != std::string::npos) {
        result.replace(pos, 20, formatRecentPerformance(context.recentShots));
        pos += formatRecentPerformance(context.recentShots).length();
    }
    
    pos = 0;
    std::string shotResult = context.isLegalShot ? "Legal and successful" : "Illegal or failed";
    while ((pos = result.find("{SHOT_RESULT}", pos)) != std::string::npos) {
        result.replace(pos, 13, shotResult);
        pos += shotResult.length();
    }
    
    pos = 0;
    std::string weaknesses = "";
    for (size_t i = 0; i < context.player.weaknesses.size(); ++i) {
        weaknesses += context.player.weaknesses[i];
        if (i < context.player.weaknesses.size() - 1) weaknesses += ", ";
    }
    while ((pos = result.find("{WEAKNESSES}", pos)) != std::string::npos) {
        result.replace(pos, 12, weaknesses);
        pos += weaknesses.length();
    }
    
    pos = 0;
    while ((pos = result.find("{CONTEXT_SUMMARY}", pos)) != std::string::npos) {
        result.replace(pos, 17, generateContextSummary(context));
        pos += generateContextSummary(context).length();
    }
    
    pos = 0;
    while ((pos = result.find("{SKILL_LEVEL}", pos)) != std::string::npos) {
        result.replace(pos, 13, formatSkillLevel(context.player.skillLevel));
        pos += formatSkillLevel(context.player.skillLevel).length();
    }
    
    pos = 0;
    while ((pos = result.find("{IDENTIFIED_MISTAKES}", pos)) != std::string::npos) {
        result.replace(pos, 21, identifyMistakes(context));
        pos += identifyMistakes(context).length();
    }
    
    return result;
}

std::string CoachingPrompts::formatSkillLevel(int level) const {
    switch (level) {
        case 1: case 2:
            return std::to_string(level) + "/10 (Beginner)";
        case 3: case 4:
            return std::to_string(level) + "/10 (Novice)";
        case 5: case 6:
            return std::to_string(level) + "/10 (Intermediate)";
        case 7: case 8:
            return std::to_string(level) + "/10 (Advanced)";
        case 9: case 10:
            return std::to_string(level) + "/10 (Expert)";
        default:
            return std::to_string(level) + "/10";
    }
}

std::string CoachingPrompts::generateContextSummary(const CoachingContext& context) const {
    std::ostringstream oss;
    
    if (!context.isLegalShot) {
        oss << "Player made an illegal shot";
        if (!context.ruleViolations.empty()) {
            oss << " (violations: ";
            for (size_t i = 0; i < context.ruleViolations.size(); ++i) {
                oss << context.ruleViolations[i];
                if (i < context.ruleViolations.size() - 1) oss << ", ";
            }
            oss << ")";
        }
        oss << ". ";
    }
    
    if (context.session.shotsAttempted > 0) {
        float successRate = static_cast<float>(context.session.successfulShots) / context.session.shotsAttempted;
        if (successRate < 0.3f) {
            oss << "Player is struggling this session. ";
        } else if (successRate > 0.7f) {
            oss << "Player is performing well this session. ";
        }
    }
    
    if (context.recentShots.size() >= 3) {
        int recentLegal = 0;
        for (const auto& shot : context.recentShots) {
            if (shot.isLegalShot) recentLegal++;
        }
        if (recentLegal == 0) {
            oss << "Recent shots have been problematic. ";
        }
    }
    
    return oss.str();
}

std::string CoachingPrompts::analyzeShotDifficulty(const pv::modern::ShotSegmentation::ShotEvent& shot) const {
    // Simple heuristic based on shot characteristics
    if (shot.shotType == pv::modern::ShotSegmentation::ShotEvent::BankShot ||
        shot.shotType == pv::modern::ShotSegmentation::ShotEvent::JumpShot ||
        shot.shotType == pv::modern::ShotSegmentation::ShotEvent::MassShot) {
        return "High difficulty";
    } else if (shot.maxSpeed > 1000.0f || shot.hasCollisions) {
        return "Medium difficulty";
    } else {
        return "Standard difficulty";
    }
}

std::string CoachingPrompts::identifyMistakes(const CoachingContext& context) const {
    std::ostringstream oss;
    
    if (!context.isLegalShot) {
        oss << "Illegal shot detected. ";
        if (!context.ruleViolations.empty()) {
            oss << "Specific violations: ";
            for (size_t i = 0; i < context.ruleViolations.size(); ++i) {
                oss << context.ruleViolations[i];
                if (i < context.ruleViolations.size() - 1) oss << ", ";
            }
            oss << ". ";
        }
    }
    
    const auto& shot = context.currentShot;
    if (shot.duration > 10.0) {
        oss << "Shot took unusually long, suggesting hesitation or uncertainty. ";
    }
    
    if (shot.maxSpeed < 100.0f) {
        oss << "Very soft shot - may indicate lack of confidence. ";
    } else if (shot.maxSpeed > 2000.0f) {
        oss << "Very hard shot - may indicate poor control. ";
    }
    
    if (!shot.hasCollisions && shot.ballsInMotion.size() > 1) {
        oss << "Multiple balls in motion without detected collisions - possible measurement issue. ";
    }
    
    return oss.str();
}

std::vector<std::string> CoachingPrompts::suggestImprovements(const CoachingContext& context) const {
    std::vector<std::string> suggestions;
    
    if (!context.isLegalShot) {
        suggestions.push_back("Review game rules to avoid future violations");
        suggestions.push_back("Practice shot planning and visualization");
    }
    
    const auto& shot = context.currentShot;
    if (shot.duration > 10.0) {
        suggestions.push_back("Work on shot selection confidence");
        suggestions.push_back("Practice pre-shot routine for consistency");
    }
    
    if (context.player.successRate < 0.5f) {
        suggestions.push_back("Focus on fundamentals: stance, bridge, and stroke");
        suggestions.push_back("Start with easier shots to build confidence");
    }
    
    if (!context.player.weaknesses.empty()) {
        for (const auto& weakness : context.player.weaknesses) {
            suggestions.push_back("Address " + weakness + " through targeted practice");
        }
    }
    
    return suggestions;
}

std::string CoachingPrompts::recommendDrills(const CoachingContext& context) const {
    std::ostringstream oss;
    
    // Basic recommendations based on skill level and weaknesses
    if (context.player.skillLevel <= 3) {
        oss << "Beginner drills: Straight-in shots, basic aim training, cue ball control practice";
    } else if (context.player.skillLevel <= 6) {
        oss << "Intermediate drills: Position play, safety shots, pattern recognition";
    } else {
        oss << "Advanced drills: Complex patterns, pressure situations, advanced cue ball control";
    }
    
    // Add specific recommendations based on weaknesses
    for (const auto& weakness : context.player.weaknesses) {
        if (weakness.find("aim") != std::string::npos) {
            oss << "\nAiming drills: Ghost ball visualization, target practice";
        } else if (weakness.find("position") != std::string::npos) {
            oss << "\nPosition drills: Stop shots, follow shots, draw shots";
        } else if (weakness.find("safety") != std::string::npos) {
            oss << "\nSafety drills: Hide the cue ball, cluster breaking";
        }
    }
    
    return oss.str();
}

// AdvancedPromptBuilder implementation
AdvancedPromptBuilder::AdvancedPromptBuilder(const PromptConfig& config) : config_(config) {}

std::string AdvancedPromptBuilder::buildCustomPrompt(
    const CoachingPrompts::CoachingContext& context,
    const std::vector<CoachingPrompts::CoachingType>& types,
    const std::string& specificInstructions) const {
    
    std::ostringstream oss;
    
    oss << "You are providing comprehensive pool coaching guidance. ";
    oss << "Focus on the following areas: ";
    
    for (size_t i = 0; i < types.size(); ++i) {
        switch (types[i]) {
            case CoachingPrompts::CoachingType::ShotAnalysis:
                oss << "shot analysis";
                break;
            case CoachingPrompts::CoachingType::TechnicalCorrection:
                oss << "technical correction";
                break;
            case CoachingPrompts::CoachingType::StrategyAdvice:
                oss << "strategy advice";
                break;
            // Add other types as needed
            default:
                oss << "general coaching";
                break;
        }
        if (i < types.size() - 1) oss << ", ";
    }
    oss << ".\n\n";
    
    if (!specificInstructions.empty()) {
        oss << "Special instructions: " << specificInstructions << "\n\n";
    }
    
    // Add context information
    oss << "Player context:\n";
    oss << "- Skill level: " << context.player.skillLevel << "/10\n";
    oss << "- Success rate: " << (context.player.successRate * 100) << "%\n";
    oss << "- Current shot legal: " << (context.isLegalShot ? "Yes" : "No") << "\n\n";
    
    oss << "Please provide coaching guidance within " << config_.maxTokens << " tokens.";
    
    return oss.str();
}

std::string AdvancedPromptBuilder::buildMultiShotAnalysis(
    const std::vector<pv::modern::ShotSegmentation::ShotEvent>& shots,
    const CoachingPrompts::CoachingContext& context) const {
    
    std::ostringstream oss;
    
    oss << "Analyze this sequence of " << shots.size() << " shots:\n\n";
    
    for (size_t i = 0; i < shots.size() && i < 5; ++i) {  // Limit to 5 shots for brevity
        oss << "Shot " << (i + 1) << ":\n";
        oss << "- Duration: " << shots[i].duration << "s\n";
        oss << "- Legal: " << (shots[i].isLegalShot ? "Yes" : "No") << "\n";
        oss << "- Speed: " << shots[i].maxSpeed << " pixels/s\n\n";
    }
    
    oss << "Player profile: Skill level " << context.player.skillLevel << "/10\n";
    oss << "Overall session success rate: ";
    if (context.session.shotsAttempted > 0) {
        oss << (static_cast<float>(context.session.successfulShots) / context.session.shotsAttempted * 100);
        oss << "%\n\n";
    } else {
        oss << "N/A\n\n";
    }
    
    oss << "Provide pattern analysis and recommendations for improvement.";
    
    return oss.str();
}

} // namespace ai
} // namespace pv