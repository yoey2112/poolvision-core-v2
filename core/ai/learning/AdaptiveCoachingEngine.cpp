#include "AdaptiveCoachingEngine.hpp"
#include <algorithm>
#include <numeric>
#include <sstream>
#include <random>
#include <iostream>
#include <iomanip>
#include <cmath>

namespace pv {
namespace ai {
namespace learning {

// StrategyEngine Implementation
std::string AdaptiveCoachingEngine::StrategyEngine::generatePersonalizedStrategy(
    const PlayerProfile& profile, const ShotAnalysisEngine::PatternAnalysis& patterns) {
    
    std::stringstream strategy;
    
    // Base strategy on skill level
    if (profile.skillLevel.overall < 0.3f) {
        strategy << "Focus on fundamentals: stance, grip, and basic aim. ";
    } else if (profile.skillLevel.overall < 0.6f) {
        strategy << "Develop consistency and position play. ";
    } else {
        strategy << "Refine advanced techniques and strategic thinking. ";
    }
    
    // Adapt to learning style
    switch (profile.learningStyle.primaryStyle) {
        case PlayerProfile::LearningStyle::Visual:
            strategy << "Use visual cues and demonstrations. ";
            break;
        case PlayerProfile::LearningStyle::Analytical:
            strategy << "Provide detailed shot analysis and statistics. ";
            break;
        case PlayerProfile::LearningStyle::Encouraging:
            strategy << "Focus on positive reinforcement and achievable goals. ";
            break;
        case PlayerProfile::LearningStyle::Technical:
            strategy << "Emphasize technical precision and mechanics. ";
            break;
        case PlayerProfile::LearningStyle::Progressive:
            strategy << "Gradually increase challenge level. ";
            break;
    }
    
    // Address specific weaknesses
    if (profile.skillLevel.accuracy < 0.4f) {
        strategy << "Priority: improve shot accuracy through aiming drills. ";
    }
    if (profile.skillLevel.positioning < 0.4f) {
        strategy << "Priority: develop position play awareness. ";
    }
    if (profile.skillLevel.consistency < 0.4f) {
        strategy << "Priority: build consistent pre-shot routine. ";
    }
    
    return strategy.str();
}

void AdaptiveCoachingEngine::StrategyEngine::adaptCoachingStrategy(
    PlayerProfile& profile, const DataCollectionEngine::ShotOutcomeData& recentShot) {
    
    // Update session state
    profile.sessionState.shotsAttempted++;
    if (recentShot.successful) {
        profile.sessionState.shotsSuccessful++;
    }
    
    // Calculate session trend
    float currentRate = static_cast<float>(profile.sessionState.shotsSuccessful) / 
                       std::max(1, profile.sessionState.shotsAttempted);
    
    // Update confidence based on recent performance
    if (recentShot.successful) {
        profile.sessionState.currentConfidence = std::min(1.0f, 
            profile.sessionState.currentConfidence + 0.05f);
    } else {
        profile.sessionState.currentConfidence = std::max(0.0f, 
            profile.sessionState.currentConfidence - 0.03f);
    }
    
    // Adapt coaching style based on confidence
    if (profile.sessionState.currentConfidence < 0.3f) {
        profile.learningStyle.primaryStyle = PlayerProfile::LearningStyle::Encouraging;
        profile.learningStyle.feedbackFrequency = 0.8f; // More frequent positive feedback
    } else if (profile.sessionState.currentConfidence > 0.8f) {
        profile.learningStyle.wantsChallenges = true; // Offer more challenges
    }
}

float AdaptiveCoachingEngine::StrategyEngine::calculateNextDifficulty(const PlayerProfile& profile) {
    float baseDifficulty = profile.skillLevel.overall;
    
    // Adjust based on session performance
    float sessionRate = static_cast<float>(profile.sessionState.shotsSuccessful) / 
                       std::max(1, profile.sessionState.shotsAttempted);
    
    if (sessionRate > 0.8f) {
        baseDifficulty += 0.1f; // Increase difficulty if performing well
    } else if (sessionRate < 0.4f) {
        baseDifficulty -= 0.1f; // Decrease difficulty if struggling
    }
    
    // Respect player preferences
    if (!profile.learningStyle.wantsChallenges) {
        baseDifficulty = std::min(baseDifficulty, profile.skillLevel.overall + 0.05f);
    }
    
    return std::clamp(baseDifficulty, 0.1f, 0.9f);
}

AdaptiveCoachingEngine::CoachingMessage AdaptiveCoachingEngine::StrategyEngine::generateContextualMessage(
    const PlayerProfile& profile, const GameState& gameState, 
    const ShotAnalysisEngine::ShotAnalysisResult& analysis) {
    
    CoachingMessage message;
    message.timestamp = std::chrono::steady_clock::now();
    message.hasVisualAid = profile.learningStyle.primaryStyle == PlayerProfile::LearningStyle::Visual;
    
    // Determine message priority and category based on analysis
    if (analysis.mainPrediction.successProbability < 0.3f) {
        message.priority = CoachingMessage::High;
        message.category = CoachingMessage::Technique;
        message.message = "This shot looks challenging. " + analysis.mainPrediction.reasoning;
        message.relevanceScore = 0.9f;
    } else if (analysis.mainPrediction.successProbability > 0.8f) {
        message.priority = CoachingMessage::Medium;
        message.category = CoachingMessage::Encouragement;
        message.message = "Great opportunity! " + analysis.mainPrediction.reasoning;
        message.relevanceScore = 0.7f;
    } else {
        message.priority = CoachingMessage::Medium;
        message.category = CoachingMessage::Strategy;
        message.message = "Consider your options. " + analysis.mainPrediction.reasoning;
        message.relevanceScore = 0.6f;
    }
    
    // Add insights from learning analysis
    if (!analysis.insights.empty()) {
        message.message += " " + analysis.insights[0].description;
    }
    
    // Adapt message style to player preference
    if (profile.learningStyle.primaryStyle == PlayerProfile::LearningStyle::Technical) {
        message.message += " (Difficulty: " + std::to_string(analysis.mainPrediction.difficultyRating) + ")";
    }
    
    message.context = "Game analysis with " + std::to_string(analysis.alternatives.size()) + " alternatives";
    
    return message;
}

// OllamaIntegration Implementation
bool AdaptiveCoachingEngine::OllamaIntegration::connectToOllama(const std::string& endpoint) {
    ollamaEndpoint_ = endpoint;
    
    // In a real implementation, this would test the connection
    // For now, assume connection is successful
    connected_ = true;
    return connected_;
}

std::string AdaptiveCoachingEngine::OllamaIntegration::generatePersonalizedPrompt(
    const PlayerProfile& profile, const std::string& context, 
    const CoachingMessage::Category& category) {
    
    std::stringstream prompt;
    
    prompt << "You are a personalized pool coach for " << profile.playerName << ". ";
    
    // Add skill context
    prompt << "Player skill level: ";
    if (profile.skillLevel.overall < 0.3f) {
        prompt << "beginner";
    } else if (profile.skillLevel.overall < 0.7f) {
        prompt << "intermediate";
    } else {
        prompt << "advanced";
    }
    prompt << " (overall: " << std::fixed << std::setprecision(1) << (profile.skillLevel.overall * 100) << "%)";
    
    // Add learning style context
    prompt << ". Learning style: ";
    switch (profile.learningStyle.primaryStyle) {
        case PlayerProfile::LearningStyle::Visual:
            prompt << "prefers visual demonstrations";
            break;
        case PlayerProfile::LearningStyle::Analytical:
            prompt << "prefers detailed analysis";
            break;
        case PlayerProfile::LearningStyle::Encouraging:
            prompt << "responds well to positive reinforcement";
            break;
        case PlayerProfile::LearningStyle::Technical:
            prompt << "enjoys technical details";
            break;
        case PlayerProfile::LearningStyle::Progressive:
            prompt << "prefers gradual challenges";
            break;
    }
    
    // Add session context
    prompt << ". Current session: " << profile.sessionState.shotsAttempted << " shots, ";
    float successRate = static_cast<float>(profile.sessionState.shotsSuccessful) / 
                       std::max(1, profile.sessionState.shotsAttempted);
    prompt << std::fixed << std::setprecision(1) << (successRate * 100) << "% success rate. ";
    
    // Add category-specific context
    switch (category) {
        case CoachingMessage::Technique:
            prompt << "Focus on technique improvement. ";
            break;
        case CoachingMessage::Strategy:
            prompt << "Provide strategic guidance. ";
            break;
        case CoachingMessage::Encouragement:
            prompt << "Provide motivational support. ";
            break;
        default:
            break;
    }
    
    prompt << "Context: " << context << ". ";
    prompt << "Provide a helpful, personalized coaching message (2-3 sentences max).";
    
    return prompt.str();
}

AdaptiveCoachingEngine::CoachingMessage AdaptiveCoachingEngine::OllamaIntegration::processOllamaResponse(
    const std::string& response, const PlayerProfile& profile) {
    
    CoachingMessage message;
    message.message = response;
    message.timestamp = std::chrono::steady_clock::now();
    message.priority = CoachingMessage::Medium;
    message.category = CoachingMessage::Strategy; // Default category
    message.relevanceScore = 0.7f; // Default relevance
    message.hasVisualAid = profile.learningStyle.primaryStyle == PlayerProfile::LearningStyle::Visual;
    message.context = "AI-generated personalized coaching";
    
    // Simple analysis of response content to determine category
    std::string lowerResponse = response;
    std::transform(lowerResponse.begin(), lowerResponse.end(), lowerResponse.begin(), ::tolower);
    
    if (lowerResponse.find("aim") != std::string::npos || 
        lowerResponse.find("stance") != std::string::npos ||
        lowerResponse.find("technique") != std::string::npos) {
        message.category = CoachingMessage::Technique;
    } else if (lowerResponse.find("good") != std::string::npos ||
               lowerResponse.find("great") != std::string::npos ||
               lowerResponse.find("excellent") != std::string::npos) {
        message.category = CoachingMessage::Encouragement;
    } else if (lowerResponse.find("position") != std::string::npos ||
               lowerResponse.find("plan") != std::string::npos) {
        message.category = CoachingMessage::Positioning;
    }
    
    return message;
}

std::string AdaptiveCoachingEngine::OllamaIntegration::enhanceCoachingWithAI(
    const std::string& baseCoaching, const PlayerProfile& profile) {
    
    if (!connected_) return baseCoaching;
    
    std::stringstream enhanced;
    enhanced << baseCoaching;
    
    // Add personalized enhancement based on profile
    if (profile.skillLevel.consistency < 0.4f) {
        enhanced << " Focus on developing a consistent pre-shot routine.";
    }
    
    if (profile.sessionState.currentConfidence < 0.3f) {
        enhanced << " Remember, improvement takes time - stay positive!";
    }
    
    return enhanced.str();
}

// PerformanceAssessor Implementation
void AdaptiveCoachingEngine::PerformanceAssessor::updateSkillAssessment(
    PlayerProfile& profile, const std::vector<DataCollectionEngine::ShotOutcomeData>& recentShots) {
    
    if (recentShots.empty()) return;
    
    // Calculate accuracy
    int successful = 0;
    float totalDifficulty = 0;
    std::map<DataCollectionEngine::ShotOutcomeData::ShotType, int> shotTypeCounts;
    std::map<DataCollectionEngine::ShotOutcomeData::ShotType, int> shotTypeSuccesses;
    
    for (const auto& shot : recentShots) {
        if (shot.successful) successful++;
        totalDifficulty += shot.shotDifficulty;
        shotTypeCounts[shot.shotType]++;
        if (shot.successful) shotTypeSuccesses[shot.shotType]++;
    }
    
    float rawAccuracy = static_cast<float>(successful) / recentShots.size();
    float avgDifficulty = totalDifficulty / recentShots.size();
    
    // Adjust accuracy for difficulty
    profile.skillLevel.accuracy = rawAccuracy * (1.0f + avgDifficulty * 0.5f);
    profile.skillLevel.accuracy = std::clamp(profile.skillLevel.accuracy, 0.0f, 1.0f);
    
    // Calculate consistency (variance in performance)
    if (recentShots.size() >= 10) {
        std::vector<float> performances;
        for (size_t i = 0; i < recentShots.size(); i += 5) {
            int batchSuccessful = 0;
            int batchSize = std::min(5, static_cast<int>(recentShots.size() - i));
            
            for (int j = 0; j < batchSize; ++j) {
                if (recentShots[i + j].successful) batchSuccessful++;
            }
            
            performances.push_back(static_cast<float>(batchSuccessful) / batchSize);
        }
        
        // Calculate variance
        float mean = std::accumulate(performances.begin(), performances.end(), 0.0f) / performances.size();
        float variance = 0;
        for (float perf : performances) {
            variance += std::pow(perf - mean, 2);
        }
        variance /= performances.size();
        
        profile.skillLevel.consistency = std::max(0.0f, 1.0f - variance);
    }
    
    // Estimate positioning skill based on shot type diversity
    float diversity = static_cast<float>(shotTypeCounts.size()) / 7.0f; // 7 shot types available
    profile.skillLevel.positioning = std::min(profile.skillLevel.accuracy + diversity * 0.2f, 1.0f);
    
    // Update overall skill
    profile.skillLevel.overall = (profile.skillLevel.accuracy * 0.4f +
                                 profile.skillLevel.positioning * 0.2f +
                                 profile.skillLevel.consistency * 0.2f +
                                 profile.skillLevel.strategy * 0.1f +
                                 profile.skillLevel.pressure * 0.1f);
}

float AdaptiveCoachingEngine::PerformanceAssessor::assessLearningProgress(const PlayerProfile& profile) {
    if (profile.history.sessionRatings.size() < 2) return 0.0f;
    
    // Calculate trend over recent sessions
    const auto& ratings = profile.history.sessionRatings;
    int recentSessions = std::min(10, static_cast<int>(ratings.size()));
    
    if (recentSessions < 2) return 0.0f;
    
    // Simple linear regression to find trend
    float sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0;
    
    for (int i = 0; i < recentSessions; ++i) {
        float x = static_cast<float>(i);
        float y = ratings[ratings.size() - recentSessions + i];
        
        sumX += x;
        sumY += y;
        sumXY += x * y;
        sumX2 += x * x;
    }
    
    float slope = (recentSessions * sumXY - sumX * sumY) / 
                  (recentSessions * sumX2 - sumX * sumX);
    
    return std::clamp(slope, -1.0f, 1.0f);
}

std::vector<std::string> AdaptiveCoachingEngine::PerformanceAssessor::detectPerformancePatterns(
    const PlayerProfile& profile) {
    
    std::vector<std::string> patterns;
    
    // Skill imbalances
    if (profile.skillLevel.accuracy > profile.skillLevel.positioning + 0.3f) {
        patterns.push_back("Strong accuracy but needs position play work");
    }
    
    if (profile.skillLevel.consistency < 0.3f && profile.skillLevel.accuracy > 0.6f) {
        patterns.push_back("Capable of good shots but lacks consistency");
    }
    
    // Session patterns
    if (profile.sessionState.shotsAttempted > 20) {
        float sessionRate = static_cast<float>(profile.sessionState.shotsSuccessful) / 
                           profile.sessionState.shotsAttempted;
        
        if (sessionRate < profile.skillLevel.accuracy - 0.2f) {
            patterns.push_back("Underperforming compared to usual skill level");
        } else if (sessionRate > profile.skillLevel.accuracy + 0.2f) {
            patterns.push_back("Having an exceptionally good session");
        }
    }
    
    // Confidence patterns
    if (profile.sessionState.currentConfidence < 0.3f) {
        patterns.push_back("Low confidence affecting performance");
    }
    
    return patterns;
}

float AdaptiveCoachingEngine::PerformanceAssessor::assessCoachingEffectiveness(
    const PlayerProfile& profile, const std::vector<CoachingMessage>& recentCoaching) {
    
    if (recentCoaching.empty() || profile.history.sessionRatings.size() < 2) {
        return 0.5f; // Neutral effectiveness
    }
    
    // Simple measure: improvement rate when receiving coaching
    float progressRate = assessLearningProgress(profile);
    float messageRelevance = 0;
    
    for (const auto& message : recentCoaching) {
        messageRelevance += message.relevanceScore;
    }
    messageRelevance /= recentCoaching.size();
    
    return std::clamp((progressRate + 1.0f) / 2.0f * messageRelevance, 0.0f, 1.0f);
}

// AdaptiveCoachingEngine Implementation
AdaptiveCoachingEngine::AdaptiveCoachingEngine(DataCollectionEngine* dataEngine,
                                               ShotAnalysisEngine* analysisEngine)
    : dataEngine_(dataEngine), analysisEngine_(analysisEngine) {
    
    strategyEngine_ = std::make_unique<StrategyEngine>();
    ollamaIntegration_ = std::make_unique<OllamaIntegration>();
    performanceAssessor_ = std::make_unique<PerformanceAssessor>();
    
    std::cout << "Adaptive Coaching Engine initialized" << std::endl;
}

AdaptiveCoachingEngine::~AdaptiveCoachingEngine() {
    stopCoaching();
}

void AdaptiveCoachingEngine::startCoaching() {
    if (coachingActive_.load()) return;
    
    coachingActive_ = true;
    coachingThread_ = std::thread(&AdaptiveCoachingEngine::coachingLoop, this);
    
    std::cout << "Adaptive Coaching Engine started" << std::endl;
}

void AdaptiveCoachingEngine::stopCoaching() {
    if (!coachingActive_.load()) return;
    
    coachingActive_ = false;
    coachingCondition_.notify_all();
    
    if (coachingThread_.joinable()) {
        coachingThread_.join();
    }
    
    std::cout << "Adaptive Coaching Engine stopped" << std::endl;
}

void AdaptiveCoachingEngine::addPlayer(int playerId, const std::string& playerName) {
    std::lock_guard<std::mutex> lock(profileMutex_);
    initializePlayerProfile(playerId, playerName);
}

void AdaptiveCoachingEngine::updatePlayerProfile(int playerId, 
                                                const DataCollectionEngine::PlayerBehaviorData& behavior) {
    std::lock_guard<std::mutex> lock(profileMutex_);
    
    auto it = playerProfiles_.find(playerId);
    if (it == playerProfiles_.end()) return;
    
    auto& profile = it->second;
    
    // Update learning style based on behavior
    if (behavior.aimingTime > 5.0f) {
        profile.learningStyle.primaryStyle = PlayerProfile::LearningStyle::Analytical;
        profile.learningStyle.detailLevel = 0.8f;
    } else if (behavior.aimingTime < 2.0f) {
        profile.learningStyle.feedbackFrequency = 0.4f; // Less frequent feedback for quick players
    }
    
    // Update pressure handling
    if (behavior.confidenceLevel < 0.3f) {
        profile.skillLevel.pressure = std::max(0.0f, profile.skillLevel.pressure - 0.1f);
    } else if (behavior.confidenceLevel > 0.8f) {
        profile.skillLevel.pressure = std::min(1.0f, profile.skillLevel.pressure + 0.05f);
    }
}

AdaptiveCoachingEngine::PlayerProfile AdaptiveCoachingEngine::getPlayerProfile(int playerId) {
    std::lock_guard<std::mutex> lock(profileMutex_);
    
    auto it = playerProfiles_.find(playerId);
    if (it != playerProfiles_.end()) {
        return it->second;
    }
    
    return PlayerProfile{}; // Return empty profile if not found
}

void AdaptiveCoachingEngine::startCoachingSession(int playerId) {
    std::lock_guard<std::mutex> lock(sessionMutex_);
    
    CoachingSession session;
    session.playerId = playerId;
    session.startTime = std::chrono::steady_clock::now();
    session.playerEngagement = 0.5f; // Start neutral
    session.lessonCompleted = false;
    
    activeSessions_[playerId] = session;
    
    // Reset session state in player profile
    std::lock_guard<std::mutex> profileLock(profileMutex_);
    auto profileIt = playerProfiles_.find(playerId);
    if (profileIt != playerProfiles_.end()) {
        profileIt->second.sessionState.shotsAttempted = 0;
        profileIt->second.sessionState.shotsSuccessful = 0;
        profileIt->second.sessionState.sessionStart = std::chrono::steady_clock::now();
        profileIt->second.sessionState.recentFeedback.clear();
    }
    
    metrics_.sessionsStarted.fetch_add(1);
}

void AdaptiveCoachingEngine::endCoachingSession(int playerId) {
    std::lock_guard<std::mutex> lock(sessionMutex_);
    
    auto it = activeSessions_.find(playerId);
    if (it != activeSessions_.end()) {
        // Update historical data
        std::lock_guard<std::mutex> profileLock(profileMutex_);
        auto profileIt = playerProfiles_.find(playerId);
        if (profileIt != playerProfiles_.end()) {
            auto& profile = profileIt->second;
            
            // Calculate session rating
            float sessionRating = 0.0f;
            if (profile.sessionState.shotsAttempted > 0) {
                sessionRating = static_cast<float>(profile.sessionState.shotsSuccessful) / 
                               profile.sessionState.shotsAttempted;
            }
            
            // Update history
            profile.history.sessionRatings.push_back(sessionRating);
            if (profile.history.sessionRatings.size() > 20) {
                profile.history.sessionRatings.erase(profile.history.sessionRatings.begin());
            }
            
            profile.history.totalSessions++;
            if (sessionRating > profile.history.bestSessionRating) {
                profile.history.bestSessionRating = sessionRating;
            }
        }
        
        activeSessions_.erase(it);
    }
}

AdaptiveCoachingEngine::CoachingSession AdaptiveCoachingEngine::getCurrentSession(int playerId) {
    std::lock_guard<std::mutex> lock(sessionMutex_);
    
    auto it = activeSessions_.find(playerId);
    if (it != activeSessions_.end()) {
        return it->second;
    }
    
    return CoachingSession{}; // Return empty session
}

AdaptiveCoachingEngine::CoachingMessage AdaptiveCoachingEngine::generateRealtimeCoaching(
    int playerId, const GameState& gameState, 
    const ShotAnalysisEngine::ShotAnalysisResult& analysis) {
    
    auto profile = getPlayerProfile(playerId);
    if (profile.playerId == 0) {
        // Return empty message for unknown player
        return CoachingMessage{};
    }
    
    // Generate contextual message
    auto message = strategyEngine_->generateContextualMessage(profile, gameState, analysis);
    
    // Enhance with Ollama if available
    if (ollamaIntegration_->isConnected()) {
        std::string context = "Shot analysis: " + analysis.mainPrediction.reasoning;
        std::string prompt = ollamaIntegration_->generatePersonalizedPrompt(profile, context, message.category);
        
        // In a real implementation, this would make an actual API call to Ollama
        // For now, enhance the existing message
        message.message = ollamaIntegration_->enhanceCoachingWithAI(message.message, profile);
    }
    
    // Update metrics
    metrics_.messagesGenerated.fetch_add(1);
    double currentAvgRelevance = metrics_.avgMessageRelevance.load();
    metrics_.avgMessageRelevance.store(currentAvgRelevance * 0.9 + message.relevanceScore * 0.1);
    
    return message;
}

void AdaptiveCoachingEngine::updateFromShotOutcome(int playerId, 
                                                  const DataCollectionEngine::ShotOutcomeData& shot) {
    // Add to processing queue
    std::lock_guard<std::mutex> lock(queueMutex_);
    shotQueue_.push(std::make_pair(playerId, shot));
    coachingCondition_.notify_one();
}

void AdaptiveCoachingEngine::integrateWithOllama(const std::string& ollamaEndpoint) {
    if (ollamaIntegration_->connectToOllama(ollamaEndpoint)) {
        std::cout << "Connected to Ollama at " << ollamaEndpoint << std::endl;
    } else {
        std::cout << "Failed to connect to Ollama" << std::endl;
    }
}

std::string AdaptiveCoachingEngine::enhanceExistingCoaching(int playerId, 
                                                          const std::string& baseCoaching) {
    auto profile = getPlayerProfile(playerId);
    return ollamaIntegration_->enhanceCoachingWithAI(baseCoaching, profile);
}

std::vector<std::string> AdaptiveCoachingEngine::generatePlayerInsights(int playerId) {
    auto profile = getPlayerProfile(playerId);
    return performanceAssessor_->detectPerformancePatterns(profile);
}

float AdaptiveCoachingEngine::calculateImprovementRate(int playerId) {
    auto profile = getPlayerProfile(playerId);
    return performanceAssessor_->assessLearningProgress(profile);
}

std::map<std::string, float> AdaptiveCoachingEngine::getSkillBreakdown(int playerId) {
    auto profile = getPlayerProfile(playerId);
    
    std::map<std::string, float> breakdown;
    breakdown["accuracy"] = profile.skillLevel.accuracy;
    breakdown["positioning"] = profile.skillLevel.positioning;
    breakdown["strategy"] = profile.skillLevel.strategy;
    breakdown["consistency"] = profile.skillLevel.consistency;
    breakdown["pressure"] = profile.skillLevel.pressure;
    breakdown["overall"] = profile.skillLevel.overall;
    
    return breakdown;
}

void AdaptiveCoachingEngine::coachingLoop() {
    while (coachingActive_.load()) {
        std::unique_lock<std::mutex> lock(queueMutex_);
        
        coachingCondition_.wait_for(lock, std::chrono::seconds(1), [this] {
            return !coachingActive_.load() || !shotQueue_.empty();
        });
        
        if (!coachingActive_.load()) break;
        
        processShotQueue();
        lock.unlock();
        
        updatePlayerProfiles();
        generateScheduledCoaching();
        
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

void AdaptiveCoachingEngine::processShotQueue() {
    while (!shotQueue_.empty()) {
        auto [playerId, shot] = shotQueue_.front();
        shotQueue_.pop();
        
        // Update player profile
        {
            std::lock_guard<std::mutex> lock(profileMutex_);
            auto it = playerProfiles_.find(playerId);
            if (it != playerProfiles_.end()) {
                strategyEngine_->adaptCoachingStrategy(it->second, shot);
                updatePlayerSkills(it->second, shot);
            }
        }
    }
}

void AdaptiveCoachingEngine::updatePlayerProfiles() {
    std::lock_guard<std::mutex> lock(profileMutex_);
    
    for (auto& [playerId, profile] : playerProfiles_) {
        // Get recent shot data
        auto recentShots = dataEngine_->getPlayerShotHistory(playerId, 50);
        if (!recentShots.empty()) {
            performanceAssessor_->updateSkillAssessment(profile, recentShots);
        }
        
        // Update average improvement
        profile.history.averageImprovement = performanceAssessor_->assessLearningProgress(profile);
    }
}

void AdaptiveCoachingEngine::generateScheduledCoaching() {
    // Generate periodic insights and lessons for active sessions
    std::lock_guard<std::mutex> sessionLock(sessionMutex_);
    
    for (auto& [playerId, session] : activeSessions_) {
        auto profile = getPlayerProfile(playerId);
        
        // Calculate session duration
        auto now = std::chrono::steady_clock::now();
        auto sessionDuration = std::chrono::duration_cast<std::chrono::minutes>(
            now - session.startTime).count();
        
        // Generate periodic encouragement for long sessions
        if (sessionDuration > 15 && sessionDuration % 10 == 0) {
            CoachingMessage encouragement;
            encouragement.category = CoachingMessage::Encouragement;
            encouragement.priority = CoachingMessage::Low;
            encouragement.message = "You're doing great! Keep up the good work.";
            encouragement.timestamp = now;
            encouragement.relevanceScore = 0.6f;
            
            session.messages.push_back(encouragement);
        }
    }
}

void AdaptiveCoachingEngine::initializePlayerProfile(int playerId, const std::string& playerName) {
    PlayerProfile profile;
    profile.playerId = playerId;
    profile.playerName = playerName;
    
    // Initialize with neutral values
    profile.skillLevel.overall = 0.5f;
    profile.skillLevel.accuracy = 0.5f;
    profile.skillLevel.positioning = 0.5f;
    profile.skillLevel.strategy = 0.5f;
    profile.skillLevel.consistency = 0.5f;
    profile.skillLevel.pressure = 0.5f;
    
    // Default learning style
    profile.learningStyle.primaryStyle = PlayerProfile::LearningStyle::Progressive;
    profile.learningStyle.detailLevel = 0.5f;
    profile.learningStyle.feedbackFrequency = 0.6f;
    profile.learningStyle.wantsChallenges = true;
    
    // Initialize session state
    profile.sessionState.shotsAttempted = 0;
    profile.sessionState.shotsSuccessful = 0;
    profile.sessionState.currentConfidence = 0.5f;
    profile.sessionState.sessionTrend = 0.0f;
    profile.sessionState.sessionStart = std::chrono::steady_clock::now();
    
    // Initialize history
    profile.history.totalSessions = 0;
    profile.history.bestSessionRating = 0.0f;
    profile.history.averageImprovement = 0.0f;
    
    playerProfiles_[playerId] = profile;
}

void AdaptiveCoachingEngine::updatePlayerSkills(PlayerProfile& profile, 
                                               const DataCollectionEngine::ShotOutcomeData& shot) {
    // Update shot type progression
    auto& progression = profile.history.shotTypeProgression[shot.shotType];
    if (shot.successful) {
        progression = std::min(1.0f, progression + 0.02f);
    } else {
        progression = std::max(0.0f, progression - 0.01f);
    }
    
    // Update strategy skill based on shot selection
    if (shot.shotDifficulty > 0.7f && !shot.successful) {
        profile.skillLevel.strategy = std::max(0.0f, profile.skillLevel.strategy - 0.01f);
    } else if (shot.shotDifficulty < 0.3f && shot.successful) {
        profile.skillLevel.strategy = std::min(1.0f, profile.skillLevel.strategy + 0.005f);
    }
}

void AdaptiveCoachingEngine::logCoachingReport() {
    auto metrics = getCoachingMetrics();
    
    std::cout << "\n=== Adaptive Coaching Engine Report ===" << std::endl;
    std::cout << "Sessions started: " << metrics.sessionsStarted.load() << std::endl;
    std::cout << "Messages generated: " << metrics.messagesGenerated.load() << std::endl;
    std::cout << "Lessons completed: " << metrics.lessonsCompleted.load() << std::endl;
    std::cout << "Avg message relevance: " << std::fixed << std::setprecision(2) 
              << metrics.avgMessageRelevance.load() << std::endl;
    std::cout << "Avg player engagement: " << metrics.avgPlayerEngagement.load() << std::endl;
    std::cout << "Avg improvement rate: " << metrics.avgImprovementRate.load() << std::endl;
    std::cout << "=========================================" << std::endl;
}

// Factory Implementation
std::unique_ptr<AdaptiveCoachingEngine> AdaptiveCoachingFactory::createWithOllama(
    DataCollectionEngine* dataEngine, ShotAnalysisEngine* analysisEngine,
    const std::string& ollamaEndpoint) {
    
    auto engine = std::make_unique<AdaptiveCoachingEngine>(dataEngine, analysisEngine);
    engine->integrateWithOllama(ollamaEndpoint);
    return engine;
}

std::unique_ptr<AdaptiveCoachingEngine> AdaptiveCoachingFactory::createStandalone(
    DataCollectionEngine* dataEngine, ShotAnalysisEngine* analysisEngine) {
    
    return std::make_unique<AdaptiveCoachingEngine>(dataEngine, analysisEngine);
}

} // namespace learning
} // namespace ai
} // namespace pv