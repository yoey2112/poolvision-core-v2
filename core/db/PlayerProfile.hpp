#pragma once
#include <string>
#include <ctime>

namespace pv {

/**
 * @brief Player skill levels
 */
enum class SkillLevel {
    Beginner = 1,
    Intermediate = 2,
    Advanced = 3,
    Expert = 4,
    Professional = 5
};

/**
 * @brief Player handedness
 */
enum class Handedness {
    Right,
    Left,
    Ambidextrous
};

/**
 * @brief Player profile information
 */
struct PlayerProfile {
    int id = -1;
    std::string name;
    std::string avatar;  // Path to avatar image or icon identifier
    SkillLevel skillLevel = SkillLevel::Intermediate;
    Handedness handedness = Handedness::Right;
    
    // Game preferences
    std::string preferredGameType = "8-Ball";
    bool soundEnabled = true;
    
    // Statistics (cached from database)
    int gamesPlayed = 0;
    int gamesWon = 0;
    int totalShots = 0;
    int successfulShots = 0;
    float winRate = 0.0f;
    float shotSuccessRate = 0.0f;
    
    // Metadata
    std::time_t createdAt = 0;
    std::time_t lastPlayedAt = 0;
    
    /**
     * @brief Check if profile is valid
     */
    bool isValid() const {
        return !name.empty();
    }
    
    /**
     * @brief Get skill level as string
     */
    std::string getSkillLevelString() const {
        switch (skillLevel) {
            case SkillLevel::Beginner: return "Beginner";
            case SkillLevel::Intermediate: return "Intermediate";
            case SkillLevel::Advanced: return "Advanced";
            case SkillLevel::Expert: return "Expert";
            case SkillLevel::Professional: return "Professional";
            default: return "Unknown";
        }
    }
    
    /**
     * @brief Get handedness as string
     */
    std::string getHandednessString() const {
        switch (handedness) {
            case Handedness::Right: return "Right";
            case Handedness::Left: return "Left";
            case Handedness::Ambidextrous: return "Ambidextrous";
            default: return "Unknown";
        }
    }
};

/**
 * @brief Game session record
 */
struct GameSession {
    int id = -1;
    int player1Id = -1;
    int player2Id = -1;
    std::string gameType = "8-Ball";  // 8-Ball, 9-Ball, 10-Ball, Straight Pool
    int winnerId = -1;
    int player1Score = 0;
    int player2Score = 0;
    int durationSeconds = 0;
    std::time_t startedAt = 0;
    std::time_t finishedAt = 0;
    
    /**
     * @brief Check if session is complete
     */
    bool isComplete() const {
        return winnerId != -1 && finishedAt > 0;
    }
    
    /**
     * @brief Get game duration in minutes
     */
    int getDurationMinutes() const {
        return durationSeconds / 60;
    }
};

/**
 * @brief Individual shot record
 */
struct ShotRecord {
    int id = -1;
    int sessionId = -1;
    int playerId = -1;
    std::string shotType;  // "break", "bank", "combo", "cut", "safety", etc.
    bool successful = false;
    float ballX = 0.0f;  // Ball position before shot
    float ballY = 0.0f;
    float targetX = 0.0f;  // Target position
    float targetY = 0.0f;
    float shotSpeed = 0.0f;  // Shot velocity
    int shotNumber = 0;  // Shot number in game
    std::time_t timestamp = 0;
    
    /**
     * @brief Check if shot is valid
     */
    bool isValid() const {
        return sessionId != -1 && playerId != -1;
    }
};

} // namespace pv
