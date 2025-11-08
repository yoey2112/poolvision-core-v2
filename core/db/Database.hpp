#pragma once
#include "PlayerProfile.hpp"
#include <string>
#include <vector>
#include <memory>
#include <sqlite3.h>
#include <opencv2/opencv.hpp>

namespace pv {

// Forward declarations
class GameRecorder;

// Drill system data structures
struct DrillSession {
    int id = -1;
    int playerId = -1;
    std::string drillId;
    int difficulty = 1;
    int attemptsTotal = 0;
    int attemptsSuccessful = 0;
    double timeSpent = 0.0;
    double bestScore = 0.0;
    std::time_t startedAt = 0;
    std::time_t finishedAt = 0;
    bool completed = false;
};

struct DrillStats {
    int totalSessions = 0;
    int completedSessions = 0;
    double averageScore = 0.0;
    double bestScore = 0.0;
    double totalTimeSpent = 0.0;
    double successRate = 0.0;
    int totalAttempts = 0;
    int successfulAttempts = 0;
    std::time_t firstAttempt = 0;
    std::time_t lastAttempt = 0;
};

// Match system data structures
struct MatchRecord {
    int id = -1;
    int player1Id = -1;
    int player2Id = -1;
    std::string matchFormat;
    std::string gameType;
    int targetGames = 0;
    int player1Wins = 0;
    int player2Wins = 0;
    int winnerId = -1;
    bool completed = false;
    double durationMinutes = 0.0;
    std::time_t startedAt = 0;
    std::time_t finishedAt = 0;
    int tournamentId = -1;
    std::string notes;
};

struct HeadToHeadRecord {
    int player1Id = -1;
    int player2Id = -1;
    int player1Wins = 0;
    int player2Wins = 0;
    int totalMatches = 0;
    std::time_t firstMatch = 0;
    std::time_t lastMatch = 0;
    double averageMatchDuration = 0.0;
};

// Tournament system data structures
struct TournamentRecord {
    int id = -1;
    std::string name;
    std::string format;
    std::string gameType;
    int maxParticipants = 0;
    int currentParticipants = 0;
    bool isActive = false;
    bool isCompleted = false;
    std::time_t startedAt = 0;
    std::time_t finishedAt = 0;
    int winnerId = -1;
    std::string description;
    std::string bracketData; // JSON bracket structure
};

/**
 * @brief SQLite database wrapper for Pool Vision
 * 
 * Manages player profiles, game sessions, shots, statistics,
 * drill sessions, matches, and tournaments
 */
class Database {
public:
    Database();
    ~Database();
    
    /**
     * @brief Open database connection
     * @param dbPath Path to SQLite database file
     * @return true if successful
     */
    bool open(const std::string& dbPath = "data/poolvision.db");
    
    /**
     * @brief Close database connection
     */
    void close();
    
    /**
     * @brief Check if database is open
     */
    bool isOpen() const { return db_ != nullptr; }
    
    /**
     * @brief Initialize database schema
     * Creates all necessary tables if they don't exist
     */
    bool initializeSchema();
    
    // Player profile operations
    
    /**
     * @brief Create a new player profile
     * @return Player ID if successful, -1 on error
     */
    int createPlayer(PlayerProfile& player);
    
    /**
     * @brief Update existing player profile
     */
    bool updatePlayer(const PlayerProfile& player);
    
    /**
     * @brief Delete player profile
     */
    bool deletePlayer(int playerId);
    
    /**
     * @brief Get player by ID
     */
    PlayerProfile getPlayer(int playerId);
    
    /**
     * @brief Get all players
     */
    std::vector<PlayerProfile> getAllPlayers();
    
    /**
     * @brief Search players by name
     */
    std::vector<PlayerProfile> searchPlayers(const std::string& query);
    
    /**
     * @brief Get recently played players
     */
    std::vector<PlayerProfile> getRecentPlayers(int limit = 10);
    
    // Game session operations
    
    /**
     * @brief Create a new game session
     * @return Session ID if successful, -1 on error
     */
    int createSession(GameSession& session);
    
    /**
     * @brief Update game session
     */
    bool updateSession(const GameSession& session);
    
    /**
     * @brief Get session by ID
     */
    GameSession getSession(int sessionId);
    
    /**
     * @brief Get all sessions for a player
     */
    std::vector<GameSession> getPlayerSessions(int playerId);
    
    // Shot record operations
    
    /**
     * @brief Add shot record
     * @return Shot ID if successful, -1 on error
     */
    int addShot(ShotRecord& shot);
    
    /**
     * @brief Get all shots for a session
     */
    std::vector<ShotRecord> getSessionShots(int sessionId);
    
    /**
     * @brief Get all shots for a player
     */
    std::vector<ShotRecord> getPlayerShots(int playerId);
    
    // Frame operations
    
    /**
     * @brief Get frame snapshots for a session
     * @return Vector of frame snapshots (currently returns empty vector as frames aren't stored to DB yet)
     */
    std::vector<cv::Mat> getSessionFrames(int sessionId);
    
    // Statistics operations
    
    /**
     * @brief Update player statistics from game history
     */
    bool updatePlayerStats(int playerId);
    
    /**
     * @brief Get player win rate
     */
    float getPlayerWinRate(int playerId);
    
    /**
     * @brief Get player shot success rate
     */
    float getPlayerShotSuccessRate(int playerId);
    
    // Drill session operations
    
    /**
     * @brief Create a new drill session
     * @return Session ID if successful, -1 on error
     */
    int createDrillSession(DrillSession& session);
    
    /**
     * @brief Update drill session
     */
    bool updateDrillSession(const DrillSession& session);
    
    /**
     * @brief Get drill sessions for a player
     */
    std::vector<DrillSession> getPlayerDrillSessions(int playerId, int limit = 50);
    
    /**
     * @brief Get drill performance statistics
     */
    DrillStats getDrillStats(int playerId, const std::string& drillId);
    
    // Match operations
    
    /**
     * @brief Create a new match
     * @return Match ID if successful, -1 on error
     */
    int createMatch(MatchRecord& match);
    
    /**
     * @brief Update match record
     */
    bool updateMatch(const MatchRecord& match);
    
    /**
     * @brief Get match by ID
     */
    MatchRecord getMatch(int matchId);
    
    /**
     * @brief Get player match history
     */
    std::vector<MatchRecord> getPlayerMatches(int playerId, int limit = 50);
    
    /**
     * @brief Get head-to-head record between two players
     */
    HeadToHeadRecord getHeadToHead(int player1Id, int player2Id);
    
    // Tournament operations
    
    /**
     * @brief Create a new tournament
     * @return Tournament ID if successful, -1 on error
     */
    int createTournament(TournamentRecord& tournament);
    
    /**
     * @brief Update tournament record
     */
    bool updateTournament(const TournamentRecord& tournament);
    
    /**
     * @brief Get tournament by ID
     */
    TournamentRecord getTournament(int tournamentId);
    
    /**
     * @brief Get active tournaments
     */
    std::vector<TournamentRecord> getActiveTournaments();
    
    /**
     * @brief Get last error message
     */
    std::string getLastError() const { return lastError_; }
    
private:
    sqlite3* db_;
    std::string lastError_;
    
    /**
     * @brief Execute SQL query
     */
    bool execute(const std::string& sql);
    
    /**
     * @brief Prepare SQL statement
     */
    sqlite3_stmt* prepare(const std::string& sql);
    
    /**
     * @brief Bind text parameter
     */
    void bindText(sqlite3_stmt* stmt, int index, const std::string& value);
    
    /**
     * @brief Bind int parameter
     */
    void bindInt(sqlite3_stmt* stmt, int index, int value);
    
    /**
     * @brief Bind double parameter
     */
    void bindDouble(sqlite3_stmt* stmt, int index, double value);
};

} // namespace pv
