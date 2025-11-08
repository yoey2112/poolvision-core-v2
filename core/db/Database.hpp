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

/**
 * @brief SQLite database wrapper for Pool Vision
 * 
 * Manages player profiles, game sessions, shots, and statistics
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
