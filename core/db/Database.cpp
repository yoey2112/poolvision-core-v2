#include "Database.hpp"
#include <iostream>
#include <sstream>
#include <ctime>

namespace pv {

Database::Database() : db_(nullptr) {
}

Database::~Database() {
    close();
}

bool Database::open(const std::string& dbPath) {
    // Create data directory if it doesn't exist
#ifdef _WIN32
    system("if not exist data mkdir data");
#else
    system("mkdir -p data");
#endif
    
    int rc = sqlite3_open(dbPath.c_str(), &db_);
    if (rc != SQLITE_OK) {
        lastError_ = sqlite3_errmsg(db_);
        sqlite3_close(db_);
        db_ = nullptr;
        return false;
    }
    
    // Enable foreign keys
    execute("PRAGMA foreign_keys = ON;");
    
    return initializeSchema();
}

void Database::close() {
    if (db_) {
        sqlite3_close(db_);
        db_ = nullptr;
    }
}

bool Database::execute(const std::string& sql) {
    char* errMsg = nullptr;
    int rc = sqlite3_exec(db_, sql.c_str(), nullptr, nullptr, &errMsg);
    
    if (rc != SQLITE_OK) {
        lastError_ = errMsg ? errMsg : "Unknown error";
        sqlite3_free(errMsg);
        return false;
    }
    
    return true;
}

sqlite3_stmt* Database::prepare(const std::string& sql) {
    sqlite3_stmt* stmt = nullptr;
    int rc = sqlite3_prepare_v2(db_, sql.c_str(), -1, &stmt, nullptr);
    
    if (rc != SQLITE_OK) {
        lastError_ = sqlite3_errmsg(db_);
        return nullptr;
    }
    
    return stmt;
}

void Database::bindText(sqlite3_stmt* stmt, int index, const std::string& value) {
    sqlite3_bind_text(stmt, index, value.c_str(), -1, SQLITE_TRANSIENT);
}

void Database::bindInt(sqlite3_stmt* stmt, int index, int value) {
    sqlite3_bind_int(stmt, index, value);
}

void Database::bindDouble(sqlite3_stmt* stmt, int index, double value) {
    sqlite3_bind_double(stmt, index, value);
}

bool Database::initializeSchema() {
    const char* createPlayersSql = R"(
        CREATE TABLE IF NOT EXISTS players (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            avatar TEXT,
            skill_level INTEGER DEFAULT 2,
            handedness INTEGER DEFAULT 0,
            preferred_game_type TEXT DEFAULT '8-Ball',
            sound_enabled INTEGER DEFAULT 1,
            games_played INTEGER DEFAULT 0,
            games_won INTEGER DEFAULT 0,
            total_shots INTEGER DEFAULT 0,
            successful_shots INTEGER DEFAULT 0,
            created_at INTEGER NOT NULL,
            last_played_at INTEGER
        );
    )";
    
    const char* createGamesSql = R"(
        CREATE TABLE IF NOT EXISTS game_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            player1_id INTEGER NOT NULL,
            player2_id INTEGER,
            game_type TEXT NOT NULL,
            winner_id INTEGER,
            player1_score INTEGER DEFAULT 0,
            player2_score INTEGER DEFAULT 0,
            duration_seconds INTEGER DEFAULT 0,
            started_at INTEGER NOT NULL,
            finished_at INTEGER,
            FOREIGN KEY (player1_id) REFERENCES players(id) ON DELETE CASCADE,
            FOREIGN KEY (player2_id) REFERENCES players(id) ON DELETE CASCADE,
            FOREIGN KEY (winner_id) REFERENCES players(id) ON DELETE SET NULL
        );
    )";
    
    const char* createShotsSql = R"(
        CREATE TABLE IF NOT EXISTS shot_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            player_id INTEGER NOT NULL,
            shot_type TEXT,
            successful INTEGER DEFAULT 0,
            ball_x REAL DEFAULT 0,
            ball_y REAL DEFAULT 0,
            target_x REAL DEFAULT 0,
            target_y REAL DEFAULT 0,
            shot_speed REAL DEFAULT 0,
            shot_number INTEGER DEFAULT 0,
            timestamp INTEGER NOT NULL,
            FOREIGN KEY (session_id) REFERENCES game_sessions(id) ON DELETE CASCADE,
            FOREIGN KEY (player_id) REFERENCES players(id) ON DELETE CASCADE
        );
    )";
    
    const char* createIndexesSql = R"(
        CREATE INDEX IF NOT EXISTS idx_players_name ON players(name);
        CREATE INDEX IF NOT EXISTS idx_sessions_player1 ON game_sessions(player1_id);
        CREATE INDEX IF NOT EXISTS idx_sessions_player2 ON game_sessions(player2_id);
        CREATE INDEX IF NOT EXISTS idx_shots_session ON shot_records(session_id);
        CREATE INDEX IF NOT EXISTS idx_shots_player ON shot_records(player_id);
    )";
    
    return execute(createPlayersSql) &&
           execute(createGamesSql) &&
           execute(createShotsSql) &&
           execute(createIndexesSql);
}

// Player profile operations

int Database::createPlayer(PlayerProfile& player) {
    if (!isOpen()) return -1;
    
    player.createdAt = std::time(nullptr);
    
    const char* sql = R"(
        INSERT INTO players (name, avatar, skill_level, handedness, 
                           preferred_game_type, sound_enabled, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?);
    )";
    
    sqlite3_stmt* stmt = prepare(sql);
    if (!stmt) return -1;
    
    bindText(stmt, 1, player.name);
    bindText(stmt, 2, player.avatar);
    bindInt(stmt, 3, static_cast<int>(player.skillLevel));
    bindInt(stmt, 4, static_cast<int>(player.handedness));
    bindText(stmt, 5, player.preferredGameType);
    bindInt(stmt, 6, player.soundEnabled ? 1 : 0);
    bindInt(stmt, 7, static_cast<int>(player.createdAt));
    
    int rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    
    if (rc != SQLITE_DONE) {
        lastError_ = sqlite3_errmsg(db_);
        return -1;
    }
    
    player.id = static_cast<int>(sqlite3_last_insert_rowid(db_));
    return player.id;
}

bool Database::updatePlayer(const PlayerProfile& player) {
    if (!isOpen() || player.id < 0) return false;
    
    const char* sql = R"(
        UPDATE players 
        SET name = ?, avatar = ?, skill_level = ?, handedness = ?,
            preferred_game_type = ?, sound_enabled = ?
        WHERE id = ?;
    )";
    
    sqlite3_stmt* stmt = prepare(sql);
    if (!stmt) return false;
    
    bindText(stmt, 1, player.name);
    bindText(stmt, 2, player.avatar);
    bindInt(stmt, 3, static_cast<int>(player.skillLevel));
    bindInt(stmt, 4, static_cast<int>(player.handedness));
    bindText(stmt, 5, player.preferredGameType);
    bindInt(stmt, 6, player.soundEnabled ? 1 : 0);
    bindInt(stmt, 7, player.id);
    
    int rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    
    return rc == SQLITE_DONE;
}

bool Database::deletePlayer(int playerId) {
    if (!isOpen()) return false;
    
    const char* sql = "DELETE FROM players WHERE id = ?;";
    
    sqlite3_stmt* stmt = prepare(sql);
    if (!stmt) return false;
    
    bindInt(stmt, 1, playerId);
    
    int rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    
    return rc == SQLITE_DONE;
}

PlayerProfile Database::getPlayer(int playerId) {
    PlayerProfile player;
    if (!isOpen()) return player;
    
    const char* sql = R"(
        SELECT id, name, avatar, skill_level, handedness, 
               preferred_game_type, sound_enabled, games_played, games_won,
               total_shots, successful_shots, created_at, last_played_at
        FROM players WHERE id = ?;
    )";
    
    sqlite3_stmt* stmt = prepare(sql);
    if (!stmt) return player;
    
    bindInt(stmt, 1, playerId);
    
    if (sqlite3_step(stmt) == SQLITE_ROW) {
        player.id = sqlite3_column_int(stmt, 0);
        player.name = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
        player.avatar = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));
        player.skillLevel = static_cast<SkillLevel>(sqlite3_column_int(stmt, 3));
        player.handedness = static_cast<Handedness>(sqlite3_column_int(stmt, 4));
        player.preferredGameType = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 5));
        player.soundEnabled = sqlite3_column_int(stmt, 6) != 0;
        player.gamesPlayed = sqlite3_column_int(stmt, 7);
        player.gamesWon = sqlite3_column_int(stmt, 8);
        player.totalShots = sqlite3_column_int(stmt, 9);
        player.successfulShots = sqlite3_column_int(stmt, 10);
        player.createdAt = sqlite3_column_int(stmt, 11);
        player.lastPlayedAt = sqlite3_column_int(stmt, 12);
        
        // Calculate rates
        if (player.gamesPlayed > 0) {
            player.winRate = static_cast<float>(player.gamesWon) / player.gamesPlayed;
        }
        if (player.totalShots > 0) {
            player.shotSuccessRate = static_cast<float>(player.successfulShots) / player.totalShots;
        }
    }
    
    sqlite3_finalize(stmt);
    return player;
}

std::vector<PlayerProfile> Database::getAllPlayers() {
    std::vector<PlayerProfile> players;
    if (!isOpen()) return players;
    
    const char* sql = R"(
        SELECT id, name, avatar, skill_level, handedness, 
               preferred_game_type, sound_enabled, games_played, games_won,
               total_shots, successful_shots, created_at, last_played_at
        FROM players ORDER BY name;
    )";
    
    sqlite3_stmt* stmt = prepare(sql);
    if (!stmt) return players;
    
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        PlayerProfile player;
        player.id = sqlite3_column_int(stmt, 0);
        player.name = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
        player.avatar = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));
        player.skillLevel = static_cast<SkillLevel>(sqlite3_column_int(stmt, 3));
        player.handedness = static_cast<Handedness>(sqlite3_column_int(stmt, 4));
        player.preferredGameType = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 5));
        player.soundEnabled = sqlite3_column_int(stmt, 6) != 0;
        player.gamesPlayed = sqlite3_column_int(stmt, 7);
        player.gamesWon = sqlite3_column_int(stmt, 8);
        player.totalShots = sqlite3_column_int(stmt, 9);
        player.successfulShots = sqlite3_column_int(stmt, 10);
        player.createdAt = sqlite3_column_int(stmt, 11);
        player.lastPlayedAt = sqlite3_column_int(stmt, 12);
        
        if (player.gamesPlayed > 0) {
            player.winRate = static_cast<float>(player.gamesWon) / player.gamesPlayed;
        }
        if (player.totalShots > 0) {
            player.shotSuccessRate = static_cast<float>(player.successfulShots) / player.totalShots;
        }
        
        players.push_back(player);
    }
    
    sqlite3_finalize(stmt);
    return players;
}

std::vector<PlayerProfile> Database::searchPlayers(const std::string& query) {
    std::vector<PlayerProfile> players;
    if (!isOpen()) return players;
    
    const char* sql = R"(
        SELECT id, name, avatar, skill_level, handedness, 
               preferred_game_type, sound_enabled, games_played, games_won,
               total_shots, successful_shots, created_at, last_played_at
        FROM players 
        WHERE name LIKE ? 
        ORDER BY name;
    )";
    
    sqlite3_stmt* stmt = prepare(sql);
    if (!stmt) return players;
    
    std::string searchPattern = "%" + query + "%";
    bindText(stmt, 1, searchPattern);
    
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        PlayerProfile player;
        player.id = sqlite3_column_int(stmt, 0);
        player.name = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
        player.avatar = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));
        player.skillLevel = static_cast<SkillLevel>(sqlite3_column_int(stmt, 3));
        player.handedness = static_cast<Handedness>(sqlite3_column_int(stmt, 4));
        player.preferredGameType = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 5));
        player.soundEnabled = sqlite3_column_int(stmt, 6) != 0;
        player.gamesPlayed = sqlite3_column_int(stmt, 7);
        player.gamesWon = sqlite3_column_int(stmt, 8);
        player.totalShots = sqlite3_column_int(stmt, 9);
        player.successfulShots = sqlite3_column_int(stmt, 10);
        player.createdAt = sqlite3_column_int(stmt, 11);
        player.lastPlayedAt = sqlite3_column_int(stmt, 12);
        
        if (player.gamesPlayed > 0) {
            player.winRate = static_cast<float>(player.gamesWon) / player.gamesPlayed;
        }
        if (player.totalShots > 0) {
            player.shotSuccessRate = static_cast<float>(player.successfulShots) / player.totalShots;
        }
        
        players.push_back(player);
    }
    
    sqlite3_finalize(stmt);
    return players;
}

std::vector<PlayerProfile> Database::getRecentPlayers(int limit) {
    std::vector<PlayerProfile> players;
    if (!isOpen()) return players;
    
    const char* sql = R"(
        SELECT id, name, avatar, skill_level, handedness, 
               preferred_game_type, sound_enabled, games_played, games_won,
               total_shots, successful_shots, created_at, last_played_at
        FROM players 
        WHERE last_played_at IS NOT NULL
        ORDER BY last_played_at DESC 
        LIMIT ?;
    )";
    
    sqlite3_stmt* stmt = prepare(sql);
    if (!stmt) return players;
    
    bindInt(stmt, 1, limit);
    
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        PlayerProfile player;
        player.id = sqlite3_column_int(stmt, 0);
        player.name = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
        player.avatar = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));
        player.skillLevel = static_cast<SkillLevel>(sqlite3_column_int(stmt, 3));
        player.handedness = static_cast<Handedness>(sqlite3_column_int(stmt, 4));
        player.preferredGameType = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 5));
        player.soundEnabled = sqlite3_column_int(stmt, 6) != 0;
        player.gamesPlayed = sqlite3_column_int(stmt, 7);
        player.gamesWon = sqlite3_column_int(stmt, 8);
        player.totalShots = sqlite3_column_int(stmt, 9);
        player.successfulShots = sqlite3_column_int(stmt, 10);
        player.createdAt = sqlite3_column_int(stmt, 11);
        player.lastPlayedAt = sqlite3_column_int(stmt, 12);
        
        if (player.gamesPlayed > 0) {
            player.winRate = static_cast<float>(player.gamesWon) / player.gamesPlayed;
        }
        if (player.totalShots > 0) {
            player.shotSuccessRate = static_cast<float>(player.successfulShots) / player.totalShots;
        }
        
        players.push_back(player);
    }
    
    sqlite3_finalize(stmt);
    return players;
}

// Game session operations

int Database::createSession(GameSession& session) {
    if (!isOpen()) return -1;
    
    session.startedAt = std::time(nullptr);
    
    const char* sql = R"(
        INSERT INTO game_sessions (player1_id, player2_id, game_type, started_at)
        VALUES (?, ?, ?, ?);
    )";
    
    sqlite3_stmt* stmt = prepare(sql);
    if (!stmt) return -1;
    
    bindInt(stmt, 1, session.player1Id);
    bindInt(stmt, 2, session.player2Id);
    bindText(stmt, 3, session.gameType);
    bindInt(stmt, 4, static_cast<int>(session.startedAt));
    
    int rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    
    if (rc != SQLITE_DONE) {
        lastError_ = sqlite3_errmsg(db_);
        return -1;
    }
    
    session.id = static_cast<int>(sqlite3_last_insert_rowid(db_));
    return session.id;
}

bool Database::updateSession(const GameSession& session) {
    if (!isOpen() || session.id < 0) return false;
    
    const char* sql = R"(
        UPDATE game_sessions 
        SET winner_id = ?, player1_score = ?, player2_score = ?,
            duration_seconds = ?, finished_at = ?
        WHERE id = ?;
    )";
    
    sqlite3_stmt* stmt = prepare(sql);
    if (!stmt) return false;
    
    bindInt(stmt, 1, session.winnerId);
    bindInt(stmt, 2, session.player1Score);
    bindInt(stmt, 3, session.player2Score);
    bindInt(stmt, 4, session.durationSeconds);
    bindInt(stmt, 5, static_cast<int>(session.finishedAt));
    bindInt(stmt, 6, session.id);
    
    int rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    
    return rc == SQLITE_DONE;
}

GameSession Database::getSession(int sessionId) {
    GameSession session;
    if (!isOpen()) return session;
    
    const char* sql = R"(
        SELECT id, player1_id, player2_id, game_type, winner_id,
               player1_score, player2_score, duration_seconds,
               started_at, finished_at
        FROM game_sessions WHERE id = ?;
    )";
    
    sqlite3_stmt* stmt = prepare(sql);
    if (!stmt) return session;
    
    bindInt(stmt, 1, sessionId);
    
    if (sqlite3_step(stmt) == SQLITE_ROW) {
        session.id = sqlite3_column_int(stmt, 0);
        session.player1Id = sqlite3_column_int(stmt, 1);
        session.player2Id = sqlite3_column_int(stmt, 2);
        session.gameType = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 3));
        session.winnerId = sqlite3_column_int(stmt, 4);
        session.player1Score = sqlite3_column_int(stmt, 5);
        session.player2Score = sqlite3_column_int(stmt, 6);
        session.durationSeconds = sqlite3_column_int(stmt, 7);
        session.startedAt = sqlite3_column_int(stmt, 8);
        session.finishedAt = sqlite3_column_int(stmt, 9);
    }
    
    sqlite3_finalize(stmt);
    return session;
}

std::vector<GameSession> Database::getPlayerSessions(int playerId) {
    std::vector<GameSession> sessions;
    if (!isOpen()) return sessions;
    
    const char* sql = R"(
        SELECT id, player1_id, player2_id, game_type, winner_id,
               player1_score, player2_score, duration_seconds,
               started_at, finished_at
        FROM game_sessions 
        WHERE player1_id = ? OR player2_id = ?
        ORDER BY started_at DESC;
    )";
    
    sqlite3_stmt* stmt = prepare(sql);
    if (!stmt) return sessions;
    
    bindInt(stmt, 1, playerId);
    bindInt(stmt, 2, playerId);
    
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        GameSession session;
        session.id = sqlite3_column_int(stmt, 0);
        session.player1Id = sqlite3_column_int(stmt, 1);
        session.player2Id = sqlite3_column_int(stmt, 2);
        session.gameType = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 3));
        session.winnerId = sqlite3_column_int(stmt, 4);
        session.player1Score = sqlite3_column_int(stmt, 5);
        session.player2Score = sqlite3_column_int(stmt, 6);
        session.durationSeconds = sqlite3_column_int(stmt, 7);
        session.startedAt = sqlite3_column_int(stmt, 8);
        session.finishedAt = sqlite3_column_int(stmt, 9);
        
        sessions.push_back(session);
    }
    
    sqlite3_finalize(stmt);
    return sessions;
}

// Shot record operations

int Database::addShot(ShotRecord& shot) {
    if (!isOpen()) return -1;
    
    shot.timestamp = std::time(nullptr);
    
    const char* sql = R"(
        INSERT INTO shot_records 
        (session_id, player_id, shot_type, successful, 
         ball_x, ball_y, target_x, target_y, shot_speed, shot_number, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
    )";
    
    sqlite3_stmt* stmt = prepare(sql);
    if (!stmt) return -1;
    
    bindInt(stmt, 1, shot.sessionId);
    bindInt(stmt, 2, shot.playerId);
    bindText(stmt, 3, shot.shotType);
    bindInt(stmt, 4, shot.successful ? 1 : 0);
    bindDouble(stmt, 5, shot.ballX);
    bindDouble(stmt, 6, shot.ballY);
    bindDouble(stmt, 7, shot.targetX);
    bindDouble(stmt, 8, shot.targetY);
    bindDouble(stmt, 9, shot.shotSpeed);
    bindInt(stmt, 10, shot.shotNumber);
    bindInt(stmt, 11, static_cast<int>(shot.timestamp));
    
    int rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    
    if (rc != SQLITE_DONE) {
        lastError_ = sqlite3_errmsg(db_);
        return -1;
    }
    
    shot.id = static_cast<int>(sqlite3_last_insert_rowid(db_));
    return shot.id;
}

std::vector<ShotRecord> Database::getSessionShots(int sessionId) {
    std::vector<ShotRecord> shots;
    if (!isOpen()) return shots;
    
    const char* sql = R"(
        SELECT id, session_id, player_id, shot_type, successful,
               ball_x, ball_y, target_x, target_y, shot_speed,
               shot_number, timestamp
        FROM shot_records 
        WHERE session_id = ?
        ORDER BY shot_number;
    )";
    
    sqlite3_stmt* stmt = prepare(sql);
    if (!stmt) return shots;
    
    bindInt(stmt, 1, sessionId);
    
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        ShotRecord shot;
        shot.id = sqlite3_column_int(stmt, 0);
        shot.sessionId = sqlite3_column_int(stmt, 1);
        shot.playerId = sqlite3_column_int(stmt, 2);
        shot.shotType = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 3));
        shot.successful = sqlite3_column_int(stmt, 4) != 0;
        shot.ballX = static_cast<float>(sqlite3_column_double(stmt, 5));
        shot.ballY = static_cast<float>(sqlite3_column_double(stmt, 6));
        shot.targetX = static_cast<float>(sqlite3_column_double(stmt, 7));
        shot.targetY = static_cast<float>(sqlite3_column_double(stmt, 8));
        shot.shotSpeed = static_cast<float>(sqlite3_column_double(stmt, 9));
        shot.shotNumber = sqlite3_column_int(stmt, 10);
        shot.timestamp = sqlite3_column_int(stmt, 11);
        
        shots.push_back(shot);
    }
    
    sqlite3_finalize(stmt);
    return shots;
}

std::vector<ShotRecord> Database::getPlayerShots(int playerId) {
    std::vector<ShotRecord> shots;
    if (!isOpen()) return shots;
    
    const char* sql = R"(
        SELECT id, session_id, player_id, shot_type, successful,
               ball_x, ball_y, target_x, target_y, shot_speed,
               shot_number, timestamp
        FROM shot_records 
        WHERE player_id = ?
        ORDER BY timestamp DESC
        LIMIT 1000;
    )";
    
    sqlite3_stmt* stmt = prepare(sql);
    if (!stmt) return shots;
    
    bindInt(stmt, 1, playerId);
    
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        ShotRecord shot;
        shot.id = sqlite3_column_int(stmt, 0);
        shot.sessionId = sqlite3_column_int(stmt, 1);
        shot.playerId = sqlite3_column_int(stmt, 2);
        shot.shotType = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 3));
        shot.successful = sqlite3_column_int(stmt, 4) != 0;
        shot.ballX = static_cast<float>(sqlite3_column_double(stmt, 5));
        shot.ballY = static_cast<float>(sqlite3_column_double(stmt, 6));
        shot.targetX = static_cast<float>(sqlite3_column_double(stmt, 7));
        shot.targetY = static_cast<float>(sqlite3_column_double(stmt, 8));
        shot.shotSpeed = static_cast<float>(sqlite3_column_double(stmt, 9));
        shot.shotNumber = sqlite3_column_int(stmt, 10);
        shot.timestamp = sqlite3_column_int(stmt, 11);
        
        shots.push_back(shot);
    }
    
    sqlite3_finalize(stmt);
    return shots;
}

std::vector<cv::Mat> Database::getSessionFrames(int sessionId) {
    std::vector<cv::Mat> frames;
    // Frame storage is not yet implemented in the database
    // This method returns empty vector for now
    // TODO: Implement frame storage and retrieval when needed
    return frames;
}

// Statistics operations

bool Database::updatePlayerStats(int playerId) {
    if (!isOpen()) return false;
    
    // Update games played and won
    const char* updateStatsSql = R"(
        UPDATE players 
        SET games_played = (
            SELECT COUNT(*) FROM game_sessions 
            WHERE player1_id = ? OR player2_id = ?
        ),
        games_won = (
            SELECT COUNT(*) FROM game_sessions 
            WHERE winner_id = ?
        ),
        total_shots = (
            SELECT COUNT(*) FROM shot_records 
            WHERE player_id = ?
        ),
        successful_shots = (
            SELECT COUNT(*) FROM shot_records 
            WHERE player_id = ? AND successful = 1
        )
        WHERE id = ?;
    )";
    
    sqlite3_stmt* stmt = prepare(updateStatsSql);
    if (!stmt) return false;
    
    bindInt(stmt, 1, playerId);
    bindInt(stmt, 2, playerId);
    bindInt(stmt, 3, playerId);
    bindInt(stmt, 4, playerId);
    bindInt(stmt, 5, playerId);
    bindInt(stmt, 6, playerId);
    
    int rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    
    return rc == SQLITE_DONE;
}

float Database::getPlayerWinRate(int playerId) {
    if (!isOpen()) return 0.0f;
    
    const char* sql = R"(
        SELECT 
            CAST(SUM(CASE WHEN winner_id = ? THEN 1 ELSE 0 END) AS REAL) /
            CAST(COUNT(*) AS REAL) as win_rate
        FROM game_sessions
        WHERE (player1_id = ? OR player2_id = ?) AND winner_id IS NOT NULL;
    )";
    
    sqlite3_stmt* stmt = prepare(sql);
    if (!stmt) return 0.0f;
    
    bindInt(stmt, 1, playerId);
    bindInt(stmt, 2, playerId);
    bindInt(stmt, 3, playerId);
    
    float winRate = 0.0f;
    if (sqlite3_step(stmt) == SQLITE_ROW) {
        winRate = static_cast<float>(sqlite3_column_double(stmt, 0));
    }
    
    sqlite3_finalize(stmt);
    return winRate;
}

float Database::getPlayerShotSuccessRate(int playerId) {
    if (!isOpen()) return 0.0f;
    
    const char* sql = R"(
        SELECT 
            CAST(SUM(CASE WHEN successful = 1 THEN 1 ELSE 0 END) AS REAL) /
            CAST(COUNT(*) AS REAL) as success_rate
        FROM shot_records
        WHERE player_id = ?;
    )";
    
    sqlite3_stmt* stmt = prepare(sql);
    if (!stmt) return 0.0f;
    
    bindInt(stmt, 1, playerId);
    
    float successRate = 0.0f;
    if (sqlite3_step(stmt) == SQLITE_ROW) {
        successRate = static_cast<float>(sqlite3_column_double(stmt, 0));
    }
    
    sqlite3_finalize(stmt);
    return successRate;
}

} // namespace pv
