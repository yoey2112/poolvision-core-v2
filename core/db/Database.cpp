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
    
    const char* createDrillSessionsSql = R"(
        CREATE TABLE IF NOT EXISTS drill_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            player_id INTEGER NOT NULL,
            drill_id TEXT NOT NULL,
            difficulty INTEGER DEFAULT 1,
            attempts_total INTEGER DEFAULT 0,
            attempts_successful INTEGER DEFAULT 0,
            time_spent REAL DEFAULT 0.0,
            best_score REAL DEFAULT 0.0,
            started_at INTEGER NOT NULL,
            finished_at INTEGER,
            completed INTEGER DEFAULT 0,
            FOREIGN KEY (player_id) REFERENCES players(id) ON DELETE CASCADE
        );
    )";
    
    const char* createMatchesSql = R"(
        CREATE TABLE IF NOT EXISTS match_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            player1_id INTEGER NOT NULL,
            player2_id INTEGER NOT NULL,
            match_format TEXT NOT NULL,
            game_type TEXT NOT NULL,
            target_games INTEGER DEFAULT 0,
            player1_wins INTEGER DEFAULT 0,
            player2_wins INTEGER DEFAULT 0,
            winner_id INTEGER,
            completed INTEGER DEFAULT 0,
            duration_minutes REAL DEFAULT 0.0,
            started_at INTEGER NOT NULL,
            finished_at INTEGER,
            tournament_id INTEGER,
            notes TEXT,
            FOREIGN KEY (player1_id) REFERENCES players(id) ON DELETE CASCADE,
            FOREIGN KEY (player2_id) REFERENCES players(id) ON DELETE CASCADE,
            FOREIGN KEY (winner_id) REFERENCES players(id) ON DELETE SET NULL,
            FOREIGN KEY (tournament_id) REFERENCES tournaments(id) ON DELETE SET NULL
        );
    )";
    
    const char* createTournamentsSql = R"(
        CREATE TABLE IF NOT EXISTS tournaments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            format TEXT NOT NULL,
            game_type TEXT NOT NULL,
            max_participants INTEGER DEFAULT 0,
            current_participants INTEGER DEFAULT 0,
            is_active INTEGER DEFAULT 0,
            is_completed INTEGER DEFAULT 0,
            started_at INTEGER NOT NULL,
            finished_at INTEGER,
            winner_id INTEGER,
            description TEXT,
            bracket_data TEXT,
            FOREIGN KEY (winner_id) REFERENCES players(id) ON DELETE SET NULL
        );
    )";
    
    const char* createIndexesSql = R"(
        CREATE INDEX IF NOT EXISTS idx_players_name ON players(name);
        CREATE INDEX IF NOT EXISTS idx_sessions_player1 ON game_sessions(player1_id);
        CREATE INDEX IF NOT EXISTS idx_sessions_player2 ON game_sessions(player2_id);
        CREATE INDEX IF NOT EXISTS idx_shots_session ON shot_records(session_id);
        CREATE INDEX IF NOT EXISTS idx_shots_player ON shot_records(player_id);
        CREATE INDEX IF NOT EXISTS idx_drill_sessions_player ON drill_sessions(player_id);
        CREATE INDEX IF NOT EXISTS idx_drill_sessions_drill ON drill_sessions(drill_id);
        CREATE INDEX IF NOT EXISTS idx_matches_player1 ON match_records(player1_id);
        CREATE INDEX IF NOT EXISTS idx_matches_player2 ON match_records(player2_id);
        CREATE INDEX IF NOT EXISTS idx_matches_tournament ON match_records(tournament_id);
        CREATE INDEX IF NOT EXISTS idx_tournaments_active ON tournaments(is_active);
    )";
    
    return execute(createPlayersSql) &&
           execute(createGamesSql) &&
           execute(createShotsSql) &&
           execute(createDrillSessionsSql) &&
           execute(createMatchesSql) &&
           execute(createTournamentsSql) &&
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

// Drill session operations

int Database::createDrillSession(DrillSession& session) {
    if (!isOpen()) return -1;
    
    session.startedAt = std::time(nullptr);
    
    const char* sql = R"(
        INSERT INTO drill_sessions (player_id, drill_id, difficulty, started_at)
        VALUES (?, ?, ?, ?);
    )";
    
    sqlite3_stmt* stmt = prepare(sql);
    if (!stmt) return -1;
    
    bindInt(stmt, 1, session.playerId);
    bindText(stmt, 2, session.drillId);
    bindInt(stmt, 3, session.difficulty);
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

bool Database::updateDrillSession(const DrillSession& session) {
    if (!isOpen() || session.id < 0) return false;
    
    const char* sql = R"(
        UPDATE drill_sessions 
        SET attempts_total = ?, attempts_successful = ?, time_spent = ?,
            best_score = ?, finished_at = ?, completed = ?
        WHERE id = ?;
    )";
    
    sqlite3_stmt* stmt = prepare(sql);
    if (!stmt) return false;
    
    bindInt(stmt, 1, session.attemptsTotal);
    bindInt(stmt, 2, session.attemptsSuccessful);
    bindDouble(stmt, 3, session.timeSpent);
    bindDouble(stmt, 4, session.bestScore);
    bindInt(stmt, 5, static_cast<int>(session.finishedAt));
    bindInt(stmt, 6, session.completed ? 1 : 0);
    bindInt(stmt, 7, session.id);
    
    int rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    
    return rc == SQLITE_DONE;
}

std::vector<DrillSession> Database::getPlayerDrillSessions(int playerId, int limit) {
    std::vector<DrillSession> sessions;
    if (!isOpen()) return sessions;
    
    const char* sql = R"(
        SELECT id, player_id, drill_id, difficulty, attempts_total,
               attempts_successful, time_spent, best_score, started_at,
               finished_at, completed
        FROM drill_sessions 
        WHERE player_id = ?
        ORDER BY started_at DESC 
        LIMIT ?;
    )";
    
    sqlite3_stmt* stmt = prepare(sql);
    if (!stmt) return sessions;
    
    bindInt(stmt, 1, playerId);
    bindInt(stmt, 2, limit);
    
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        DrillSession session;
        session.id = sqlite3_column_int(stmt, 0);
        session.playerId = sqlite3_column_int(stmt, 1);
        session.drillId = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));
        session.difficulty = sqlite3_column_int(stmt, 3);
        session.attemptsTotal = sqlite3_column_int(stmt, 4);
        session.attemptsSuccessful = sqlite3_column_int(stmt, 5);
        session.timeSpent = sqlite3_column_double(stmt, 6);
        session.bestScore = sqlite3_column_double(stmt, 7);
        session.startedAt = sqlite3_column_int(stmt, 8);
        session.finishedAt = sqlite3_column_int(stmt, 9);
        session.completed = sqlite3_column_int(stmt, 10) != 0;
        
        sessions.push_back(session);
    }
    
    sqlite3_finalize(stmt);
    return sessions;
}

DrillStats Database::getDrillStats(int playerId, const std::string& drillId) {
    DrillStats stats;
    if (!isOpen()) return stats;
    
    const char* sql = R"(
        SELECT 
            COUNT(*) as total_sessions,
            SUM(CASE WHEN completed = 1 THEN 1 ELSE 0 END) as completed_sessions,
            AVG(best_score) as average_score,
            MAX(best_score) as best_score,
            SUM(time_spent) as total_time_spent,
            SUM(attempts_total) as total_attempts,
            SUM(attempts_successful) as successful_attempts,
            MIN(started_at) as first_attempt,
            MAX(started_at) as last_attempt
        FROM drill_sessions
        WHERE player_id = ? AND drill_id = ?;
    )";
    
    sqlite3_stmt* stmt = prepare(sql);
    if (!stmt) return stats;
    
    bindInt(stmt, 1, playerId);
    bindText(stmt, 2, drillId);
    
    if (sqlite3_step(stmt) == SQLITE_ROW) {
        stats.totalSessions = sqlite3_column_int(stmt, 0);
        stats.completedSessions = sqlite3_column_int(stmt, 1);
        stats.averageScore = sqlite3_column_double(stmt, 2);
        stats.bestScore = sqlite3_column_double(stmt, 3);
        stats.totalTimeSpent = sqlite3_column_double(stmt, 4);
        stats.totalAttempts = sqlite3_column_int(stmt, 5);
        stats.successfulAttempts = sqlite3_column_int(stmt, 6);
        stats.firstAttempt = sqlite3_column_int(stmt, 7);
        stats.lastAttempt = sqlite3_column_int(stmt, 8);
        
        if (stats.totalAttempts > 0) {
            stats.successRate = static_cast<double>(stats.successfulAttempts) / stats.totalAttempts;
        }
    }
    
    sqlite3_finalize(stmt);
    return stats;
}

// Match operations

int Database::createMatch(MatchRecord& match) {
    if (!isOpen()) return -1;
    
    match.startedAt = std::time(nullptr);
    
    const char* sql = R"(
        INSERT INTO match_records (player1_id, player2_id, match_format, game_type,
                                  target_games, started_at, tournament_id)
        VALUES (?, ?, ?, ?, ?, ?, ?);
    )";
    
    sqlite3_stmt* stmt = prepare(sql);
    if (!stmt) return -1;
    
    bindInt(stmt, 1, match.player1Id);
    bindInt(stmt, 2, match.player2Id);
    bindText(stmt, 3, match.matchFormat);
    bindText(stmt, 4, match.gameType);
    bindInt(stmt, 5, match.targetGames);
    bindInt(stmt, 6, static_cast<int>(match.startedAt));
    bindInt(stmt, 7, match.tournamentId);
    
    int rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    
    if (rc != SQLITE_DONE) {
        lastError_ = sqlite3_errmsg(db_);
        return -1;
    }
    
    match.id = static_cast<int>(sqlite3_last_insert_rowid(db_));
    return match.id;
}

bool Database::updateMatch(const MatchRecord& match) {
    if (!isOpen() || match.id < 0) return false;
    
    const char* sql = R"(
        UPDATE match_records 
        SET player1_wins = ?, player2_wins = ?, winner_id = ?, completed = ?,
            duration_minutes = ?, finished_at = ?, notes = ?
        WHERE id = ?;
    )";
    
    sqlite3_stmt* stmt = prepare(sql);
    if (!stmt) return false;
    
    bindInt(stmt, 1, match.player1Wins);
    bindInt(stmt, 2, match.player2Wins);
    bindInt(stmt, 3, match.winnerId);
    bindInt(stmt, 4, match.completed ? 1 : 0);
    bindDouble(stmt, 5, match.durationMinutes);
    bindInt(stmt, 6, static_cast<int>(match.finishedAt));
    bindText(stmt, 7, match.notes);
    bindInt(stmt, 8, match.id);
    
    int rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    
    return rc == SQLITE_DONE;
}

MatchRecord Database::getMatch(int matchId) {
    MatchRecord match;
    if (!isOpen()) return match;
    
    const char* sql = R"(
        SELECT id, player1_id, player2_id, match_format, game_type,
               target_games, player1_wins, player2_wins, winner_id,
               completed, duration_minutes, started_at, finished_at,
               tournament_id, notes
        FROM match_records WHERE id = ?;
    )";
    
    sqlite3_stmt* stmt = prepare(sql);
    if (!stmt) return match;
    
    bindInt(stmt, 1, matchId);
    
    if (sqlite3_step(stmt) == SQLITE_ROW) {
        match.id = sqlite3_column_int(stmt, 0);
        match.player1Id = sqlite3_column_int(stmt, 1);
        match.player2Id = sqlite3_column_int(stmt, 2);
        match.matchFormat = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 3));
        match.gameType = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 4));
        match.targetGames = sqlite3_column_int(stmt, 5);
        match.player1Wins = sqlite3_column_int(stmt, 6);
        match.player2Wins = sqlite3_column_int(stmt, 7);
        match.winnerId = sqlite3_column_int(stmt, 8);
        match.completed = sqlite3_column_int(stmt, 9) != 0;
        match.durationMinutes = sqlite3_column_double(stmt, 10);
        match.startedAt = sqlite3_column_int(stmt, 11);
        match.finishedAt = sqlite3_column_int(stmt, 12);
        match.tournamentId = sqlite3_column_int(stmt, 13);
        const char* notesText = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 14));
        if (notesText) match.notes = notesText;
    }
    
    sqlite3_finalize(stmt);
    return match;
}

std::vector<MatchRecord> Database::getPlayerMatches(int playerId, int limit) {
    std::vector<MatchRecord> matches;
    if (!isOpen()) return matches;
    
    const char* sql = R"(
        SELECT id, player1_id, player2_id, match_format, game_type,
               target_games, player1_wins, player2_wins, winner_id,
               completed, duration_minutes, started_at, finished_at,
               tournament_id, notes
        FROM match_records 
        WHERE player1_id = ? OR player2_id = ?
        ORDER BY started_at DESC 
        LIMIT ?;
    )";
    
    sqlite3_stmt* stmt = prepare(sql);
    if (!stmt) return matches;
    
    bindInt(stmt, 1, playerId);
    bindInt(stmt, 2, playerId);
    bindInt(stmt, 3, limit);
    
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        MatchRecord match;
        match.id = sqlite3_column_int(stmt, 0);
        match.player1Id = sqlite3_column_int(stmt, 1);
        match.player2Id = sqlite3_column_int(stmt, 2);
        match.matchFormat = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 3));
        match.gameType = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 4));
        match.targetGames = sqlite3_column_int(stmt, 5);
        match.player1Wins = sqlite3_column_int(stmt, 6);
        match.player2Wins = sqlite3_column_int(stmt, 7);
        match.winnerId = sqlite3_column_int(stmt, 8);
        match.completed = sqlite3_column_int(stmt, 9) != 0;
        match.durationMinutes = sqlite3_column_double(stmt, 10);
        match.startedAt = sqlite3_column_int(stmt, 11);
        match.finishedAt = sqlite3_column_int(stmt, 12);
        match.tournamentId = sqlite3_column_int(stmt, 13);
        const char* notesText = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 14));
        if (notesText) match.notes = notesText;
        
        matches.push_back(match);
    }
    
    sqlite3_finalize(stmt);
    return matches;
}

HeadToHeadRecord Database::getHeadToHead(int player1Id, int player2Id) {
    HeadToHeadRecord record;
    record.player1Id = player1Id;
    record.player2Id = player2Id;
    
    if (!isOpen()) return record;
    
    const char* sql = R"(
        SELECT 
            COUNT(*) as total_matches,
            SUM(CASE WHEN winner_id = ? THEN 1 ELSE 0 END) as player1_wins,
            SUM(CASE WHEN winner_id = ? THEN 1 ELSE 0 END) as player2_wins,
            MIN(started_at) as first_match,
            MAX(started_at) as last_match,
            AVG(duration_minutes) as average_duration
        FROM match_records
        WHERE (player1_id = ? AND player2_id = ?) OR (player1_id = ? AND player2_id = ?)
        AND completed = 1;
    )";
    
    sqlite3_stmt* stmt = prepare(sql);
    if (!stmt) return record;
    
    bindInt(stmt, 1, player1Id);
    bindInt(stmt, 2, player2Id);
    bindInt(stmt, 3, player1Id);
    bindInt(stmt, 4, player2Id);
    bindInt(stmt, 5, player2Id);
    bindInt(stmt, 6, player1Id);
    
    if (sqlite3_step(stmt) == SQLITE_ROW) {
        record.totalMatches = sqlite3_column_int(stmt, 0);
        record.player1Wins = sqlite3_column_int(stmt, 1);
        record.player2Wins = sqlite3_column_int(stmt, 2);
        record.firstMatch = sqlite3_column_int(stmt, 3);
        record.lastMatch = sqlite3_column_int(stmt, 4);
        record.averageMatchDuration = sqlite3_column_double(stmt, 5);
    }
    
    sqlite3_finalize(stmt);
    return record;
}

// Tournament operations

int Database::createTournament(TournamentRecord& tournament) {
    if (!isOpen()) return -1;
    
    tournament.startedAt = std::time(nullptr);
    
    const char* sql = R"(
        INSERT INTO tournaments (name, format, game_type, max_participants,
                               started_at, description, bracket_data)
        VALUES (?, ?, ?, ?, ?, ?, ?);
    )";
    
    sqlite3_stmt* stmt = prepare(sql);
    if (!stmt) return -1;
    
    bindText(stmt, 1, tournament.name);
    bindText(stmt, 2, tournament.format);
    bindText(stmt, 3, tournament.gameType);
    bindInt(stmt, 4, tournament.maxParticipants);
    bindInt(stmt, 5, static_cast<int>(tournament.startedAt));
    bindText(stmt, 6, tournament.description);
    bindText(stmt, 7, tournament.bracketData);
    
    int rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    
    if (rc != SQLITE_DONE) {
        lastError_ = sqlite3_errmsg(db_);
        return -1;
    }
    
    tournament.id = static_cast<int>(sqlite3_last_insert_rowid(db_));
    return tournament.id;
}

bool Database::updateTournament(const TournamentRecord& tournament) {
    if (!isOpen() || tournament.id < 0) return false;
    
    const char* sql = R"(
        UPDATE tournaments 
        SET current_participants = ?, is_active = ?, is_completed = ?,
            finished_at = ?, winner_id = ?, bracket_data = ?
        WHERE id = ?;
    )";
    
    sqlite3_stmt* stmt = prepare(sql);
    if (!stmt) return false;
    
    bindInt(stmt, 1, tournament.currentParticipants);
    bindInt(stmt, 2, tournament.isActive ? 1 : 0);
    bindInt(stmt, 3, tournament.isCompleted ? 1 : 0);
    bindInt(stmt, 4, static_cast<int>(tournament.finishedAt));
    bindInt(stmt, 5, tournament.winnerId);
    bindText(stmt, 6, tournament.bracketData);
    bindInt(stmt, 7, tournament.id);
    
    int rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    
    return rc == SQLITE_DONE;
}

TournamentRecord Database::getTournament(int tournamentId) {
    TournamentRecord tournament;
    if (!isOpen()) return tournament;
    
    const char* sql = R"(
        SELECT id, name, format, game_type, max_participants,
               current_participants, is_active, is_completed,
               started_at, finished_at, winner_id, description,
               bracket_data
        FROM tournaments WHERE id = ?;
    )";
    
    sqlite3_stmt* stmt = prepare(sql);
    if (!stmt) return tournament;
    
    bindInt(stmt, 1, tournamentId);
    
    if (sqlite3_step(stmt) == SQLITE_ROW) {
        tournament.id = sqlite3_column_int(stmt, 0);
        tournament.name = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
        tournament.format = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));
        tournament.gameType = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 3));
        tournament.maxParticipants = sqlite3_column_int(stmt, 4);
        tournament.currentParticipants = sqlite3_column_int(stmt, 5);
        tournament.isActive = sqlite3_column_int(stmt, 6) != 0;
        tournament.isCompleted = sqlite3_column_int(stmt, 7) != 0;
        tournament.startedAt = sqlite3_column_int(stmt, 8);
        tournament.finishedAt = sqlite3_column_int(stmt, 9);
        tournament.winnerId = sqlite3_column_int(stmt, 10);
        const char* descText = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 11));
        if (descText) tournament.description = descText;
        const char* bracketText = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 12));
        if (bracketText) tournament.bracketData = bracketText;
    }
    
    sqlite3_finalize(stmt);
    return tournament;
}

std::vector<TournamentRecord> Database::getActiveTournaments() {
    std::vector<TournamentRecord> tournaments;
    if (!isOpen()) return tournaments;
    
    const char* sql = R"(
        SELECT id, name, format, game_type, max_participants,
               current_participants, is_active, is_completed,
               started_at, finished_at, winner_id, description,
               bracket_data
        FROM tournaments 
        WHERE is_active = 1 OR (is_completed = 0 AND current_participants > 0)
        ORDER BY started_at DESC;
    )";
    
    sqlite3_stmt* stmt = prepare(sql);
    if (!stmt) return tournaments;
    
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        TournamentRecord tournament;
        tournament.id = sqlite3_column_int(stmt, 0);
        tournament.name = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
        tournament.format = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));
        tournament.gameType = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 3));
        tournament.maxParticipants = sqlite3_column_int(stmt, 4);
        tournament.currentParticipants = sqlite3_column_int(stmt, 5);
        tournament.isActive = sqlite3_column_int(stmt, 6) != 0;
        tournament.isCompleted = sqlite3_column_int(stmt, 7) != 0;
        tournament.startedAt = sqlite3_column_int(stmt, 8);
        tournament.finishedAt = sqlite3_column_int(stmt, 9);
        tournament.winnerId = sqlite3_column_int(stmt, 10);
        const char* descText = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 11));
        if (descText) tournament.description = descText;
        const char* bracketText = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 12));
        if (bracketText) tournament.bracketData = bracketText;
        
        tournaments.push_back(tournament);
    }
    
    sqlite3_finalize(stmt);
    return tournaments;
}

} // namespace pv
