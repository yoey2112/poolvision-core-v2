#pragma once
#include "../util/Types.hpp"
#include "../db/Database.hpp"
#include "GameState.hpp"
#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <chrono>

namespace pv {

/**
 * @brief Comprehensive match management system
 * 
 * Handles match setup, state management, scoring, and tournament-style
 * competitive play with enhanced UI integration.
 */
class MatchSystem {
public:
    /**
     * @brief Match formats for competitive play
     */
    enum class MatchFormat {
        RaceToN,        // First to N games wins
        BestOfN,        // Best of N games
        Timed,          // Time-based matches
        Tournament      // Tournament bracket play
    };

    /**
     * @brief Game types supported
     */
    enum class GameType {
        EightBall,
        NineBall,
        TenBall,
        Straight,
        OnePocket,
        BankPool
    };

    /**
     * @brief Match states
     */
    enum class MatchState {
        Setup,
        PreMatch,
        InProgress,
        Paused,
        Completed,
        Cancelled
    };

    /**
     * @brief Player types for match setup
     */
    enum class PlayerType {
        Human,
        AI,
        Guest
    };

    /**
     * @brief Match configuration
     */
    struct MatchConfig {
        int matchId;
        MatchFormat format;
        GameType gameType;
        int targetGames;        // For RaceToN or BestOfN
        double timeLimit;       // For timed matches (minutes)
        bool handicapEnabled;
        int player1Handicap;
        int player2Handicap;
        bool shotClock;
        double shotTimeLimit;   // Seconds per shot
        bool allowUndo;
        bool recordSession;
        std::string venue;
        std::string notes;
        
        MatchConfig() : matchId(0), format(MatchFormat::RaceToN), gameType(GameType::EightBall),
                       targetGames(7), timeLimit(0.0), handicapEnabled(false),
                       player1Handicap(0), player2Handicap(0), shotClock(false),
                       shotTimeLimit(30.0), allowUndo(true), recordSession(true) {}
    };

    /**
     * @brief Player information for match
     */
    struct MatchPlayer {
        int playerId;
        std::string name;
        PlayerType type;
        int skillLevel;
        int handicap;
        bool isBreaking;
        
        MatchPlayer() : playerId(0), type(PlayerType::Human), skillLevel(1),
                       handicap(0), isBreaking(false) {}
    };

    /**
     * @brief Game result within a match
     */
    struct GameResult {
        int gameNumber;
        int winnerId;
        int player1Score;
        int player2Score;
        double duration;        // Minutes
        int shotCount;
        bool wasEarlyFinish;    // 8-ball break, etc.
        std::vector<int> fouls;
        std::chrono::steady_clock::time_point startTime;
        std::chrono::steady_clock::time_point endTime;
        
        GameResult() : gameNumber(0), winnerId(0), player1Score(0), player2Score(0),
                      duration(0.0), shotCount(0), wasEarlyFinish(false) {}
    };

    /**
     * @brief Complete match record
     */
    struct MatchRecord {
        int matchId;
        MatchConfig config;
        MatchPlayer player1;
        MatchPlayer player2;
        MatchState state;
        std::vector<GameResult> games;
        int currentGame;
        int player1Wins;
        int player2Wins;
        int winnerId;
        std::chrono::steady_clock::time_point startTime;
        std::chrono::steady_clock::time_point endTime;
        double totalDuration;
        std::string venue;
        
        MatchRecord() : matchId(0), state(MatchState::Setup), currentGame(0),
                       player1Wins(0), player2Wins(0), winnerId(0), totalDuration(0.0) {}
    };

    /**
     * @brief Shot clock information
     */
    struct ShotClock {
        bool active;
        double timeRemaining;
        double timeLimit;
        bool warning;           // 10 seconds or less
        int violations;
        
        ShotClock() : active(false), timeRemaining(0.0), timeLimit(30.0),
                     warning(false), violations(0) {}
    };

    /**
     * @brief Live match statistics
     */
    struct LiveStats {
        double avgShotTime;
        double maxShotTime;
        int totalShots;
        int successfulShots;
        double shotSuccessRate;
        int fouls;
        int timeouts;
        int safetiesPlayed;
        std::vector<double> recentShotTimes;
        
        LiveStats() : avgShotTime(0.0), maxShotTime(0.0), totalShots(0),
                     successfulShots(0), shotSuccessRate(0.0), fouls(0),
                     timeouts(0), safetiesPlayed(0) {}
    };

    /**
     * @brief Construct a new Match System
     */
    MatchSystem(Database& database, GameState& gameState);
    ~MatchSystem() = default;

    // Match setup and management
    /**
     * @brief Create new match
     */
    int createMatch(const MatchConfig& config, const MatchPlayer& player1, const MatchPlayer& player2);
    
    /**
     * @brief Start the match
     */
    bool startMatch(int matchId);
    
    /**
     * @brief Pause the match
     */
    void pauseMatch();
    
    /**
     * @brief Resume the match
     */
    void resumeMatch();
    
    /**
     * @brief End the match
     */
    void endMatch();
    
    /**
     * @brief Cancel the match
     */
    void cancelMatch();

    // Game management within match
    /**
     * @brief Start new game within match
     */
    bool startGame();
    
    /**
     * @brief End current game
     */
    void endGame(int winnerId, int player1Score, int player2Score);
    
    /**
     * @brief Process shot during game
     */
    void processShot(const cv::Point2f& shotPosition, bool successful, double shotTime);
    
    /**
     * @brief Record foul
     */
    void recordFoul(int playerId, const std::string& foulType);

    // Shot clock management
    /**
     * @brief Start shot clock
     */
    void startShotClock();
    
    /**
     * @brief Stop shot clock
     */
    void stopShotClock();
    
    /**
     * @brief Reset shot clock
     */
    void resetShotClock();
    
    /**
     * @brief Update shot clock (call every frame)
     */
    void updateShotClock();

    // Match queries
    /**
     * @brief Get current match record
     */
    const MatchRecord& getCurrentMatch() const { return currentMatch_; }
    
    /**
     * @brief Get current shot clock state
     */
    const ShotClock& getShotClock() const { return shotClock_; }
    
    /**
     * @brief Get live statistics
     */
    const LiveStats& getLiveStats(int playerId) const;
    
    /**
     * @brief Check if match is active
     */
    bool isMatchActive() const { return currentMatch_.state == MatchState::InProgress; }
    
    /**
     * @brief Check if match is paused
     */
    bool isMatchPaused() const { return currentMatch_.state == MatchState::Paused; }
    
    /**
     * @brief Get current game number
     */
    int getCurrentGameNumber() const { return currentMatch_.currentGame; }

    // Statistics and history
    /**
     * @brief Get match history for player
     */
    std::vector<MatchRecord> getMatchHistory(int playerId, int limit = 20) const;
    
    /**
     * @brief Get head-to-head record
     */
    struct HeadToHeadRecord {
        int player1Wins;
        int player2Wins;
        int totalMatches;
        double avgMatchDuration;
        GameType mostPlayedType;
    };
    
    HeadToHeadRecord getHeadToHeadRecord(int player1Id, int player2Id) const;
    
    /**
     * @brief Get player performance in matches
     */
    struct PlayerMatchStats {
        int totalMatches;
        int wins;
        int losses;
        double winRate;
        double avgMatchDuration;
        double bestWinStreak;
        double currentWinStreak;
        GameType favoriteGameType;
    };
    
    PlayerMatchStats getPlayerMatchStats(int playerId) const;

    // Tournament support
    /**
     * @brief Create tournament bracket
     */
    struct TournamentBracket {
        int tournamentId;
        std::string name;
        std::vector<int> participants;
        std::vector<MatchRecord> matches;
        int currentRound;
        int winnerId;
        
        TournamentBracket() : tournamentId(0), currentRound(1), winnerId(0) {}
    };
    
    int createTournament(const std::string& name, const std::vector<int>& playerIds);
    bool advanceTournament(int tournamentId);
    TournamentBracket getTournament(int tournamentId) const;

    // Callbacks for UI updates
    /**
     * @brief Set callback for match state changes
     */
    void setMatchStateCallback(std::function<void(MatchState)> callback) {
        matchStateCallback_ = callback;
    }
    
    /**
     * @brief Set callback for game completion
     */
    void setGameCompleteCallback(std::function<void(const GameResult&)> callback) {
        gameCompleteCallback_ = callback;
    }
    
    /**
     * @brief Set callback for shot clock updates
     */
    void setShotClockCallback(std::function<void(const ShotClock&)> callback) {
        shotClockCallback_ = callback;
    }

    // Save/Load functionality
    /**
     * @brief Save current match state
     */
    void saveMatchState();
    
    /**
     * @brief Load match state
     */
    bool loadMatchState(int matchId);
    
    /**
     * @brief Export match data
     */
    std::string exportMatchData(int matchId, const std::string& format = "json") const;

    // Utility functions
    /**
     * @brief Convert match format to string
     */
    static std::string formatToString(MatchFormat format);
    
    /**
     * @brief Convert game type to string
     */
    static std::string gameTypeToString(GameType gameType);
    
    /**
     * @brief Convert match state to string
     */
    static std::string stateToString(MatchState state);

private:
    Database& database_;
    GameState& gameState_;
    MatchRecord currentMatch_;
    ShotClock shotClock_;
    LiveStats player1Stats_;
    LiveStats player2Stats_;
    
    // Callbacks for UI updates
    std::function<void(MatchState)> matchStateCallback_;
    std::function<void(const GameResult&)> gameCompleteCallback_;
    std::function<void(const ShotClock&)> shotClockCallback_;
    
    // Timing
    std::chrono::steady_clock::time_point lastShotTime_;
    std::chrono::steady_clock::time_point gameStartTime_;
    std::chrono::steady_clock::time_point shotClockStart_;
    
    // Internal methods
    void changeState(MatchState newState);
    void updateLiveStats(int playerId, bool shotSuccessful, double shotTime);
    bool checkMatchWinCondition();
    void recordGameToDatabase(const GameResult& game);
    void recordMatchToDatabase();
    int getNextMatchId();
    void initializeLiveStats();
    void resetGameStats();
};

} // namespace pv