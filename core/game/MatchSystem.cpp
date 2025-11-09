#include "MatchSystem.hpp"
#include "../util/Config.hpp"
#include <algorithm>
#include <cmath>
#include <sstream>
#include <iomanip>

namespace pv {

MatchSystem::MatchSystem(Database& database, GameState& gameState)
    : database_(database), gameState_(gameState) {
    initializeLiveStats();
}

int MatchSystem::createMatch(const MatchConfig& config, const MatchPlayer& player1, const MatchPlayer& player2) {
    // Create new match record
    currentMatch_ = MatchRecord();
    currentMatch_.matchId = getNextMatchId();
    currentMatch_.config = config;
    currentMatch_.config.matchId = currentMatch_.matchId;
    currentMatch_.player1 = player1;
    currentMatch_.player2 = player2;
    currentMatch_.state = MatchState::Setup;
    
    // Initialize match timing
    currentMatch_.startTime = std::chrono::steady_clock::now();
    
    // Reset statistics
    initializeLiveStats();
    
    return currentMatch_.matchId;
}

bool MatchSystem::startMatch(int matchId) {
    if (currentMatch_.matchId != matchId) {
        if (!loadMatchState(matchId)) {
            return false;
        }
    }
    
    // Initialize shot clock if enabled
    if (currentMatch_.config.shotClock) {
        shotClock_.timeLimit = currentMatch_.config.shotTimeLimit;
        shotClock_.active = false;
    }
    
    changeState(MatchState::PreMatch);
    
    // Start first game
    if (startGame()) {
        changeState(MatchState::InProgress);
        return true;
    }
    
    return false;
}

void MatchSystem::pauseMatch() {
    if (currentMatch_.state == MatchState::InProgress) {
        changeState(MatchState::Paused);
        stopShotClock();
    }
}

void MatchSystem::resumeMatch() {
    if (currentMatch_.state == MatchState::Paused) {
        changeState(MatchState::InProgress);
        if (currentMatch_.config.shotClock) {
            startShotClock();
        }
    }
}

void MatchSystem::endMatch() {
    if (currentMatch_.state == MatchState::InProgress || currentMatch_.state == MatchState::Paused) {
        currentMatch_.endTime = std::chrono::steady_clock::now();
        currentMatch_.totalDuration = std::chrono::duration<double>(
            currentMatch_.endTime - currentMatch_.startTime).count() / 60.0; // Convert to minutes
        
        // Determine winner based on current scores
        if (currentMatch_.player1Wins > currentMatch_.player2Wins) {
            currentMatch_.winnerId = currentMatch_.player1.playerId;
        } else if (currentMatch_.player2Wins > currentMatch_.player1Wins) {
            currentMatch_.winnerId = currentMatch_.player2.playerId;
        }
        
        changeState(MatchState::Completed);
        recordMatchToDatabase();
    }
}

void MatchSystem::cancelMatch() {
    changeState(MatchState::Cancelled);
}

bool MatchSystem::startGame() {
    if (currentMatch_.state != MatchState::PreMatch && currentMatch_.state != MatchState::InProgress) {
        return false;
    }
    
    currentMatch_.currentGame++;
    gameStartTime_ = std::chrono::steady_clock::now();
    resetGameStats();
    
    // Initialize game state for new game
    // This would interface with GameState to set up new game
    
    if (currentMatch_.config.shotClock) {
        startShotClock();
    }
    
    return true;
}

void MatchSystem::endGame(int winnerId, int player1Score, int player2Score) {
    if (currentMatch_.state != MatchState::InProgress) {
        return;
    }
    
    // Create game result
    GameResult result;
    result.gameNumber = currentMatch_.currentGame;
    result.winnerId = winnerId;
    result.player1Score = player1Score;
    result.player2Score = player2Score;
    result.startTime = gameStartTime_;
    result.endTime = std::chrono::steady_clock::now();
    result.duration = std::chrono::duration<double>(result.endTime - result.startTime).count() / 60.0;
    
    // Calculate shot count from live stats
    result.shotCount = player1Stats_.totalShots + player2Stats_.totalShots;
    
    // Update match scores
    if (winnerId == currentMatch_.player1.playerId) {
        currentMatch_.player1Wins++;
    } else if (winnerId == currentMatch_.player2.playerId) {
        currentMatch_.player2Wins++;
    }
    
    // Add game to match record
    currentMatch_.games.push_back(result);
    
    // Record game to database
    recordGameToDatabase(result);
    
    // Notify UI
    if (gameCompleteCallback_) {
        gameCompleteCallback_(result);
    }
    
    // Check for match completion
    if (checkMatchWinCondition()) {
        endMatch();
    } else {
        // Start next game
        startGame();
    }
}

void MatchSystem::processShot(const cv::Point2f& shotPosition, bool successful, double shotTime) {
    if (currentMatch_.state != MatchState::InProgress) {
        return;
    }
    
    // Determine current player (would need integration with GameState)
    int currentPlayerId = gameState_.getCurrentTurn() == PlayerTurn::Player1 ? 
                         currentMatch_.player1.playerId : currentMatch_.player2.playerId;
    
    // Update live statistics
    updateLiveStats(currentPlayerId, successful, shotTime);
    
    // Reset shot clock
    if (currentMatch_.config.shotClock) {
        resetShotClock();
        startShotClock();
    }
    
    lastShotTime_ = std::chrono::steady_clock::now();
}

void MatchSystem::recordFoul(int playerId, const std::string& foulType) {
    if (playerId == currentMatch_.player1.playerId) {
        player1Stats_.fouls++;
    } else if (playerId == currentMatch_.player2.playerId) {
        player2Stats_.fouls++;
    }
    
    // Add to current game fouls
    if (!currentMatch_.games.empty()) {
        currentMatch_.games.back().fouls.push_back(playerId);
    }
}

void MatchSystem::startShotClock() {
    if (!currentMatch_.config.shotClock) {
        return;
    }
    
    shotClock_.active = true;
    shotClock_.timeRemaining = shotClock_.timeLimit;
    shotClock_.warning = false;
    shotClockStart_ = std::chrono::steady_clock::now();
}

void MatchSystem::stopShotClock() {
    shotClock_.active = false;
}

void MatchSystem::resetShotClock() {
    shotClock_.timeRemaining = shotClock_.timeLimit;
    shotClock_.warning = false;
    if (shotClock_.active) {
        shotClockStart_ = std::chrono::steady_clock::now();
    }
}

void MatchSystem::updateShotClock() {
    if (!shotClock_.active) {
        return;
    }
    
    auto now = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(now - shotClockStart_).count();
    shotClock_.timeRemaining = std::max(0.0, shotClock_.timeLimit - elapsed);
    
    // Check for warning (10 seconds or less)
    if (shotClock_.timeRemaining <= 10.0 && !shotClock_.warning) {
        shotClock_.warning = true;
    }
    
    // Check for violation
    if (shotClock_.timeRemaining <= 0.0) {
        shotClock_.violations++;
        // This would typically trigger a foul
        resetShotClock();
    }
    
    // Notify UI if callback is set
    if (shotClockCallback_) {
        shotClockCallback_(shotClock_);
    }
}

const MatchSystem::LiveStats& MatchSystem::getLiveStats(int playerId) const {
    if (playerId == currentMatch_.player1.playerId) {
        return player1Stats_;
    } else {
        return player2Stats_;
    }
}

std::vector<MatchSystem::MatchRecord> MatchSystem::getMatchHistory(int playerId, int limit) const {
    std::vector<MatchRecord> history;
    
    // Query database for match history
    // Implementation would involve SQL queries
    
    return history;
}

MatchSystem::HeadToHeadRecord MatchSystem::getHeadToHeadRecord(int player1Id, int player2Id) const {
    HeadToHeadRecord record;
    
    // Query database for head-to-head statistics
    // Implementation would involve SQL queries
    
    return record;
}

MatchSystem::PlayerMatchStats MatchSystem::getPlayerMatchStats(int playerId) const {
    PlayerMatchStats stats;
    
    // Query database for player match statistics
    // Implementation would involve SQL queries
    
    return stats;
}

int MatchSystem::createTournament(const std::string& name, const std::vector<int>& playerIds) {
    // Tournament creation logic
    // This would create bracket structure and initial matches
    return 0;
}

bool MatchSystem::advanceTournament(int tournamentId) {
    // Tournament advancement logic
    return false;
}

MatchSystem::TournamentBracket MatchSystem::getTournament(int tournamentId) const {
    TournamentBracket bracket;
    
    // Query database for tournament information
    
    return bracket;
}

void MatchSystem::saveMatchState() {
    // Save current match state to database
    recordMatchToDatabase();
}

bool MatchSystem::loadMatchState(int matchId) {
    // Load match state from database
    // Implementation would involve SQL queries
    return false;
}

std::string MatchSystem::exportMatchData(int matchId, const std::string& format) const {
    std::ostringstream output;
    
    if (format == "json") {
        // Export as JSON
        output << "{\n";
        output << "  \"matchId\": " << currentMatch_.matchId << ",\n";
        output << "  \"player1\": \"" << currentMatch_.player1.name << "\",\n";
        output << "  \"player2\": \"" << currentMatch_.player2.name << "\",\n";
        output << "  \"player1Wins\": " << currentMatch_.player1Wins << ",\n";
        output << "  \"player2Wins\": " << currentMatch_.player2Wins << ",\n";
        output << "  \"games\": [\n";
        
        for (size_t i = 0; i < currentMatch_.games.size(); ++i) {
            const auto& game = currentMatch_.games[i];
            if (i > 0) output << ",\n";
            output << "    {\n";
            output << "      \"gameNumber\": " << game.gameNumber << ",\n";
            output << "      \"winnerId\": " << game.winnerId << ",\n";
            output << "      \"duration\": " << std::fixed << std::setprecision(2) << game.duration << "\n";
            output << "    }";
        }
        
        output << "\n  ]\n";
        output << "}\n";
    }
    
    return output.str();
}

std::string MatchSystem::formatToString(MatchFormat format) {
    switch (format) {
        case MatchFormat::RaceToN: return "Race to N";
        case MatchFormat::BestOfN: return "Best of N";
        case MatchFormat::Timed: return "Timed";
        case MatchFormat::Tournament: return "Tournament";
        default: return "Unknown";
    }
}

std::string MatchSystem::gameTypeToString(GameType gameType) {
    switch (gameType) {
        case GameType::EightBall: return "8-Ball";
        case GameType::NineBall: return "9-Ball";
        case GameType::TenBall: return "10-Ball";
        case GameType::Straight: return "Straight Pool";
        case GameType::OnePocket: return "One Pocket";
        case GameType::BankPool: return "Bank Pool";
        default: return "Unknown";
    }
}

std::string MatchSystem::stateToString(MatchState state) {
    switch (state) {
        case MatchState::Setup: return "Setup";
        case MatchState::PreMatch: return "Pre-Match";
        case MatchState::InProgress: return "In Progress";
        case MatchState::Paused: return "Paused";
        case MatchState::Completed: return "Completed";
        case MatchState::Cancelled: return "Cancelled";
        default: return "Unknown";
    }
}

void MatchSystem::changeState(MatchState newState) {
    currentMatch_.state = newState;
    
    if (matchStateCallback_) {
        matchStateCallback_(newState);
    }
}

void MatchSystem::updateLiveStats(int playerId, bool shotSuccessful, double shotTime) {
    LiveStats* stats = nullptr;
    
    if (playerId == currentMatch_.player1.playerId) {
        stats = &player1Stats_;
    } else if (playerId == currentMatch_.player2.playerId) {
        stats = &player2Stats_;
    }
    
    if (!stats) {
        return;
    }
    
    // Update shot statistics
    stats->totalShots++;
    if (shotSuccessful) {
        stats->successfulShots++;
    }
    
    stats->shotSuccessRate = static_cast<double>(stats->successfulShots) / stats->totalShots;
    
    // Update timing statistics
    stats->recentShotTimes.push_back(shotTime);
    if (stats->recentShotTimes.size() > 20) {
        stats->recentShotTimes.erase(stats->recentShotTimes.begin());
    }
    
    if (shotTime > stats->maxShotTime) {
        stats->maxShotTime = shotTime;
    }
    
    // Calculate average shot time
    double totalTime = 0.0;
    for (double time : stats->recentShotTimes) {
        totalTime += time;
    }
    stats->avgShotTime = totalTime / stats->recentShotTimes.size();
}

bool MatchSystem::checkMatchWinCondition() {
    switch (currentMatch_.config.format) {
        case MatchFormat::RaceToN:
            return (currentMatch_.player1Wins >= currentMatch_.config.targetGames ||
                    currentMatch_.player2Wins >= currentMatch_.config.targetGames);
            
        case MatchFormat::BestOfN: {
            int gamesNeededToWin = (currentMatch_.config.targetGames / 2) + 1;
            return (currentMatch_.player1Wins >= gamesNeededToWin ||
                    currentMatch_.player2Wins >= gamesNeededToWin);
        }
        
        case MatchFormat::Timed: {
            auto now = std::chrono::steady_clock::now();
            double elapsed = std::chrono::duration<double>(now - currentMatch_.startTime).count() / 60.0;
            return elapsed >= currentMatch_.config.timeLimit;
        }
        
        default:
            return false;
    }
}

void MatchSystem::recordGameToDatabase(const GameResult& game) {
    // Record individual game to database
    // Implementation would involve SQL INSERT statements
}

void MatchSystem::recordMatchToDatabase() {
    // Record complete match to database
    // Implementation would involve SQL INSERT statements
}

int MatchSystem::getNextMatchId() {
    // Generate next available match ID
    // Implementation would query database for highest ID
    return 1;
}

void MatchSystem::initializeLiveStats() {
    player1Stats_ = LiveStats();
    player2Stats_ = LiveStats();
}

void MatchSystem::resetGameStats() {
    // Reset per-game statistics while keeping match-level stats
    // Could be used to track per-game vs overall match performance
}

} // namespace pv
