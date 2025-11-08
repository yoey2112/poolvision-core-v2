#pragma once
#include "track/Tracker.hpp"
#include "events/EventEngine.hpp"
#include <vector>
#include <map>

namespace pv {

enum class GameType {
    EightBall,
    NineBall,
    // Add more game types as needed
};

enum class PlayerTurn {
    Player1,
    Player2
};

enum class BallGroup {
    Solids,
    Stripes,
    Black,
    None
};

enum class GameEvent {
    LegalPot,        // Legal ball potted
    IllegalPot,      // Wrong ball potted
    Scratch,         // Cue ball potted
    Foul,           // Any other foul
    Break,          // Break shot
    GameWon,        // Game won
    GameLost        // Game lost due to illegal 8-ball shot
};

struct Shot {
    int ballPotted;      // ID of the potted ball (-1 if none)
    bool isLegal;        // Whether the shot was legal
    bool isScratch;      // Whether the cue ball was potted
    bool isFoul;         // Whether a foul occurred
};

class GameState {
public:
    GameState(GameType type = GameType::EightBall);
    
    // Process new events and update game state
    void update(const std::vector<Track>& tracks, const std::vector<Event>& events);
    
    // Get current game status
    bool isBreakShot() const { return isBreak; }
    PlayerTurn getCurrentTurn() const { return currentTurn; }
    BallGroup getPlayerGroup(PlayerTurn player) const;
    bool isGameOver() const { return gameOver; }
    PlayerTurn getWinner() const { return winner; }
    
    // Get remaining balls
    std::vector<int> getRemainingBalls(BallGroup group) const;
    
    // Get game statistics
    int getScore(PlayerTurn player) const;
    const std::vector<Shot>& getShotHistory() const { return shotHistory; }
    
private:
    bool validateShot(const Event& event);
    void processEvent(const Event& event);
    void switchTurn();
    void checkGameEnd();
    
    GameType gameType;
    PlayerTurn currentTurn;
    PlayerTurn winner;
    bool isBreak;
    bool groupsAssigned;
    bool gameOver;
    
    std::map<PlayerTurn, BallGroup> playerGroups;
    std::map<BallGroup, std::vector<int>> remainingBalls;
    std::vector<Shot> shotHistory;
    
    // Track the last position of each ball
    std::map<int, cv::Point2f> lastPositions;
};

} // namespace pv
