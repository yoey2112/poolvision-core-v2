#include "GameState.hpp"
#include <algorithm>

using namespace pv;

GameState::GameState(GameType type) 
    : gameType(type)
    , currentTurn(PlayerTurn::Player1)
    , isBreak(true)
    , groupsAssigned(false)
    , gameOver(false)
{
    // Initialize remaining balls based on game type
    if (gameType == GameType::EightBall) {
        remainingBalls[BallGroup::Solids] = {1, 2, 3, 4, 5, 6, 7};
        remainingBalls[BallGroup::Stripes] = {9, 10, 11, 12, 13, 14, 15};
        remainingBalls[BallGroup::Black] = {8};
    } else if (gameType == GameType::NineBall) {
        remainingBalls[BallGroup::None] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    }
}

void GameState::update(const std::vector<Track>& tracks, const std::vector<Event>& events) {
    // Update last known positions
    for (const auto& track : tracks) {
        lastPositions[track.id] = track.c;
    }
    
    // Process each event
    for (const auto& event : events) {
        if (event.type == EventType::Pocket) {
            Shot shot;
            shot.ballPotted = event.ballId;
            shot.isLegal = validateShot(event);
            shot.isScratch = (event.ballId == 0); // Assuming 0 is cue ball
            shot.isFoul = !shot.isLegal || shot.isScratch;
            
            // Update game state based on the shot
            if (shot.isLegal && !shot.isScratch) {
                // Remove potted ball from remaining balls
                for (auto& group : remainingBalls) {
                    auto& balls = group.second;
                    balls.erase(std::remove(balls.begin(), balls.end(), shot.ballPotted), balls.end());
                }
                
                // Assign groups on first legal pot after break
                if (isBreak && !shot.isFoul) {
                    isBreak = false;
                    if (!groupsAssigned && gameType == GameType::EightBall) {
                        BallGroup pottedGroup = (shot.ballPotted < 8) ? BallGroup::Solids : BallGroup::Stripes;
                        playerGroups[currentTurn] = pottedGroup;
                        playerGroups[currentTurn == PlayerTurn::Player1 ? PlayerTurn::Player2 : PlayerTurn::Player1] = 
                            (pottedGroup == BallGroup::Solids) ? BallGroup::Stripes : BallGroup::Solids;
                        groupsAssigned = true;
                    }
                }
            } else {
                // Switch turns on fouls
                switchTurn();
            }
            
            shotHistory.push_back(shot);
            checkGameEnd();
        }
    }
}

bool GameState::validateShot(const Event& event) {
    if (gameType == GameType::EightBall) {
        if (!groupsAssigned) {
            // During or right after break, any ball is legal except 8-ball
            return event.ballId != 8;
        }
        
        BallGroup playerGroup = playerGroups[currentTurn];
        int ballId = event.ballId;
        
        // Check if player pocketed their assigned group
        if (playerGroup == BallGroup::Solids) {
            return ballId >= 1 && ballId <= 7;
        } else if (playerGroup == BallGroup::Stripes) {
            return ballId >= 9 && ballId <= 15;
        }
        
        // 8-ball is only legal when all other balls of player's group are potted
        if (ballId == 8) {
            return remainingBalls[playerGroup].empty();
        }
    } else if (gameType == GameType::NineBall) {
        // In 9-ball, must hit lowest numbered ball first
        auto& balls = remainingBalls[BallGroup::None];
        return !balls.empty() && event.ballId >= balls.front();
    }
    
    return false;
}

void GameState::switchTurn() {
    currentTurn = (currentTurn == PlayerTurn::Player1) ? PlayerTurn::Player2 : PlayerTurn::Player1;
}

void GameState::checkGameEnd() {
    if (gameType == GameType::EightBall) {
        // Check if 8-ball was potted
        const auto& lastShot = shotHistory.back();
        if (lastShot.ballPotted == 8) {
            gameOver = true;
            if (lastShot.isLegal && !lastShot.isScratch) {
                winner = currentTurn;
            } else {
                winner = (currentTurn == PlayerTurn::Player1) ? PlayerTurn::Player2 : PlayerTurn::Player1;
            }
        }
    } else if (gameType == GameType::NineBall) {
        // Game ends when 9-ball is legally potted
        const auto& lastShot = shotHistory.back();
        if (lastShot.ballPotted == 9) {
            gameOver = true;
            if (lastShot.isLegal && !lastShot.isScratch) {
                winner = currentTurn;
            } else {
                winner = (currentTurn == PlayerTurn::Player1) ? PlayerTurn::Player2 : PlayerTurn::Player1;
            }
        }
    }
}

BallGroup GameState::getPlayerGroup(PlayerTurn player) const {
    auto it = playerGroups.find(player);
    return (it != playerGroups.end()) ? it->second : BallGroup::None;
}

std::vector<int> GameState::getRemainingBalls(BallGroup group) const {
    auto it = remainingBalls.find(group);
    return (it != remainingBalls.end()) ? it->second : std::vector<int>();
}

int GameState::getScore(PlayerTurn player) const {
    int score = 0;
    for (const auto& shot : shotHistory) {
        if (shot.isLegal && !shot.isScratch && shot.ballPotted > 0) {
            score++;
        }
    }
    return score;
}