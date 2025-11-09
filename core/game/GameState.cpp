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

bool GameState::isLegalTarget(int ballId) const {
    if (gameType == GameType::EightBall) {
        if (!groupsAssigned) {
            // During or right after break, any ball is legal except 8-ball
            return ballId != 8;
        }
        
        BallGroup playerGroup = playerGroups.at(currentTurn);
        // Check if ball belongs to player's group
        if (playerGroup == BallGroup::Solids) {
            return ballId >= 1 && ballId <= 7;
        } else if (playerGroup == BallGroup::Stripes) {
            return ballId >= 9 && ballId <= 15;
        }
        
        // 8-ball is only legal when all other balls of player's group are potted
        if (ballId == 8) {
            return remainingBalls.at(playerGroup).empty();
        }
    } else if (gameType == GameType::NineBall) {
        // In 9-ball, must hit lowest numbered ball first
        const auto& balls = remainingBalls.at(BallGroup::None);
        return !balls.empty() && ballId >= balls.front();
    }
    return false;
}

std::string GameState::getCurrentPlayer() const {
    return (currentTurn == PlayerTurn::Player1) ? "Player 1" : "Player 2";
}

std::string GameState::getStateString() const {
    if (gameOver) {
        return std::string("Game Over - ") + getCurrentPlayer() + " Wins";
    }
    if (isBreak) {
        return "Break Shot";
    }
    if (!groupsAssigned && gameType == GameType::EightBall) {
        return "Open Table";
    }
    if (gameType == GameType::EightBall) {
        auto group = playerGroups.at(currentTurn);
        return getCurrentPlayer() + " - " + 
               (group == BallGroup::Solids ? "Solids" : 
                group == BallGroup::Stripes ? "Stripes" : "8-Ball");
    }
    return getCurrentPlayer() + "'s Turn";
}

bool GameState::hasFoul() const {
    return !shotHistory.empty() && shotHistory.back().isFoul;
}

std::string GameState::getFoulReason() const {
    if (!hasFoul()) return "";
    
    const auto& lastShot = shotHistory.back();
    if (lastShot.isScratch) {
        return "Scratch - Cue ball potted";
    }
    if (!lastShot.isLegal) {
        if (lastShot.ballPotted == 8) {
            return "Illegal 8-ball shot";
        }
        return "Illegal ball";
    }
    return "Foul";
}

std::vector<int> GameState::getLegalTargets() const {
    std::vector<int> targets;
    
    if (gameType == GameType::EightBall) {
        if (!groupsAssigned) {
            // During or right after break, any ball is legal except 8-ball
            targets = {1,2,3,4,5,6,7,9,10,11,12,13,14,15};
        } else {
            BallGroup playerGroup = playerGroups.at(currentTurn);
            targets = remainingBalls.at(playerGroup);
            // Add 8-ball if all group balls are potted
            if (targets.empty()) {
                targets = remainingBalls.at(BallGroup::Black);
            }
        }
    } else if (gameType == GameType::NineBall) {
        const auto& balls = remainingBalls.at(BallGroup::None);
        if (!balls.empty()) {
            targets.push_back(balls.front());  // Only lowest numbered ball is legal
        }
    }
    
    return targets;
}

std::vector<Shot> GameState::getSuggestedShots() const {
    std::vector<Shot> suggestions;
    
    // Get legal target balls for current player
    std::vector<int> legalTargets = getLegalTargets();
    if (legalTargets.empty()) {
        return suggestions;
    }
    
    // Find cue ball position from current ball positions
    cv::Point2f cueBallPos;
    bool cueBallFound = false;
    for (const auto& pair : lastPositions) {
        if (pair.first == 0) { // Cue ball ID is 0
            cueBallPos = pair.second;
            cueBallFound = true;
            break;
        }
    }
    
    if (!cueBallFound) {
        return suggestions; // Cannot suggest shots without cue ball position
    }
    
    // Generate suggestions for each legal target ball
    for (int targetBallId : legalTargets) {
        auto ballIter = lastPositions.find(targetBallId);
        if (ballIter == lastPositions.end()) {
            continue; // Target ball not found
        }
        
        cv::Point2f targetBallPos = ballIter->second;
        
        // Calculate shot parameters
        cv::Point2f direction = targetBallPos - cueBallPos;
        float distance = cv::norm(direction);
        
        if (distance < 10.0f) {
            continue; // Too close, skip this shot
        }
        
        // Create shot suggestion
        Shot suggestion;
        suggestion.ballPotted = targetBallId;
        suggestion.isLegal = true;
        suggestion.isScratch = false;
        suggestion.isFoul = false;
        
        // Calculate shot difficulty based on distance and angle
        float difficultyScore = calculateShotDifficulty(cueBallPos, targetBallPos);
        
        // Only suggest shots with reasonable difficulty (not impossible)
        if (difficultyScore <= 0.9f) {
            suggestions.push_back(suggestion);
        }
    }
    
    // Sort suggestions by difficulty (easier shots first)
    std::sort(suggestions.begin(), suggestions.end(), 
              [this, cueBallPos](const Shot& a, const Shot& b) {
                  auto posA = lastPositions.find(a.ballPotted);
                  auto posB = lastPositions.find(b.ballPotted);
                  if (posA == lastPositions.end() || posB == lastPositions.end()) {
                      return false;
                  }
                  float diffA = calculateShotDifficulty(cueBallPos, posA->second);
                  float diffB = calculateShotDifficulty(cueBallPos, posB->second);
                  return diffA < diffB;
              });
    
    // Limit to top 3 suggestions to avoid overwhelming the player
    if (suggestions.size() > 3) {
        suggestions.resize(3);
    }
    
    return suggestions;
}

float GameState::calculateShotDifficulty(const cv::Point2f& cueBallPos, 
                                        const cv::Point2f& targetBallPos) const {
    // Calculate basic distance factor
    float distance = cv::norm(targetBallPos - cueBallPos);
    float normalizedDistance = std::min(distance / 500.0f, 1.0f); // Normalize to 500 pixels max
    
    // Check for obstacles (other balls in the way)
    float obstacleFactor = 0.0f;
    cv::Point2f direction = targetBallPos - cueBallPos;
    float pathLength = cv::norm(direction);
    
    if (pathLength > 0.0f) {
        direction /= pathLength;
        
        // Check for balls along the shot path
        for (const auto& pair : lastPositions) {
            if (pair.first == 0 || pair.first == (targetBallPos.x + targetBallPos.y)) { // Skip cue ball and target
                continue;
            }
            
            cv::Point2f ballPos = pair.second;
            cv::Point2f toBall = ballPos - cueBallPos;
            
            // Project ball position onto shot line
            float projection = toBall.dot(direction);
            if (projection > 0 && projection < pathLength) {
                cv::Point2f closestPoint = cueBallPos + direction * projection;
                float distanceToLine = cv::norm(ballPos - closestPoint);
                
                // If ball is within 2 ball radii of shot line, it's an obstacle
                if (distanceToLine < 60.0f) { // 2 * BALL_RADIUS (approximate)
                    obstacleFactor += 0.3f; // Increase difficulty
                }
            }
        }
    }
    
    // Calculate angle difficulty (shots requiring extreme angles are harder)
    float angle = 0.0f;
    if (pathLength > 0.0f) {
        // Simplified angle calculation - shots requiring sharp angles to pockets are harder
        // For now, assume straight shots are easier
        angle = 0.1f; // Base angle difficulty
    }
    
    // Combine factors
    float totalDifficulty = normalizedDistance * 0.5f + obstacleFactor + angle * 0.2f;
    
    return std::min(totalDifficulty, 1.0f); // Cap at 1.0
}
