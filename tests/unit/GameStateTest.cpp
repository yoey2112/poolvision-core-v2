#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "game/GameState.hpp"

using namespace pv;

class GameStateTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Set up default game state for 8-ball
        gameState = std::make_unique<GameState>(GameType::EightBall);
    }
    
    std::unique_ptr<GameState> gameState;
    
    Event createPocketEvent(int ballId) {
        Event e;
        e.type = EventType::Pocket;
        e.ballId = ballId;
        e.timestamp = 0.0;
        return e;
    }
    
    Track createTrack(int id, cv::Point2f pos) {
        Track t;
        t.id = id;
        t.c = pos;
        t.v = cv::Point2f(0, 0);
        t.r = 1.0f;
        return t;
    }
};

TEST_F(GameStateTest, InitialState) {
    EXPECT_TRUE(gameState->isBreakShot());
    EXPECT_EQ(gameState->getCurrentTurn(), PlayerTurn::Player1);
    EXPECT_FALSE(gameState->isGameOver());
}

TEST_F(GameStateTest, GroupAssignment) {
    std::vector<Track> tracks = {createTrack(1, cv::Point2f(0, 0))};
    
    // Pot a solid ball after break
    std::vector<Event> events = {createPocketEvent(1)};
    gameState->update(tracks, events);
    
    // Player 1 should be assigned solids
    EXPECT_EQ(gameState->getPlayerGroup(PlayerTurn::Player1), BallGroup::Solids);
    EXPECT_EQ(gameState->getPlayerGroup(PlayerTurn::Player2), BallGroup::Stripes);
}

TEST_F(GameStateTest, IllegalShot) {
    std::vector<Track> tracks = {createTrack(8, cv::Point2f(0, 0))};
    
    // Try to pot 8-ball on break (illegal)
    std::vector<Event> events = {createPocketEvent(8)};
    gameState->update(tracks, events);
    
    // Should switch turns due to illegal shot
    EXPECT_EQ(gameState->getCurrentTurn(), PlayerTurn::Player2);
}

TEST_F(GameStateTest, GameWinCondition) {
    // Simulate clearing all solids
    for (int i = 1; i <= 7; i++) {
        std::vector<Track> tracks = {createTrack(i, cv::Point2f(0, 0))};
        std::vector<Event> events = {createPocketEvent(i)};
        gameState->update(tracks, events);
    }
    
    // Now pot the 8-ball
    std::vector<Track> tracks = {createTrack(8, cv::Point2f(0, 0))};
    std::vector<Event> events = {createPocketEvent(8)};
    gameState->update(tracks, events);
    
    EXPECT_TRUE(gameState->isGameOver());
}

TEST_F(GameStateTest, NineBallGameplay) {
    GameState nineBallState(GameType::NineBall);
    
    // Legal shot - hitting lowest ball first
    std::vector<Track> tracks = {createTrack(1, cv::Point2f(0, 0))};
    std::vector<Event> events = {createPocketEvent(1)};
    nineBallState.update(tracks, events);
    
    // Verify remaining balls
    auto remaining = nineBallState.getRemainingBalls(BallGroup::None);
    EXPECT_EQ(remaining.size(), 8);
    EXPECT_EQ(remaining.front(), 2);
}

TEST_F(GameStateTest, Scratches) {
    // Simulate a scratch (potting the cue ball)
    std::vector<Track> tracks = {createTrack(0, cv::Point2f(0, 0))};
    std::vector<Event> events = {createPocketEvent(0)};
    gameState->update(tracks, events);
    
    // Should switch turns on scratch
    EXPECT_EQ(gameState->getCurrentTurn(), PlayerTurn::Player2);
}

TEST_F(GameStateTest, ScoreTracking) {
    // Legal pot of ball 1
    std::vector<Track> tracks = {createTrack(1, cv::Point2f(0, 0))};
    std::vector<Event> events = {createPocketEvent(1)};
    gameState->update(tracks, events);
    
    // Score should increment for legal pot
    EXPECT_EQ(gameState->getScore(PlayerTurn::Player1), 1);
}