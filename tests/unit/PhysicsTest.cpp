#include <gtest/gtest.h>
#include "../../core/track/Physics.hpp"
#include <opencv2/opencv.hpp>

using namespace pv;

TEST(PhysicsTest, CushionReflection) {
    cv::Size tableSize(1000, 500);
    cv::Point2f pos(990, 250);
    cv::Point2f vel(20, 0);
    float radius = 10;
    
    // Ball moving right should reflect off right cushion
    Physics::cushionReflect(pos, vel, tableSize, radius);
    EXPECT_LT(vel.x, 0); // Velocity should be reversed
    EXPECT_FLOAT_EQ(pos.x, tableSize.width - radius); // Position should be at boundary
    
    // Test left cushion
    pos = cv::Point2f(10, 250);
    vel = cv::Point2f(-20, 0);
    Physics::cushionReflect(pos, vel, tableSize, radius);
    EXPECT_GT(vel.x, 0);
    EXPECT_FLOAT_EQ(pos.x, radius);
}

TEST(PhysicsTest, BallCollisionDetection) {
    std::vector<Track> tracks = {
        {1, {100, 100}, {10, 0}, 10}, // Moving right
        {2, {120, 100}, {-10, 0}, 10} // Moving left
    };
    
    auto collisions = Physics::detectCollisions(tracks);
    ASSERT_EQ(collisions.size(), 1);
    EXPECT_EQ(collisions[0].id1, 1);
    EXPECT_EQ(collisions[0].id2, 2);
}

TEST(PhysicsTest, CollisionResolution) {
    Track a{1, {100, 100}, {10, 0}, 10};
    Track b{2, {120, 100}, {-10, 0}, 10};
    cv::Point2f normal(1, 0); // Collision along x-axis
    
    float va_before = a.v.x;
    float vb_before = b.v.x;
    
    Physics::resolveCollision(a, b, normal);
    
    // Conservation of momentum check
    float total_momentum_before = Physics::BALL_MASS * va_before + Physics::BALL_MASS * vb_before;
    float total_momentum_after = Physics::BALL_MASS * a.v.x + Physics::BALL_MASS * b.v.x;
    EXPECT_NEAR(total_momentum_before, total_momentum_after, 0.001f);
}

TEST(PhysicsTest, PhysicsUpdate) {
    std::vector<Track> tracks = {
        {1, {100, 100}, {10, 0}, 10}, // Moving right
        {2, {300, 100}, {-10, 0}, 10} // Moving left
    };
    
    const float dt = 1.0f/30.0f;
    Physics::update(tracks, dt);
    
    // Check friction effect
    for(const auto &t: tracks) {
        float speed = cv::norm(t.v);
        float initial_speed = 10.0f;
        EXPECT_LT(speed, initial_speed); // Speed should decrease due to friction
    }
}

TEST(PhysicsTest, MultipleCollisions) {
    std::vector<Track> tracks = {
        {1, {100, 100}, {10, 0}, 10},  // Moving right
        {2, {120, 100}, {0, 0}, 10},   // Stationary
        {3, {140, 100}, {-10, 0}, 10}  // Moving left
    };
    
    auto collisions = Physics::detectCollisions(tracks);
    EXPECT_EQ(collisions.size(), 2); // Should detect both collisions
    
    const float dt = 1.0f/30.0f;
    Physics::update(tracks, dt);
    
    // Middle ball should now be moving
    EXPECT_NE(tracks[1].v.x, 0);
}