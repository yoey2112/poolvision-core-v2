#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include "../util/Types.hpp"

namespace pv {

class Physics {
public:
    struct Collision {
        int id1, id2;        // Ball IDs
        cv::Point2f normal;  // Collision normal
        float overlap;       // Overlap distance
    };
    
    // Physical constants
    static constexpr float DEFAULT_DAMPING = 0.97f;
    static constexpr float BALL_MASS = 0.17f; // kg
    static constexpr float FRICTION = 0.01f;  // Rolling friction coefficient
    static constexpr float RESTITUTION = 0.95f; // Coefficient of restitution
    
    // Reflect velocity on table bounds
    static void cushionReflect(cv::Point2f &pos, cv::Point2f &vel, 
                             const cv::Size &tableSize, float radius, 
                             float damping=DEFAULT_DAMPING);
    
    // Detect and resolve collisions between balls
    static std::vector<Collision> detectCollisions(const std::vector<Track> &tracks);
    static void resolveCollision(Track &a, Track &b, const cv::Point2f &normal);
    
    // Apply physics updates
    static void update(std::vector<Track> &tracks, float dt);
};

}
