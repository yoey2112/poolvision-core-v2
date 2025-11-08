#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace pv {

constexpr float BALL_RADIUS = 28.5f;  // Pool ball radius in pixels

struct Ball {
    int id = -1;
    cv::Point2f c;
    float r = BALL_RADIUS;
    float stripeScore = 0.0f; // 0..1
    int label = -1; // numeric label or -1

    cv::Point2f position() const { return c; }
    float radius() const { return r; }
};

struct Detection { Ball b; };

struct Track {
    int id = -1;
    cv::Point2f c;
    cv::Point2f v;
    float r = BALL_RADIUS;
    
    cv::Point2f position() const { return c; }
    float radius() const { return r; }
};

struct PocketEvent {
    int ball_id;
    int pocket_idx;
    double timestamp;
};

enum class EventType {
    Pocket,
    Collision,
    Foul
};

struct Event {
    EventType type;
    int ballId;
    double timestamp;
    // Additional event-specific data can be added here
};

struct FrameState {
    double timestamp = 0.0;
    std::vector<Ball> balls;
    std::vector<Track> tracks;
    std::vector<Event> events;
    std::string gameStatus; // JSON string containing game state information
};

// minimal json helpers
inline std::string json_escape(const std::string &s){
    std::string out;
    for(char c: s){
        if(c=='"') out += "\\\"";
        else if(c=='\\') out += "\\\\";
        else out += c;
    }
    return out;
}

} // namespace pv
