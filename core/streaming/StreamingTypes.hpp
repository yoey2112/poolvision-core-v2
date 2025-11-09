#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <map>

namespace pv {
namespace streaming {

// Element types
enum class ElementType {
    TEXT,
    IMAGE,
    VIDEO,
    SCORE,
    TIMER,
    PLAYER_NAME,
    GAME_STATE,
    CUSTOM
};

// Template styling
struct TemplateStyle {
    cv::Scalar primaryColor = cv::Scalar(255, 255, 255);      // White
    cv::Scalar secondaryColor = cv::Scalar(200, 200, 200);    // Light gray
    cv::Scalar backgroundColor = cv::Scalar(0, 0, 0, 128);    // Semi-transparent black
    std::string fontFamily = "Arial";
    int fontSize = 16;
    bool glassMorphism = false;
    float borderRadius = 5.0f;
    float opacity = 1.0f;
};

// Individual overlay element
struct OverlayElement {
    std::string id;
    ElementType type;
    cv::Point2f position;
    cv::Size2f size;
    std::string text;
    std::string imagePath;
    bool visible = true;
    float opacity = 1.0f;
    cv::Scalar color = cv::Scalar(255, 255, 255);
    std::map<std::string, std::string> properties;
};

// Complete overlay template
struct OverlayTemplate {
    std::string id;
    std::string name;
    std::string description;
    std::string author;
    std::vector<OverlayElement> elements;
    TemplateStyle style;
    cv::Size2f canvasSize = cv::Size2f(1920, 1080);  // Default 1080p
};

// Real-time overlay data
struct OverlayData {
    // Player information
    std::string player1Name;
    std::string player2Name;
    int player1Score = 0;
    int player2Score = 0;
    
    // Game state
    std::string gameType = "8-Ball";
    int currentRack = 1;
    std::string currentPlayer;
    std::string matchStatus = "In Progress";
    float gameProgress = 0.0f;
    bool isBreakShot = false;
    
    // Timing information
    std::chrono::steady_clock::time_point gameStartTime;
    std::chrono::steady_clock::time_point shotStartTime;
    std::chrono::steady_clock::time_point startTime; // Legacy compatibility
    
    // Ball positions and state
    std::vector<cv::Point2f> ballPositions;
    std::vector<bool> ballsPocketed;
    std::vector<int> ballsRemaining;
    
    // Shot information
    std::string suggestedShot;
    float shotDifficulty = 0.0f;
    std::string lastShotResult;
    
    // Statistics
    int totalShots = 0;
    int successfulShots = 0;
    float averageShotTime = 0.0f;
    
    // Streaming metadata
    std::string streamTitle;
    std::string eventName;
    bool isLive = false;
};

} // namespace streaming
} // namespace pv