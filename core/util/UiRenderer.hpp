#pragma once
#include <opencv2/opencv.hpp>
#include "game/GameState.hpp"
#include "../track/Tracker.hpp"
#include <string>

namespace pv {

class UiRenderer {
public:
    struct Theme {
        cv::Scalar primaryColor{0, 153, 255};      // Orange
        cv::Scalar secondaryColor{255, 255, 255};  // White
        cv::Scalar accentColor{0, 255, 153};       // Green
        cv::Scalar warningColor{0, 0, 255};        // Red
        cv::Scalar textColor{255, 255, 255};       // White
        cv::Scalar overlayBg{0, 0, 0, 180};        // Semi-transparent black
        
        int fontSize = 1;
        int thickness = 2;
        int lineType = cv::LINE_AA;
    };
    
    UiRenderer();
    
    // Main render function
    cv::Mat render(const cv::Mat& frame, const GameState& gameState,
                  const std::vector<Ball>& balls, const std::vector<Track>& tracks);
    
private:
    // UI Components
    void renderGameStatus(cv::Mat& output, const GameState& gameState);
    void renderBalls(cv::Mat& output, const std::vector<Ball>& balls, const std::vector<Track>& tracks);
    void renderScoreboard(cv::Mat& output, const GameState& gameState);
    void renderTurnIndicator(cv::Mat& output, const GameState& gameState);
    
    // Helper functions
    void drawText(cv::Mat& img, const std::string& text, cv::Point pos, 
                 const cv::Scalar& color, double scale = 1.0, int thickness = 1);
    void drawOverlayBox(cv::Mat& img, cv::Rect box, double alpha = 0.7);
    
    Theme theme;
    const int margin = 20;
    const int padding = 10;
};

} // namespace pv
