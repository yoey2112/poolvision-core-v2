#pragma once
#include "../detect/classical/BallDetector.hpp"
#include "../track/Tracker.hpp"
#include "../game/GameState.hpp"
#include "../util/Types.hpp"
#include "UITheme.hpp"
#include <opencv2/opencv.hpp>
#include <memory>
#include <vector>
#include <algorithm>

namespace pv {

/**
 * @brief Real-time overlay renderer for game visualization
 * 
 * Handles all visual overlays including:
 * - Ball highlighting and numbering
 * - Shot trajectory prediction
 * - Game state HUD
 * - Position aids and ghost balls
 * - Statistics and suggestions
 */
class OverlayRenderer {
public:
    /**
     * @brief Construct a new Overlay Renderer
     * 
     * @param gameState Current game state for rules and scoring
     * @param tracker Ball tracker for velocity and prediction
     */
    OverlayRenderer(std::shared_ptr<GameState> gameState, std::shared_ptr<Tracker> tracker);
    ~OverlayRenderer() = default;
    
    /**
     * @brief Main rendering entry point
     * 
     * @param frame Input frame to draw overlays on
     * @param detectedBalls Current frame's detected balls
     * @param cueBallPos Position of cue ball if detected
     * @return cv::Mat Frame with overlays applied
     */
    cv::Mat render(cv::Mat& frame, 
                   const std::vector<Ball>& detectedBalls,
                   const cv::Point2f& cueBallPos);
    
    /**
     * @brief Handle mouse events for shot line and targeting
     */
    void onMouse(int event, int x, int y, int flags);
    
    /**
     * @brief Set window size for HUD scaling
     */
    void setWindowSize(int width, int height);
    
    /**
     * @brief Enable/disable specific overlay features
     */
    void setOverlayFlags(bool showTrajectory, bool showGhostBall, 
                        bool showPositionAids, bool showStats);
    
private:
    // Drawing functions
    void drawBallHighlights(cv::Mat& frame, const std::vector<Ball>& balls);
    void drawShotLine(cv::Mat& frame, const cv::Point2f& start, const cv::Point2f& end);
    void drawTrajectoryPrediction(cv::Mat& frame, const cv::Point2f& cueBallPos,
                                const cv::Point2f& targetPos);
    void drawGameStateHUD(cv::Mat& frame);
    void drawGhostBall(cv::Mat& frame, const cv::Point2f& pos);
    void drawPositionAids(cv::Mat& frame, const cv::Point2f& cueBallPos);
    void drawStatistics(cv::Mat& frame);
    
    // Prediction helpers
    std::vector<cv::Point2f> predictTrajectory(const cv::Point2f& startPos,
                                             const cv::Point2f& velocity,
                                             float power);
    cv::Point2f calculateGhostBallPosition(const cv::Point2f& cueBallPos,
                                         const cv::Point2f& targetPos);
    float evaluatePosition(const cv::Point2f& pos);
    
    // Member variables
    std::shared_ptr<GameState> gameState_;
    std::shared_ptr<Tracker> tracker_;
    
    // Window dimensions
    int windowWidth_ = 1280;
    int windowHeight_ = 720;
    
    // Mouse interaction
    cv::Point mousePos_;
    bool isDragging_ = false;
    cv::Point2f shotStart_;
    cv::Point2f shotEnd_;
    
    // Feature flags
    bool showTrajectory_ = true;
    bool showGhostBall_ = true;
    bool showPositionAids_ = true;
    bool showStats_ = true;
    
    // Constants
    const float GHOST_BALL_ALPHA = 0.5f;
    const int MAX_TRAJECTORY_POINTS = 50;
    const float MIN_SHOT_POWER = 0.1f;
    const float MAX_SHOT_POWER = 1.0f;
};

} // namespace pv