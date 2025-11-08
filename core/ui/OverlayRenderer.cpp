#include "OverlayRenderer.hpp"
#include "../track/Physics.hpp"
#include <iostream>

namespace pv {

OverlayRenderer::OverlayRenderer(std::shared_ptr<GameState> gameState,
                               std::shared_ptr<Tracker> tracker)
    : gameState_(gameState)
    , tracker_(tracker) {
}

cv::Mat OverlayRenderer::render(cv::Mat& frame,
                              const std::vector<Ball>& detectedBalls,
                              const cv::Point2f& cueBallPos) {
    // Create overlay layer
    cv::Mat overlay = frame.clone();
    
    // Draw features based on flags
    if (showTrajectory_ && isDragging_) {
        drawShotLine(overlay, shotStart_, shotEnd_);
        drawTrajectoryPrediction(overlay, cueBallPos, shotEnd_);
    }
    
    drawBallHighlights(overlay, detectedBalls);
    
    if (showGhostBall_ && isDragging_) {
        drawGhostBall(overlay, calculateGhostBallPosition(cueBallPos, shotEnd_));
    }
    
    if (showPositionAids_) {
        drawPositionAids(overlay, cueBallPos);
    }
    
    drawGameStateHUD(overlay);
    
    if (showStats_) {
        drawStatistics(overlay);
    }
    
    return overlay;
}

void OverlayRenderer::drawBallHighlights(cv::Mat& frame, const std::vector<Ball>& balls) {
    for (const auto& ball : balls) {
        // Draw ball outline
        cv::Scalar color;
        int thickness = 2;
        
        if (ball.id == 0) {  // Cue ball
            color = UITheme::Colors::NeonCyan;
            thickness = 3;  // More prominent
        } else if (gameState_->isLegalTarget(ball.id)) {
            color = UITheme::Colors::NeonGreen;
        } else {
            color = UITheme::Colors::NeonRed;
        }
        
        // Ball circle with glow effect
        cv::circle(frame, ball.c, ball.r + 2, color, thickness);
        cv::circle(frame, ball.c, ball.r + 4, color, 1, cv::LINE_AA);
        
        // Draw ball number
        std::string number = std::to_string(ball.id);
        int fontFace = UITheme::Fonts::FontFace;
        double fontScale = UITheme::Fonts::SmallSize * 0.8;
        int textThickness = UITheme::Fonts::BodyThickness;
        
        cv::Size textSize = cv::getTextSize(number, fontFace, fontScale, thickness, nullptr);
        cv::Point textPos(ball.c.x - textSize.width/2,
                         ball.c.y + textSize.height/3);
        
        // Draw text with glow
        cv::putText(frame, number, textPos, fontFace, fontScale,
                   UITheme::Colors::TextShadow, thickness + 2, cv::LINE_AA);
        cv::putText(frame, number, textPos, fontFace, fontScale,
                   UITheme::Colors::TextPrimary, thickness, cv::LINE_AA);
    }
}

void OverlayRenderer::drawShotLine(cv::Mat& frame, const cv::Point2f& start,
                                const cv::Point2f& end) {
    // Draw aiming line
    cv::line(frame, start, end, UITheme::Colors::NeonYellow, 2, cv::LINE_AA);
    
    // Draw direction arrow
    float angle = atan2(end.y - start.y, end.x - start.x);
    float arrowLength = 20;
    float arrowAngle = CV_PI / 6;  // 30 degrees
    
    cv::Point2f arrowP1(end.x - arrowLength * cos(angle + arrowAngle),
                        end.y - arrowLength * sin(angle + arrowAngle));
    cv::Point2f arrowP2(end.x - arrowLength * cos(angle - arrowAngle),
                        end.y - arrowLength * sin(angle - arrowAngle));
    
    cv::line(frame, end, arrowP1, UITheme::Colors::NeonYellow, 2, cv::LINE_AA);
    cv::line(frame, end, arrowP2, UITheme::Colors::NeonYellow, 2, cv::LINE_AA);
    
    // Draw power indicator
    float power = std::min(1.0f, static_cast<float>(cv::norm(end - start)) / 200.0f);  // Normalize to 0-1
    cv::Rect powerBar(10, frame.rows - 40, 200, 20);
    
    // Background
    cv::rectangle(frame, powerBar, UITheme::Colors::DarkBg, cv::FILLED);
    cv::rectangle(frame, powerBar, UITheme::Colors::BorderColor, 1);
    
    // Fill bar
    cv::Rect fillBar = powerBar;
    fillBar.width = static_cast<int>(powerBar.width * power);
    cv::rectangle(frame, fillBar, UITheme::Colors::NeonYellow, cv::FILLED);
    
    // Power text
    std::string powerText = std::to_string(static_cast<int>(power * 100)) + "%";
    cv::putText(frame, powerText, cv::Point(powerBar.x + powerBar.width + 10,
               powerBar.y + 15), UITheme::Fonts::FontFace,
               UITheme::Fonts::SmallSize, UITheme::Colors::TextSecondary,
               UITheme::Fonts::BodyThickness, cv::LINE_AA);
}

void OverlayRenderer::drawTrajectoryPrediction(cv::Mat& frame,
                                             const cv::Point2f& cueBallPos,
                                             const cv::Point2f& targetPos) {
    // Calculate initial velocity vector
    cv::Point2f direction = targetPos - cueBallPos;
    float power = std::min(1.0f, static_cast<float>(cv::norm(direction)) / 200.0f);
    cv::Point2f velocity = direction * (power / cv::norm(direction)) * 50.0f;
    
    // Get predicted trajectory
    std::vector<cv::Point2f> trajectory = predictTrajectory(cueBallPos, velocity, power);
    
    // Draw trajectory line with fade-out effect
    for (size_t i = 1; i < trajectory.size(); ++i) {
        float alpha = 1.0f - (static_cast<float>(i) / trajectory.size());
        cv::Scalar color = UITheme::Colors::NeonCyan * alpha;
        
        cv::line(frame, trajectory[i-1], trajectory[i], color, 2, cv::LINE_AA);
        
        // Draw bounce indicators
        if (i > 1) {
            cv::Point2f v1 = trajectory[i-1] - trajectory[i-2];
            cv::Point2f v2 = trajectory[i] - trajectory[i-1];
            float angle1 = atan2(v1.y, v1.x);
            float angle2 = atan2(v2.y, v2.x);
            
            if (std::abs(angle1 - angle2) > 0.1) {  // Significant direction change
                cv::circle(frame, trajectory[i-1], 4, color, -1, cv::LINE_AA);
            }
        }
    }
}

void OverlayRenderer::drawGameStateHUD(cv::Mat& frame) {
    // Player info at top
    std::string playerText = gameState_->getCurrentPlayer() + "'s Turn";
    cv::putText(frame, playerText, cv::Point(20, 40), UITheme::Fonts::FontFaceBold,
               UITheme::Fonts::HeadingSize, UITheme::Colors::NeonCyan,
               UITheme::Fonts::HeadingThickness, cv::LINE_AA);
    
    // Remaining balls
    int y = 80;
    auto group = gameState_->getPlayerGroup(gameState_->getCurrentTurn());
    std::vector<int> remainingBalls = gameState_->getRemainingBalls(group);
    std::string ballsText = "Remaining: ";
    for (int id : remainingBalls) {
        ballsText += std::to_string(id) + " ";
    }
    cv::putText(frame, ballsText, cv::Point(20, y), UITheme::Fonts::FontFace,
               UITheme::Fonts::BodySize, UITheme::Colors::TextPrimary,
               UITheme::Fonts::BodyThickness, cv::LINE_AA);
    
    // Game state
    y += 30;
    std::string stateText = gameState_->getStateString();
    cv::putText(frame, stateText, cv::Point(20, y), UITheme::Fonts::FontFace,
               UITheme::Fonts::BodySize, UITheme::Colors::TextSecondary,
               UITheme::Fonts::BodyThickness, cv::LINE_AA);
    
    // Last foul if any
    if (gameState_->hasFoul()) {
        y += 30;
        std::string foulText = "FOUL: " + gameState_->getFoulReason();
        cv::putText(frame, foulText, cv::Point(20, y), UITheme::Fonts::FontFace,
                   UITheme::Fonts::BodySize, UITheme::Colors::NeonRed,
                   UITheme::Fonts::BodyThickness, cv::LINE_AA);
    }
}

void OverlayRenderer::drawGhostBall(cv::Mat& frame, const cv::Point2f& pos) {
    // Draw semi-transparent ghost ball
    cv::Mat overlay;
    frame.copyTo(overlay);
    
    cv::circle(overlay, pos, BALL_RADIUS, UITheme::Colors::NeonCyan, -1, cv::LINE_AA);
    cv::circle(overlay, pos, BALL_RADIUS, UITheme::Colors::NeonCyan, 1, cv::LINE_AA);
    
    cv::addWeighted(overlay, GHOST_BALL_ALPHA, frame, 1 - GHOST_BALL_ALPHA, 0, frame);
}

void OverlayRenderer::drawPositionAids(cv::Mat& frame, const cv::Point2f& cueBallPos) {
    if (!isDragging_) return;
    
    // Get next legal targets
    std::vector<int> legalTargets = gameState_->getLegalTargets();
    
    for (const auto& targetId : legalTargets) {
        // Find ball in tracker
        auto targetBall = tracker_->getBall(targetId);
        if (!targetBall) continue;
        
        // Draw position zones
        cv::Mat heatmap = cv::Mat::zeros(frame.size(), CV_8UC3);
        
        // Generate position quality heatmap
        for (int y = 0; y < frame.rows; y += 20) {
            for (int x = 0; x < frame.cols; x += 20) {
                cv::Point2f pos(x, y);
                float quality = evaluatePosition(pos);
                
                if (quality > 0.3f) {
                    cv::Scalar color = UITheme::Colors::NeonGreen * quality;
                    cv::circle(heatmap, pos, 10, color, -1, cv::LINE_AA);
                }
            }
        }
        
        // Blend heatmap with frame
        cv::addWeighted(frame, 1.0, heatmap, 0.3, 0, frame);
    }
}

void OverlayRenderer::drawStatistics(cv::Mat& frame) {
    // Show shot difficulty
    float difficulty = evaluatePosition(shotEnd_);
    std::string diffText = "Shot Difficulty: " + 
                          std::to_string(static_cast<int>(difficulty * 100)) + "%";
    
    cv::Point pos(frame.cols - 250, frame.rows - 40);
    cv::putText(frame, diffText, pos, UITheme::Fonts::FontFace,
               UITheme::Fonts::BodySize, UITheme::Colors::TextSecondary,
               UITheme::Fonts::BodyThickness, cv::LINE_AA);
    
    // Show suggested shots when not dragging
    if (!isDragging_) {
        // Convert Shot objects to point/score pairs
        auto shots = gameState_->getSuggestedShots();
        std::vector<std::pair<cv::Point2f, float>> suggestions;
        for (const auto& shot : shots) {
            // Use ball position as shot target point and legality as score
            if (auto targetBall = tracker_->getBall(shot.ballPotted)) {
                suggestions.emplace_back(targetBall->c, shot.isLegal ? 1.0f : 0.0f);
            }
        }
        int y = 120;
        
        for (const auto& [pos, score] : suggestions) {
            std::string text = "Try shot to (" + 
                             std::to_string(static_cast<int>(pos.x)) + ", " +
                             std::to_string(static_cast<int>(pos.y)) + ") - " +
                             std::to_string(static_cast<int>(score * 100)) + "%";
            
            cv::putText(frame, text, cv::Point(frame.cols - 350, y),
                       UITheme::Fonts::FontFace, UITheme::Fonts::SmallSize,
                       UITheme::Colors::TextSecondary, UITheme::Fonts::BodyThickness,
                       cv::LINE_AA);
            y += 25;
        }
    }
}

void OverlayRenderer::onMouse(int event, int x, int y, int flags) {
    mousePos_ = cv::Point(x, y);
    
    if (event == cv::EVENT_LBUTTONDOWN) {
        isDragging_ = true;
        shotStart_ = mousePos_;
        shotEnd_ = mousePos_;
    }
    else if (event == cv::EVENT_MOUSEMOVE && isDragging_) {
        shotEnd_ = mousePos_;
    }
    else if (event == cv::EVENT_LBUTTONUP) {
        isDragging_ = false;
    }
}

void OverlayRenderer::setWindowSize(int width, int height) {
    windowWidth_ = width;
    windowHeight_ = height;
}

void OverlayRenderer::setOverlayFlags(bool showTrajectory, bool showGhostBall,
                                   bool showPositionAids, bool showStats) {
    showTrajectory_ = showTrajectory;
    showGhostBall_ = showGhostBall;
    showPositionAids_ = showPositionAids;
    showStats_ = showStats;
}

std::vector<cv::Point2f> OverlayRenderer::predictTrajectory(
    const cv::Point2f& startPos, const cv::Point2f& velocity, float power) {
    std::vector<cv::Point2f> points;
    points.push_back(startPos);
    
    cv::Point2f pos = startPos;
    cv::Point2f vel = velocity;
    float friction = 0.98f;  // Air resistance
    
    for (int i = 0; i < MAX_TRAJECTORY_POINTS && cv::norm(vel) > 0.1f; ++i) {
        // Update position
        pos += vel;
        points.push_back(pos);
        
        // Apply friction
        vel *= friction;
        
        // Check table boundaries (simplified)
        if (pos.x < 0 || pos.x > windowWidth_) vel.x *= -0.8f;
        if (pos.y < 0 || pos.y > windowHeight_) vel.y *= -0.8f;
    }
    
    return points;
}

cv::Point2f OverlayRenderer::calculateGhostBallPosition(
    const cv::Point2f& cueBallPos, const cv::Point2f& targetPos) {
    cv::Point2f direction = targetPos - cueBallPos;
    float distance = cv::norm(direction);
    
    if (distance < 1e-6) return cueBallPos;
    
    direction *= (1.0f / distance);  // Normalize
    return targetPos - direction * (BALL_RADIUS * 2);  // Two ball radii away
}

float OverlayRenderer::evaluatePosition(const cv::Point2f& pos) {
    // Simple position evaluation based on:
    // 1. Distance from cushions
    // 2. Distance from other balls
    // 3. Angle to potential targets
    
    float score = 1.0f;
    
    // Penalize positions too close to cushions
    float cushionDist = std::min({
        pos.x, pos.y,
        static_cast<float>(windowWidth_) - pos.x,
        static_cast<float>(windowHeight_) - pos.y
    });
    score *= std::min(1.0f, cushionDist / (BALL_RADIUS * 3));
    
    // Penalize positions too close to other balls
    for (const auto& ball : tracker_->getBalls()) {
        float dist = cv::norm(pos - ball.c);
        if (dist < BALL_RADIUS * 3) {
            score *= dist / (BALL_RADIUS * 3);
        }
    }
    
    return score;
}

} // namespace pv