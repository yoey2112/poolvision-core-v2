#include "UiRenderer.hpp"
#include <sstream>
#include <iomanip>

using namespace pv;

UiRenderer::UiRenderer() {}

cv::Mat UiRenderer::render(const cv::Mat& frame, const GameState& gameState,
                          const std::vector<Ball>& balls, const std::vector<Track>& tracks) {
    cv::Mat output = frame.clone();
    
    // Add semi-transparent overlay at the top
    cv::Rect headerRect(0, 0, output.cols, 60);
    drawOverlayBox(output, headerRect);
    
    // Add semi-transparent overlay at the bottom
    cv::Rect footerRect(0, output.rows - 60, output.cols, 60);
    drawOverlayBox(output, footerRect);
    
    // Render UI components
    renderGameStatus(output, gameState);
    renderBalls(output, balls, tracks);
    renderScoreboard(output, gameState);
    renderTurnIndicator(output, gameState);
    
    return output;
}

void UiRenderer::renderGameStatus(cv::Mat& output, const GameState& gameState) {
    std::string status;
    cv::Scalar statusColor;
    
    if (gameState.isGameOver()) {
        status = "Game Over - " + std::string(gameState.getWinner() == PlayerTurn::Player1 ? "Player 1" : "Player 2") + " Wins!";
        statusColor = theme.warningColor;
    } else if (gameState.isBreakShot()) {
        status = "Break Shot";
        statusColor = theme.accentColor;
    } else {
        status = std::string(gameState.getCurrentTurn() == PlayerTurn::Player1 ? "Player 1" : "Player 2") + "'s Turn";
        statusColor = theme.primaryColor;
    }
    
    // Draw centered status text
    int baseline = 0;
    cv::Size textSize = cv::getTextSize(status, cv::FONT_HERSHEY_SIMPLEX, 1.0, theme.thickness, &baseline);
    cv::Point textPos((output.cols - textSize.width) / 2, 40);
    drawText(output, status, textPos, statusColor, 1.0, theme.thickness);
}

void UiRenderer::renderBalls(cv::Mat& output, const std::vector<Ball>& balls, const std::vector<Track>& tracks) {
    // Draw predicted trajectories
    for (const auto& track : tracks) {
        // Draw velocity vector
        cv::Point2f endPoint = track.c + track.v * 20; // Scale velocity for visualization
        cv::arrowedLine(output, track.c, endPoint, theme.primaryColor, 2, theme.lineType);
        
        // Draw tracking circle
        cv::circle(output, track.c, track.r, theme.secondaryColor, 2, theme.lineType);
    }
    
    // Draw detected balls with labels
    for (const auto& ball : balls) {
        // Draw ball circle
        cv::Scalar ballColor = (ball.stripeScore > 0.5) ? theme.accentColor : theme.primaryColor;
        cv::circle(output, ball.c, ball.r, ballColor, -1, theme.lineType);
        
        // Draw ball number if available
        if (ball.label > 0) {
            std::string label = std::to_string(ball.label);
            int baseline = 0;
            cv::Size textSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
            cv::Point textPos(ball.c.x - textSize.width/2, ball.c.y + textSize.height/2);
            drawText(output, label, textPos, theme.textColor, 0.5, 1);
        }
    }
}

void UiRenderer::renderScoreboard(cv::Mat& output, const GameState& gameState) {
    // Create scoreboard in bottom left
    std::stringstream ss;
    ss << "P1: " << std::setw(2) << gameState.getScore(PlayerTurn::Player1) << " | "
       << "P2: " << std::setw(2) << gameState.getScore(PlayerTurn::Player2);
    
    cv::Point scorePos(margin, output.rows - margin);
    drawText(output, ss.str(), scorePos, theme.secondaryColor);
    
    // Show remaining balls
    std::string remaining;
    if (gameState.getPlayerGroup(PlayerTurn::Player1) == BallGroup::Solids) {
        remaining = "P1: Solids | P2: Stripes";
    } else if (gameState.getPlayerGroup(PlayerTurn::Player1) == BallGroup::Stripes) {
        remaining = "P1: Stripes | P2: Solids";
    }
    
    if (!remaining.empty()) {
        cv::Point remainingPos(margin, output.rows - margin - 30);
        drawText(output, remaining, remainingPos, theme.secondaryColor);
    }
}

void UiRenderer::renderTurnIndicator(cv::Mat& output, const GameState& gameState) {
    // Draw turn indicator in top right
    std::string turnText = "TURN";
    cv::Point turnPos(output.cols - margin - 100, margin + 15);
    drawText(output, turnText, turnPos, theme.secondaryColor);
    
    // Draw player indicator
    cv::Rect indicator(output.cols - margin - 50, margin, 30, 30);
    cv::Scalar playerColor = (gameState.getCurrentTurn() == PlayerTurn::Player1) ? 
                            theme.primaryColor : theme.accentColor;
    cv::circle(output, cv::Point(indicator.x + indicator.width/2, 
                                indicator.y + indicator.height/2),
               indicator.width/2, playerColor, -1, theme.lineType);
}

void UiRenderer::drawText(cv::Mat& img, const std::string& text, cv::Point pos, 
                         const cv::Scalar& color, double scale, int thickness) {
    // Draw text with outline for better visibility
    cv::putText(img, text, pos, cv::FONT_HERSHEY_SIMPLEX, scale, cv::Scalar(0,0,0), 
                thickness + 2, theme.lineType);
    cv::putText(img, text, pos, cv::FONT_HERSHEY_SIMPLEX, scale, color, 
                thickness, theme.lineType);
}

void UiRenderer::drawOverlayBox(cv::Mat& img, cv::Rect box, double alpha) {
    cv::Mat overlay = img.clone();
    cv::rectangle(overlay, box, theme.overlayBg, -1);
    cv::addWeighted(overlay, alpha, img, 1.0 - alpha, 0.0, img);
}