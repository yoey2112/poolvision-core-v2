#include "MatchUI.hpp"
#include <algorithm>
#include <sstream>
#include <iomanip>
#include <cmath>

namespace pv {

void MatchUI::UIConfig::initializeDefaultPanels() {
    // Birds-eye view panel (top-right)
    panels[0].rect = cv::Rect(1200, 100, 300, 250);
    
    // Game stats panel (bottom-right)  
    panels[1].rect = cv::Rect(1200, 360, 300, 200);
    
    // Shot clock panel (top-center)
    panels[2].rect = cv::Rect(500, 10, 200, 80);
    
    // Match info panel (top-left)
    panels[3].rect = cv::Rect(10, 100, 280, 150);
    
    // Player profiles panel (left)
    panels[4].rect = cv::Rect(10, 260, 280, 300);
    
    // Chat log panel (bottom-left)
    panels[5].rect = cv::Rect(10, 570, 280, 150);
    
    // Controls panel (bottom-center)
    panels[6].rect = cv::Rect(400, 650, 400, 80);
}

MatchUI::MatchUI(std::shared_ptr<MatchSystem> matchSystem, GameState& gameState, Tracker& tracker)
    : matchSystem_(matchSystem), gameState_(gameState), tracker_(tracker),
      activePanelType_(PanelType::BirdsEyeView), isDragging_(false), isResizing_(false) {
    
    lastActivity_ = std::chrono::steady_clock::now();
    
    // Initialize animation states
    for (int i = 0; i < 7; ++i) {
        animations_[i] = AnimationState();
    }
}

void MatchUI::render(cv::Mat& frame) {
    // Clear with dark background
    if (config_.fullscreen) {
        frame = cv::Scalar(26, 26, 26);  // DarkBg
    }
    
    // Render main camera view
    renderMainView(frame);
    
    // Render enabled panels
    for (int i = 0; i < 7; ++i) {
        PanelType panelType = static_cast<PanelType>(i);
        if (config_.panels[i].enabled) {
            renderPanel(frame, config_.panels[i].rect, panelType);
        }
    }
    
    // Render match overlays if match is active
    if (matchSystem_->isMatchActive()) {
        renderScoreboard(frame, cv::Rect(frame.cols/2 - 150, 10, 300, 60));
    }
}

void MatchUI::handleMouseDown(const cv::Point& pos) {
    lastMousePos_ = pos;
    lastActivity_ = std::chrono::steady_clock::now();
    
    // Check if clicking on a panel
    PanelType clickedPanel = findPanelAtPosition(pos);
    if (clickedPanel != PanelType(7)) {  // Valid panel found
        activePanelType_ = clickedPanel;
        
        // Check for resize handle
        if (isResizeHandle(pos, clickedPanel)) {
            startPanelResize(clickedPanel, pos);
        } else {
            startPanelDrag(clickedPanel, pos);
        }
    }
}

void MatchUI::handleMouseMove(const cv::Point& pos) {
    if (isDragging_) {
        updatePanelDrag(pos);
    } else if (isResizing_) {
        updatePanelResize(pos);
    }
    
    lastMousePos_ = pos;
}

void MatchUI::handleMouseUp(const cv::Point& pos) {
    if (isDragging_) {
        endPanelDrag();
    } else if (isResizing_) {
        endPanelResize();
    }
}

void MatchUI::handleKeyPress(int key) {
    switch (key) {
        case 'f': case 'F':
            config_.fullscreen = !config_.fullscreen;
            break;
        case 'h': case 'H':
            config_.showAllPanels = !config_.showAllPanels;
            for (int i = 0; i < 7; ++i) {
                config_.panels[i].enabled = config_.showAllPanels;
            }
            break;
        case '1': case '2': case '3': case '4': case '5': case '6': case '7':
            {
                int panelIndex = key - '1';
                togglePanel(static_cast<PanelType>(panelIndex));
            }
            break;
        case 'r': case 'R':
            resetPanelLayout();
            break;
    }
}

void MatchUI::update(double deltaTime) {
    updateAnimations(deltaTime);
    
    // Auto-hide panels if enabled and inactive
    if (config_.autoHide) {
        auto now = std::chrono::steady_clock::now();
        double inactiveTime = std::chrono::duration<double>(now - lastActivity_).count();
        
        if (inactiveTime > config_.autoHideDelay) {
            for (int i = 0; i < 7; ++i) {
                config_.panels[i].alpha = std::max(0.3f, config_.panels[i].alpha - 0.02f);
            }
        } else {
            for (int i = 0; i < 7; ++i) {
                config_.panels[i].alpha = std::min(0.9f, config_.panels[i].alpha + 0.02f);
            }
        }
    }
}

void MatchUI::renderMainView(cv::Mat& frame) {
    // The main camera view would be rendered here
    // This is typically the full frame with overlay elements
    
    if (matchSystem_->isMatchActive()) {
        // Render game-specific overlays
        // Ball highlighting, shot lines, etc. would go here
    }
}

void MatchUI::renderPanel(cv::Mat& frame, const cv::Rect& rect, PanelType type) {
    // Apply animation transform
    cv::Rect animatedRect = rect;
    int panelIndex = static_cast<int>(type);
    float alpha = config_.panels[panelIndex].alpha * animations_[panelIndex].fadeProgress;
    
    if (alpha < 0.1f) return;  // Don't render if too transparent
    
    // Render panel based on type
    switch (type) {
        case PanelType::BirdsEyeView:
            renderBirdsEyePanel(frame, animatedRect);
            break;
        case PanelType::GameStats:
            renderGameStatsPanel(frame, animatedRect);
            break;
        case PanelType::ShotClock:
            renderShotClockPanel(frame, animatedRect);
            break;
        case PanelType::MatchInfo:
            renderMatchInfoPanel(frame, animatedRect);
            break;
        case PanelType::PlayerProfiles:
            renderPlayerProfilesPanel(frame, animatedRect);
            break;
        case PanelType::ChatLog:
            renderChatLogPanel(frame, animatedRect);
            break;
        case PanelType::Controls:
            renderControlsPanel(frame, animatedRect);
            break;
    }
}

void MatchUI::renderBirdsEyePanel(cv::Mat& frame, const cv::Rect& panelRect) {
    renderPanelFrame(frame, panelRect, "Table Overview", activePanelType_ == PanelType::BirdsEyeView);
    
    cv::Rect contentRect = getPanelContentRect(panelRect);
    
    // Draw simplified top-down table view
    cv::Rect tableRect(contentRect.x + 20, contentRect.y + 20, 
                      contentRect.width - 40, contentRect.height - 40);
    
    // Table background
    cv::rectangle(frame, tableRect, UITheme::Colors::TableGreen, -1);
    cv::rectangle(frame, tableRect, UITheme::Colors::BorderColor, 2);
    
    // Draw pockets
    std::vector<cv::Point> pockets = {
        {tableRect.x, tableRect.y},                           // Top-left
        {tableRect.x + tableRect.width/2, tableRect.y},       // Top-center
        {tableRect.x + tableRect.width, tableRect.y},         // Top-right
        {tableRect.x, tableRect.y + tableRect.height},        // Bottom-left
        {tableRect.x + tableRect.width/2, tableRect.y + tableRect.height}, // Bottom-center
        {tableRect.x + tableRect.width, tableRect.y + tableRect.height}    // Bottom-right
    };
    
    for (const auto& pocket : pockets) {
        cv::circle(frame, pocket, 8, cv::Scalar(0, 0, 0), -1);
    }
    
    // Draw balls if available
    auto balls = tracker_.getBalls();
    for (const auto& ball : balls) {
        cv::Point ballPos(tableRect.x + (ball.x / 1920.0f) * tableRect.width,
                         tableRect.y + (ball.y / 1080.0f) * tableRect.height);
        
        cv::Scalar ballColor = (ball.label == 0) ? UITheme::Colors::TextPrimary : UITheme::Colors::NeonYellow;
        cv::circle(frame, ballPos, 4, ballColor, -1);
        
        if (ball.label > 0) {
            cv::putText(frame, std::to_string(ball.label), 
                       cv::Point(ballPos.x - 3, ballPos.y + 3),
                       cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(0, 0, 0), 1);
        }
    }
}

void MatchUI::renderGameStatsPanel(cv::Mat& frame, const cv::Rect& panelRect) {
    renderPanelFrame(frame, panelRect, "Live Statistics", activePanelType_ == PanelType::GameStats);
    
    cv::Rect contentRect = getPanelContentRect(panelRect);
    
    if (!matchSystem_->isMatchActive()) {
        cv::putText(frame, "No active match", 
                   cv::Point(contentRect.x + 10, contentRect.y + 30),
                   cv::FONT_HERSHEY_SIMPLEX, 0.6, UITheme::Colors::TextSecondary, 1);
        return;
    }
    
    const auto& match = matchSystem_->getCurrentMatch();
    const auto& player1Stats = matchSystem_->getLiveStats(match.player1.playerId);
    const auto& player2Stats = matchSystem_->getLiveStats(match.player2.playerId);
    
    int yPos = contentRect.y + 20;
    int lineHeight = 25;
    
    // Player 1 stats
    std::ostringstream p1Stream;
    p1Stream << match.player1.name << ": " << std::fixed << std::setprecision(1) 
             << (player1Stats.shotSuccessRate * 100) << "%";
    cv::putText(frame, p1Stream.str(), cv::Point(contentRect.x + 10, yPos),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, UITheme::Colors::TextPrimary, 1);
    
    yPos += lineHeight;
    
    // Player 2 stats  
    std::ostringstream p2Stream;
    p2Stream << match.player2.name << ": " << std::fixed << std::setprecision(1)
             << (player2Stats.shotSuccessRate * 100) << "%";
    cv::putText(frame, p2Stream.str(), cv::Point(contentRect.x + 10, yPos),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, UITheme::Colors::TextPrimary, 1);
    
    yPos += lineHeight * 2;
    
    // Shot count comparison
    cv::putText(frame, "Total Shots:", cv::Point(contentRect.x + 10, yPos),
                cv::FONT_HERSHEY_SIMPLEX, 0.4, UITheme::Colors::TextSecondary, 1);
    
    yPos += 20;
    std::string shotText = std::to_string(player1Stats.totalShots) + " vs " + 
                          std::to_string(player2Stats.totalShots);
    cv::putText(frame, shotText, cv::Point(contentRect.x + 10, yPos),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, UITheme::Colors::TextPrimary, 1);
}

void MatchUI::renderShotClockPanel(cv::Mat& frame, const cv::Rect& panelRect) {
    if (!matchSystem_->isMatchActive()) return;
    
    const auto& shotClock = matchSystem_->getShotClock();
    if (!shotClock.active) return;
    
    renderPanelFrame(frame, panelRect, "Shot Clock", false);
    
    cv::Rect contentRect = getPanelContentRect(panelRect);
    cv::Point center(contentRect.x + contentRect.width/2, contentRect.y + contentRect.height/2);
    
    // Draw circular shot clock
    double progress = shotClock.timeRemaining / shotClock.timeLimit;
    cv::Scalar clockColor = shotClock.warning ? UITheme::Colors::NeonRed : UITheme::Colors::NeonCyan;
    
    renderProgressRing(frame, center, 25, progress, clockColor);
    
    // Draw time remaining
    std::ostringstream timeStream;
    timeStream << std::fixed << std::setprecision(1) << shotClock.timeRemaining;
    cv::Size textSize = cv::getTextSize(timeStream.str(), cv::FONT_HERSHEY_SIMPLEX, 0.7, 2);
    
    cv::Point textPos(center.x - textSize.width/2, center.y + textSize.height/2);
    cv::putText(frame, timeStream.str(), textPos, cv::FONT_HERSHEY_SIMPLEX, 0.7, 
                UITheme::Colors::TextPrimary, 2);
}

void MatchUI::renderMatchInfoPanel(cv::Mat& frame, const cv::Rect& panelRect) {
    renderPanelFrame(frame, panelRect, "Match Info", activePanelType_ == PanelType::MatchInfo);
    
    cv::Rect contentRect = getPanelContentRect(panelRect);
    
    if (!matchSystem_->isMatchActive()) {
        cv::putText(frame, "No active match", 
                   cv::Point(contentRect.x + 10, contentRect.y + 30),
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, UITheme::Colors::TextSecondary, 1);
        return;
    }
    
    const auto& match = matchSystem_->getCurrentMatch();
    int yPos = contentRect.y + 20;
    int lineHeight = 20;
    
    // Match format
    std::string formatText = MatchSystem::formatToString(match.config.format);
    if (match.config.format == MatchSystem::MatchFormat::RaceToN) {
        formatText += " " + std::to_string(match.config.targetGames);
    }
    cv::putText(frame, formatText, cv::Point(contentRect.x + 10, yPos),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, UITheme::Colors::TextPrimary, 1);
    
    yPos += lineHeight;
    
    // Game type
    std::string gameTypeText = MatchSystem::gameTypeToString(match.config.gameType);
    cv::putText(frame, gameTypeText, cv::Point(contentRect.x + 10, yPos),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, UITheme::Colors::TextSecondary, 1);
    
    yPos += lineHeight * 2;
    
    // Current score
    std::string scoreText = std::to_string(match.player1Wins) + " - " + 
                           std::to_string(match.player2Wins);
    cv::putText(frame, scoreText, cv::Point(contentRect.x + 10, yPos),
                cv::FONT_HERSHEY_SIMPLEX, 0.8, UITheme::Colors::NeonCyan, 2);
    
    yPos += lineHeight * 2;
    
    // Game number
    std::string gameText = "Game " + std::to_string(match.currentGame);
    cv::putText(frame, gameText, cv::Point(contentRect.x + 10, yPos),
                cv::FONT_HERSHEY_SIMPLEX, 0.4, UITheme::Colors::TextSecondary, 1);
}

void MatchUI::renderPlayerProfilesPanel(cv::Mat& frame, const cv::Rect& panelRect) {
    renderPanelFrame(frame, panelRect, "Players", activePanelType_ == PanelType::PlayerProfiles);
    
    cv::Rect contentRect = getPanelContentRect(panelRect);
    
    if (!matchSystem_->isMatchActive()) return;
    
    const auto& match = matchSystem_->getCurrentMatch();
    const auto& player1Stats = matchSystem_->getLiveStats(match.player1.playerId);
    const auto& player2Stats = matchSystem_->getLiveStats(match.player2.playerId);
    
    // Split content area for two players
    int playerHeight = contentRect.height / 2 - 10;
    cv::Rect player1Rect(contentRect.x, contentRect.y, contentRect.width, playerHeight);
    cv::Rect player2Rect(contentRect.x, contentRect.y + playerHeight + 10, 
                        contentRect.width, playerHeight);
    
    bool player1Active = (gameState_.getCurrentPlayer() == PlayerTurn::Player1);
    bool player2Active = (gameState_.getCurrentPlayer() == PlayerTurn::Player2);
    
    renderPlayerCard(frame, player1Rect, match.player1, player1Stats, player1Active);
    renderPlayerCard(frame, player2Rect, match.player2, player2Stats, player2Active);
}

void MatchUI::renderChatLogPanel(cv::Mat& frame, const cv::Rect& panelRect) {
    renderPanelFrame(frame, panelRect, "Commentary", activePanelType_ == PanelType::ChatLog);
    
    cv::Rect contentRect = getPanelContentRect(panelRect);
    
    // This would display match commentary, notes, or chat messages
    cv::putText(frame, "Match started", cv::Point(contentRect.x + 10, contentRect.y + 20),
                cv::FONT_HERSHEY_SIMPLEX, 0.4, UITheme::Colors::TextSecondary, 1);
}

void MatchUI::renderControlsPanel(cv::Mat& frame, const cv::Rect& panelRect) {
    renderPanelFrame(frame, panelRect, "Controls", activePanelType_ == PanelType::Controls);
    
    cv::Rect contentRect = getPanelContentRect(panelRect);
    
    // Draw control buttons
    int buttonWidth = 80;
    int buttonHeight = 35;
    int spacing = 10;
    int startX = contentRect.x + 10;
    int yPos = contentRect.y + 10;
    
    std::vector<std::string> buttons = {"Pause", "Reset", "Save", "Settings"};
    
    for (size_t i = 0; i < buttons.size(); ++i) {
        cv::Rect buttonRect(startX + i * (buttonWidth + spacing), yPos, buttonWidth, buttonHeight);
        UITheme::drawButton(frame, buttons[i], buttonRect);
    }
}

void MatchUI::renderPanelFrame(cv::Mat& frame, const cv::Rect& rect, const std::string& title, bool isActive) {
    cv::Scalar bgColor = config_.theme.panelColor;
    cv::Scalar borderColor = isActive ? config_.theme.accentColor : config_.theme.borderColor;
    
    // Apply glass effect
    renderGlassEffect(frame, rect, config_.theme.glassOpacity);
    
    // Draw border
    cv::rectangle(frame, rect, borderColor, 2);
    
    // Draw title bar
    cv::Rect titleRect(rect.x, rect.y, rect.width, PANEL_HEADER_HEIGHT);
    cv::rectangle(frame, titleRect, borderColor, -1);
    
    // Draw title text
    cv::Size textSize = cv::getTextSize(title, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1);
    cv::Point titlePos(titleRect.x + 8, titleRect.y + (titleRect.height + textSize.height) / 2);
    cv::putText(frame, title, titlePos, cv::FONT_HERSHEY_SIMPLEX, 0.5, 
                UITheme::Colors::TextPrimary, 1);
    
    // Draw close button
    cv::Point closePos(titleRect.x + titleRect.width - 15, titlePos.y);
    cv::putText(frame, "×", closePos, cv::FONT_HERSHEY_SIMPLEX, 0.6, 
                UITheme::Colors::TextSecondary, 1);
}

cv::Rect MatchUI::getPanelContentRect(const cv::Rect& panelRect) const {
    return cv::Rect(panelRect.x + 5, panelRect.y + PANEL_HEADER_HEIGHT + 5,
                   panelRect.width - 10, panelRect.height - PANEL_HEADER_HEIGHT - 10);
}

void MatchUI::renderPlayerCard(cv::Mat& frame, const cv::Rect& rect, 
                              const MatchSystem::MatchPlayer& player,
                              const MatchSystem::LiveStats& stats, bool isActive) {
    cv::Scalar cardColor = isActive ? UITheme::Colors::ButtonActive : UITheme::Colors::LightBg;
    cv::rectangle(frame, rect, cardColor, -1);
    cv::rectangle(frame, rect, UITheme::Colors::BorderColor, 1);
    
    int yPos = rect.y + 15;
    
    // Player name
    cv::putText(frame, player.name, cv::Point(rect.x + 10, yPos),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, UITheme::Colors::TextPrimary, 2);
    
    yPos += 25;
    
    // Success rate
    std::ostringstream rateStream;
    rateStream << "Success: " << std::fixed << std::setprecision(1) 
               << (stats.shotSuccessRate * 100) << "%";
    cv::putText(frame, rateStream.str(), cv::Point(rect.x + 10, yPos),
                cv::FONT_HERSHEY_SIMPLEX, 0.4, UITheme::Colors::TextSecondary, 1);
    
    yPos += 20;
    
    // Shot count
    std::string shotText = "Shots: " + std::to_string(stats.totalShots);
    cv::putText(frame, shotText, cv::Point(rect.x + 10, yPos),
                cv::FONT_HERSHEY_SIMPLEX, 0.4, UITheme::Colors::TextSecondary, 1);
}

void MatchUI::renderProgressRing(cv::Mat& frame, const cv::Point& center, int radius, 
                                double progress, cv::Scalar color) {
    // Draw background circle
    cv::circle(frame, center, radius, UITheme::Colors::LightBg, 3);
    
    // Draw progress arc
    if (progress > 0) {
        double startAngle = -90;  // Start at top
        double endAngle = startAngle + (360 * progress);
        
        // OpenCV ellipse uses degrees
        cv::ellipse(frame, center, cv::Size(radius, radius), 0, 
                   startAngle, endAngle, color, 3);
    }
}

void MatchUI::renderGlassEffect(cv::Mat& frame, const cv::Rect& rect, float opacity) {
    cv::Mat overlay = frame(rect).clone();
    cv::addWeighted(frame(rect), 1.0 - opacity, overlay, opacity, 0, frame(rect));
}

void MatchUI::renderScoreboard(cv::Mat& frame, const cv::Rect& rect) {
    if (!matchSystem_->isMatchActive()) return;
    
    const auto& match = matchSystem_->getCurrentMatch();
    
    renderGlassEffect(frame, rect, 0.8f);
    cv::rectangle(frame, rect, UITheme::Colors::BorderColor, 2);
    
    // Draw match score
    std::string scoreText = match.player1.name + " " + 
                           std::to_string(match.player1Wins) + " - " +
                           std::to_string(match.player2Wins) + " " +
                           match.player2.name;
    
    cv::Size textSize = cv::getTextSize(scoreText, cv::FONT_HERSHEY_SIMPLEX, 0.6, 2);
    cv::Point textPos(rect.x + (rect.width - textSize.width) / 2, 
                     rect.y + (rect.height + textSize.height) / 2);
    
    UITheme::drawTextWithShadow(frame, scoreText, textPos, cv::FONT_HERSHEY_SIMPLEX, 0.6,
                                UITheme::Colors::TextPrimary, 2);
}

// Simplified implementations for remaining methods...
void MatchUI::setPanelEnabled(PanelType panel, bool enabled) {
    config_.panels[static_cast<int>(panel)].enabled = enabled;
}

void MatchUI::togglePanel(PanelType panel) {
    int index = static_cast<int>(panel);
    config_.panels[index].enabled = !config_.panels[index].enabled;
}

void MatchUI::resetPanelLayout() {
    config_.initializeDefaultPanels();
}

void MatchUI::updateAnimations(double deltaTime) {
    // Update panel animations
    for (int i = 0; i < 7; ++i) {
        if (animations_[i].isTransitioning) {
            animations_[i].fadeProgress += deltaTime * 2.0f; // 2 seconds for full transition
            if (animations_[i].fadeProgress >= 1.0f) {
                animations_[i].fadeProgress = 1.0f;
                animations_[i].isTransitioning = false;
            }
        }
        
        // Update pulse animation
        animations_[i].pulsePhase += deltaTime * 2.0f;
        if (animations_[i].pulsePhase > 2 * M_PI) {
            animations_[i].pulsePhase -= 2 * M_PI;
        }
    }
}

MatchUI::PanelType MatchUI::findPanelAtPosition(const cv::Point& pos) const {
    for (int i = 6; i >= 0; --i) { // Check from top to bottom (reverse order)
        if (config_.panels[i].enabled && UITheme::isPointInRect(pos, config_.panels[i].rect)) {
            return static_cast<PanelType>(i);
        }
    }
    return static_cast<PanelType>(7); // Invalid panel
}

bool MatchUI::isResizeHandle(const cv::Point& pos, PanelType panel) const {
    cv::Rect rect = config_.panels[static_cast<int>(panel)].rect;
    cv::Rect handleRect(rect.x + rect.width - RESIZE_HANDLE_SIZE, 
                       rect.y + rect.height - RESIZE_HANDLE_SIZE,
                       RESIZE_HANDLE_SIZE, RESIZE_HANDLE_SIZE);
    return UITheme::isPointInRect(pos, handleRect);
}

void MatchUI::startPanelDrag(PanelType panel, const cv::Point& pos) {
    isDragging_ = true;
    activePanelType_ = panel;
    cv::Rect panelRect = config_.panels[static_cast<int>(panel)].rect;
    dragOffset_ = cv::Point(pos.x - panelRect.x, pos.y - panelRect.y);
}

void MatchUI::updatePanelDrag(const cv::Point& pos) {
    if (!isDragging_) return;
    
    int panelIndex = static_cast<int>(activePanelType_);
    config_.panels[panelIndex].rect.x = pos.x - dragOffset_.x;
    config_.panels[panelIndex].rect.y = pos.y - dragOffset_.y;
    
    clampPanelToScreen(activePanelType_);
}

void MatchUI::endPanelDrag() {
    isDragging_ = false;
}

void MatchUI::startPanelResize(PanelType panel, const cv::Point& pos) {
    isResizing_ = true;
    activePanelType_ = panel;
}

void MatchUI::updatePanelResize(const cv::Point& pos) {
    if (!isResizing_) return;
    
    int panelIndex = static_cast<int>(activePanelType_);
    cv::Rect& rect = config_.panels[panelIndex].rect;
    
    rect.width = std::max(MIN_PANEL_WIDTH, pos.x - rect.x);
    rect.height = std::max(MIN_PANEL_HEIGHT, pos.y - rect.y);
}

void MatchUI::endPanelResize() {
    isResizing_ = false;
}

void MatchUI::clampPanelToScreen(PanelType panel) {
    // Ensure panels stay within screen bounds
    // Implementation would clamp panel positions to screen dimensions
}

std::string MatchUI::formatTime(double seconds) const {
    int mins = static_cast<int>(seconds) / 60;
    int secs = static_cast<int>(seconds) % 60;
    std::ostringstream stream;
    stream << mins << ":" << std::setfill('0') << std::setw(2) << secs;
    return stream.str();
}

} // namespace pv