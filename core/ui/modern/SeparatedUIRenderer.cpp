#include "SeparatedUIRenderer.hpp"
#include <algorithm>
#include <iostream>
#include <sstream>
#include <iomanip>

#ifdef _WIN32
#include <Windows.h>
#elif defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif

namespace pv {
namespace modern {

// OverlayRenderer Implementation
SeparatedUIRenderer::OverlayRenderer::OverlayRenderer() {
    // Initialize ball colors for different ball types
    ballColors_ = {
        cv::Scalar(255, 255, 255),  // 0 - Cue ball (white)
        cv::Scalar(0, 255, 255),    // 1 - Yellow
        cv::Scalar(0, 0, 255),      // 2 - Blue  
        cv::Scalar(0, 0, 255),      // 3 - Red
        cv::Scalar(128, 0, 128),    // 4 - Purple
        cv::Scalar(255, 165, 0),    // 5 - Orange
        cv::Scalar(0, 128, 0),      // 6 - Green
        cv::Scalar(128, 0, 0),      // 7 - Maroon
        cv::Scalar(0, 0, 0),        // 8 - Black
        cv::Scalar(0, 255, 255),    // 9 - Yellow stripe
        cv::Scalar(0, 0, 255),      // 10 - Blue stripe
        cv::Scalar(0, 0, 255),      // 11 - Red stripe
        cv::Scalar(128, 0, 128),    // 12 - Purple stripe
        cv::Scalar(255, 165, 0),    // 13 - Orange stripe
        cv::Scalar(0, 128, 0),      // 14 - Green stripe
        cv::Scalar(128, 0, 0),      // 15 - Maroon stripe
    };
}

void SeparatedUIRenderer::OverlayRenderer::renderBallDetections(cv::Mat& output, const std::vector<Ball>& detections) {
    for (const auto& ball : detections) {
        cv::Scalar color = ballColors_[std::min(ball.id, static_cast<int>(ballColors_.size() - 1))];
        drawBall(output, ball, color);
        
        // Draw ball ID
        std::string ballText = std::to_string(ball.id);
        cv::Point textPos(ball.c.x - 10, ball.c.y - ball.r - 5);
        drawText(output, ballText, textPos, color, 0.6f);
    }
}

void SeparatedUIRenderer::OverlayRenderer::renderTrackingOverlays(cv::Mat& output, 
    const std::vector<pv::modern::ByteTrackMOT::Track>& tracks) {
    
    for (const auto& track : tracks) {
        drawTrack(output, track);
        
        // Draw track ID and confidence
        std::stringstream ss;
        ss << "T:" << track.trackId << " (" << std::fixed << std::setprecision(2) << track.confidence << ")";
        cv::Point textPos(track.bbox.x, track.bbox.y - 5);
        drawText(output, ss.str(), textPos, trackColor_, 0.5f);
    }
}

void SeparatedUIRenderer::OverlayRenderer::renderShotAnalysis(cv::Mat& output, 
    const pv::modern::ShotSegmentation::ShotEvent& shot) {
    
    if (shot.isActive) {
        // Draw shot in progress indicator
        std::string shotText = "SHOT IN PROGRESS";
        cv::Size textSize = cv::getTextSize(shotText, cv::FONT_HERSHEY_SIMPLEX, 0.8, 2, nullptr);
        cv::Point textPos((output.cols - textSize.width) / 2, 50);
        
        // Background rectangle for better visibility
        cv::Rect textBg(textPos.x - 10, textPos.y - textSize.height - 10, 
                       textSize.width + 20, textSize.height + 20);
        cv::rectangle(output, textBg, cv::Scalar(0, 0, 0), -1);
        
        drawText(output, shotText, textPos, cv::Scalar(0, 255, 255), 0.8f);
    }
}

void SeparatedUIRenderer::OverlayRenderer::renderGameHUD(cv::Mat& output, const GameState& state) {
    // Game state information in top-left corner
    std::vector<std::string> hudLines;
    
    // Current turn
    std::string turnText = "Turn: " + 
        (state.getCurrentTurn() == PlayerTurn::Player1 ? "Player 1" : "Player 2");
    hudLines.push_back(turnText);
    
    // Scores
    std::stringstream scoreText;
    scoreText << "Score: P1=" << state.getScore(PlayerTurn::Player1) 
              << " P2=" << state.getScore(PlayerTurn::Player2);
    hudLines.push_back(scoreText.str());
    
    // Game over status
    if (state.isGameOver()) {
        std::string winnerText = "Winner: " + 
            (state.getWinner() == PlayerTurn::Player1 ? "Player 1" : "Player 2");
        hudLines.push_back(winnerText);
    }
    
    // Render HUD background
    int lineHeight = 25;
    int padding = 10;
    int hudHeight = hudLines.size() * lineHeight + 2 * padding;
    int hudWidth = 250;
    
    cv::Rect hudBg(10, 10, hudWidth, hudHeight);
    cv::rectangle(output, hudBg, cv::Scalar(0, 0, 0, 128), -1);
    cv::rectangle(output, hudBg, cv::Scalar(255, 255, 255), 2);
    
    // Render HUD text
    for (size_t i = 0; i < hudLines.size(); ++i) {
        cv::Point textPos(20, 35 + i * lineHeight);
        drawText(output, hudLines[i], textPos, cv::Scalar(255, 255, 255), 0.6f);
    }
}

void SeparatedUIRenderer::OverlayRenderer::renderPerformanceMetrics(cv::Mat& output, 
    float inferenceTime, float trackingTime, float renderTime) {
    
    std::vector<std::string> perfLines;
    
    std::stringstream ss1;
    ss1 << "GPU: " << std::fixed << std::setprecision(1) << inferenceTime << "ms";
    perfLines.push_back(ss1.str());
    
    std::stringstream ss2;
    ss2 << "CPU: " << std::fixed << std::setprecision(1) << trackingTime << "ms";
    perfLines.push_back(ss2.str());
    
    std::stringstream ss3;
    ss3 << "UI: " << std::fixed << std::setprecision(1) << renderTime << "ms";
    perfLines.push_back(ss3.str());
    
    // Position in top-right corner
    int lineHeight = 20;
    int padding = 10;
    int perfHeight = perfLines.size() * lineHeight + 2 * padding;
    int perfWidth = 120;
    
    cv::Rect perfBg(output.cols - perfWidth - 10, 10, perfWidth, perfHeight);
    cv::rectangle(output, perfBg, cv::Scalar(0, 0, 0, 128), -1);
    cv::rectangle(output, perfBg, cv::Scalar(0, 255, 0), 1);
    
    for (size_t i = 0; i < perfLines.size(); ++i) {
        cv::Point textPos(output.cols - perfWidth, 30 + i * lineHeight);
        drawText(output, perfLines[i], textPos, cv::Scalar(0, 255, 0), 0.5f);
    }
}

#ifdef USE_OLLAMA
void SeparatedUIRenderer::OverlayRenderer::renderCoachingOverlay(cv::Mat& output, const std::string& advice) {
    if (advice.empty()) return;
    
    // Position coaching advice in bottom area
    int maxWidth = output.cols - 40;
    int lineHeight = 25;
    int padding = 15;
    
    // Word wrap the advice text
    std::vector<std::string> wrappedLines;
    std::istringstream iss(advice);
    std::string word;
    std::string currentLine;
    
    while (iss >> word) {
        std::string testLine = currentLine.empty() ? word : currentLine + " " + word;
        cv::Size textSize = cv::getTextSize(testLine, cv::FONT_HERSHEY_SIMPLEX, 0.6, 1, nullptr);
        
        if (textSize.width > maxWidth && !currentLine.empty()) {
            wrappedLines.push_back(currentLine);
            currentLine = word;
        } else {
            currentLine = testLine;
        }
    }
    if (!currentLine.empty()) {
        wrappedLines.push_back(currentLine);
    }
    
    // Limit to maximum 4 lines
    if (wrappedLines.size() > 4) {
        wrappedLines.resize(4);
        if (!wrappedLines.empty()) {
            wrappedLines.back() += "...";
        }
    }
    
    // Background for coaching advice
    int coachHeight = wrappedLines.size() * lineHeight + 2 * padding;
    cv::Rect coachBg(20, output.rows - coachHeight - 20, output.cols - 40, coachHeight);
    cv::rectangle(output, coachBg, cv::Scalar(0, 0, 128, 180), -1);
    cv::rectangle(output, coachBg, cv::Scalar(255, 255, 255), 2);
    
    // Coaching label
    drawText(output, "[AI Coach]", cv::Point(30, output.rows - coachHeight - 5), coachingColor_, 0.7f);
    
    // Coaching text
    for (size_t i = 0; i < wrappedLines.size(); ++i) {
        cv::Point textPos(30, output.rows - coachHeight + padding + i * lineHeight);
        drawText(output, wrappedLines[i], textPos, coachingColor_, 0.6f);
    }
}
#endif

void SeparatedUIRenderer::OverlayRenderer::drawBall(cv::Mat& output, const Ball& ball, const cv::Scalar& color) {
    // Draw ball circle
    cv::circle(output, ball.c, ball.r, color, 2);
    
    // Draw center point
    cv::circle(output, ball.c, 2, color, -1);
}

void SeparatedUIRenderer::OverlayRenderer::drawTrack(cv::Mat& output, const pv::modern::ByteTrackMOT::Track& track) {
    // Draw bounding box
    cv::rectangle(output, track.bbox, trackColor_, 2);
    
    // Draw velocity vector if significant
    if (track.velocity.x * track.velocity.x + track.velocity.y * track.velocity.y > 1.0f) {
        cv::Point center(track.bbox.x + track.bbox.width / 2, track.bbox.y + track.bbox.height / 2);
        cv::Point endpoint(center.x + track.velocity.x * 5, center.y + track.velocity.y * 5);
        cv::arrowedLine(output, center, endpoint, predictionColor_, 2);
    }
}

void SeparatedUIRenderer::OverlayRenderer::drawText(cv::Mat& output, const std::string& text, 
    cv::Point position, const cv::Scalar& color, float scale) {
    
    // Add text shadow for better readability
    cv::putText(output, text, cv::Point(position.x + 1, position.y + 1), 
               cv::FONT_HERSHEY_SIMPLEX, scale, cv::Scalar(0, 0, 0), 2);
    cv::putText(output, text, position, cv::FONT_HERSHEY_SIMPLEX, scale, color, 1);
}

// BirdsEyeRenderer Implementation
SeparatedUIRenderer::BirdsEyeRenderer::BirdsEyeRenderer() {
    tableBackground_ = cv::Mat::zeros(tableSize_, CV_8UC3);
    birdsEyeBuffer_ = cv::Mat::zeros(tableSize_, CV_8UC3);
    
    // Initialize pocket locations (standard pool table)
    pocketLocations_ = {
        cv::Point2f(50, 50),     // Top-left
        cv::Point2f(400, 30),    // Top-center
        cv::Point2f(750, 50),    // Top-right
        cv::Point2f(50, 350),    // Bottom-left
        cv::Point2f(400, 370),   // Bottom-center
        cv::Point2f(750, 350)    // Bottom-right
    };
}

void SeparatedUIRenderer::BirdsEyeRenderer::initializeTableView(const cv::Size& tableSize) {
    tableSize_ = tableSize;
    tableBackground_ = cv::Mat::zeros(tableSize_, CV_8UC3);
    birdsEyeBuffer_ = cv::Mat::zeros(tableSize_, CV_8UC3);
    
    // Draw table layout
    drawTable(tableBackground_);
    drawPockets(tableBackground_);
    
    initialized_ = true;
}

void SeparatedUIRenderer::BirdsEyeRenderer::renderTableLayout(cv::Mat& output) {
    if (!initialized_) return;
    
    tableBackground_.copyTo(output);
}

void SeparatedUIRenderer::BirdsEyeRenderer::renderBallPositions(cv::Mat& output, 
    const std::vector<pv::modern::ByteTrackMOT::Track>& tracks) {
    
    if (!initialized_) return;
    
    for (const auto& track : tracks) {
        // Transform camera coordinates to table coordinates
        cv::Point2f tablePos = transformToTableCoords(cv::Point2f(
            track.bbox.x + track.bbox.width / 2,
            track.bbox.y + track.bbox.height / 2
        ));
        
        // Draw ball on table
        cv::Scalar ballColor = (track.trackId == 0) ? cv::Scalar(255, 255, 255) : cv::Scalar(0, 255, 255);
        cv::circle(output, tablePos, 8, ballColor, -1);
        cv::circle(output, tablePos, 8, cv::Scalar(0, 0, 0), 1);
        
        // Draw track ID
        std::string trackText = std::to_string(track.trackId);
        cv::putText(output, trackText, cv::Point(tablePos.x - 5, tablePos.y + 3), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 0), 1);
    }
}

void SeparatedUIRenderer::BirdsEyeRenderer::renderShotPrediction(cv::Mat& output, 
    const pv::modern::ByteTrackMOT::Track& cueBall, const cv::Point2f& targetPoint) {
    
    if (!initialized_) return;
    
    cv::Point2f cuePos = transformToTableCoords(cv::Point2f(
        cueBall.bbox.x + cueBall.bbox.width / 2,
        cueBall.bbox.y + cueBall.bbox.height / 2
    ));
    cv::Point2f targetPos = transformToTableCoords(targetPoint);
    
    // Draw shot line
    cv::line(output, cuePos, targetPos, cv::Scalar(255, 255, 0), 2);
    
    // Draw target indicator
    cv::circle(output, targetPos, 12, cv::Scalar(255, 255, 0), 2);
}

void SeparatedUIRenderer::BirdsEyeRenderer::renderPocketProbabilities(cv::Mat& output, 
    const std::vector<float>& probabilities) {
    
    if (!initialized_ || probabilities.size() != pocketLocations_.size()) return;
    
    for (size_t i = 0; i < pocketLocations_.size(); ++i) {
        float prob = probabilities[i];
        if (prob > 0.1f) {  // Only show significant probabilities
            cv::Scalar color(0, prob * 255, (1.0f - prob) * 255);
            cv::circle(output, pocketLocations_[i], 15, color, -1);
            
            // Show probability percentage
            std::stringstream ss;
            ss << std::fixed << std::setprecision(0) << prob * 100 << "%";
            cv::putText(output, ss.str(), cv::Point(pocketLocations_[i].x - 15, pocketLocations_[i].y - 20),
                       cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255), 1);
        }
    }
}

cv::Point2f SeparatedUIRenderer::BirdsEyeRenderer::transformToTableCoords(const cv::Point2f& cameraPoint) {
    // Simple linear transformation - in real implementation, would use homography
    float scaleX = static_cast<float>(tableSize_.width) / 1920.0f;
    float scaleY = static_cast<float>(tableSize_.height) / 1080.0f;
    
    return cv::Point2f(cameraPoint.x * scaleX, cameraPoint.y * scaleY);
}

void SeparatedUIRenderer::BirdsEyeRenderer::drawPockets(cv::Mat& output) {
    for (const auto& pocket : pocketLocations_) {
        cv::circle(output, pocket, 20, cv::Scalar(0, 0, 0), -1);
        cv::circle(output, pocket, 20, cv::Scalar(128, 128, 128), 2);
    }
}

void SeparatedUIRenderer::BirdsEyeRenderer::drawTable(cv::Mat& output) {
    // Table surface
    cv::rectangle(output, cv::Rect(30, 30, output.cols - 60, output.rows - 60), 
                 cv::Scalar(0, 100, 0), -1);
    
    // Table rails
    cv::rectangle(output, cv::Rect(30, 30, output.cols - 60, output.rows - 60), 
                 cv::Scalar(139, 69, 19), 8);
    
    // Center line (for 9-ball)
    cv::line(output, cv::Point(output.cols / 2, 35), cv::Point(output.cols / 2, output.rows - 35),
            cv::Scalar(255, 255, 255), 1);
}

// SeparatedUIRenderer Main Implementation
SeparatedUIRenderer::SeparatedUIRenderer(const UIRenderConfig& config) 
    : config_(config) {
    
    overlayRenderer_ = std::make_unique<OverlayRenderer>();
    birdsEyeRenderer_ = std::make_unique<BirdsEyeRenderer>();
    
    birdsEyeRenderer_->initializeTableView(cv::Size(800, 400));
}

SeparatedUIRenderer::~SeparatedUIRenderer() {
    stopUIRendering();
}

void SeparatedUIRenderer::submitFrameData(const FrameData& frameData) {
    std::unique_lock<std::mutex> lock(frameQueueMutex_);
    
    // Drop old frames if queue is full
    while (frameQueue_.size() >= static_cast<size_t>(config_.maxFrameQueueSize)) {
        frameQueue_.pop();
        performanceMetrics_.droppedFrames++;
    }
    
    frameQueue_.push(frameData);
    frameCondition_.notify_one();
}

cv::Mat SeparatedUIRenderer::getCurrentCompositeFrame() {
    std::lock_guard<std::mutex> lock(outputFramesMutex_);
    return latestCompositeFrame_.clone();
}

cv::Mat SeparatedUIRenderer::getCurrentBirdsEyeView() {
    std::lock_guard<std::mutex> lock(outputFramesMutex_);
    return latestBirdsEyeFrame_.clone();
}

cv::Mat SeparatedUIRenderer::getCurrentOverlayFrame() {
    std::lock_guard<std::mutex> lock(outputFramesMutex_);
    return latestOverlayFrame_.clone();
}

bool SeparatedUIRenderer::startUIRendering() {
    if (renderingActive_.load()) return true;
    
    renderingActive_ = true;
    uiRenderThread_ = std::thread(&SeparatedUIRenderer::renderingLoop, this);
    
    return true;
}

void SeparatedUIRenderer::stopUIRendering() {
    if (!renderingActive_.load()) return;
    
    renderingActive_ = false;
    frameCondition_.notify_all();
    
    if (uiRenderThread_.joinable()) {
        uiRenderThread_.join();
    }
    
    // Clear queues
    std::lock_guard<std::mutex> lock(frameQueueMutex_);
    while (!frameQueue_.empty()) {
        frameQueue_.pop();
    }
}

void SeparatedUIRenderer::setRenderConfig(const UIRenderConfig& newConfig) {
    config_ = newConfig;
}

void SeparatedUIRenderer::enableOverlay(const std::string& overlayType, bool enable) {
    if (overlayType == "detections") config_.enableOverlays = enable;
    else if (overlayType == "performance") config_.enablePerformanceHUD = enable;
    else if (overlayType == "birdsEye") config_.enableBirdsEyeView = enable;
#ifdef USE_OLLAMA
    else if (overlayType == "coaching") config_.enableCoachingOverlay = enable;
#endif
}

void SeparatedUIRenderer::setTableGeometry(const cv::Size& tableSize) {
    birdsEyeRenderer_->initializeTableView(tableSize);
}

SeparatedUIRenderer::UIPerformanceMetrics SeparatedUIRenderer::getPerformanceMetrics() {
    std::lock_guard<std::mutex> lock(metricsMutex_);
    return performanceMetrics_;
}

float SeparatedUIRenderer::getRenderingFPS() {
    std::lock_guard<std::mutex> lock(metricsMutex_);
    return performanceMetrics_.currentFPS;
}

int SeparatedUIRenderer::getQueuedFrames() {
    std::lock_guard<std::mutex> lock(frameQueueMutex_);
    return static_cast<int>(frameQueue_.size());
}

int SeparatedUIRenderer::getDroppedFrames() {
    std::lock_guard<std::mutex> lock(metricsMutex_);
    return performanceMetrics_.droppedFrames;
}

cv::Mat SeparatedUIRenderer::getCompositeFrame() {
    return getCurrentCompositeFrame();
}

cv::Mat SeparatedUIRenderer::getBirdsEyeFrame() {
    return getCurrentBirdsEyeView();
}

cv::Mat SeparatedUIRenderer::getSideBySideFrame() {
    cv::Mat composite = getCurrentCompositeFrame();
    cv::Mat birdsEye = getCurrentBirdsEyeView();
    
    if (composite.empty() || birdsEye.empty()) {
        return composite;
    }
    
    // Create side-by-side layout
    int totalWidth = composite.cols + birdsEye.cols;
    int totalHeight = std::max(composite.rows, birdsEye.rows);
    
    cv::Mat sideBySide(totalHeight, totalWidth, CV_8UC3, cv::Scalar(0, 0, 0));
    
    // Copy composite to left side
    cv::Rect leftROI(0, 0, composite.cols, composite.rows);
    composite.copyTo(sideBySide(leftROI));
    
    // Copy birds-eye to right side
    cv::Rect rightROI(composite.cols, 0, birdsEye.cols, birdsEye.rows);
    birdsEye.copyTo(sideBySide(rightROI));
    
    return sideBySide;
}

#ifdef USE_OLLAMA
void SeparatedUIRenderer::updateCoachingAdvice(const std::string& advice) {
    // This would be called from the coaching system to update the display
    // For now, we'll handle this through the FrameData structure
}
#endif

void SeparatedUIRenderer::renderingLoop() {
    setCpuAffinity();
    
    auto targetFrameTime = std::chrono::microseconds(1000000 / config_.targetFps);
    auto lastFrameTime = std::chrono::steady_clock::now();
    
    while (renderingActive_.load()) {
        FrameData frameData;
        bool hasFrame = false;
        
        // Wait for frame data with timeout
        {
            std::unique_lock<std::mutex> lock(frameQueueMutex_);
            if (frameCondition_.wait_for(lock, std::chrono::milliseconds(50), 
                [this] { return !frameQueue_.empty() || !renderingActive_.load(); })) {
                
                if (!frameQueue_.empty()) {
                    frameData = frameQueue_.front();
                    frameQueue_.pop();
                    hasFrame = true;
                }
            }
        }
        
        if (hasFrame && frameData.hasValidData) {
            auto renderStart = std::chrono::high_resolution_clock::now();
            
            processFrameData(frameData);
            
            auto renderEnd = std::chrono::high_resolution_clock::now();
            float renderTime = std::chrono::duration<float, std::milli>(renderEnd - renderStart).count();
            updatePerformanceMetrics(renderTime);
        }
        
        // Frame rate limiting
        auto currentTime = std::chrono::steady_clock::now();
        auto frameTime = currentTime - lastFrameTime;
        if (frameTime < targetFrameTime) {
            std::this_thread::sleep_for(targetFrameTime - frameTime);
        }
        lastFrameTime = std::chrono::steady_clock::now();
    }
}

void SeparatedUIRenderer::processFrameData(const FrameData& frameData) {
    if (frameData.originalFrame.empty()) return;
    
    // Render main view
    cv::Mat compositeView = frameData.originalFrame.clone();
    renderMainView(frameData, compositeView);
    
    // Render birds-eye view
    cv::Mat birdsEyeView;
    renderBirdsEyeView(frameData, birdsEyeView);
    
    // Render overlay-only view
    cv::Mat overlayView = cv::Mat::zeros(frameData.originalFrame.size(), CV_8UC3);
    renderMainView(frameData, overlayView);
    
    // Update output frames
    {
        std::lock_guard<std::mutex> lock(outputFramesMutex_);
        latestCompositeFrame_ = compositeView;
        latestBirdsEyeFrame_ = birdsEyeView;
        latestOverlayFrame_ = overlayView;
    }
}

void SeparatedUIRenderer::renderMainView(const FrameData& frameData, cv::Mat& output) {
    if (config_.enableOverlays) {
        // Render ball detections
        overlayRenderer_->renderBallDetections(output, frameData.detections);
        
        // Render tracking overlays
        overlayRenderer_->renderTrackingOverlays(output, frameData.tracks);
        
        // Render shot analysis
        overlayRenderer_->renderShotAnalysis(output, frameData.currentShot);
    }
    
    // Render game HUD
    overlayRenderer_->renderGameHUD(output, frameData.gameState);
    
    // Render performance metrics
    if (config_.enablePerformanceHUD) {
        std::lock_guard<std::mutex> lock(metricsMutex_);
        overlayRenderer_->renderPerformanceMetrics(output, 
            frameData.inferenceTime, frameData.trackingTime, performanceMetrics_.averageRenderTime);
    }
    
#ifdef USE_OLLAMA
    // Render coaching overlay
    if (config_.enableCoachingOverlay && frameData.hasNewCoaching) {
        overlayRenderer_->renderCoachingOverlay(output, frameData.latestCoachingAdvice);
    }
#endif
}

void SeparatedUIRenderer::renderBirdsEyeView(const FrameData& frameData, cv::Mat& output) {
    if (!config_.enableBirdsEyeView) {
        output = cv::Mat();
        return;
    }
    
    // Create birds-eye view
    birdsEyeRenderer_->renderTableLayout(output);
    birdsEyeRenderer_->renderBallPositions(output, frameData.tracks);
    
    // Add shot prediction if available
    for (const auto& track : frameData.tracks) {
        if (track.trackId == 0) {  // Cue ball
            // Simple prediction - in real implementation would use physics
            cv::Point2f targetPoint(400, 200);  // Mock target
            birdsEyeRenderer_->renderShotPrediction(output, track, targetPoint);
            break;
        }
    }
}

void SeparatedUIRenderer::updatePerformanceMetrics(float renderTime) {
    std::lock_guard<std::mutex> lock(metricsMutex_);
    
    performanceMetrics_.totalFramesRendered++;
    
    // Update rolling average render time
    float alpha = 0.1f;
    performanceMetrics_.averageRenderTime = alpha * renderTime + 
        (1.0f - alpha) * performanceMetrics_.averageRenderTime;
    
    // Calculate current FPS
    auto currentTime = std::chrono::steady_clock::now();
    auto timeDiff = currentTime - performanceMetrics_.lastFrameTime;
    if (timeDiff.count() > 0) {
        float fps = 1000.0f / std::chrono::duration<float, std::milli>(timeDiff).count();
        performanceMetrics_.currentFPS = alpha * fps + (1.0f - alpha) * performanceMetrics_.currentFPS;
    }
    performanceMetrics_.lastFrameTime = currentTime;
}

void SeparatedUIRenderer::setCpuAffinity() {
#ifdef _WIN32
    HANDLE thread = GetCurrentThread();
    DWORD_PTR affinity = 1ULL << config_.renderThreadCpuCore;
    SetThreadAffinityMask(thread, affinity);
#elif defined(__linux__)
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(config_.renderThreadCpuCore, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
#endif
}

// UIRendererFactory Implementation
std::unique_ptr<SeparatedUIRenderer> UIRendererFactory::createDefault() {
    return std::make_unique<SeparatedUIRenderer>(getDefaultConfig());
}

std::unique_ptr<SeparatedUIRenderer> UIRendererFactory::createHighPerformance() {
    return std::make_unique<SeparatedUIRenderer>(getHighPerformanceConfig());
}

std::unique_ptr<SeparatedUIRenderer> UIRendererFactory::createLowLatency() {
    return std::make_unique<SeparatedUIRenderer>(getLowLatencyConfig());
}

std::unique_ptr<SeparatedUIRenderer> UIRendererFactory::createStreamingOptimized() {
    return std::make_unique<SeparatedUIRenderer>(getStreamingConfig());
}

SeparatedUIRenderer::UIRenderConfig UIRendererFactory::getDefaultConfig() {
    SeparatedUIRenderer::UIRenderConfig config;
    config.targetFps = 60;
    config.enableVSync = true;
    config.renderThreadCpuCore = 3;
    config.enableBirdsEyeView = true;
    config.enableOverlays = true;
    config.overlayResolution = cv::Size(1920, 1080);
    config.enablePerformanceHUD = true;
    config.enableCoachingOverlay = true;
    config.maxFrameQueueSize = 3;
    return config;
}

SeparatedUIRenderer::UIRenderConfig UIRendererFactory::getHighPerformanceConfig() {
    auto config = getDefaultConfig();
    config.targetFps = 120;
    config.renderThreadCpuCore = 2;
    config.maxFrameQueueSize = 2;
    config.enablePerformanceHUD = false;
    return config;
}

SeparatedUIRenderer::UIRenderConfig UIRendererFactory::getLowLatencyConfig() {
    auto config = getDefaultConfig();
    config.targetFps = 30;
    config.enableVSync = false;
    config.maxFrameQueueSize = 1;
    return config;
}

SeparatedUIRenderer::UIRenderConfig UIRendererFactory::getStreamingConfig() {
    auto config = getDefaultConfig();
    config.targetFps = 60;
    config.overlayResolution = cv::Size(1920, 1080);
    config.enableBirdsEyeView = true;
    config.enableOverlays = true;
    config.enablePerformanceHUD = false;
    return config;
}

} // namespace modern
} // namespace pv