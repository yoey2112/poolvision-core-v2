#include "SessionPlayback.hpp"
#include "../ui/UITheme.hpp"
#include <algorithm>
#include <sstream>
#include <iomanip>

using namespace pv;

SessionPlayback::SessionPlayback(Database& database, std::shared_ptr<SessionVideoManager> videoManager)
    : database_(database)
    , videoManager_(videoManager)
    , sessionId_(-1)
    , state_(PlaybackState::Stopped)
    , speed_(PlaybackSpeed::Normal)
    , currentTime_(0.0)
    , sessionDuration_(0.0)
    , playbackRate_(1.0)
    , isDraggingTimeline_(false) {
}

bool SessionPlayback::loadSession(int sessionId) {
    // Load session info
    sessionInfo_ = database_.getSession(sessionId);
    if (sessionInfo_.id == 0) {
        return false;
    }
    
    sessionId_ = sessionId;
    
    // Load associated data
    loadFrames();
    loadShots();
    
    // Calculate session duration from shots or frames
    sessionDuration_ = 0.0;
    if (!shots_.empty()) {
        // Use shot timestamps to determine duration
        auto lastShot = std::max_element(shots_.begin(), shots_.end(),
            [](const ShotRecord& a, const ShotRecord& b) {
                return a.shotNumber < b.shotNumber;
            });
        // Estimate 30 seconds per shot for duration
        sessionDuration_ = static_cast<double>(lastShot->shotNumber * 30000);
    } else {
        // Default duration if no shots
        sessionDuration_ = sessionInfo_.durationSeconds * 1000.0;
    }
    
    // Reset playback state
    currentTime_ = 0.0;
    state_ = PlaybackState::Stopped;
    
    return true;
}

void SessionPlayback::play() {
    if (sessionId_ < 0) return;
    state_ = PlaybackState::Playing;
}

void SessionPlayback::pause() {
    if (state_ == PlaybackState::Playing) {
        state_ = PlaybackState::Paused;
    }
}

void SessionPlayback::stop() {
    state_ = PlaybackState::Stopped;
    currentTime_ = 0.0;
}

void SessionPlayback::seekTo(double timeMs) {
    if (sessionId_ < 0) return;
    currentTime_ = std::clamp(timeMs, 0.0, sessionDuration_);
}

void SessionPlayback::seekBy(double deltaMs) {
    seekTo(currentTime_ + deltaMs);
}

void SessionPlayback::setPlaybackSpeed(PlaybackSpeed speed) {
    speed_ = speed;
    playbackRate_ = getPlaybackRate();
}

cv::Mat SessionPlayback::update(double deltaTime) {
    if (sessionId_ < 0 || state_ != PlaybackState::Playing) {
        return cv::Mat();
    }
    
    // Update playback time
    currentTime_ += deltaTime * playbackRate_;
    
    // Check if we reached the end
    if (currentTime_ >= sessionDuration_) {
        currentTime_ = sessionDuration_;
        state_ = PlaybackState::Stopped;
    }
    
    // Get current frame
    const auto* frame = getCurrentFrame();
    if (frame) {
        // Create a copy and return it
        return frame->image.clone();
    }
    
    return cv::Mat();
}

void SessionPlayback::renderControls(cv::Mat& frame, const cv::Rect& rect) {
    controlsRect_ = rect;
    
    // Background
    UITheme::drawCard(frame, rect, UITheme::ComponentState::Normal, UITheme::Colors::DarkBg, 200);
    
    // Title
    std::string title = "Session Playback";
    if (sessionId_ >= 0) {
        title += " - Session " + std::to_string(sessionId_);
    }
    cv::putText(frame, title, cv::Point(rect.x + 20, rect.y + 30),
               UITheme::Fonts::FontFaceBold, UITheme::Fonts::BodySize,
               UITheme::Colors::TextPrimary, UITheme::Fonts::ButtonThickness);
    
    if (sessionId_ < 0) {
        cv::putText(frame, "No session loaded", cv::Point(rect.x + 20, rect.y + 60),
                   UITheme::Fonts::FontFace, UITheme::Fonts::BodySize,
                   UITheme::Colors::TextSecondary, UITheme::Fonts::BodyThickness);
        return;
    }
    
    // Timeline
    timelineRect_ = cv::Rect(rect.x + 20, rect.y + 50, rect.width - 40, 20);
    renderTimeline(frame);
    
    // Control buttons
    renderPlaybackButtons(frame);
    
    // Session info
    renderSessionInfo(frame);
}

void SessionPlayback::onMouse(int event, int x, int y, int flags) {
    mousePos_ = cv::Point(x, y);
    
    if (event == cv::EVENT_LBUTTONDOWN) {
        if (isInTimeline(mousePos_)) {
            isDraggingTimeline_ = true;
            double time = timelineXToTime(x);
            seekTo(time);
        }
        
        // Check control buttons (simplified - would need proper button rects)
        // For now, just handle basic play/pause in timeline area
        if (controlsRect_.contains(mousePos_)) {
            if (state_ == PlaybackState::Playing) {
                pause();
            } else {
                play();
            }
        }
    }
    
    if (event == cv::EVENT_LBUTTONUP) {
        isDraggingTimeline_ = false;
    }
    
    if (event == cv::EVENT_MOUSEMOVE && isDraggingTimeline_) {
        double time = timelineXToTime(x);
        seekTo(time);
    }
}

bool SessionPlayback::onKeyboard(int key) {
    if (sessionId_ < 0) return false;
    
    switch (key) {
        case ' ':  // Spacebar - play/pause
            if (state_ == PlaybackState::Playing) {
                pause();
            } else {
                play();
            }
            return true;
            
        case 's':  // S - stop
            stop();
            return true;
            
        case 'a':  // A - seek backward 10 seconds
            seekBy(-10000);
            return true;
            
        case 'd':  // D - seek forward 10 seconds
            seekBy(10000);
            return true;
            
        case '1':  // 1 - quarter speed
            setPlaybackSpeed(PlaybackSpeed::Quarter);
            return true;
            
        case '2':  // 2 - half speed
            setPlaybackSpeed(PlaybackSpeed::Half);
            return true;
            
        case '3':  // 3 - normal speed
            setPlaybackSpeed(PlaybackSpeed::Normal);
            return true;
            
        case '4':  // 4 - double speed
            setPlaybackSpeed(PlaybackSpeed::Double);
            return true;
            
        case '5':  // 5 - quadruple speed
            setPlaybackSpeed(PlaybackSpeed::Quadruple);
            return true;
    }
    
    return false;
}

double SessionPlayback::getPosition() const {
    if (sessionDuration_ <= 0.0) return 0.0;
    return currentTime_ / sessionDuration_;
}

void SessionPlayback::loadFrames() {
    frames_.clear();
    
    // Try to get frames from video manager if available and it's the current session
    if (videoManager_ && sessionId_ == videoManager_->getCurrentSessionId()) {
        auto videoFrames = videoManager_->getSessionFrames();
        
        for (const auto& videoFrame : videoFrames) {
            GameRecorder::FrameSnapshot snapshot;
            snapshot.timestamp = videoFrame.timestamp;
            snapshot.image = videoFrame.frame;
            // Note: Other game state fields are not stored in video manager
            frames_.push_back(snapshot);
        }
        
        std::cout << "Loaded " << frames_.size() << " frames from video manager" << std::endl;
    } else {
        // TODO: Load frames from saved video files or database
        std::cout << "No video frames available for session " << sessionId_ << std::endl;
    }
}

void SessionPlayback::loadShots() {
    shots_ = database_.getSessionShots(sessionId_);
}

double SessionPlayback::getPlaybackRate() const {
    switch (speed_) {
        case PlaybackSpeed::Quarter:   return 0.25;
        case PlaybackSpeed::Half:      return 0.5;
        case PlaybackSpeed::Normal:    return 1.0;
        case PlaybackSpeed::Double:    return 2.0;
        case PlaybackSpeed::Quadruple: return 4.0;
        default: return 1.0;
    }
}

const GameRecorder::FrameSnapshot* SessionPlayback::getCurrentFrame() const {
    if (frames_.empty()) return nullptr;
    
    // Find frame closest to current time
    auto it = std::lower_bound(frames_.begin(), frames_.end(), currentTime_,
        [](const GameRecorder::FrameSnapshot& frame, double time) {
            return frame.timestamp < time;
        });
    
    if (it != frames_.end()) {
        return &(*it);
    }
    
    // Return last frame if we're past the end
    return &frames_.back();
}

void SessionPlayback::renderTimeline(cv::Mat& frame) {
    // Timeline background
    cv::rectangle(frame, timelineRect_, UITheme::Colors::MediumBg, -1);
    cv::rectangle(frame, timelineRect_, UITheme::Colors::TextSecondary, 1);
    
    if (sessionDuration_ > 0.0) {
        // Progress indicator
        int progressX = timeToTimelineX(currentTime_);
        cv::Rect progressRect(timelineRect_.x, timelineRect_.y,
                             progressX - timelineRect_.x, timelineRect_.height);
        cv::rectangle(frame, progressRect, UITheme::Colors::NeonCyan, -1);
        
        // Current position marker
        cv::line(frame,
                cv::Point(progressX, timelineRect_.y),
                cv::Point(progressX, timelineRect_.y + timelineRect_.height),
                UITheme::Colors::NeonYellow, 2);
    }
    
    // Time labels
    std::ostringstream currentStream;
    currentStream << std::fixed << std::setprecision(1) << (currentTime_ / 1000.0) << "s";
    
    std::ostringstream durationStream;
    durationStream << std::fixed << std::setprecision(1) << (sessionDuration_ / 1000.0) << "s";
    
    cv::putText(frame, currentStream.str(),
               cv::Point(timelineRect_.x, timelineRect_.y + timelineRect_.height + 20),
               UITheme::Fonts::FontFace, UITheme::Fonts::SmallSize,
               UITheme::Colors::TextPrimary, 1);
    
    cv::putText(frame, durationStream.str(),
               cv::Point(timelineRect_.x + timelineRect_.width - 50,
                        timelineRect_.y + timelineRect_.height + 20),
               UITheme::Fonts::FontFace, UITheme::Fonts::SmallSize,
               UITheme::Colors::TextPrimary, 1);
}

void SessionPlayback::renderPlaybackButtons(cv::Mat& frame) {
    int y = timelineRect_.y + timelineRect_.height + 40;
    int buttonWidth = 80;
    int buttonHeight = 30;
    int spacing = 10;
    int x = controlsRect_.x + 20;
    
    // Play/Pause button
    std::string playText = (state_ == PlaybackState::Playing) ? "Pause" : "Play";
    cv::Rect playButton(x, y, buttonWidth, buttonHeight);
    UITheme::drawButton(frame, playText, playButton, UITheme::ComponentState::Normal);
    x += buttonWidth + spacing;
    
    // Stop button
    cv::Rect stopButton(x, y, buttonWidth, buttonHeight);
    UITheme::drawButton(frame, "Stop", stopButton, UITheme::ComponentState::Normal);
    x += buttonWidth + spacing;
    
    // Speed indicator
    std::string speedText = "Speed: " + std::to_string(static_cast<int>(playbackRate_ * 100)) + "%";
    cv::putText(frame, speedText, cv::Point(x, y + 20),
               UITheme::Fonts::FontFace, UITheme::Fonts::SmallSize,
               UITheme::Colors::TextSecondary, 1);
}

void SessionPlayback::renderSessionInfo(cv::Mat& frame) {
    int x = controlsRect_.x + controlsRect_.width - 300;
    int y = timelineRect_.y + 20;
    
    // Session details
    std::ostringstream info;
    info << "Game: " << sessionInfo_.gameType;
    cv::putText(frame, info.str(), cv::Point(x, y),
               UITheme::Fonts::FontFace, UITheme::Fonts::SmallSize,
               UITheme::Colors::TextSecondary, 1);
    
    y += 20;
    info.str("");
    info << "Players: " << sessionInfo_.player1Id << " vs " << sessionInfo_.player2Id;
    cv::putText(frame, info.str(), cv::Point(x, y),
               UITheme::Fonts::FontFace, UITheme::Fonts::SmallSize,
               UITheme::Colors::TextSecondary, 1);
    
    y += 20;
    info.str("");
    info << "Shots: " << shots_.size();
    cv::putText(frame, info.str(), cv::Point(x, y),
               UITheme::Fonts::FontFace, UITheme::Fonts::SmallSize,
               UITheme::Colors::TextSecondary, 1);
}

bool SessionPlayback::isInTimeline(const cv::Point& point) const {
    return timelineRect_.contains(point);
}

double SessionPlayback::timelineXToTime(int x) const {
    if (timelineRect_.width <= 0) return 0.0;
    
    double ratio = static_cast<double>(x - timelineRect_.x) / timelineRect_.width;
    return std::clamp(ratio * sessionDuration_, 0.0, sessionDuration_);
}

int SessionPlayback::timeToTimelineX(double time) const {
    if (sessionDuration_ <= 0.0) return timelineRect_.x;
    
    double ratio = time / sessionDuration_;
    return timelineRect_.x + static_cast<int>(ratio * timelineRect_.width);
}
