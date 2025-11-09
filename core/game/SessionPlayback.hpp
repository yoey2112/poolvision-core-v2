#pragma once
#include "../db/Database.hpp"
#include "../video/SessionVideoManager.hpp"
#include "GameRecorder.hpp"
#include <opencv2/opencv.hpp>
#include <memory>
#include <vector>
#include <chrono>

namespace pv {

/**
 * @brief Controls playback of recorded game sessions
 * 
 * Provides timeline-based playback with controls for play/pause/speed/seek
 */
class SessionPlayback {
public:
    enum class PlaybackState {
        Stopped,
        Playing,
        Paused
    };

    /**
     * @brief Controls for playback speed
     */
    enum class PlaybackSpeed {
        Quarter = 0,  // 0.25x
        Half = 1,     // 0.5x
        Normal = 2,   // 1.0x
        Double = 3,   // 2.0x
        Quadruple = 4 // 4.0x
    };

public:
    /**
     * @brief Constructor
     * @param database Database reference for loading session data
     * @param videoManager Optional video manager for frame access
     */
    explicit SessionPlayback(Database& database, std::shared_ptr<SessionVideoManager> videoManager = nullptr);
    
    /**
     * @brief Load a game session for playback
     * @param sessionId Session ID to load
     * @return true if session loaded successfully
     */
    bool loadSession(int sessionId);
    
    /**
     * @brief Start playback from current position
     */
    void play();
    
    /**
     * @brief Pause playback
     */
    void pause();
    
    /**
     * @brief Stop playback and return to beginning
     */
    void stop();
    
    /**
     * @brief Seek to specific time in the session
     * @param timeMs Time in milliseconds from start
     */
    void seekTo(double timeMs);
    
    /**
     * @brief Seek forward/backward by specified amount
     * @param deltaMs Time delta in milliseconds (negative for backward)
     */
    void seekBy(double deltaMs);
    
    /**
     * @brief Set playback speed
     * @param speed Playback speed multiplier
     */
    void setPlaybackSpeed(PlaybackSpeed speed);
    
    /**
     * @brief Update playback and get current frame
     * @param deltaTime Time elapsed since last update
     * @return Current frame to display, or empty Mat if no frame
     */
    cv::Mat update(double deltaTime);
    
    /**
     * @brief Render playback controls overlay
     * @param frame Frame to draw controls on
     * @param rect Area for the controls
     */
    void renderControls(cv::Mat& frame, const cv::Rect& rect);
    
    /**
     * @brief Handle mouse events for playback controls
     * @param event OpenCV mouse event
     * @param x Mouse X coordinate
     * @param y Mouse Y coordinate
     * @param flags Mouse event flags
     */
    void onMouse(int event, int x, int y, int flags);
    
    /**
     * @brief Handle keyboard events for playback controls
     * @param key Key code
     * @return true if key was handled
     */
    bool onKeyboard(int key);
    
    /**
     * @brief Get current playback state
     */
    PlaybackState getState() const { return state_; }
    
    /**
     * @brief Get current playback position (0.0 to 1.0)
     */
    double getPosition() const;
    
    /**
     * @brief Get session duration in milliseconds
     */
    double getDuration() const { return sessionDuration_; }
    
    /**
     * @brief Get current time in milliseconds
     */
    double getCurrentTime() const { return currentTime_; }
    
    /**
     * @brief Get loaded session info
     */
    const GameSession& getSessionInfo() const { return sessionInfo_; }
    
    /**
     * @brief Check if a session is loaded
     */
    bool hasSession() const { return sessionId_ >= 0; }

private:
    Database& database_;
    std::shared_ptr<SessionVideoManager> videoManager_;
    
    // Session data
    int sessionId_;
    GameSession sessionInfo_;
    std::vector<GameRecorder::FrameSnapshot> frames_;
    std::vector<ShotRecord> shots_;
    
    // Playback state
    PlaybackState state_;
    PlaybackSpeed speed_;
    double currentTime_;        // Current playback time (ms)
    double sessionDuration_;    // Total session duration (ms)
    double playbackRate_;       // Actual rate multiplier
    
    // Timeline control
    cv::Rect timelineRect_;
    cv::Rect controlsRect_;
    cv::Point mousePos_;
    bool isDraggingTimeline_;
    
    /**
     * @brief Load frame snapshots for the session
     */
    void loadFrames();
    
    /**
     * @brief Load shot records for the session
     */
    void loadShots();
    
    /**
     * @brief Get playback rate from speed enum
     */
    double getPlaybackRate() const;
    
    /**
     * @brief Find frame at current time
     */
    const GameRecorder::FrameSnapshot* getCurrentFrame() const;
    
    /**
     * @brief Render timeline control
     */
    void renderTimeline(cv::Mat& frame);
    
    /**
     * @brief Render playback buttons
     */
    void renderPlaybackButtons(cv::Mat& frame);
    
    /**
     * @brief Render session info display
     */
    void renderSessionInfo(cv::Mat& frame);
    
    /**
     * @brief Check if point is in timeline
     */
    bool isInTimeline(const cv::Point& point) const;
    
    /**
     * @brief Convert timeline X coordinate to time
     */
    double timelineXToTime(int x) const;
    
    /**
     * @brief Convert time to timeline X coordinate
     */
    int timeToTimelineX(double time) const;
};

} // namespace pv