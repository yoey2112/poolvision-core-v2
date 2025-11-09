#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <memory>
#include <chrono>

namespace pv {

/**
 * @brief Manages temporary video storage for game sessions with user save/delete workflow
 * 
 * Provides session-based video recording with user prompts to save or delete
 * recordings after each session. Enables frame-by-frame analysis without
 * consuming permanent storage.
 */
class SessionVideoManager {
public:
    /**
     * @brief Video frame with timestamp for session playback
     */
    struct VideoFrame {
        cv::Mat frame;
        double timestamp;  // Time in milliseconds since session start
        
        VideoFrame() : timestamp(0.0) {}
        VideoFrame(const cv::Mat& f, double t) : frame(f.clone()), timestamp(t) {}
    };
    
    /**
     * @brief Session metadata for video storage
     */
    struct SessionMetadata {
        int sessionId;
        std::string gameType;
        int player1Id;
        int player2Id;
        std::chrono::system_clock::time_point startTime;
        std::chrono::system_clock::time_point endTime;
        std::string description;
        
        SessionMetadata() : sessionId(-1), player1Id(0), player2Id(0) {}
    };
    
    /**
     * @brief Result of user save/delete prompt
     */
    enum class UserChoice {
        Save,     // User wants to save the session video
        Delete,   // User wants to delete the session video
        Cancel    // User cancelled the prompt (keep temporarily)
    };

public:
    /**
     * @brief Constructor
     * @param tempDirectory Directory for temporary video storage
     */
    explicit SessionVideoManager(const std::string& tempDirectory = "temp_sessions");
    
    /**
     * @brief Destructor - cleans up any remaining temporary files
     */
    ~SessionVideoManager();
    
    /**
     * @brief Start recording a new session
     * @param sessionId Unique session identifier
     * @param metadata Session information
     * @return true if recording started successfully
     */
    bool startSessionRecording(int sessionId, const SessionMetadata& metadata);
    
    /**
     * @brief Stop current session recording
     * @return true if recording stopped successfully
     */
    bool stopSessionRecording();
    
    /**
     * @brief Add a frame to the current recording session
     * @param frame Frame to record
     * @param timestamp Timestamp in milliseconds since session start
     */
    void recordFrame(const cv::Mat& frame, double timestamp);
    
    /**
     * @brief Prompt user to save or delete the session video
     * @param metadata Session information for the prompt
     * @return User's choice (Save, Delete, or Cancel)
     */
    UserChoice promptUserForVideoSave(const SessionMetadata& metadata);
    
    /**
     * @brief Save the current session video to permanent storage
     * @param filename Filename for the saved video (without extension)
     * @param permanentDirectory Directory for permanent storage
     * @return true if save was successful
     */
    bool saveSessionVideo(const std::string& filename, const std::string& permanentDirectory = "saved_sessions");
    
    /**
     * @brief Delete the current session video from temporary storage
     * @return true if deletion was successful
     */
    bool deleteSessionVideo();
    
    /**
     * @brief Get all recorded frames for the current session
     * @return Vector of video frames with timestamps
     */
    std::vector<VideoFrame> getSessionFrames() const;
    
    /**
     * @brief Get frame at specific timestamp
     * @param timestamp Time in milliseconds since session start
     * @return Frame at requested time, or empty Mat if not found
     */
    cv::Mat getFrameAtTimestamp(double timestamp) const;
    
    /**
     * @brief Get session recording duration in milliseconds
     * @return Duration of current session, or 0.0 if no session
     */
    double getSessionDuration() const;
    
    /**
     * @brief Check if currently recording a session
     * @return true if recording is active
     */
    bool isRecording() const { return isRecording_; }
    
    /**
     * @brief Get current session ID
     * @return Session ID, or -1 if no active session
     */
    int getCurrentSessionId() const { return currentSessionId_; }
    
    /**
     * @brief Get current session metadata
     * @return Session metadata, or default if no active session
     */
    const SessionMetadata& getCurrentSessionMetadata() const { return currentMetadata_; }
    
    /**
     * @brief Clean up old temporary session files
     * @param maxAgeHours Maximum age in hours before cleanup (default 24)
     */
    void cleanupOldSessions(int maxAgeHours = 24);
    
    /**
     * @brief Get temporary storage usage in bytes
     * @return Total bytes used by temporary session storage
     */
    size_t getTempStorageUsage() const;
    
    /**
     * @brief Set maximum session recording time
     * @param maxDurationMinutes Maximum recording time in minutes (default 60)
     */
    void setMaxRecordingDuration(int maxDurationMinutes) { maxDurationMinutes_ = maxDurationMinutes; }

private:
    // Recording state
    bool isRecording_;
    int currentSessionId_;
    SessionMetadata currentMetadata_;
    std::string tempDirectory_;
    
    // Video recording
    std::vector<VideoFrame> sessionFrames_;
    double sessionStartTime_;
    int maxFrames_;
    int frameSkipCount_;  // Skip frames to manage storage
    
    // Configuration
    int maxDurationMinutes_;
    size_t maxStorageMB_;
    
    // Private methods
    std::string getSessionTempPath(int sessionId) const;
    std::string getSessionFilename(int sessionId) const;
    bool createTempDirectory();
    void clearCurrentSession();
    double getCurrentTimestamp() const;
    bool shouldRecordFrame() const;  // Frame rate limiting
    void limitFrameCount();  // Keep storage reasonable
};

} // namespace pv