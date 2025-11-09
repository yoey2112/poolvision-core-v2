#include "SessionVideoManager.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <sstream>
#include <iomanip>

using namespace pv;
namespace fs = std::filesystem;

SessionVideoManager::SessionVideoManager(const std::string& tempDirectory)
    : isRecording_(false)
    , currentSessionId_(-1)
    , tempDirectory_(tempDirectory)
    , sessionStartTime_(0.0)
    , maxFrames_(3600)  // Approximately 2 minutes at 30 FPS
    , frameSkipCount_(0)
    , maxDurationMinutes_(60)
    , maxStorageMB_(500) {
    
    createTempDirectory();
    cleanupOldSessions();
}

SessionVideoManager::~SessionVideoManager() {
    if (isRecording_) {
        stopSessionRecording();
    }
    
    // Clean up temporary session if it wasn't saved
    if (currentSessionId_ >= 0 && !sessionFrames_.empty()) {
        deleteSessionVideo();
    }
}

bool SessionVideoManager::startSessionRecording(int sessionId, const SessionMetadata& metadata) {
    if (isRecording_) {
        stopSessionRecording();
    }
    
    currentSessionId_ = sessionId;
    currentMetadata_ = metadata;
    sessionFrames_.clear();
    frameSkipCount_ = 0;
    
    sessionStartTime_ = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now().time_since_epoch()
    ).count();
    
    isRecording_ = true;
    
    std::cout << "Started video recording for session " << sessionId << std::endl;
    return true;
}

bool SessionVideoManager::stopSessionRecording() {
    if (!isRecording_) {
        return false;
    }
    
    isRecording_ = false;
    currentMetadata_.endTime = std::chrono::system_clock::now();
    
    std::cout << "Stopped video recording for session " << currentSessionId_ 
              << " (" << sessionFrames_.size() << " frames recorded)" << std::endl;
    
    return true;
}

void SessionVideoManager::recordFrame(const cv::Mat& frame, double timestamp) {
    if (!isRecording_ || frame.empty()) {
        return;
    }
    
    // Check storage limits
    if (getTempStorageUsage() > maxStorageMB_ * 1024 * 1024) {
        std::cout << "Warning: Temporary storage limit reached, skipping frame recording" << std::endl;
        return;
    }
    
    // Frame rate limiting - record every 2nd frame to save space
    if (!shouldRecordFrame()) {
        frameSkipCount_++;
        return;
    }
    frameSkipCount_ = 0;
    
    // Limit total frames to prevent memory issues
    if (sessionFrames_.size() >= maxFrames_) {
        limitFrameCount();
    }
    
    // Add frame with relative timestamp
    double relativeTimestamp = timestamp - sessionStartTime_;
    sessionFrames_.emplace_back(frame, relativeTimestamp);
}

SessionVideoManager::UserChoice SessionVideoManager::promptUserForVideoSave(const SessionMetadata& metadata) {
    if (currentSessionId_ < 0 || sessionFrames_.empty()) {
        return UserChoice::Delete;
    }
    
    // For now, implement a simple console prompt
    // In a full GUI application, this would be a proper dialog
    std::cout << "\\n=== Session Video Save Prompt ===" << std::endl;
    std::cout << "Session ID: " << metadata.sessionId << std::endl;
    std::cout << "Game Type: " << metadata.gameType << std::endl;
    std::cout << "Players: " << metadata.player1Id << " vs " << metadata.player2Id << std::endl;
    std::cout << "Duration: " << std::fixed << std::setprecision(1) 
              << (getSessionDuration() / 1000.0) << " seconds" << std::endl;
    std::cout << "Frames recorded: " << sessionFrames_.size() << std::endl;
    
    std::cout << "\\nDo you want to save this session video?" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  s - Save video" << std::endl;
    std::cout << "  d - Delete video" << std::endl;
    std::cout << "  c - Cancel (keep temporarily)" << std::endl;
    std::cout << "Choice (s/d/c): ";
    
    char choice;
    std::cin >> choice;
    
    switch (std::tolower(choice)) {
        case 's':
            return UserChoice::Save;
        case 'd':
            return UserChoice::Delete;
        case 'c':
        default:
            return UserChoice::Cancel;
    }
}

bool SessionVideoManager::saveSessionVideo(const std::string& filename, const std::string& permanentDirectory) {
    if (currentSessionId_ < 0 || sessionFrames_.empty()) {
        return false;
    }
    
    // Create permanent directory if it doesn't exist
    fs::create_directories(permanentDirectory);
    
    // Generate filename with timestamp if not provided
    std::string actualFilename = filename;
    if (actualFilename.empty()) {
        auto now = std::chrono::system_clock::now();
        auto timeT = std::chrono::system_clock::to_time_t(now);
        std::ostringstream oss;
        oss << "session_" << currentSessionId_ << "_" 
            << std::put_time(std::localtime(&timeT), "%Y%m%d_%H%M%S");
        actualFilename = oss.str();
    }
    
    // Save as video file using OpenCV VideoWriter
    std::string videoPath = permanentDirectory + "/" + actualFilename + ".avi";
    
    if (sessionFrames_.empty()) {
        std::cerr << "No frames to save" << std::endl;
        return false;
    }
    
    // Get video properties from first frame
    cv::Size frameSize = sessionFrames_[0].frame.size();
    double fps = 30.0; // Target FPS for saved video
    
    cv::VideoWriter writer;
    int fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
    writer.open(videoPath, fourcc, fps, frameSize);
    
    if (!writer.isOpened()) {
        std::cerr << "Failed to open video writer for " << videoPath << std::endl;
        return false;
    }
    
    // Write frames to video
    for (const auto& videoFrame : sessionFrames_) {
        writer.write(videoFrame.frame);
    }
    
    writer.release();
    
    // Save session metadata as JSON
    std::string metadataPath = permanentDirectory + "/" + actualFilename + ".json";
    std::ofstream metaFile(metadataPath);
    if (metaFile.is_open()) {
        auto startTimeT = std::chrono::system_clock::to_time_t(currentMetadata_.startTime);
        auto endTimeT = std::chrono::system_clock::to_time_t(currentMetadata_.endTime);
        
        metaFile << "{" << std::endl;
        metaFile << "  \"sessionId\": " << currentSessionId_ << "," << std::endl;
        metaFile << "  \"gameType\": \"" << currentMetadata_.gameType << "\"," << std::endl;
        metaFile << "  \"player1Id\": " << currentMetadata_.player1Id << "," << std::endl;
        metaFile << "  \"player2Id\": " << currentMetadata_.player2Id << "," << std::endl;
        metaFile << "  \"startTime\": " << startTimeT << "," << std::endl;
        metaFile << "  \"endTime\": " << endTimeT << "," << std::endl;
        metaFile << "  \"durationMs\": " << getSessionDuration() << "," << std::endl;
        metaFile << "  \"frameCount\": " << sessionFrames_.size() << "," << std::endl;
        metaFile << "  \"videoFile\": \"" << actualFilename << ".avi\"" << std::endl;
        metaFile << "}" << std::endl;
        metaFile.close();
    }
    
    std::cout << "Session video saved to: " << videoPath << std::endl;
    
    // Clean up temporary data
    clearCurrentSession();
    
    return true;
}

bool SessionVideoManager::deleteSessionVideo() {
    if (currentSessionId_ < 0) {
        return false;
    }
    
    std::cout << "Deleting session video for session " << currentSessionId_ << std::endl;
    
    // Clear memory
    clearCurrentSession();
    
    return true;
}

std::vector<SessionVideoManager::VideoFrame> SessionVideoManager::getSessionFrames() const {
    return sessionFrames_;
}

cv::Mat SessionVideoManager::getFrameAtTimestamp(double timestamp) const {
    if (sessionFrames_.empty()) {
        return cv::Mat();
    }
    
    // Find frame closest to requested timestamp
    auto it = std::lower_bound(sessionFrames_.begin(), sessionFrames_.end(), timestamp,
        [](const VideoFrame& frame, double time) {
            return frame.timestamp < time;
        });
    
    if (it != sessionFrames_.end()) {
        return it->frame.clone();
    }
    
    // Return last frame if timestamp is beyond the end
    return sessionFrames_.back().frame.clone();
}

double SessionVideoManager::getSessionDuration() const {
    if (sessionFrames_.empty()) {
        return 0.0;
    }
    
    return sessionFrames_.back().timestamp - sessionFrames_.front().timestamp;
}

void SessionVideoManager::cleanupOldSessions(int maxAgeHours) {
    try {
        auto cutoffTime = std::chrono::system_clock::now() - std::chrono::hours(maxAgeHours);
        
        if (!fs::exists(tempDirectory_)) {
            return;
        }
        
        for (const auto& entry : fs::directory_iterator(tempDirectory_)) {
            if (entry.is_regular_file()) {
                auto fileTime = fs::last_write_time(entry);
                auto sctp = std::chrono::time_point_cast<std::chrono::system_clock::duration>(
                    fileTime - fs::file_time_type::clock::now() + std::chrono::system_clock::now()
                );
                
                if (sctp < cutoffTime) {
                    fs::remove(entry);
                    std::cout << "Cleaned up old temporary file: " << entry.path() << std::endl;
                }
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error cleaning up old sessions: " << e.what() << std::endl;
    }
}

size_t SessionVideoManager::getTempStorageUsage() const {
    size_t totalSize = 0;
    
    // Calculate memory usage of current session frames
    for (const auto& frame : sessionFrames_) {
        totalSize += frame.frame.total() * frame.frame.elemSize();
    }
    
    // Add disk usage if temp directory exists
    try {
        if (fs::exists(tempDirectory_)) {
            for (const auto& entry : fs::recursive_directory_iterator(tempDirectory_)) {
                if (entry.is_regular_file()) {
                    totalSize += entry.file_size();
                }
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error calculating storage usage: " << e.what() << std::endl;
    }
    
    return totalSize;
}

std::string SessionVideoManager::getSessionTempPath(int sessionId) const {
    return tempDirectory_ + "/session_" + std::to_string(sessionId);
}

std::string SessionVideoManager::getSessionFilename(int sessionId) const {
    return "session_" + std::to_string(sessionId) + ".tmp";
}

bool SessionVideoManager::createTempDirectory() {
    try {
        fs::create_directories(tempDirectory_);
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Failed to create temp directory: " << e.what() << std::endl;
        return false;
    }
}

void SessionVideoManager::clearCurrentSession() {
    sessionFrames_.clear();
    currentSessionId_ = -1;
    currentMetadata_ = SessionMetadata();
    sessionStartTime_ = 0.0;
}

double SessionVideoManager::getCurrentTimestamp() const {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now().time_since_epoch()
    ).count();
}

bool SessionVideoManager::shouldRecordFrame() const {
    // Record every 2nd frame to reduce storage
    return (frameSkipCount_ % 2) == 0;
}

void SessionVideoManager::limitFrameCount() {
    if (sessionFrames_.size() > maxFrames_) {
        // Remove oldest frames to stay within limit
        size_t framesToRemove = sessionFrames_.size() - maxFrames_ + 100; // Remove extra for buffer
        sessionFrames_.erase(sessionFrames_.begin(), sessionFrames_.begin() + framesToRemove);
        
        std::cout << "Removed " << framesToRemove << " old frames to stay within storage limits" << std::endl;
    }
}