#include "GameRecorder.hpp"
#include <chrono>
#include <sstream>
#include <iostream>

using namespace pv;

GameRecorder::GameRecorder(Database& database, std::shared_ptr<SessionVideoManager> videoManager)
    : database_(database), videoManager_(videoManager) {
}

int GameRecorder::startRecording(int player1Id, int player2Id, const std::string& gameType) {
    if (isRecording_) {
        stopRecording(-1, 0, 0); // Finalize previous session
    }
    
    // Get current timestamp
    auto now = std::chrono::system_clock::now();
    auto timestamp = std::chrono::duration_cast<std::chrono::seconds>(
        now.time_since_epoch()).count();
    
    // Convert timestamp to string
    std::ostringstream oss;
    oss << timestamp;
    std::string playedAt = oss.str();
    
    // Create game session in database
    GameSession session;
    session.player1Id = player1Id;
    session.player2Id = player2Id;
    session.gameType = gameType;
    session.startedAt = timestamp;
    session.winnerId = -1; // TBD
    session.player1Score = 0;
    session.player2Score = 0;
    session.durationSeconds = 0;
    
    currentSessionId_ = database_.createSession(session);
    currentGameType_ = gameType;
    isRecording_ = true;
    sessionStartTime_ = static_cast<double>(timestamp);
    frameBuffer_.clear();
    
    // Start video recording if video manager is available
    if (videoManager_) {
        SessionVideoManager::SessionMetadata videoMetadata;
        videoMetadata.sessionId = currentSessionId_;
        videoMetadata.gameType = gameType;
        videoMetadata.player1Id = player1Id;
        videoMetadata.player2Id = player2Id;
        videoMetadata.startTime = now;
        
        videoManager_->startSessionRecording(currentSessionId_, videoMetadata);
    }
    
    return currentSessionId_;
}

void GameRecorder::stopRecording(int winnerId, int player1Score, int player2Score) {
    if (!isRecording_) return;
    
    // Flush any remaining frames
    flushFrameBuffer();
    
    // Stop video recording if video manager is available
    if (videoManager_) {
        videoManager_->stopSessionRecording();
        
        // Prompt user to save or delete video
        auto choice = videoManager_->promptUserForVideoSave(videoManager_->getCurrentSessionMetadata());
        
        switch (choice) {
            case SessionVideoManager::UserChoice::Save:
                saveSessionVideo();
                break;
            case SessionVideoManager::UserChoice::Delete:
                deleteSessionVideo();
                break;
            case SessionVideoManager::UserChoice::Cancel:
                std::cout << "Session video kept temporarily (will be cleaned up after 24 hours)" << std::endl;
                break;
        }
    }
    
    // Calculate duration
    auto now = std::chrono::system_clock::now();
    auto timestamp = std::chrono::duration_cast<std::chrono::seconds>(
        now.time_since_epoch()).count();
    int duration = static_cast<int>(static_cast<double>(timestamp) - sessionStartTime_);
    
    // Update session with final stats
    // Note: Database class doesn't have updateGameSession yet, 
    // so we'll need to add that method or handle this differently
    // For now, we'll create a workaround by re-creating the session
    
    isRecording_ = false;
    currentSessionId_ = -1;
}

void GameRecorder::recordFrame(const FrameSnapshot& snapshot) {
    if (!isRecording_) return;
    
    // Record frame to video manager if available
    if (videoManager_) {
        videoManager_->recordFrame(snapshot.image, snapshot.timestamp);
    }
    
    // Add to buffer for database storage (if needed)
    frameBuffer_.push_back(snapshot);
    
    // Flush if buffer is full
    if (frameBuffer_.size() >= MAX_BUFFER_SIZE) {
        flushFrameBuffer();
    }
}

void GameRecorder::recordShot(int playerId, const std::string& shotType, bool success,
                             const cv::Point2f& cueBallPos, const cv::Point2f& targetBallPos,
                             float speed) {
    if (!isRecording_) return;
    
    // Record shot to database
    ShotRecord shot;
    shot.sessionId = currentSessionId_;
    shot.playerId = playerId;
    shot.shotNumber = 0; // Will be auto-incremented or managed
    shot.shotType = shotType;
    shot.successful = success;
    shot.ballX = cueBallPos.x;
    shot.ballY = cueBallPos.y;
    shot.targetX = targetBallPos.x;
    shot.targetY = targetBallPos.y;
    shot.shotSpeed = speed;
    
    database_.addShot(shot);
}

void GameRecorder::flushFrameBuffer() {
    if (frameBuffer_.empty()) return;
    
    // For now, we'll just clear the buffer since we're using SessionVideoManager
    // for frame storage. The frame buffer could be used for database storage
    // if detailed frame-by-frame analysis is needed in the future.
    
    frameBuffer_.clear();
}

std::vector<GameRecorder::FrameSnapshot> GameRecorder::getSessionFrames(int sessionId) {
    // If video manager is available and this is the current session, get frames from it
    if (videoManager_ && sessionId == currentSessionId_) {
        std::vector<FrameSnapshot> result;
        auto videoFrames = videoManager_->getSessionFrames();
        
        for (const auto& videoFrame : videoFrames) {
            FrameSnapshot snapshot;
            snapshot.timestamp = videoFrame.timestamp;
            snapshot.image = videoFrame.frame;
            // Note: Other fields (balls, tracks, events, etc.) are not stored in video manager
            // They would need to be reconstructed from database if needed
            result.push_back(snapshot);
        }
        
        return result;
    }
    
    // TODO: For other sessions, retrieve from database/file storage
    // For now, return empty vector
    return std::vector<FrameSnapshot>();
}

SessionVideoManager::UserChoice GameRecorder::promptUserForVideoSave() {
    if (!videoManager_) {
        return SessionVideoManager::UserChoice::Delete;
    }
    
    return videoManager_->promptUserForVideoSave(videoManager_->getCurrentSessionMetadata());
}

bool GameRecorder::saveSessionVideo(const std::string& filename) {
    if (!videoManager_) {
        return false;
    }
    
    return videoManager_->saveSessionVideo(filename);
}

bool GameRecorder::deleteSessionVideo() {
    if (!videoManager_) {
        return false;
    }
    
    return videoManager_->deleteSessionVideo();
}

GameSession GameRecorder::getSessionInfo(int sessionId) {
    return database_.getSession(sessionId);
}
