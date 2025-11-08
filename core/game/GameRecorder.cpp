#include "GameRecorder.hpp"
#include <chrono>
#include <sstream>

using namespace pv;

GameRecorder::GameRecorder(Database& database)
    : database_(database) {
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
    
    return currentSessionId_;
}

void GameRecorder::stopRecording(int winnerId, int player1Score, int player2Score) {
    if (!isRecording_) return;
    
    // Flush any remaining frames
    flushFrameBuffer();
    
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
    
    // Add to buffer
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
    
    // TODO: Store frame snapshots to database or file
    // For now, we'll just clear the buffer since we don't have
    // a frame_snapshots table in the database yet
    // This could be added in a future enhancement
    
    frameBuffer_.clear();
}

std::vector<GameRecorder::FrameSnapshot> GameRecorder::getSessionFrames(int sessionId) {
    // TODO: Retrieve frames from database/file storage
    // For now, return empty vector
    return std::vector<FrameSnapshot>();
}

GameSession GameRecorder::getSessionInfo(int sessionId) {
    return database_.getSession(sessionId);
}
