#pragma once
#include "../db/Database.hpp"
#include "../db/PlayerProfile.hpp"
#include "../util/Types.hpp"
#include "../video/SessionVideoManager.hpp"
#include "GameState.hpp"
#include <memory>
#include <vector>
#include <string>

namespace pv {

/**
 * @brief Records game sessions for historical analysis and replay
 * 
 * Captures all game events, ball positions, and player actions
 * for later playback, analysis, and training purposes.
 */
class GameRecorder {
public:
    /**
     * @brief Frame snapshot containing complete game state
     */
    struct FrameSnapshot {
        double timestamp;
        std::vector<Ball> balls;
        std::vector<Track> tracks;
        std::vector<Event> events;
        PlayerTurn currentPlayer;
        int player1Score;
        int player2Score;
        bool isGameOver;
        cv::Mat image;  // Frame image for playback
    };
    
    /**
     * @brief Construct a new Game Recorder
     * 
     * @param database Database instance for persistence
     * @param videoManager Session video manager for frame storage (optional)
     */
    GameRecorder(Database& database, std::shared_ptr<SessionVideoManager> videoManager = nullptr);
    ~GameRecorder() = default;
    
    /**
     * @brief Start recording a new game session
     * 
     * @param player1Id First player ID (0 for guest)
     * @param player2Id Second player ID (0 for guest)
     * @param gameType Type of game being played
     * @return int Session ID for this recording
     */
    int startRecording(int player1Id, int player2Id, const std::string& gameType);
    
    /**
     * @brief Stop recording and finalize the session
     * 
     * @param winnerId ID of the winning player
     * @param player1Score Final score for player 1
     * @param player2Score Final score for player 2
     */
    void stopRecording(int winnerId, int player1Score, int player2Score);
    
    /**
     * @brief Record a frame snapshot during gameplay
     * 
     * @param snapshot Complete game state at this moment
     */
    void recordFrame(const FrameSnapshot& snapshot);
    
    /**
     * @brief Record a shot event
     * 
     * @param playerId Player who took the shot
     * @param shotType Type of shot (break, bank, etc.)
     * @param success Whether the shot was successful
     * @param cueBallPos Position of cue ball before shot
     * @param targetBallPos Position of target ball
     * @param speed Shot speed/power
     */
    void recordShot(int playerId, const std::string& shotType, bool success,
                   const cv::Point2f& cueBallPos, const cv::Point2f& targetBallPos,
                   float speed);
    
    /**
     * @brief Check if currently recording
     */
    bool isRecording() const { return isRecording_; }
    
    /**
     * @brief Get current session ID
     */
    int getSessionId() const { return currentSessionId_; }
    
    /**
     * @brief Get all frame snapshots for a session
     * 
     * @param sessionId Session to retrieve
     * @return std::vector<FrameSnapshot> All recorded frames
     */
    std::vector<FrameSnapshot> getSessionFrames(int sessionId);
    
    /**
     * @brief Prompt user to save or delete session video
     * 
     * @return SessionVideoManager::UserChoice User's choice
     */
    SessionVideoManager::UserChoice promptUserForVideoSave();
    
    /**
     * @brief Save current session video
     * 
     * @param filename Filename for saved video
     * @return true if saved successfully
     */
    bool saveSessionVideo(const std::string& filename = "");
    
    /**
     * @brief Delete current session video
     * 
     * @return true if deleted successfully
     */
    bool deleteSessionVideo();
    
    /**
     * @brief Get session metadata
     * 
     * @param sessionId Session to query
     * @return GameSession Session information
     */
    GameSession getSessionInfo(int sessionId);

private:
    Database& database_;
    std::shared_ptr<SessionVideoManager> videoManager_;
    bool isRecording_ = false;
    int currentSessionId_ = -1;
    double sessionStartTime_ = 0.0;
    std::string currentGameType_;
    
    // Frame storage for efficient batch saving
    std::vector<FrameSnapshot> frameBuffer_;
    const size_t MAX_BUFFER_SIZE = 1000;
    
    void flushFrameBuffer();
};

} // namespace pv
