#pragma once
#include "../db/Database.hpp"
#include "SessionPlayback.hpp"
#include <opencv2/opencv.hpp>
#include <memory>
#include <vector>
#include <string>

namespace pv {

/**
 * @brief Training mode for practicing specific shots
 * 
 * Provides practice scenarios, shot comparison, instant replay, and skill assessment
 */
class TrainingMode {
public:
    /**
     * @brief Different training exercise types
     */
    enum class ExerciseType {
        CueBallControl,    // Practice cue ball positioning
        BankShots,         // Practice bank shot accuracy
        CombinationShots,  // Practice combination shots
        BreakShots,        // Practice break shot power and accuracy
        SafetyPlay,        // Practice defensive/safety shots
        CutShots,          // Practice cutting balls at various angles
        DrawShots,         // Practice draw (backspin) shots
        FollowShots,       // Practice follow (topspin) shots
        CustomDrill        // Custom practice scenario
    };

    /**
     * @brief Training session state
     */
    enum class SessionState {
        Setup,      // Setting up exercise parameters
        Ready,      // Ready to start shot
        Aiming,     // Player is aiming
        Shooting,   // Shot is being executed
        Reviewing,  // Reviewing shot result
        Comparing   // Comparing with reference shot
    };

    /**
     * @brief Shot evaluation metrics
     */
    struct ShotEvaluation {
        bool successful;              // Did the shot achieve its objective?
        float accuracyScore;          // 0.0 to 1.0 accuracy rating
        float speedScore;             // 0.0 to 1.0 speed rating  
        float positionScore;          // 0.0 to 1.0 positioning rating
        float overallScore;           // 0.0 to 1.0 overall rating
        std::string feedback;         // Textual feedback for improvement
        cv::Point2f targetPosition;   // Where the ball should have gone
        cv::Point2f actualPosition;   // Where the ball actually went
        cv::Point2f cueBallTarget;    // Where cue ball should have ended
        cv::Point2f cueBallActual;    // Where cue ball actually ended
    };

    /**
     * @brief Practice drill definition
     */
    struct TrainingDrill {
        ExerciseType type;
        std::string name;
        std::string description;
        std::string instructions;
        cv::Point2f cueBallStart;     // Starting cue ball position
        cv::Point2f targetBall;       // Target ball position
        cv::Point2f objectiveBall;    // Objective ball position (for cuts/combinations)
        cv::Point2f idealCueBallEnd;  // Ideal ending cue ball position
        float difficultyLevel;        // 1.0 (easy) to 5.0 (expert)
        int attempts;                 // Number of attempts in this session
        int successes;                // Number of successful attempts
        float bestScore;              // Best score achieved
        std::vector<ShotEvaluation> history; // Shot history for this drill
    };

public:
    /**
     * @brief Constructor
     * @param database Database reference for storing training data
     */
    explicit TrainingMode(Database& database);
    
    /**
     * @brief Start a training session
     * @param exerciseType Type of exercise to practice
     * @param playerId ID of the player training
     * @return true if session started successfully
     */
    bool startSession(ExerciseType exerciseType, int playerId);
    
    /**
     * @brief Start a custom drill
     * @param drill Custom drill configuration
     * @param playerId ID of the player training
     */
    bool startCustomDrill(const TrainingDrill& drill, int playerId);
    
    /**
     * @brief End the current training session
     */
    void endSession();
    
    /**
     * @brief Update training state
     * @param deltaTime Time elapsed since last update
     */
    void update(double deltaTime);
    
    /**
     * @brief Process a shot attempt
     * @param cueBallEnd Final cue ball position
     * @param targetBallEnd Final target ball position  
     * @param shotSpeed Speed of the shot
     * @param shotType Type of shot attempted
     */
    void processShotAttempt(const cv::Point2f& cueBallEnd,
                           const cv::Point2f& targetBallEnd,
                           float shotSpeed,
                           const std::string& shotType);
    
    /**
     * @brief Render training interface
     * @param frame Frame to draw on
     */
    void render(cv::Mat& frame);
    
    /**
     * @brief Render training overlay on table view
     * @param frame Table frame to draw overlay on
     */
    void renderTableOverlay(cv::Mat& frame);
    
    /**
     * @brief Handle mouse events
     * @param event OpenCV mouse event
     * @param x Mouse X coordinate
     * @param y Mouse Y coordinate
     * @param flags Mouse event flags
     */
    void onMouse(int event, int x, int y, int flags);
    
    /**
     * @brief Handle keyboard events
     * @param key Key code
     * @return true if key was handled
     */
    bool onKeyboard(int key);
    
    /**
     * @brief Get current session state
     */
    SessionState getState() const { return sessionState_; }
    
    /**
     * @brief Get current drill
     */
    const TrainingDrill& getCurrentDrill() const { return currentDrill_; }
    
    /**
     * @brief Get session statistics
     */
    struct SessionStats {
        int totalAttempts;
        int successfulAttempts;
        float averageScore;
        float bestScore;
        float averageAccuracy;
        float improvementRate;
    };
    SessionStats getSessionStats() const;
    
    /**
     * @brief Show instant replay of last shot
     */
    void showInstantReplay();
    
    /**
     * @brief Compare current shot with reference shot
     * @param referenceSessionId Session ID containing reference shot
     * @param referenceShotId Shot ID to compare against
     */
    void compareWithReference(int referenceSessionId, int referenceShotId);
    
    /**
     * @brief Get available training exercises
     */
    static std::vector<std::pair<ExerciseType, std::string>> getAvailableExercises();

private:
    Database& database_;
    
    // Session state
    SessionState sessionState_;
    ExerciseType currentExercise_;
    TrainingDrill currentDrill_;
    int playerId_;
    
    // Shot tracking
    ShotEvaluation lastEvaluation_;
    std::vector<cv::Point2f> shotTrajectory_;
    std::chrono::steady_clock::time_point shotStartTime_;
    bool isRecordingShot_;
    
    // Replay system
    SessionPlayback replaySystem_;
    bool showingReplay_;
    int replaySessionId_;
    
    // UI state
    cv::Point mousePos_;
    cv::Rect drillInfoRect_;
    cv::Rect statsRect_;
    cv::Rect controlsRect_;
    std::vector<cv::Rect> clickableAreas_;
    
    /**
     * @brief Create drill for specific exercise type
     */
    TrainingDrill createDrillForExercise(ExerciseType type);
    
    /**
     * @brief Evaluate shot performance
     */
    ShotEvaluation evaluateShot(const cv::Point2f& cueBallEnd,
                                const cv::Point2f& targetBallEnd,
                                float shotSpeed,
                                const std::string& shotType);
    
    /**
     * @brief Save shot attempt to database
     */
    void saveShotAttempt(const ShotEvaluation& eval);
    
    /**
     * @brief Load player's training history for this exercise
     */
    void loadTrainingHistory();
    
    /**
     * @brief Calculate improvement rate over recent attempts
     */
    float calculateImprovementRate() const;
    
    /**
     * @brief Render drill instructions
     */
    void renderDrillInfo(cv::Mat& frame);
    
    /**
     * @brief Render session statistics
     */
    void renderSessionStats(cv::Mat& frame);
    
    /**
     * @brief Render training controls
     */
    void renderControls(cv::Mat& frame);
    
    /**
     * @brief Render shot evaluation feedback
     */
    void renderShotEvaluation(cv::Mat& frame);
    
    /**
     * @brief Render ideal shot visualization on table
     */
    void renderIdealShot(cv::Mat& frame);
    
    /**
     * @brief Render target zones on table
     */
    void renderTargetZones(cv::Mat& frame);
    
    /**
     * @brief Generate practice feedback based on performance
     */
    std::string generateFeedback(const ShotEvaluation& eval) const;
    
    /**
     * @brief Calculate distance between two points
     */
    float calculateDistance(const cv::Point2f& p1, const cv::Point2f& p2) const;
    
    /**
     * @brief Map exercise type to string
     */
    static std::string exerciseTypeToString(ExerciseType type);
    
    /**
     * @brief Render exercise selection screen  
     */
    void renderExerciseSelection(cv::Mat& frame);
    
    /**
     * @brief Handle button click events
     */
    void handleButtonClick(int buttonIndex);
};

} // namespace pv