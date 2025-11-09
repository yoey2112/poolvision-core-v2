#include "TrainingMode.hpp"
#include "../ui/UITheme.hpp"
#include <algorithm>
#include <cmath>
#include <sstream>
#include <iomanip>
#include <chrono>

using namespace pv;

TrainingMode::TrainingMode(Database& database)
    : database_(database)
    , replaySystem_(database)
    , sessionState_(SessionState::Setup)
    , currentExercise_(ExerciseType::CueBallControl)
    , playerId_(-1)
    , isRecordingShot_(false)
    , showingReplay_(false)
    , replaySessionId_(-1) {
}

bool TrainingMode::startSession(ExerciseType exerciseType, int playerId) {
    currentExercise_ = exerciseType;
    playerId_ = playerId;
    currentDrill_ = createDrillForExercise(exerciseType);
    sessionState_ = SessionState::Ready;
    
    loadTrainingHistory();
    
    return true;
}

bool TrainingMode::startCustomDrill(const TrainingDrill& drill, int playerId) {
    currentDrill_ = drill;
    playerId_ = playerId;
    currentExercise_ = ExerciseType::CustomDrill;
    sessionState_ = SessionState::Ready;
    
    return true;
}

void TrainingMode::endSession() {
    sessionState_ = SessionState::Setup;
    playerId_ = -1;
    isRecordingShot_ = false;
    showingReplay_ = false;
}

void TrainingMode::update(double deltaTime) {
    if (showingReplay_) {
        replaySystem_.update(deltaTime);
    }
    
    // Update session state based on time or events
    if (sessionState_ == SessionState::Shooting && isRecordingShot_) {
        auto currentTime = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            currentTime - shotStartTime_).count();
        
        // Automatically transition to review after 5 seconds of shooting
        if (elapsed > 5000) {
            sessionState_ = SessionState::Reviewing;
            isRecordingShot_ = false;
        }
    }
}

void TrainingMode::processShotAttempt(const cv::Point2f& cueBallEnd,
                                     const cv::Point2f& targetBallEnd,
                                     float shotSpeed,
                                     const std::string& shotType) {
    if (sessionState_ != SessionState::Shooting) return;
    
    // Evaluate the shot
    lastEvaluation_ = evaluateShot(cueBallEnd, targetBallEnd, shotSpeed, shotType);
    
    // Update drill statistics
    currentDrill_.attempts++;
    if (lastEvaluation_.successful) {
        currentDrill_.successes++;
    }
    
    if (lastEvaluation_.overallScore > currentDrill_.bestScore) {
        currentDrill_.bestScore = lastEvaluation_.overallScore;
    }
    
    currentDrill_.history.push_back(lastEvaluation_);
    
    // Save to database
    saveShotAttempt(lastEvaluation_);
    
    // Transition to review state
    sessionState_ = SessionState::Reviewing;
    isRecordingShot_ = false;
}

void TrainingMode::render(cv::Mat& frame) {
    // Background
    frame = cv::Mat(720, 1280, CV_8UC3, UITheme::Colors::DarkBg);
    
    // Title bar
    UITheme::drawTitleBar(frame, "Training Mode");
    
    switch (sessionState_) {
        case SessionState::Setup:
            // Show exercise selection
            renderExerciseSelection(frame);
            break;
            
        case SessionState::Ready:
        case SessionState::Aiming:
        case SessionState::Shooting:
            // Show drill info and stats during active training
            renderDrillInfo(frame);
            renderSessionStats(frame);
            break;
            
        case SessionState::Reviewing:
            // Show shot evaluation and feedback
            renderShotEvaluation(frame);
            break;
            
        case SessionState::Comparing:
            // Show comparison with reference shot (simplified for now)
            renderDrillInfo(frame);
            break;
    }
    
    // Always render controls at bottom
    controlsRect_ = cv::Rect(0, frame.rows - 100, frame.cols, 100);
    renderControls(frame);
}

void TrainingMode::renderTableOverlay(cv::Mat& frame) {
    if (sessionState_ == SessionState::Setup) return;
    
    // Render ideal shot visualization
    renderIdealShot(frame);
    
    // Render target zones
    renderTargetZones(frame);
    
    // Show trajectory if recording shot
    if (isRecordingShot_ && !shotTrajectory_.empty()) {
        for (size_t i = 1; i < shotTrajectory_.size(); ++i) {
            cv::line(frame, shotTrajectory_[i-1], shotTrajectory_[i],
                    UITheme::Colors::NeonYellow, 2, cv::LINE_AA);
        }
    }
    
    // Show last shot evaluation if reviewing
    if (sessionState_ == SessionState::Reviewing) {
        // Draw actual vs target positions
        cv::circle(frame, lastEvaluation_.targetPosition, 15,
                  UITheme::Colors::NeonGreen, 2, cv::LINE_AA);
        cv::circle(frame, lastEvaluation_.actualPosition, 15,
                  UITheme::Colors::NeonRed, 2, cv::LINE_AA);
        
        // Draw line showing difference
        cv::line(frame, lastEvaluation_.targetPosition, lastEvaluation_.actualPosition,
                UITheme::Colors::NeonOrange, 1, cv::LINE_AA);
    }
}

void TrainingMode::onMouse(int event, int x, int y, int flags) {
    mousePos_ = cv::Point(x, y);
    
    if (event == cv::EVENT_LBUTTONDOWN) {
        // Check clickable areas
        for (size_t i = 0; i < clickableAreas_.size(); ++i) {
            if (clickableAreas_[i].contains(mousePos_)) {
                handleButtonClick(i);
                break;
            }
        }
    }
}

bool TrainingMode::onKeyboard(int key) {
    switch (key) {
        case ' ':  // Spacebar - start/reset shot
            if (sessionState_ == SessionState::Ready) {
                sessionState_ = SessionState::Aiming;
                return true;
            }
            break;
            
        case 13: // Enter - begin shot execution
            if (sessionState_ == SessionState::Aiming) {
                sessionState_ = SessionState::Shooting;
                isRecordingShot_ = true;
                shotStartTime_ = std::chrono::steady_clock::now();
                shotTrajectory_.clear();
                return true;
            }
            break;
            
        case 'r': // R - show replay
            if (sessionState_ == SessionState::Reviewing) {
                showInstantReplay();
                return true;
            }
            break;
            
        case 'n': // N - next attempt
            if (sessionState_ == SessionState::Reviewing) {
                sessionState_ = SessionState::Ready;
                return true;
            }
            break;
            
        case 27: // ESC - back to setup
            sessionState_ = SessionState::Setup;
            return true;
    }
    
    return false;
}

TrainingMode::SessionStats TrainingMode::getSessionStats() const {
    SessionStats stats;
    stats.totalAttempts = currentDrill_.attempts;
    stats.successfulAttempts = currentDrill_.successes;
    stats.bestScore = currentDrill_.bestScore;
    
    if (currentDrill_.history.empty()) {
        stats.averageScore = 0.0f;
        stats.averageAccuracy = 0.0f;
        stats.improvementRate = 0.0f;
    } else {
        float totalScore = 0.0f;
        float totalAccuracy = 0.0f;
        
        for (const auto& eval : currentDrill_.history) {
            totalScore += eval.overallScore;
            totalAccuracy += eval.accuracyScore;
        }
        
        stats.averageScore = totalScore / currentDrill_.history.size();
        stats.averageAccuracy = totalAccuracy / currentDrill_.history.size();
        stats.improvementRate = calculateImprovementRate();
    }
    
    return stats;
}

void TrainingMode::showInstantReplay() {
    if (replaySessionId_ >= 0) {
        showingReplay_ = true;
        replaySystem_.loadSession(replaySessionId_);
        replaySystem_.play();
    }
}

void TrainingMode::compareWithReference(int referenceSessionId, int referenceShotId) {
    sessionState_ = SessionState::Comparing;
    // Load reference shot data for comparison
    // This would need additional database methods to load specific shot data
}

std::vector<std::pair<TrainingMode::ExerciseType, std::string>> TrainingMode::getAvailableExercises() {
    return {
        {ExerciseType::CueBallControl, "Cue Ball Control"},
        {ExerciseType::BankShots, "Bank Shots"},
        {ExerciseType::CombinationShots, "Combination Shots"},
        {ExerciseType::BreakShots, "Break Shots"},
        {ExerciseType::SafetyPlay, "Safety Play"},
        {ExerciseType::CutShots, "Cut Shots"},
        {ExerciseType::DrawShots, "Draw Shots"},
        {ExerciseType::FollowShots, "Follow Shots"},
        {ExerciseType::CustomDrill, "Custom Drill"}
    };
}

TrainingMode::TrainingDrill TrainingMode::createDrillForExercise(ExerciseType type) {
    TrainingDrill drill;
    drill.type = type;
    drill.attempts = 0;
    drill.successes = 0;
    drill.bestScore = 0.0f;
    
    switch (type) {
        case ExerciseType::CueBallControl:
            drill.name = "Cue Ball Position";
            drill.description = "Practice controlling cue ball position after contact";
            drill.instructions = "Hit the target ball and stop the cue ball in the marked zone";
            drill.cueBallStart = cv::Point2f(320, 360); // Center of table
            drill.targetBall = cv::Point2f(640, 360);   // Target ball
            drill.idealCueBallEnd = cv::Point2f(320, 200); // Where cue ball should end
            drill.difficultyLevel = 2.0f;
            break;
            
        case ExerciseType::BankShots:
            drill.name = "Bank Shot Accuracy";
            drill.description = "Practice bank shots using the rail";
            drill.instructions = "Use the rail to pocket the target ball";
            drill.cueBallStart = cv::Point2f(200, 500);
            drill.targetBall = cv::Point2f(800, 200);
            drill.objectiveBall = cv::Point2f(1100, 150); // Corner pocket
            drill.difficultyLevel = 3.5f;
            break;
            
        case ExerciseType::CutShots:
            drill.name = "Cut Shot Practice";
            drill.description = "Practice cutting balls at various angles";
            drill.instructions = "Cut the target ball into the corner pocket";
            drill.cueBallStart = cv::Point2f(400, 400);
            drill.targetBall = cv::Point2f(700, 300);
            drill.objectiveBall = cv::Point2f(1100, 150); // Corner pocket
            drill.difficultyLevel = 2.5f;
            break;
            
        // Add more drill types...
        default:
            drill.name = "Basic Practice";
            drill.description = "General practice drill";
            drill.instructions = "Practice your shot";
            drill.cueBallStart = cv::Point2f(320, 360);
            drill.targetBall = cv::Point2f(640, 360);
            drill.difficultyLevel = 1.0f;
            break;
    }
    
    return drill;
}

TrainingMode::ShotEvaluation TrainingMode::evaluateShot(const cv::Point2f& cueBallEnd,
                                                       const cv::Point2f& targetBallEnd,
                                                       float shotSpeed,
                                                       const std::string& shotType) {
    ShotEvaluation eval;
    eval.actualPosition = targetBallEnd;
    eval.targetPosition = currentDrill_.objectiveBall;
    eval.cueBallActual = cueBallEnd;
    eval.cueBallTarget = currentDrill_.idealCueBallEnd;
    
    // Calculate accuracy based on distance to target
    float targetDistance = calculateDistance(targetBallEnd, currentDrill_.objectiveBall);
    eval.accuracyScore = std::max(0.0f, 1.0f - (targetDistance / 100.0f)); // 100 pixel tolerance
    
    // Calculate position score for cue ball
    float positionDistance = calculateDistance(cueBallEnd, currentDrill_.idealCueBallEnd);
    eval.positionScore = std::max(0.0f, 1.0f - (positionDistance / 150.0f)); // 150 pixel tolerance
    
    // Speed evaluation (ideal speed is drill-dependent)
    float idealSpeed = 50.0f; // Default ideal speed
    float speedDiff = std::abs(shotSpeed - idealSpeed);
    eval.speedScore = std::max(0.0f, 1.0f - (speedDiff / 50.0f));
    
    // Overall score (weighted combination)
    eval.overallScore = (eval.accuracyScore * 0.5f) + 
                       (eval.positionScore * 0.3f) + 
                       (eval.speedScore * 0.2f);
    
    // Determine success (overall score > 0.6)
    eval.successful = eval.overallScore > 0.6f;
    
    // Generate feedback
    eval.feedback = generateFeedback(eval);
    
    return eval;
}

void TrainingMode::saveShotAttempt(const ShotEvaluation& eval) {
    // Create a shot record for the database
    ShotRecord shot;
    shot.playerId = playerId_;
    shot.sessionId = replaySessionId_; // If we have an associated session
    shot.shotType = exerciseTypeToString(currentExercise_);
    shot.successful = eval.successful;
    shot.ballX = eval.actualPosition.x;
    shot.ballY = eval.actualPosition.y;
    shot.targetX = eval.targetPosition.x;
    shot.targetY = eval.targetPosition.y;
    shot.shotSpeed = eval.speedScore * 100.0f; // Convert back to speed estimate
    
    database_.addShot(shot);
}

void TrainingMode::loadTrainingHistory() {
    // Load previous attempts for this exercise type
    auto shots = database_.getPlayerShots(playerId_);
    
    // Filter shots by exercise type and populate drill history
    std::string exerciseString = exerciseTypeToString(currentExercise_);
    for (const auto& shot : shots) {
        if (shot.shotType == exerciseString) {
            // Convert shot record back to evaluation format
            ShotEvaluation eval;
            eval.successful = shot.successful;
            eval.actualPosition = cv::Point2f(shot.ballX, shot.ballY);
            eval.targetPosition = cv::Point2f(shot.targetX, shot.targetY);
            // Note: Other fields would need to be stored/retrieved as well
            
            currentDrill_.history.push_back(eval);
            currentDrill_.attempts++;
            if (eval.successful) {
                currentDrill_.successes++;
            }
        }
    }
}

float TrainingMode::calculateImprovementRate() const {
    if (currentDrill_.history.size() < 5) return 0.0f; // Need at least 5 attempts
    
    // Compare average of first 5 attempts with last 5 attempts
    size_t historySize = currentDrill_.history.size();
    float earlyAverage = 0.0f;
    float recentAverage = 0.0f;
    
    for (size_t i = 0; i < 5; ++i) {
        earlyAverage += currentDrill_.history[i].overallScore;
        recentAverage += currentDrill_.history[historySize - 5 + i].overallScore;
    }
    
    earlyAverage /= 5.0f;
    recentAverage /= 5.0f;
    
    return (recentAverage - earlyAverage) * 100.0f; // Return as percentage improvement
}

void TrainingMode::renderDrillInfo(cv::Mat& frame) {
    drillInfoRect_ = cv::Rect(40, 120, 600, 200);
    UITheme::drawCard(frame, drillInfoRect_, UITheme::ComponentState::Normal, UITheme::Colors::MediumBg, 180);
    
    // Title
    cv::putText(frame, currentDrill_.name, cv::Point(drillInfoRect_.x + 20, drillInfoRect_.y + 40),
               UITheme::Fonts::FontFaceBold, UITheme::Fonts::HeadingSize,
               UITheme::Colors::NeonCyan, UITheme::Fonts::HeadingThickness);
    
    // Description
    cv::putText(frame, currentDrill_.description, cv::Point(drillInfoRect_.x + 20, drillInfoRect_.y + 80),
               UITheme::Fonts::FontFace, UITheme::Fonts::BodySize,
               UITheme::Colors::TextPrimary, UITheme::Fonts::BodyThickness);
    
    // Instructions
    cv::putText(frame, currentDrill_.instructions, cv::Point(drillInfoRect_.x + 20, drillInfoRect_.y + 120),
               UITheme::Fonts::FontFace, UITheme::Fonts::BodySize,
               UITheme::Colors::NeonGreen, UITheme::Fonts::BodyThickness);
    
    // Difficulty
    std::stringstream diff;
    diff << "Difficulty: " << std::fixed << std::setprecision(1) << currentDrill_.difficultyLevel << "/5.0";
    cv::putText(frame, diff.str(), cv::Point(drillInfoRect_.x + 20, drillInfoRect_.y + 160),
               UITheme::Fonts::FontFace, UITheme::Fonts::BodySize,
               UITheme::Colors::NeonOrange, UITheme::Fonts::BodyThickness);
}

void TrainingMode::renderSessionStats(cv::Mat& frame) {
    auto stats = getSessionStats();
    
    statsRect_ = cv::Rect(680, 120, 560, 200);
    UITheme::drawCard(frame, statsRect_, UITheme::ComponentState::Normal, UITheme::Colors::MediumBg, 180);
    
    // Title
    cv::putText(frame, "Session Statistics", cv::Point(statsRect_.x + 20, statsRect_.y + 40),
               UITheme::Fonts::FontFaceBold, UITheme::Fonts::BodySize,
               UITheme::Colors::TextPrimary, UITheme::Fonts::ButtonThickness);
    
    // Attempts
    std::string attemptsText = "Attempts: " + std::to_string(stats.totalAttempts);
    cv::putText(frame, attemptsText, cv::Point(statsRect_.x + 20, statsRect_.y + 80),
               UITheme::Fonts::FontFace, UITheme::Fonts::BodySize,
               UITheme::Colors::TextSecondary, UITheme::Fonts::BodyThickness);
    
    // Success rate
    float successRate = stats.totalAttempts > 0 ? 
                       (float)stats.successfulAttempts / stats.totalAttempts * 100.0f : 0.0f;
    std::stringstream success;
    success << "Success: " << std::fixed << std::setprecision(1) << successRate << "%";
    cv::putText(frame, success.str(), cv::Point(statsRect_.x + 20, statsRect_.y + 110),
               UITheme::Fonts::FontFace, UITheme::Fonts::BodySize,
               UITheme::Colors::NeonGreen, UITheme::Fonts::BodyThickness);
    
    // Best score
    std::stringstream best;
    best << "Best Score: " << std::fixed << std::setprecision(1) << (stats.bestScore * 100.0f) << "%";
    cv::putText(frame, best.str(), cv::Point(statsRect_.x + 20, statsRect_.y + 140),
               UITheme::Fonts::FontFace, UITheme::Fonts::BodySize,
               UITheme::Colors::NeonCyan, UITheme::Fonts::BodyThickness);
    
    // Improvement
    if (stats.improvementRate != 0.0f) {
        std::stringstream improvement;
        improvement << "Improvement: " << std::fixed << std::setprecision(1) 
                   << stats.improvementRate << "%";
        cv::Scalar improvementColor = stats.improvementRate > 0 ? 
                                     UITheme::Colors::NeonGreen : UITheme::Colors::NeonRed;
        cv::putText(frame, improvement.str(), cv::Point(statsRect_.x + 20, statsRect_.y + 170),
                   UITheme::Fonts::FontFace, UITheme::Fonts::BodySize,
                   improvementColor, UITheme::Fonts::BodyThickness);
    }
}

void TrainingMode::renderControls(cv::Mat& frame) {
    UITheme::drawCard(frame, controlsRect_, UITheme::ComponentState::Normal, UITheme::Colors::DarkBg, 200);
    
    clickableAreas_.clear();
    
    int x = 40;
    int y = controlsRect_.y + 20;
    int buttonWidth = 100;
    int buttonHeight = 40;
    int spacing = 20;
    
    // State-dependent buttons
    switch (sessionState_) {
        case SessionState::Ready:
            {
                cv::Rect startButton(x, y, buttonWidth, buttonHeight);
                UITheme::drawButton(frame, "Start", startButton, UITheme::ComponentState::Normal);
                clickableAreas_.push_back(startButton);
                x += buttonWidth + spacing;
            }
            break;
            
        case SessionState::Reviewing:
            {
                cv::Rect replayButton(x, y, buttonWidth, buttonHeight);
                UITheme::drawButton(frame, "Replay", replayButton, UITheme::ComponentState::Normal);
                clickableAreas_.push_back(replayButton);
                x += buttonWidth + spacing;
                
                cv::Rect nextButton(x, y, buttonWidth, buttonHeight);
                UITheme::drawButton(frame, "Next", nextButton, UITheme::ComponentState::Normal);
                clickableAreas_.push_back(nextButton);
                x += buttonWidth + spacing;
            }
            break;
    }
    
    // Always available buttons
    cv::Rect endButton(frame.cols - 140, y, buttonWidth, buttonHeight);
    UITheme::drawButton(frame, "End Session", endButton, UITheme::ComponentState::Normal);
    clickableAreas_.push_back(endButton);
}

std::string TrainingMode::generateFeedback(const ShotEvaluation& eval) const {
    std::stringstream feedback;
    
    if (eval.successful) {
        feedback << "Great shot! ";
    } else {
        feedback << "Keep practicing! ";
    }
    
    if (eval.accuracyScore < 0.5f) {
        feedback << "Focus on aim and target alignment. ";
    }
    
    if (eval.positionScore < 0.5f) {
        feedback << "Work on cue ball control and position play. ";
    }
    
    if (eval.speedScore < 0.5f) {
        feedback << "Adjust your stroke speed. ";
    }
    
    return feedback.str();
}

float TrainingMode::calculateDistance(const cv::Point2f& p1, const cv::Point2f& p2) const {
    float dx = p1.x - p2.x;
    float dy = p1.y - p2.y;
    return std::sqrt(dx*dx + dy*dy);
}

std::string TrainingMode::exerciseTypeToString(ExerciseType type) {
    switch (type) {
        case ExerciseType::CueBallControl: return "CueBallControl";
        case ExerciseType::BankShots: return "BankShots";
        case ExerciseType::CombinationShots: return "CombinationShots";
        case ExerciseType::BreakShots: return "BreakShots";
        case ExerciseType::SafetyPlay: return "SafetyPlay";
        case ExerciseType::CutShots: return "CutShots";
        case ExerciseType::DrawShots: return "DrawShots";
        case ExerciseType::FollowShots: return "FollowShots";
        case ExerciseType::CustomDrill: return "CustomDrill";
        default: return "Unknown";
    }
}

void TrainingMode::renderExerciseSelection(cv::Mat& frame) {
    // Title
    cv::putText(frame, "Select Training Exercise", cv::Point(40, 180),
               UITheme::Fonts::FontFaceBold, UITheme::Fonts::HeadingSize,
               UITheme::Colors::NeonCyan, UITheme::Fonts::HeadingThickness);
    
    // Exercise list
    auto exercises = getAvailableExercises();
    clickableAreas_.clear();
    
    int y = 220;
    for (size_t i = 0; i < exercises.size(); ++i) {
        cv::Rect exerciseRect(40, y, 600, 50);
        UITheme::drawCard(frame, exerciseRect, UITheme::ComponentState::Normal, UITheme::Colors::MediumBg, 150);
        
        cv::putText(frame, exercises[i].second, cv::Point(exerciseRect.x + 20, exerciseRect.y + 30),
                   UITheme::Fonts::FontFace, UITheme::Fonts::BodySize,
                   UITheme::Colors::TextPrimary, UITheme::Fonts::BodyThickness);
        
        clickableAreas_.push_back(exerciseRect);
        y += 60;
    }
}

void TrainingMode::renderShotEvaluation(cv::Mat& frame) {
    // Show drill info
    renderDrillInfo(frame);
    
    // Show evaluation results
    cv::Rect evalRect(680, 120, 560, 300);
    UITheme::drawCard(frame, evalRect, UITheme::ComponentState::Normal, UITheme::Colors::MediumBg, 180);
    
    // Title
    cv::putText(frame, "Shot Evaluation", cv::Point(evalRect.x + 20, evalRect.y + 40),
               UITheme::Fonts::FontFaceBold, UITheme::Fonts::BodySize,
               UITheme::Colors::TextPrimary, UITheme::Fonts::ButtonThickness);
    
    // Scores
    int y = evalRect.y + 80;
    
    std::stringstream accuracy;
    accuracy << "Accuracy: " << std::fixed << std::setprecision(1) 
             << (lastEvaluation_.accuracyScore * 100.0f) << "%";
    cv::putText(frame, accuracy.str(), cv::Point(evalRect.x + 20, y),
               UITheme::Fonts::FontFace, UITheme::Fonts::BodySize,
               UITheme::Colors::NeonGreen, UITheme::Fonts::BodyThickness);
    y += 30;
    
    std::stringstream position;
    position << "Position: " << std::fixed << std::setprecision(1) 
             << (lastEvaluation_.positionScore * 100.0f) << "%";
    cv::putText(frame, position.str(), cv::Point(evalRect.x + 20, y),
               UITheme::Fonts::FontFace, UITheme::Fonts::BodySize,
               UITheme::Colors::NeonCyan, UITheme::Fonts::BodyThickness);
    y += 30;
    
    std::stringstream speed;
    speed << "Speed: " << std::fixed << std::setprecision(1) 
          << (lastEvaluation_.speedScore * 100.0f) << "%";
    cv::putText(frame, speed.str(), cv::Point(evalRect.x + 20, y),
               UITheme::Fonts::FontFace, UITheme::Fonts::BodySize,
               UITheme::Colors::NeonOrange, UITheme::Fonts::BodyThickness);
    y += 30;
    
    std::stringstream overall;
    overall << "Overall: " << std::fixed << std::setprecision(1) 
            << (lastEvaluation_.overallScore * 100.0f) << "%";
    cv::Scalar overallColor = lastEvaluation_.successful ? 
                             UITheme::Colors::NeonGreen : UITheme::Colors::NeonRed;
    cv::putText(frame, overall.str(), cv::Point(evalRect.x + 20, y),
               UITheme::Fonts::FontFaceBold, UITheme::Fonts::BodySize,
               overallColor, UITheme::Fonts::ButtonThickness);
    y += 50;
    
    // Feedback
    cv::putText(frame, "Feedback:", cv::Point(evalRect.x + 20, y),
               UITheme::Fonts::FontFaceBold, UITheme::Fonts::BodySize,
               UITheme::Colors::TextPrimary, UITheme::Fonts::BodyThickness);
    y += 30;
    
    // Break feedback into lines (simple word wrapping)
    std::string feedback = lastEvaluation_.feedback;
    if (!feedback.empty()) {
        cv::putText(frame, feedback, cv::Point(evalRect.x + 20, y),
                   UITheme::Fonts::FontFace, UITheme::Fonts::SmallSize,
                   UITheme::Colors::TextSecondary, 1);
    }
}

void TrainingMode::renderIdealShot(cv::Mat& frame) {
    // Draw ideal shot path visualization
    cv::line(frame, currentDrill_.cueBallStart, currentDrill_.targetBall,
            UITheme::Colors::NeonGreen, 2, cv::LINE_AA);
    
    if (currentDrill_.type == ExerciseType::BankShots || 
        currentDrill_.type == ExerciseType::CombinationShots) {
        cv::line(frame, currentDrill_.targetBall, currentDrill_.objectiveBall,
                UITheme::Colors::NeonGreen, 1, cv::LINE_AA);
    }
    
    // Mark ideal cue ball ending position
    cv::circle(frame, currentDrill_.idealCueBallEnd, 20,
              UITheme::Colors::NeonBlue, 2, cv::LINE_AA);
    cv::putText(frame, "IDEAL", 
               cv::Point(currentDrill_.idealCueBallEnd.x - 15, currentDrill_.idealCueBallEnd.y + 5),
               UITheme::Fonts::FontFace, 0.3, UITheme::Colors::NeonBlue, 1);
}

void TrainingMode::renderTargetZones(cv::Mat& frame) {
    // Draw target zone for cue ball positioning
    cv::circle(frame, currentDrill_.idealCueBallEnd, 50,
              UITheme::Colors::NeonBlue, 1, cv::LINE_AA);
    
    // Draw objective target (pocket or position)
    if (currentDrill_.type == ExerciseType::BankShots || 
        currentDrill_.type == ExerciseType::CutShots) {
        cv::circle(frame, currentDrill_.objectiveBall, 30,
                  UITheme::Colors::NeonYellow, 2, cv::LINE_AA);
        cv::putText(frame, "TARGET", 
                   cv::Point(currentDrill_.objectiveBall.x - 20, currentDrill_.objectiveBall.y + 5),
                   UITheme::Fonts::FontFace, 0.3, UITheme::Colors::NeonYellow, 1);
    }
}

void TrainingMode::handleButtonClick(int buttonIndex) {
    switch (sessionState_) {
        case SessionState::Setup:
            // Button corresponds to exercise selection
            if (buttonIndex < getAvailableExercises().size()) {
                auto exercises = getAvailableExercises();
                startSession(exercises[buttonIndex].first, playerId_);
            }
            break;
            
        case SessionState::Ready:
            if (buttonIndex == 0) { // Start button
                sessionState_ = SessionState::Aiming;
            } else if (buttonIndex == 1) { // End session button
                endSession();
            }
            break;
            
        case SessionState::Reviewing:
            if (buttonIndex == 0) { // Replay button
                showInstantReplay();
            } else if (buttonIndex == 1) { // Next button
                sessionState_ = SessionState::Ready;
            } else if (buttonIndex == 2) { // End session button
                endSession();
            }
            break;
            
        default:
            // Last button is always end session
            if (buttonIndex == clickableAreas_.size() - 1) {
                endSession();
            }
            break;
    }
}
